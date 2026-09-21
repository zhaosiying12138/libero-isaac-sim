#!/usr/bin/env python
"""Isaac 相机投影探针：把世界点投到 agentview 像素坐标。

用法: ... python project_isaac_camera.py --points '{"name": [x,y,z], ...}'
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/home/zhaosiying/codebase/libero-isaac-sim/source")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--points", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from libero_isaac_sim.compat import isaaclab_ea_fixes

isaaclab_ea_fixes.apply()

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from libero_isaac_sim.envs.libero_env_cfg import make_libero_env_cfg

CACHE = os.path.expanduser("~/.cache/libero_isaac_sim")
TASK = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"


def main():
    points = json.loads(args.points)
    env = ManagerBasedRLEnv(cfg=make_libero_env_cfg(TASK, CACHE))
    env.reset()

    cam = env.unwrapped.scene["agentview"]
    if os.environ.get("LIBERO_CAM_DIRECT_WRITE") == "1":
        # 直接把 MuJoCo 相机世界位姿写进 USD prim（opengl=USD 相机约定）
        from pxr import Gf, UsdGeom
        from libero_isaac_sim.semantics.task_spec import load_task
        task = load_task(TASK, CACHE)
        camdef = task.cameras["agentview"]
        stage = env.unwrapped.scene.stage
        prim = None
        for cand in ["/World/envs/env_0/agentview", "/World/agentview"]:
            prim = stage.GetPrimAtPath(cand)
            if prim:
                break
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        m = np.eye(4)
        m[:3, :3] = np.asarray(camdef["rotmat"]).reshape(3, 3)
        m[:3, 3] = camdef["pos"]
        op = xf.AddTransformOp()
        op.Set(Gf.Matrix4d(*[float(v) for v in m.reshape(-1)]))
        print("[cam] 直接写入世界矩阵")
        # 让传感器重新读位姿
        env.sim.step(render=True)
        env.scene.update(1/120)
    # 世界位姿 + 内参
    pos_w = cam.data.pos_w.torch[0].cpu().numpy()
    # 直接从 USD prim 读世界变换（不信 sensor data 属性的约定）
    from pxr import UsdGeom
    stage = env.unwrapped.scene.stage
    prim = stage.GetPrimAtPath("/World/envs/env_0/agentview")
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    M = np.array([[m[i][j] for j in range(4)] for i in range(4)])
    pos_w = M[:3, 3]
    R = M[:3, :3]
    print("[cam] USD prim 世界矩阵:\n", np.round(M, 4))
    K = cam.data.intrinsic_matrices.torch[0].cpu().numpy()

    out = {"_cam_pos_w": np.asarray(pos_w).tolist(), "_cam_matrix": M.tolist(),
           "_K": K.tolist()}

    # R 已在上面从 USD 矩阵读出
    W = H = 128
    for name, pt in points.items():
        p_cam = R.T @ (np.array(pt, dtype=float) - pos_w)
        if abs(p_cam[2]) < 1e-9:
            out[name] = [float("nan"), float("nan")]
            continue
        u = K[0, 2] + K[0, 0] * p_cam[0] / (-p_cam[2])
        v = K[1, 2] - K[1, 1] * p_cam[1] / (-p_cam[2])
        out[name] = [float(u), float(v)]

    print("__PROJ_JSON__" + json.dumps(out))
    env.close()
    app.close()


if __name__ == "__main__":
    main()
