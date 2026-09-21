"""相机对齐事件：startup 时把 manifest 的相机位姿直接写入 USD prim。

背景：Isaac Lab 3.0 EA 的 CameraCfg.OffsetCfg 写路径会把手性矩阵转置
（实测：输入 R 上屏后读回 R.T）。本事件在 startup 一次性用 pxr 直接写
世界变换矩阵，绕开 OffsetCfg 路径。

- agentview：世界固定相机，写世界矩阵；
- robot0_eye_in_hand：挂在 robot0_right_hand 下，写父系局部变换。

约定：MuJoCo 相机帧 == USD 相机帧（-Z 朝前、+Y 朝上），直接写 manifest rotmat。
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs import ManagerBasedEnv

from libero_isaac_sim.semantics.task_spec import load_task


def _write_world_matrix(stage, prim_path: str, mat4: np.ndarray) -> bool:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return False
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op = xf.AddTransformOp()
    op.Set(Gf.Matrix4d(*[float(v) for v in mat4.reshape(-1)]))
    return True


def align_cameras_to_manifest(
    env: ManagerBasedEnv, env_ids: torch.Tensor, task_name: str, cache_dir: str
) -> None:
    """startup 事件：按 manifest 写相机位姿。"""
    task = load_task(task_name, cache_dir)
    stage = env.scene.stage

    # agentview：世界固定
    cam = task.cameras["agentview"]
    m = np.eye(4)
    m[:3, :3] = np.asarray(cam["rotmat"]).reshape(3, 3)
    m[:3, 3] = cam["pos"]
    ok = False
    for cand in ["{ENV_REGEX_NS}/agentview", "/World/agentview"]:
        path = cand.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        ok = _write_world_matrix(stage, path, m) or ok
    print(f"[cameras] agentview 世界矩阵写入: {'OK' if ok else 'MISSING'}")

    # 腕部相机：局部变换（相对 robot0_right_hand prim）
    wrist = task.cameras["robot0_eye_in_hand"]
    m = np.eye(4)
    m[:3, :3] = np.asarray(wrist["rotmat"]).reshape(3, 3)
    m[:3, 3] = wrist["pos"]
    parent = wrist["body"]  # robot0_right_hand
    ok = False
    for cand in [
        f"/World/envs/env_0/Robot/Geometry/robot0_base/robot0_link0/robot0_link1/robot0_link2/robot0_link3/robot0_link4/robot0_link5/robot0_link6/robot0_link7/robot0_right_hand/robot0_eye_in_hand",
        f"/World/envs/env_0/Robot/robot0_right_hand/robot0_eye_in_hand",
    ]:
        ok = _write_world_matrix(stage, cand, m) or ok  # 注意：挂接 prim 的局部变换
    print(f"[cameras] wrist 局部矩阵写入: {'OK' if ok else 'MISSING'}")
