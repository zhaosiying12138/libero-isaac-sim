#!/usr/bin/env python
"""dt 收敛性测试：同一动作序列在 dt=1/500 与 dt=1/1000 下各跑一遍，比较终局位姿。

用法（IsaacLab 仓目录）::

    OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python -u \
        /home/zhaosiying/codebase/libero-isaac-sim/scripts/dt_convergence.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/home/zhaosiying/codebase/libero-isaac-sim/source")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
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


def run_once(dt: float, decimation: int) -> np.ndarray:
    cfg = make_libero_env_cfg(TASK, CACHE)
    cfg.sim.dt = dt
    cfg.decimation = decimation
    env = ManagerBasedRLEnv(cfg=cfg)
    term = env.cfg.events.reset_to_init
    term.params["init_state_id"] = 0
    term.params["cycle"] = False
    env.reset()
    robot = env.scene["robot"]
    ee_idx = robot.find_bodies("robot0_right_hand")[0][0]
    origin = env.scene.env_origins[0].cpu().numpy()
    # 标准动作探针：+z 0.3 走 20 步
    action = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]])
    for _ in range(20):
        env.step(action)
    pose = robot.data.body_pose_w.torch[0, ee_idx].cpu().numpy()
    env.close()
    return pose[:3] - origin, pose[3:7]


def main():
    results = {}
    for dt, dec in [(1 / 500, 25), (1 / 1000, 50)]:
        pos, quat = run_once(dt, dec)
        results[f"dt={dt:.5f}"] = {"pos": pos.tolist(), "quat_xyzw": quat.tolist()}
        print(f"[dt] dt={dt:.5f}: EE 终局 pos={np.round(pos, 4)}")

    p1 = np.array(results["dt=0.00200"]["pos"])
    p2 = np.array(results["dt=0.00100"]["pos"])
    drift_mm = float(np.linalg.norm(p1 - p2) * 1000)
    results["drift_mm"] = drift_mm
    print(f"[dt] dt 减半终局漂移: {drift_mm:.1f} mm")

    out = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "dt_convergence.json"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[dt] -> {out}")
    app.close()


if __name__ == "__main__":
    main()
