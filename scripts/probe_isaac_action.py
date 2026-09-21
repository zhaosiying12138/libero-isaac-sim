#!/usr/bin/env python
"""Isaac 侧批量动作探针：一次进程跑多组动作，输出每组 EE 轨迹。

--actions-json: JSON 字符串，{"probe名": [7维动作], ...}
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/home/zhaosiying/codebase/libero-isaac-sim/source")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--actions-json", type=str, required=True)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--init-state-id", type=int, default=0)
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
    probes = json.loads(args.actions_json)
    env_cfg = make_libero_env_cfg(TASK, CACHE)
    env = ManagerBasedRLEnv(cfg=env_cfg)

    origin = env.scene.env_origins[0].cpu().numpy()
    robot = env.scene["robot"]
    ee_idx = robot.find_bodies("robot0_right_hand")[0][0]

    results = {}
    for name, act in probes.items():
        env.reset()
        action = torch.tensor([act], dtype=torch.float32)
        traj = []
        for _ in range(args.steps):
            env.step(action)
            pose = robot.data.body_pose_w.torch[0, ee_idx].cpu().numpy()
            traj.append({
                "pos": (pose[:3] - origin).tolist(),
                "quat_xyzw": pose[3:7].tolist(),
            })
        results[name] = traj

    print("__PROBE_JSON__" + json.dumps({"probes": results, "dt_control": 0.05}))
    env.close()
    app.close()


if __name__ == "__main__":
    main()
