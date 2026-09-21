#!/usr/bin/env python
"""任务 0 环境端到端冒烟：建 env → reset（装载 init state）→ settle → 渲染 + 打印状态。

用法（IsaacLab 仓目录）::

    OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python -u \
        /home/zhaosiying/codebase/libero-isaac-sim/scripts/env_smoke.py
"""

import argparse
import os
import sys

sys.path.insert(0, "/home/zhaosiying/codebase/libero-isaac-sim/source")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
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
from libero_isaac_sim.semantics.task_spec import load_task

CACHE = os.path.expanduser("~/.cache/libero_isaac_sim")
TASK = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"


def main():
    env_cfg = make_libero_env_cfg(TASK, CACHE)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("[env] created OK")

    obs, info = env.reset()
    print("[env] reset OK; obs groups:", list(obs.keys()))
    for k, v in obs["policy"].items():
        print(f"[env] policy.{k}:", np.round(v[0].cpu().numpy(), 3))

    # settle 5 步零动作（与官方协议一致）
    zero = torch.zeros((1, 7), device="cpu")
    for _ in range(5):
        obs, rew, terminated, truncated, info = env.step(zero)
    print("[env] settled; success terminated =", bool(terminated[0]))

    # 渲染 agentview
    cam = env.unwrapped.scene["agentview"]
    rgb = cam.data.output["rgb"][0].cpu().numpy()
    import imageio

    imageio.imwrite("/tmp/isaac_task0_agentview.png", rgb)
    wrist = env.unwrapped.scene["robot0_eye_in_hand"].data.output["rgb"][0].cpu().numpy()
    imageio.imwrite("/tmp/isaac_task0_wrist.png", wrist)
    print("[env] saved /tmp/isaac_task0_agentview.png + wrist.png")

    # 与 MuJoCo 语义状态对比（init_state 0）
    task = load_task(TASK, CACHE)
    s0 = task.get_init_state(args.init_state_id)
    for name in ["alphabet_soup_1", "basket_1"]:
        asset = env.unwrapped.scene[name]
        pos = asset.data.root_pose_w.torch[0, :3].cpu().numpy()
        origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
        pos = pos - origin
        print(f"[env] {name} isaac={np.round(pos, 3)} mujoco={np.round(s0.entities[name].pos, 3)}")

    env.close()
    app.close()
    print("[env] OK")


if __name__ == "__main__":
    main()
