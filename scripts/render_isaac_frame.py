#!/usr/bin/env python
"""渲染 Isaac 侧单帧（agentview + wrist），供对比脚本子进程调用。"""

import argparse
import os
import sys

sys.path.insert(0, "/home/zhaosiying/codebase/libero-isaac-sim/source")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--init-state-id", type=int, default=0)
parser.add_argument("--out", type=str, required=True)
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
    env_cfg = make_libero_env_cfg(TASK, CACHE)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    zero = torch.zeros((1, 7))
    for _ in range(5):
        env.step(zero)

    import imageio

    cam = env.unwrapped.scene["agentview"]
    rgb = cam.data.output["rgb"][0].cpu().numpy()
    imageio.imwrite(args.out, rgb)
    if "normals" in cam.data.output:
        nrm = cam.data.output["normals"][0].cpu().numpy()
        nrm = ((nrm * 0.5 + 0.5) * 255).astype("uint8")
        imageio.imwrite(args.out.replace(".png", "_normals.png"), nrm)
    wrist = env.unwrapped.scene["robot0_eye_in_hand"].data.output["rgb"][0].cpu().numpy()
    imageio.imwrite(args.out.replace("agentview", "wrist"), wrist)
    print(f"[render] saved {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
