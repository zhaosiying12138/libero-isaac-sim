#!/usr/bin/env python
"""Isaac Lab 3.0 EA 冒烟测试：headless 起仿真、spawn Franka、步进、渲染一帧相机。

用法（在 IsaacLab 仓目录下）:
    OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python \
        /home/zhaosiying/codebase/libero-isaac-sim/scripts/smoke_isaac.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_assets import FRANKA_PANDA_CFG


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
    )

    robot = Articulation(FRANKA_PANDA_CFG.replace(prim_path="/World/Franka"))

    cam_cfg = CameraCfg(
        prim_path="/World/agentview",
        update_period=0,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
    )
    camera = Camera(cam_cfg)

    sim.reset()
    print("[smoke] sim reset OK; franka joints:", robot.data.joint_pos.shape)

    for i in range(60):
        robot.set_joint_position_target(
            torch.zeros_like(robot.data.joint_pos)
        )
        robot.write_data_to_sim()
        sim.step()
    robot.update(sim_cfg.dt)
    camera.update(sim_cfg.dt)

    rgb = camera.data.output["rgb"][0].cpu().numpy()
    print("[smoke] camera rgb:", rgb.shape, rgb.dtype)
    import imageio

    imageio.imwrite("/tmp/isaac_smoke_agentview.png", rgb)
    print("[smoke] saved /tmp/isaac_smoke_agentview.png")
    app.close()


if __name__ == "__main__":
    main()
