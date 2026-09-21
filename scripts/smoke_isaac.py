#!/usr/bin/env python
"""Isaac Lab 3.0 EA 冒烟测试 v2：自转 Panda USD spawn + 位置控制 + 相机渲染。

用法（在 IsaacLab 仓目录下）:
    OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python -u \
        /home/zhaosiying/codebase/libero-isaac-sim/scripts/smoke_isaac.py
"""

import argparse
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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass

CACHE = os.path.expanduser("~/.cache/libero_isaac_sim")
ROBOT_USD = os.path.join(
    CACHE, "usd/robots/panda_on_ground/panda_on_ground/panda_on_ground.usda"
)

INIT_QPOS = [0, -0.161037389, 0.0, -2.44459747, 0.0, 2.22675220, 0.785398]


@configclass
class SmokeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.02),
            visible=True,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.01)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.51, 0.0, 0.42),
            joint_pos={f"robot0_joint{i+1}": INIT_QPOS[i] for i in range(7)},
        ),
        actuators={
            "panda_arm": ImplicitActuatorCfg(
                joint_names_expr=["robot0_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper0_finger_joint[12]"],
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(SmokeSceneCfg(num_envs=4, env_spacing=2.0))

    sim.reset()
    scene.reset()
    print("[smoke] sim reset OK")

    robot = scene["robot"]
    print("[smoke] joints:", robot.data.joint_names)
    print("[smoke] joint_pos:", [round(v, 3) for v in robot.data.joint_pos.torch[0].tolist()])

    for i in range(120):
        robot.set_joint_position_target(robot.data.default_joint_pos.torch.clone())
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_cfg.dt)
        scene.update(sim_cfg.dt)

    hand_idx = robot.find_bodies("robot0_right_hand")[0][0]
    hand_pos = robot.data.body_pose_w.torch[0, hand_idx, :3]
    print("[smoke] panda_hand pos:", [round(float(v), 3) for v in hand_pos])
    app.close()
    print("[smoke] OK")


if __name__ == "__main__":
    main()
