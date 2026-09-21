#!/usr/bin/env python
"""OSC 直驱探针：绕过 ManagerBased 动作管道，手工驱动 OperationalSpaceController。

命令：当前 EE 位姿 + z 方向 5cm 的绝对目标，保持 2 秒（240 物理步），
逐 20 步打印 EE 位置。用于判定漂移来自控制器/资产还是动作管道。

用法（IsaacLab 仓目录）::

    OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python -u \
        /home/zhaosiying/codebase/libero-isaac-sim/scripts/osc_direct_probe.py
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

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from libero_isaac_sim.envs.libero_scene_cfg import build_scene_cfg
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

CACHE = os.path.expanduser("~/.cache/libero_isaac_sim")
ROBOT_USD = os.path.join(
    CACHE, "usd/robots/panda_on_ground/panda_on_ground/panda_on_ground.usda"
)
INIT_QPOS = [0, -0.161037389, 0.0, -2.44459747, 0.0, 2.22675220, 0.785398]


FULL_SCENE = os.environ.get("LIBERO_TEST_FULL_SCENE", "0") == "1"

@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg()),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=ROBOT_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.51, 0.0, 0.42),
            joint_pos={f"robot0_joint{i+1}": INIT_QPOS[i] for i in range(7)},
        ),
        actuators={
            # OSC 力矩通道：stiffness=0/damping=0 时隐式执行器直通 effort target
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["robot0_joint[1-7]"], stiffness=0.0, damping=0.0
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper0_finger_joint[12]"],
                stiffness=2000.0, damping=100.0,
            ),
        },
    )


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
    if FULL_SCENE:
        from libero_isaac_sim.envs.libero_env_cfg import make_libero_env_cfg
        from libero_isaac_sim.semantics.task_spec import load_task
        from libero_isaac_sim.converters.asset_manifest import AssetRegistry
        TASK = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
        task = load_task(TASK, CACHE)
        scene_cfg = build_scene_cfg(task, AssetRegistry(CACHE), num_envs=1)
        scene = InteractiveScene(scene_cfg)
        print("[osc] full LIBERO scene")
    else:
        scene = InteractiveScene(ProbeSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    # InitialStateCfg.joint_pos 对该转换 USD 不生效（疑似 EA 版本问题），
    # 显式写关节态（与 LIBERO reset 协议一致）
    joint_pos = robot.data.joint_pos.torch[0].clone()
    arm_ids_w = robot.find_joints(["robot0_joint[1-7]"])[0]
    joint_pos[arm_ids_w] = torch.tensor(INIT_QPOS, dtype=torch.float32)
    robot.write_joint_state_to_sim(
        joint_pos.unsqueeze(0), torch.zeros_like(joint_pos).unsqueeze(0),
        env_ids=torch.tensor([0]),
    )
    print("[osc] joints after explicit write:", np.round(robot.data.joint_pos.torch[0, arm_ids_w].cpu().numpy(), 3))
    ee_name = "robot0_right_hand"
    ee_frame_idx = robot.find_bodies(ee_name)[0][0]
    arm_joint_ids = robot.find_joints(["robot0_joint[1-7]"])[0]
    ee_jacobi_idx = ee_frame_idx - 1
    jacobi_joint_ids = [j + robot.num_base_dofs for j in arm_joint_ids]

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=os.environ.get("LIBERO_TEST_INERTIAL", "1") == "1",
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=os.environ.get("LIBERO_TEST_GRAVCOMP", "1") == "1",
        motion_stiffness_task=float(os.environ.get("LIBERO_TEST_KP", "150.0")),
        motion_damping_ratio_task=1.0,
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],
        nullspace_control="none",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=1, device=sim.device)

    sim_dt = sim.get_physics_dt()
    robot.update(sim_dt)

    def states():
        jacobian_w = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx][:, :, jacobi_joint_ids]
        mass_matrix = robot.data.mass_matrix.torch[:, jacobi_joint_ids][:, :, jacobi_joint_ids]
        gravity = robot.data.gravity_compensation_forces.torch[:, jacobi_joint_ids]
        root_rot = matrix_from_quat(quat_inv(robot.data.root_quat_w.torch))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(root_rot, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(root_rot, jacobian_b[:, 3:, :])
        ee_pos_w = robot.data.body_pos_w.torch[:, ee_frame_idx]
        ee_quat_w = robot.data.body_quat_w.torch[:, ee_frame_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, ee_pos_w, ee_quat_w
        )
        ee_vel_w = robot.data.body_vel_w.torch[:, ee_frame_idx]
        root_vel_w = robot.data.root_vel_w.torch
        rel = ee_vel_w - root_vel_w
        ee_vel_b = torch.cat(
            [
                quat_apply_inverse(robot.data.root_quat_w.torch, rel[:, 0:3]),
                quat_apply_inverse(robot.data.root_quat_w.torch, rel[:, 3:6]),
            ],
            dim=-1,
        )
        joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]
        joint_vel = robot.data.joint_vel.torch[:, arm_joint_ids]
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
        return jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, joint_pos, joint_vel, ee_pos_w[0]

    # 目标：当前 EE + z 5cm（base 系）
    jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, joint_pos, joint_vel, ee_pos_w0 = states()
    target = ee_pose_b.clone()
    dz = float(os.environ.get("LIBERO_TEST_DZ", "0.05"))
    target[0, 2] += dz
    print(f"[osc] start ee_pos_w={np.round(ee_pos_w0.cpu().numpy(), 4)}, target z+{dz}m")

    command = torch.zeros(1, osc.action_dim)
    command[:, :7] = target
    osc.reset()
    osc.set_command(command=command, current_ee_pose_b=ee_pose_b)

    zero_efforts = torch.zeros(1, robot.num_joints)
    for step in range(240):
        jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, joint_pos, joint_vel, ee_pos_w = states()
        efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=None,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
        )
        full = zero_efforts.clone()
        full[:, arm_joint_ids] = efforts
        # 夹爪保持
        g_ids = robot.find_joints(["gripper0_finger_joint[12]"])[0]
        full[:, g_ids] = 0.0
        robot.set_joint_effort_target(full)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_dt)
        scene.update(sim_dt)
        if step % 40 == 0 or step == 239:
            print(f"[osc] step={step:3d} ee_pos_w={np.round(ee_pos_w.cpu().numpy(), 4)}")

    app.close()
    print("[osc] OK")


if __name__ == "__main__":
    main()
