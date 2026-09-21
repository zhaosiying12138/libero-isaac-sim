"""LIBERO schema 观测项（Isaac Lab ManagerBased mdp terms）。

对外暴露的键与 LIBERO 数据集/评测完全一致：
    agentview_image / robot0_eye_in_hand_image  —— uint8 (H, W, 3)
    robot0_joint_pos                            —— float (7,)
    robot0_gripper_qpos                         —— float (2,)
    robot0_eef_pos / robot0_eef_quat            —— float (3,) / (4,)

注意约定：
- Isaac Lab 3.0 四元数为 xyzw；``robot0_eef_quat`` 与 robosuite 观测一致
  使用 wxyz（robosuite obs 里 eef_quat 是 wxyz 约定——通过 transform_utils
  输出）。**本模块输出的 eef_quat 为 wxyz**，与 LIBERO HDF5/策略输入一致。
- 相机帧：MuJoCo 离屏帧是上下翻转的（LIBERO 下游使用时统一 flip），
  Isaac 相机输出为正常朝向，本模块不再翻转；facade 层做对齐处理。
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs import ManagerBasedEnv

from libero_isaac_sim.semantics.math_utils import mat_to_quat_wxyz, quat_xyzw_to_mat


def _robot(env: ManagerBasedEnv):
    return env.scene["robot"]


def _eef_body_index(env: ManagerBasedEnv) -> int:
    if not hasattr(env, "_libero_eef_idx"):
        env._libero_eef_idx = _robot(env).find_bodies("robot0_right_hand")[0][0]
    return env._libero_eef_idx


def robot0_joint_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """7 个手臂关节角。"""
    robot = _robot(env)
    arm_ids, _ = robot.find_joints(["robot0_joint[1-7]"])
    return robot.data.joint_pos.torch[:, arm_ids]


def robot0_gripper_qpos(env: ManagerBasedEnv) -> torch.Tensor:
    """夹爪 2 关节角。"""
    robot = _robot(env)
    g_ids, _ = robot.find_joints(["gripper0_finger_joint[12]"])
    return robot.data.joint_pos.torch[:, g_ids]


def robot0_eef_pos(env: ManagerBasedEnv) -> torch.Tensor:
    robot = _robot(env)
    return robot.data.body_pose_w.torch[:, _eef_body_index(env), 0:3]


def robot0_eef_quat(env: ManagerBasedEnv) -> torch.Tensor:
    """wxyz 约定（与 LIBERO obs 一致）。"""
    robot = _robot(env)
    quat_xyzw = robot.data.body_pose_w.torch[:, _eef_body_index(env), 3:7]
    w = quat_xyzw[:, 3:4]
    return torch.cat([w, quat_xyzw[:, 0:3]], dim=-1)


def agentview_image(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["agentview"].data.output["rgb"]


def eye_in_hand_image(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot0_eye_in_hand"].data.output["rgb"]
