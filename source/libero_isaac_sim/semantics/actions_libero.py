"""LIBERO 7D 动作语义的核心数学（纯 numpy 参考实现，仿真器无关）。

robosuite OSC_POSE 的动作语义（LIBERO 默认，20 Hz）：
- 输入 7 维，逐维 clip 到 [-1, 1]；
- 位置增量：dpos = action[0:3] * 0.05（米），世界系；
- 旋转增量：action[3:6] * 0.5（弧度）解释为轴角向量，按**世界系**左乘当前
  末端姿态：``goal_R = delta_R @ R_current``；
- 夹爪：action[6]，>0 张开，<=0 闭合（robosuite PandaGripper 二值约定）。

本模块给出目标位姿的合成计算，Isaac 侧的 action term 与等价性实验
（探针序列）共用同一份实现，避免两处各写一遍导致约定漂移。

方向性约定以「探针双仿真实验」（experiments/calibrate_action_scale.py）为最终
仲裁：若实测 robosuite 的旋转增量为体坐标系右乘，则改 compose_delta_rotation
一处即可。
"""

from __future__ import annotations

import numpy as np

from libero_isaac_sim.semantics.math_utils import axisangle_to_mat

POS_SCALE = 0.05  # 米，robosuite OSC_POSE output_max/min = ±0.05
ROT_SCALE = 0.5  # 弧度，±0.5 rad
GRIPPER_OPEN_THRESHOLD = 0.0


def clip_action(action) -> np.ndarray:
    a = np.asarray(action, dtype=float).copy()
    assert a.shape == (7,), f"LIBERO 动作应为 7 维，实际 {a.shape}"
    return np.clip(a, -1.0, 1.0)


def compute_goal_pose(
    action, current_eef_pos, current_eef_rotmat
) -> tuple[np.ndarray, np.ndarray, float]:
    """把 7D LIBERO 动作合成为世界系目标位姿。

    返回 (goal_pos, goal_rotmat, gripper_cmd)，gripper_cmd ∈ {+1.0 开, -1.0 合}。
    """
    a = clip_action(action)
    dpos = a[0:3] * POS_SCALE
    drot_aa = a[3:6] * ROT_SCALE

    goal_pos = np.asarray(current_eef_pos, dtype=float) + dpos
    delta_R = axisangle_to_mat(drot_aa)
    goal_rotmat = delta_R @ np.asarray(current_eef_rotmat, dtype=float)

    gripper_cmd = 1.0 if a[6] > GRIPPER_OPEN_THRESHOLD else -1.0
    return goal_pos, goal_rotmat, gripper_cmd


def compose_delta_rotation(current_rotmat, delta_rotmat) -> np.ndarray:
    """旋转增量合成：世界系左乘（robosuite OSC_POSE 约定，探针实验仲裁）。"""
    return np.asarray(delta_rotmat) @ np.asarray(current_rotmat)
