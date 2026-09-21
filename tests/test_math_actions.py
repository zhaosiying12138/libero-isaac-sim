"""math_utils 与 actions_libero 的单元测试（纯 numpy，无需仿真器）。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from libero_isaac_sim.semantics.actions_libero import compute_goal_pose
from libero_isaac_sim.semantics.math_utils import (
    axisangle_to_mat,
    mat_to_axisangle,
    mat_to_quat_wxyz,
    mat_to_quat_xyzw,
    quat_wxyz_to_mat,
    quat_xyzw_to_mat,
)


def test_quat_convention_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        aa = rng.normal(size=3)
        m = axisangle_to_mat(aa)
        q_wxyz = mat_to_quat_wxyz(m)
        q_xyzw = mat_to_quat_xyzw(m)
        m2 = quat_wxyz_to_mat(q_wxyz)
        m3 = quat_xyzw_to_mat(q_xyzw)
        assert np.allclose(m, m2, atol=1e-9)
        assert np.allclose(m, m3, atol=1e-9)
        assert abs(np.linalg.det(m) - 1.0) < 1e-9


def test_axisangle_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        aa = rng.uniform(-0.4, 0.4, size=3)
        m = axisangle_to_mat(aa)
        aa2 = mat_to_axisangle(m)
        m2 = axisangle_to_mat(aa2)
        assert np.allclose(m, m2, atol=1e-9)


def test_wxyz_xyzw_distinction():
    """同一组数字按不同约定解释必须得到不同矩阵（防止约定混用漏检）。"""
    q = [0.5, 0.5, 0.5, 0.5]
    m_wxyz = quat_wxyz_to_mat(q)
    m_xyzw = quat_xyzw_to_mat(q)
    # 0.5,0.5,0.5,0.5 两约定下恰好都是 120° 轴角（对称），换个不对称的
    q2 = [0.8, 0.4, 0.4, 0.2]
    assert not np.allclose(quat_wxyz_to_mat(q2), quat_xyzw_to_mat(q2))


def test_action_pose_composition():
    pos = np.array([0.3, 0.0, 0.9])
    rot = np.eye(3)
    # 满幅 x 方向平移
    goal_pos, _, g = compute_goal_pose([1, 0, 0, 0, 0, 0, 1], pos, rot)
    assert np.allclose(goal_pos, pos + [0.05, 0, 0])
    assert g == 1.0
    # 超界 clip
    goal_pos, _, g = compute_goal_pose([5, 0, 0, 0, 0, 0, -1], pos, rot)
    assert np.allclose(goal_pos, pos + [0.05, 0, 0])
    assert g == -1.0
    # 绕世界 z 转 0.5 rad（action 旋转分量满幅）
    _, goal_rot, _ = compute_goal_pose([0, 0, 0, 0, 0, 1, 0], pos, rot)
    expected = axisangle_to_mat([0, 0, 0.5])
    assert np.allclose(goal_rot, expected, atol=1e-12)
    # 世界系左乘语义：当前姿态已绕 x 转 90° 时，再绕世界 z 转
    rot_x90 = axisangle_to_mat([np.pi / 2, 0, 0])
    _, goal_rot2, _ = compute_goal_pose([0, 0, 0, 0, 0, 1, 0], pos, rot_x90)
    assert np.allclose(goal_rot2, axisangle_to_mat([0, 0, 0.5]) @ rot_x90)


if __name__ == "__main__":
    test_quat_convention_roundtrip()
    test_axisangle_roundtrip()
    test_wxyz_xyzw_distinction()
    test_action_pose_composition()
    print("全部通过")
