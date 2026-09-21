"""旋转与位姿的数学工具库：所有约定显式化。

项目内统一约定：
- 模块间传递旋转一律用 3x3 旋转矩阵（行优先）；
- 四元数只在边界处出现，且必须显式标注 wxyz（MuJoCo/Isaac Lab 内部）
  或 xyzw（robosuite obs / 部分 USD API）；
- 轴角向量 direction=轴、norm=角度（rad）。

本模块不依赖 Isaac/MuJoCo，可独立单元测试。
"""

from __future__ import annotations

import numpy as np


def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return quat_wxyz_to_mat([w, x, y, z])


def mat_to_quat_wxyz(m) -> np.ndarray:
    m = np.asarray(m, dtype=float).reshape(3, 3)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        return np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ]
        )
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    if i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array(
        [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    )


def mat_to_quat_xyzw(m) -> np.ndarray:
    w, x, y, z = mat_to_quat_wxyz(m)
    return np.array([x, y, z, w])


def axisangle_to_mat(aa) -> np.ndarray:
    """轴角（方向=轴，模长=角度 rad）转旋转矩阵（Rodrigues）。"""
    aa = np.asarray(aa, dtype=float)
    angle = np.linalg.norm(aa)
    if angle < 1e-12:
        return np.eye(3)
    axis = aa / angle
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def mat_to_axisangle(m) -> np.ndarray:
    """旋转矩阵转轴角向量。"""
    m = np.asarray(m, dtype=float).reshape(3, 3)
    cos_angle = np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # 180 度附近的病态分支：取对角线最大者定轴
        axis = np.sqrt(np.clip(np.diag(m) + 1.0, 0, None)) / np.sqrt(2.0)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        # 修正符号
        if m[2, 1] - m[1, 2] < 0:
            axis[0] = -axis[0]
        if m[0, 2] - m[2, 0] < 0:
            axis[1] = -axis[1]
        if m[1, 0] - m[0, 1] < 0:
            axis[2] = -axis[2]
        return axis * angle
    axis = np.array(
        [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]
    ) / (2.0 * np.sin(angle))
    return axis * angle


def pose_to_mat(pos, rotmat) -> np.ndarray:
    """位置 + 旋转矩阵 → 4x4 齐次变换。"""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotmat, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def mat_to_pose(T) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    return T[:3, 3].copy(), T[:3, :3].copy()
