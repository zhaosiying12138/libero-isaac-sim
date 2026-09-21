"""BDDL region 的采样与几何工具。

两类 region：
1. 桌面矩形 region（``:ranges`` + ``:yaw_rotation``）：reset 时在矩形内均匀采样
   xy 与偏航角，与官方 ``SequentialCompositeSampler`` 语义一致；
2. 物体附着 site region（无 ``:ranges``）：位姿由父物体的内嵌 site 决定，
   本模块只负责解析与坐标变换。

官方放置语义（bddl_base_domain._reset_internal + placement sampler）：
- fixture 与物体在各自 init region 内均匀采样（x, y, yaw）；
- z 由 ``z_offset`` 给出（桌面厚度修正），物体从略高处落下，由物理引擎沉降；
- 采样顺序按 BDDL :init 声明顺序，碰撞检测失败重试（Isaac 侧用固定初始状态
  时不需要采样，本模块服务于随机 reset 模式与对照实验）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class TableRegion:
    """桌面矩形采样区域（世界系）。"""

    name: str
    center_xy: np.ndarray  # (2,) 世界系中心
    half_size_xy: np.ndarray  # (2,) 半宽
    yaw_range: tuple[float, float]
    z: float  # 放置高度（桌面顶面 + z_offset）

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        xy = self.center_xy + rng.uniform(-1.0, 1.0, size=2) * self.half_size_xy
        yaw = rng.uniform(*self.yaw_range)
        return np.array([xy[0], xy[1], self.z]), yaw


def build_table_regions(
    regions: Dict[str, dict],
    table_keyword: str,
    workspace_offset: np.ndarray,
    table_top_z: float,
    z_offset: float,
) -> Dict[str, TableRegion]:
    """从 BDDL region 定义构建桌面采样区域（世界系）。

    参数
    ----
    regions : manifest/bddl 解析出的 region 字典（key 已带 target 前缀）
    table_keyword : 桌面 fixture 名（如 "living_room_table"），用于筛出桌面 region
    workspace_offset : 桌子世界系偏移（LIBERO 各 problem 类的 workspace_offset）
    table_top_z : 桌面顶面世界 z（桌子 offset z + 半厚度）
    z_offset : LIBERO 放置 z 修正（0.01 - 桌板厚度，见各 problem 类）
    """
    out: Dict[str, TableRegion] = {}
    for name, region in regions.items():
        if region["target"] != table_keyword:
            continue
        if not region["ranges"]:
            continue
        r = region["ranges"][0]  # (xmin, ymin, xmax, ymax)
        center_xy = np.array(
            [
                (r[2] + r[0]) / 2 + workspace_offset[0],
                (r[3] + r[1]) / 2 + workspace_offset[1],
            ]
        )
        half_xy = np.array([(r[2] - r[0]) / 2, (r[3] - r[1]) / 2])
        yaw = region.get("yaw_rotation", [0.0, 0.0])
        # 物体放置高度：桌面顶面 + z_offset（官方 z_offset = 0.01 - thickness）
        z = table_top_z + z_offset
        out[name] = TableRegion(name, center_xy, half_xy, (float(yaw[0]), float(yaw[1])), z)
    return out


def yaw_to_rotmat(yaw: float) -> np.ndarray:
    """绕世界 z 轴的偏航角转 3x3 旋转矩阵。"""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
