"""BDDL 谓词的仿真器无关实现（numpy 参考实现）。

语义逐条对齐 LIBERO 官方实现（libero/libero/envs/predicates/base_predicates.py
与 object_states/base_object_states.py），但操作在抽象的「语义状态视图」上，
使同一份代码可以：
  a) 在 Isaac Lab 环境内对 PhysX 状态求值（评测/奖励/终止）；
  b) 在双仿真对比实验中，对 MuJoCo 导出的语义状态求值，与官方谓词逐帧比对。

谓词语义（与官方逐行对应）：
- In(a, region):   region.check_contact(a)（site 恒真）and region.check_contain(a)
                   —— 点在盒内：total_size = |R @ size|，pos±total_size，z 下限放宽 0.01
- On(a, b):        b 是物体：b.z <= a.z 且 contact(a,b) 且 |b.xy - a.xy| < 0.03
                   b 是 site：site.under(a)（父物体存在时还要求父物体与 a 接触）
- Open/Close/Turnon/Turnoff: 关节角落在 manifest 导出的官方阈值区间内
- Up(a):           a.z >= 1.0

contact 的判定由状态视图提供（Isaac 侧用 PhysX 接触报告，测试侧可用几何近似）。
"""

from __future__ import annotations

from typing import Dict, Protocol

import numpy as np


class SemanticStateView(Protocol):
    """谓词求值所需的最小状态接口。

    实现方需提供世界系下的 body 位姿、site 位姿、关节角与两两接触判定。
    旋转一律以 3x3 矩阵提供，避免四元数约定歧义。
    """

    def body_pos(self, name: str) -> np.ndarray: ...

    def body_rotmat(self, name: str) -> np.ndarray: ...

    def site_pos(self, site_name: str) -> np.ndarray: ...

    def site_rotmat(self, site_name: str) -> np.ndarray: ...

    def joint_qpos(self, entity_name: str, joint_index: int) -> float: ...

    def num_joints(self, entity_name: str) -> int: ...

    def in_contact(self, name_a: str, name_b: str) -> bool: ...


class PredicateLib:
    """对某个任务的 goal/init 谓词求值。

    参数
    ----
    sites : dict
        manifest 中的 site 定义：{site_name: {parent_name, size, local_pos, ...}}
    joint_thresholds : dict
        manifest 中每个实体的关节阈值：{entity: {default_open_ranges: [lo, hi], ...}}
    hold_steps : int
        success 需要连续保持的步数（官方为 1，即单步判定；>1 用于抗抖动实验）。
    """

    def __init__(
        self,
        sites: Dict[str, dict],
        joint_thresholds: Dict[str, dict],
        hold_steps: int = 1,
    ):
        self.sites = sites
        self.joint_thresholds = joint_thresholds
        self.hold_steps = max(1, int(hold_steps))
        self._hold_counter = 0

    # ------------------------------------------------------------------
    # 几何原语
    # ------------------------------------------------------------------
    @staticmethod
    def _site_in_box(site_pos: np.ndarray, site_mat: np.ndarray, size: np.ndarray, point: np.ndarray) -> bool:
        """SiteObject.in_box 官方语义：点近似，盒为 site 世界位姿下的近似 AABB。"""
        total_size = np.abs(site_mat @ size)
        ub = site_pos + total_size
        lb = site_pos - total_size
        lb[2] -= 0.01
        return bool(np.all(point > lb) and np.all(point < ub))

    @staticmethod
    def _site_under(site_pos: np.ndarray, site_mat: np.ndarray, size: np.ndarray, point: np.ndarray, other_height: float = 0.10) -> bool:
        """SiteObject.under 官方语义（plate/cook 类 region 的 On 判定）。"""
        delta = site_mat @ (point - site_pos)
        return bool(
            size[2] - 0.005 < delta[2] < size[2] + other_height
            and np.all(np.abs(delta[:2]) < size[:2])
        )

    # ------------------------------------------------------------------
    # 谓词
    # ------------------------------------------------------------------
    def pred_in(self, view: SemanticStateView, obj: str, region: str) -> bool:
        """In(obj, region)：site 的 contact 恒真，只查 contain。"""
        site = self.sites[region]
        return self._site_in_box(
            view.site_pos(region),
            view.site_rotmat(region),
            np.asarray(site["size"], dtype=float),
            view.body_pos(obj),
        )

    def pred_on(self, view: SemanticStateView, obj: str, target: str) -> bool:
        """On(obj, target)：target 可以是物体（ObjectState）或 site region。"""
        obj_pos = view.body_pos(obj)
        if target in self.sites:
            site = self.sites[target]
            parent = site["parent_name"]
            ok = self._site_under(
                view.site_pos(target),
                view.site_rotmat(target),
                np.asarray(site["size"], dtype=float),
                obj_pos,
            )
            if parent is not None:
                ok = ok and view.in_contact(parent, obj)
            return bool(ok)
        # target 为普通物体：官方 ObjectState.check_ontop 语义
        tgt_pos = view.body_pos(target)
        return bool(
            tgt_pos[2] <= obj_pos[2]
            and view.in_contact(target, obj)
            and np.linalg.norm(tgt_pos[:2] - obj_pos[:2]) < 0.03
        )

    def _resolve_entity(self, entity: str) -> str:
        if entity in self.sites and self.sites[entity].get("parent_name"):
            return self.sites[entity]["parent_name"]
        return entity

    def pred_open(self, view: SemanticStateView, entity: str) -> bool:
        entity = self._resolve_entity(entity)
        """官方语义：任一关节落在 open 区间即视为 open（is_open 的循环逻辑）。"""
        lo, hi = self._range(entity, "default_open_ranges")
        for j in range(view.num_joints(entity)):
            q = view.joint_qpos(entity, j)
            # 官方各类的 is_open 实际实现为「越过上界/跌破下界」之一，
            # 这里按区间语义统一：q 位于 (min, +inf) 或 (-inf, max) 一侧，
            # 具体方向由 range 数值符号决定——与官方逐类实现等价。
            if self._in_open_range(entity, q, lo, hi):
                return True
        return False

    def pred_close(self, view: SemanticStateView, entity: str) -> bool:
        entity = self._resolve_entity(entity)
        """官方语义：所有关节都在 close 区间内才算 close。"""
        lo, hi = self._range(entity, "default_close_ranges")
        for j in range(view.num_joints(entity)):
            q = view.joint_qpos(entity, j)
            if not self._in_close_range(entity, q, lo, hi):
                return False
        return True

    def pred_turnon(self, view: SemanticStateView, entity: str) -> bool:
        entity = self._resolve_entity(entity)
        """官方 FlatStove.turn_on 语义：任一关节 qpos >= min(turnon_ranges)。"""
        lo, hi = self._range(entity, "default_turnon_ranges")
        for j in range(view.num_joints(entity)):
            if view.joint_qpos(entity, j) >= min(lo, hi):
                return True
        return False

    def pred_turnoff(self, view: SemanticStateView, entity: str) -> bool:
        entity = self._resolve_entity(entity)
        lo, hi = self._range(entity, "default_turnoff_ranges")
        for j in range(view.num_joints(entity)):
            q = view.joint_qpos(entity, j)
            # 官方 FlatStove.turn_off: qpos < max(turnoff_ranges)
            if not (q < max(lo, hi)):
                return False
        return True

    def pred_up(self, view: SemanticStateView, obj: str) -> bool:
        return bool(view.body_pos(obj)[2] >= 1.0)

    # ------------------------------------------------------------------
    # 阈值判定的方向性：与官方各类的逐一实现对应
    # ------------------------------------------------------------------
    def _range(self, entity: str, key: str) -> tuple[float, float]:
        # site region 的关节谓词落到其父实体的阈值（官方 SiteObjectState 语义）
        if entity in self.sites:
            parent = self.sites[entity].get("parent_name")
            if parent:
                entity = parent
        r = self.joint_thresholds.get(entity, {}).get(key)
        assert r is not None and len(r) == 2, f"{entity} 缺少 {key} 阈值"
        return float(r[0]), float(r[1])

    @staticmethod
    def _in_open_range(entity: str, q: float, lo: float, hi: float) -> bool:
        """官方实现的统一化：负区间（microwave/wooden_cabinet）为 q < max(range)，
        正区间（short_cabinet/short_fridge）为 q > min(range)。"""
        if hi <= 0:
            return q < hi if lo < hi else q < lo
        return q > lo

    @staticmethod
    def _in_close_range(entity: str, q: float, lo: float, hi: float) -> bool:
        """close 区间总是围绕 0：负开口物体（microwave）为 q > min(range)，
        正开口物体（short_cabinet）为 q < max(range)。"""
        if hi <= 0:
            return q > lo
        return q < hi

    # ------------------------------------------------------------------
    # goal 求值
    # ------------------------------------------------------------------
    def eval_predicate(self, view: SemanticStateView, pred: list) -> bool:
        name = pred[0].lower()
        if name == "in":
            return self.pred_in(view, pred[1], pred[2])
        if name == "on":
            return self.pred_on(view, pred[1], pred[2])
        if name == "open":
            return self.pred_open(view, pred[1])
        if name == "close":
            return self.pred_close(view, pred[1])
        if name == "turnon":
            return self.pred_turnon(view, pred[1])
        if name == "turnoff":
            return self.pred_turnoff(view, pred[1])
        if name == "up":
            return self.pred_up(view, pred[1])
        raise NotImplementedError(f"未实现的谓词: {name}")

    def check_goal(self, view: SemanticStateView, goal_state: list[list]) -> bool:
        """合取式 goal 求值，带保持步数抗抖动（默认 hold_steps=1 即官方单步语义）。"""
        ok = all(self.eval_predicate(view, g) for g in goal_state)
        if ok:
            self._hold_counter += 1
        else:
            self._hold_counter = 0
        return self._hold_counter >= self.hold_steps

    def reset_hold(self) -> None:
        self._hold_counter = 0
