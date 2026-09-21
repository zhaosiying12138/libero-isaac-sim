"""语义层装载器：把导出的 manifest + init_states 装载为内存中的任务语义对象。

这是连接「LIBERO 官方定义」与「Isaac Lab 环境」的纽带：
- converters/export_libero_task.py 在 MuJoCo 侧导出 manifest.json + init_states.npz；
- 本模块在 Isaac 侧（或测试侧）装载它们，提供统一的查询接口。

旋转一律以 3x3 矩阵在模块间传递（见 manifest["convention"]）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

DEFAULT_CACHE_DIR = os.environ.get(
    "LIBERO_ISAAC_ASSETS_DIR", os.path.expanduser("~/.cache/libero_isaac_sim")
)


@dataclass
class EntityState:
    """单个实体在某一初始状态下的语义状态。"""

    name: str
    pos: np.ndarray  # (3,) 世界系
    rotmat: np.ndarray  # (3,3)
    joints: Optional[np.ndarray] = None  # (n_joints,) 关节物体的关节角


@dataclass
class InitState:
    """一组官方固定初始状态（语义形式）。"""

    robot_joint_pos: np.ndarray  # (7,)
    gripper_qpos: np.ndarray  # (2,)
    entities: Dict[str, EntityState] = field(default_factory=dict)


@dataclass
class TaskSemantics:
    """一个 LIBERO 任务的全部仿真器无关语义。

    _npz：官方评测协议的 50 组固定状态（.pruned_init 导出）。
    _demo_npz：每条示范录制时的真实起始状态（demo HDF5 attrs.init_state 导出），
    回放实验必须用后者（动作序列与起始态配套）。
    """

    task_name: str
    manifest: dict
    _npz: dict  # 延迟数组字典
    _demo_npz: dict = field(default_factory=dict)

    # ---- 便捷访问 ----
    @property
    def bddl(self) -> dict:
        return self.manifest["bddl"]

    @property
    def language(self) -> str:
        return self.manifest["bddl"]["language"]

    @property
    def entities(self) -> dict:
        return self.manifest["entities"]

    @property
    def sites(self) -> dict:
        return self.manifest["sites"]

    @property
    def cameras(self) -> dict:
        return self.manifest["cameras"]

    @property
    def arena(self) -> dict:
        return self.manifest["arena"]

    @property
    def robot(self) -> dict:
        return self.manifest["robot"]

    @property
    def goal_state(self) -> list:
        return self.manifest["bddl"]["goal_state"]

    @property
    def num_init_states(self) -> int:
        return int(self.manifest["num_init_states"])

    def movable_objects(self) -> List[str]:
        return [n for n, e in self.entities.items() if e["kind"] == "object"]

    def fixtures(self) -> List[str]:
        return [n for n, e in self.entities.items() if e["kind"] == "fixture"]

    def joint_thresholds(self) -> Dict[str, dict]:
        return {n: e.get("joint_thresholds", {}) for n, e in self.entities.items()}

    def get_init_state(self, index: int, source: str = "pruned") -> InitState:
        """取初始状态。source: "pruned"（官方评测协议）| "demo"（示范录制起始态）。"""
        npz = self._npz
        n = self.num_init_states
        if source == "demo":
            demo_path = os.path.join(
                os.path.dirname(str(self.manifest.get("_manifest_dir", ""))) or ".", ""
            )
        i = index % n
        state = InitState(
            robot_joint_pos=np.asarray(self._npz["robot_joint_pos"][i]),
            gripper_qpos=np.asarray(self._npz["gripper_qpos"][i]),
        )
        for name in self.entities:
            state.entities[name] = EntityState(
                name=name,
                pos=np.asarray(self._npz[f"pos::{name}"][i]),
                rotmat=np.asarray(self._npz[f"rotmat::{name}"][i]).reshape(3, 3),
                joints=np.asarray(self._npz[f"joint::{name}"][i])
                if f"joint::{name}" in self._npz
                else None,
            )
        return state


def load_task(task_name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> TaskSemantics:
    """按任务名装载语义（manifest + init_states）。"""
    manifest_path = os.path.join(cache_dir, f"{task_name}_manifest.json")
    npz_path = os.path.join(cache_dir, f"{task_name}_init_states.npz")
    assert os.path.exists(manifest_path), (
        f"manifest 不存在: {manifest_path}；请先在 libero-mujoco 环境运行 "
        f"converters/export_libero_task.py"
    )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    npz = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
    demo_npz_path = os.path.join(cache_dir, f"{task_name}_demo_init_states.npz")
    demo_npz = dict(np.load(demo_npz_path)) if os.path.exists(demo_npz_path) else {}
    return TaskSemantics(
        task_name=task_name, manifest=manifest, _npz=npz, _demo_npz=demo_npz
    )


def list_exported_tasks(cache_dir: str = DEFAULT_CACHE_DIR) -> List[str]:
    """列出缓存目录中已导出的任务名。"""
    out = []
    for fn in sorted(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else []:
        if fn.endswith("_manifest.json"):
            out.append(fn[: -len("_manifest.json")])
    return out
