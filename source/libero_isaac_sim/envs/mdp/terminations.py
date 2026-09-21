"""Isaac 场景状态 → 谓词语义视图 + BDDL 成功判定终止项。

``IsaacSemanticView`` 实现 semantics.predicates.SemanticStateView 协议：
- body 位姿：RigidObject/Articulation 根位姿（减去 env 原点，回到世界语义系）
- site 位姿：父 body 位姿 @ manifest 中的局部变换（逐帧计算）
- 关节角：fixture articulation 的关节位置
- 接触：几何近似（碰撞半径包围球 + 1.5mm 容差）；需要精确接触判定的任务
  （如 mug-on-plate）在 L6 阶段可切换为 ContactSensor 实现。

终止项 ``bddl_success`` 对每个 env 独立维护谓词保持计数（hold_steps 抗抖动，
默认 1 即官方单步语义）。
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs import ManagerBasedEnv

from libero_isaac_sim.semantics.math_utils import quat_xyzw_to_mat
from libero_isaac_sim.semantics.predicates import PredicateLib
from libero_isaac_sim.semantics.task_spec import load_task

_PREDICATE_LIBS: dict[tuple[str, int], PredicateLib] = {}
_TASK_CACHE2: dict[str, object] = {}


def _task(task_name: str, cache_dir: str):
    if task_name not in _TASK_CACHE2:
        _TASK_CACHE2[task_name] = load_task(task_name, cache_dir)
    return _TASK_CACHE2[task_name]


class IsaacSemanticView:
    """单个 env 的语义状态视图（numpy，评测/回放规模下开销可忽略）。"""

    def __init__(self, env: ManagerBasedEnv, env_id: int, task_name: str, cache_dir: str):
        self.env = env
        self.env_id = env_id
        self.task = _task(task_name, cache_dir)
        self.origin = env.scene.env_origins[env_id].cpu().numpy()

    # ------------------------------------------------------------------
    def _asset_pose_world(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        asset = self.env.scene[name]
        if hasattr(asset.data, "root_pose_w"):
            pose = asset.data.root_pose_w.torch[self.env_id].cpu().numpy()
        else:
            pose = asset.data.body_pose_w.torch[self.env_id, 0].cpu().numpy()
        pos = pose[:3] - self.origin
        rotmat = quat_xyzw_to_mat(pose[3:7])
        return pos, rotmat

    def body_pos(self, name: str) -> np.ndarray:
        return self._asset_pose_world(name)[0]

    def body_rotmat(self, name: str) -> np.ndarray:
        return self._asset_pose_world(name)[1]

    # ------------------------------------------------------------------
    def _site_world(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site = self.task.sites[site_name]
        parent_body = site.get("parent_body")
        if parent_body:
            # parent_body 是 MuJoCo body 名（如 basket_1_main）；实体名去掉 _main
            entity = parent_body[:-5] if parent_body.endswith("_main") else parent_body
            if entity in self.env.scene.keys():
                p_pos, p_rot = self._asset_pose_world(entity)
            else:
                # 桌面等 arena 内 body：arena 静止于原点，用参考位姿
                p_pos = np.zeros(3)
                p_rot = np.eye(3)
                if "world_pos_ref" in site and parent_body in ("table", "living_room_table"):
                    # 桌面 region：直接用导出时的世界参考位姿（桌子不动）
                    return (
                        np.asarray(site["world_pos_ref"], dtype=float),
                        np.eye(3),
                    )
            local_pos = np.asarray(site["local_pos"], dtype=float)
            local_mat = np.asarray(site["local_rotmat"], dtype=float).reshape(3, 3)
            return p_rot @ local_pos + p_pos, p_rot @ local_mat
        # 无父体：世界固定
        return (
            np.asarray(site["world_pos_ref"], dtype=float),
            np.asarray(site["local_rotmat"], dtype=float).reshape(3, 3),
        )

    def site_pos(self, site_name: str) -> np.ndarray:
        return self._site_world(site_name)[0]

    def site_rotmat(self, site_name: str) -> np.ndarray:
        return self._site_world(site_name)[1]

    # ------------------------------------------------------------------
    def joint_qpos(self, entity_name: str, joint_index: int) -> float:
        asset = self.env.scene[entity_name]
        return float(asset.data.joint_pos.torch[self.env_id, joint_index].cpu())

    def num_joints(self, entity_name: str) -> int:
        asset = self.env.scene[entity_name]
        if not hasattr(asset.data, "joint_pos"):
            return 0
        return int(asset.data.joint_pos.torch.shape[-1])

    # ------------------------------------------------------------------
    def in_contact(self, name_a: str, name_b: str) -> bool:
        """几何近似接触判定：包围球距离。L6 阶段可替换为接触传感器。"""
        pa, _ = self._asset_pose_world(name_a)
        pb, _ = self._asset_pose_world(name_b)
        ra = self._bounding_radius(name_a)
        rb = self._bounding_radius(name_b)
        return bool(np.linalg.norm(pa - pb) < (ra + rb) + 0.0015)

    def _bounding_radius(self, name: str) -> float:
        if not hasattr(self, "_radii"):
            self._radii = {}
        if name not in self._radii:
            # 从 manifest 的 region 尺寸或质量粗估；默认 5cm 兜底
            ent = self.task.entities.get(name, {})
            self._radii[name] = 0.05
        return self._radii[name]


def bddl_success(
    env: ManagerBasedEnv, task_name: str, cache_dir: str, hold_steps: int = 1
) -> torch.Tensor:
    """BDDL goal 合取谓词终止项（每个 env 独立计数）。"""
    task = _task(task_name, cache_dir)
    out = torch.zeros(env.num_envs, dtype=torch.bool)
    for env_id in range(env.num_envs):
        key = (task_name, env_id)
        if key not in _PREDICATE_LIBS:
            _PREDICATE_LIBS[key] = PredicateLib(
                sites=task.sites,
                joint_thresholds=task.joint_thresholds(),
                hold_steps=hold_steps,
            )
        lib = _PREDICATE_LIBS[key]
        view = IsaacSemanticView(env, env_id, task_name, cache_dir)
        out[env_id] = lib.check_goal(view, task.goal_state)
    return out


def reset_predicate_state(task_name: str, cache_dir: str) -> None:
    """供 reset 事件调用：清空该任务的谓词保持计数。"""
    for key in [k for k in _PREDICATE_LIBS if k[0] == task_name]:
        _PREDICATE_LIBS[key].reset_hold()
