"""IsaacLiberoEnv：对 VLA/策略侧暴露 LIBERO 原生接口的 facade。

设计目标：现依赖 LIBERO 的策略代码（OpenVLA / GR00T / WALL-X 等）不改一行
即可把后端从 MuJoCo 切到 PhysX::

    env = IsaacLiberoEnv("LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket")
    obs = env.reset(init_state_id=17)
    obs, reward, done, info = env.step(action_7d)   # action: 7 维 OSC_POSE + 夹爪
    info["success"]                                  # BDDL 谓词判决

观测字典与 LIBERO 数据集键一致：
    agentview_image          uint8 (128,128,3)
    robot0_eye_in_hand_image uint8 (128,128,3)
    robot0_joint_pos         float (7,)
    robot0_gripper_qpos      float (2,)
    robot0_eef_pos           float (3,)
    robot0_eef_quat          float (4,) wxyz
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

DEFAULT_CACHE_DIR = os.environ.get(
    "LIBERO_ISAAC_ASSETS_DIR", os.path.expanduser("~/.cache/libero_isaac_sim")
)


class IsaacLiberoEnv:
    """单环境 LIBERO 兼容 facade（评测/回放场景，num_envs=1）。"""

    def __init__(
        self,
        task_name: str,
        cache_dir: str = DEFAULT_CACHE_DIR,
        hold_steps: int = 1,
        max_steps: int = 600,
        settle_steps: int = 5,
        init_source: str = "pruned",
    ):
        # 延迟到首次使用时创建，避免 import 期拉起 Isaac Sim
        self.task_name = task_name
        self.cache_dir = cache_dir
        self.hold_steps = hold_steps
        self.max_steps = max_steps
        self.settle_steps = settle_steps
        self.init_source = init_source
        self._env = None
        self._step_count = 0
        self._current_init_id = 0

    # ------------------------------------------------------------------
    def _ensure_env(self):
        if self._env is None:
            from isaaclab.envs import ManagerBasedRLEnv

            from libero_isaac_sim.compat import isaaclab_ea_fixes
            from libero_isaac_sim.envs.libero_env_cfg import make_libero_env_cfg

            isaaclab_ea_fixes.apply()
            cfg = make_libero_env_cfg(
                self.task_name, self.cache_dir, num_envs=1, hold_steps=self.hold_steps
            )
            self._env = ManagerBasedRLEnv(cfg=cfg)

    # ------------------------------------------------------------------
    def reset(self, init_state_id: Optional[int] = None) -> dict:
        """复位到官方固定初始状态（init_state_id 取模 50）。

        官方协议：装载状态后先走 settle_steps 步零动作稳定物理。
        """
        self._ensure_env()
        # 固定 id：写入事件参数再 reset（事件在 env.reset 内部触发）
        if init_state_id is not None:
            self._current_init_id = init_state_id
            term = self._env.cfg.events.reset_to_init
            term.params["init_state_id"] = init_state_id
            term.params["cycle"] = False
            term.params["init_source"] = self.init_source
        else:
            term = self._env.cfg.events.reset_to_init
            term.params["init_state_id"] = None
            term.params["cycle"] = True
            term.params["init_source"] = self.init_source

        obs, _ = self._env.reset()
        zero = torch.zeros((1, 7))
        for _ in range(self.settle_steps):
            obs, _, _, _, _ = self._env.step(zero)
        self._step_count = 0
        return self._to_libero_obs(obs)

    def step(self, action) -> tuple[dict, float, bool, dict]:
        """7 维 LIBERO 动作（OSC_POSE 语义）。"""
        assert self._env is not None, "先 reset"
        a = torch.as_tensor(action, dtype=torch.float32).reshape(1, 7)
        obs, reward, terminated, truncated, info = self._env.step(a)
        self._step_count += 1
        success = bool(terminated[0])
        done = success or self._step_count >= self.max_steps or bool(truncated[0])
        info = {"success": success, "step": self._step_count}
        return self._to_libero_obs(obs), float(reward[0]), done, info

    def render(self, camera: str = "agentview") -> np.ndarray:
        assert self._env is not None
        cam = self._env.unwrapped.scene[camera]
        return cam.data.output["rgb"][0].cpu().numpy()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    # ------------------------------------------------------------------
    def _to_libero_obs(self, obs: dict) -> dict:
        policy = obs["policy"]
        rgb = obs["rgb"]
        return {
            "agentview_image": rgb["agentview_image"][0].cpu().numpy().astype(np.uint8),
            "robot0_eye_in_hand_image": rgb["robot0_eye_in_hand_image"][0]
            .cpu()
            .numpy()
            .astype(np.uint8),
            "robot0_joint_pos": policy["robot0_joint_pos"][0].cpu().numpy(),
            "robot0_gripper_qpos": policy["robot0_gripper_qpos"][0].cpu().numpy(),
            "robot0_eef_pos": policy["robot0_eef_pos"][0].cpu().numpy(),
            "robot0_eef_quat": policy["robot0_eef_quat"][0].cpu().numpy(),
        }

    @property
    def language_instruction(self) -> str:
        from libero_isaac_sim.semantics.task_spec import load_task

        return load_task(self.task_name, self.cache_dir).language

    @property
    def unwrapped(self):
        return self._env
