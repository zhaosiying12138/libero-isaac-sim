"""LIBERO 任务的 ManagerBasedRLEnvCfg 工厂。

对照 LIBERO 官方环境参数：
- 控制频率 20 Hz：物理 dt=1/120，decimation=6
- 评测步数上限 600 步 → episode_length_s=30
- 动作：OSC pose_rel（±0.05m / ±0.5rad，与 robosuite OSC_POSE 输出限幅一致）
  + 夹爪二值（>0 开）
- 奖励：稀疏 success +1（官方语义）
- 终止：bddl_success（谓词合取）或超时
"""

from __future__ import annotations

import os

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg

from libero_isaac_sim.envs.mdp.actions import LiberoOscActionCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from libero_isaac_sim.converters.asset_manifest import AssetRegistry
from libero_isaac_sim.envs.libero_scene_cfg import build_scene_cfg
from libero_isaac_sim.envs.mdp import cameras as libero_cameras
from libero_isaac_sim.envs.mdp import events as libero_events
from libero_isaac_sim.envs.mdp import observations as libero_obs
from libero_isaac_sim.envs.mdp import terminations as libero_terms
from libero_isaac_sim.semantics.task_spec import load_task

DEFAULT_CACHE_DIR = os.environ.get(
    "LIBERO_ISAAC_ASSETS_DIR", os.path.expanduser("~/.cache/libero_isaac_sim")
)

# robosuite OSC_POSE 默认增益（controllers/config/osc_pose.json）：kp=150, damping_ratio=1
OSC_KP = 150.0
OSC_DAMPING_RATIO = 1.0


@configclass
class LiberoActionsCfg:
    """7 维 LIBERO 动作：6 维 EE 位姿增量（OSC）+ 1 维夹爪二值。"""

    arm: LiberoOscActionCfg = LiberoOscActionCfg(
        asset_name="robot",
        joint_names=["robot0_joint[1-7]"],
        body_name="robot0_right_hand",
        # robosuite OSC_POSE 的控制帧是 gripper0_grip_site（gripper0_eef 体），
        # 位于 hand 下方 9.7cm 处（实测 MuJoCo 模型 body_pos z=0.097）
        body_offset=LiberoOscActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.097)),
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=os.environ.get("LIBERO_OSC_INERTIAL", "1") == "1",
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=os.environ.get("LIBERO_OSC_GRAVCOMP", "1") == "1",
            motion_stiffness_task=OSC_KP,
            motion_damping_ratio_task=OSC_DAMPING_RATIO,
            motion_control_axes_task=[1, 1, 1, 1, 1, 1],
            nullspace_control=os.environ.get("LIBERO_OSC_NULLSPACE", "position"),
        ),
        position_scale=0.05,
        orientation_scale=0.5,
        nullspace_joint_pos_target=(
            "default" if os.environ.get("LIBERO_OSC_NULLSPACE", "position") == "position" else "none"
        ),
    )

    gripper: BinaryJointPositionActionCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper0_finger_joint[12]"],
        open_command_expr={"gripper0_finger_joint1": 0.04, "gripper0_finger_joint2": -0.04},
        close_command_expr={"gripper0_finger_joint1": 0.0, "gripper0_finger_joint2": 0.0},
    )


@configclass
class LiberoPolicyObsCfg(ObservationGroupCfg):
    """低维本体观测（LIBERO schema 同名键）。"""

    robot0_joint_pos: ObservationTermCfg = ObservationTermCfg(
        func=libero_obs.robot0_joint_pos
    )
    robot0_gripper_qpos: ObservationTermCfg = ObservationTermCfg(
        func=libero_obs.robot0_gripper_qpos
    )
    robot0_eef_pos: ObservationTermCfg = ObservationTermCfg(func=libero_obs.robot0_eef_pos)
    robot0_eef_quat: ObservationTermCfg = ObservationTermCfg(
        func=libero_obs.robot0_eef_quat
    )

    def __post_init__(self):
        self.concatenate_terms = False
        self.enable_corruption = False


@configclass
class LiberoRgbObsCfg(ObservationGroupCfg):
    """图像观测（agentview + 腕部，128×128 uint8）。"""

    agentview_image: ObservationTermCfg = ObservationTermCfg(
        func=libero_obs.agentview_image
    )
    robot0_eye_in_hand_image: ObservationTermCfg = ObservationTermCfg(
        func=libero_obs.eye_in_hand_image
    )

    def __post_init__(self):
        self.concatenate_terms = False
        self.enable_corruption = False


@configclass
class LiberoObservationsCfg:
    policy: LiberoPolicyObsCfg = LiberoPolicyObsCfg()
    rgb: LiberoRgbObsCfg = LiberoRgbObsCfg()


def _success_reward(env, task_name: str, cache_dir: str) -> torch.Tensor:
    return libero_terms.bddl_success(
        env, task_name=task_name, cache_dir=cache_dir
    ).float()


def make_libero_env_cfg(
    task_name: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_envs: int = 1,
    hold_steps: int = 1,
) -> ManagerBasedRLEnvCfg:
    """为指定 LIBERO 任务生成环境配置。"""
    task = load_task(task_name, cache_dir)
    registry = AssetRegistry(cache_dir)
    scene_cfg = build_scene_cfg(task, registry, num_envs=num_envs)

    @configclass
    class LiberoEventsCfg:
        align_cameras = EventTermCfg(
            func=libero_cameras.align_cameras_to_manifest,
            mode="startup",
            params={"task_name": task_name, "cache_dir": cache_dir},
        )
        reset_to_init = EventTermCfg(
            func=libero_events.reset_to_libero_init_state,
            mode="reset",
            params={"task_name": task_name, "cache_dir": cache_dir, "cycle": True},
        )

    @configclass
    class LiberoTerminationsCfg:
        success = DoneTerm(
            func=libero_terms.bddl_success,
            params={
                "task_name": task_name,
                "cache_dir": cache_dir,
                "hold_steps": hold_steps,
            },
        )
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    @configclass
    class LiberoRewardsCfg:
        success = RewardTermCfg(
            func=_success_reward,
            weight=1.0,
            params={"task_name": task_name, "cache_dir": cache_dir},
        )

    cfg = ManagerBasedRLEnvCfg()
    cfg.scene = scene_cfg
    cfg.actions = LiberoActionsCfg()
    if os.environ.get("LIBERO_TEST_NO_GRIPPER") == "1":
        del cfg.actions.gripper
    cfg.observations = LiberoObservationsCfg()
    if os.environ.get("LIBERO_TEST_NO_CAM") == "1":
        del cfg.observations.rgb
    cfg.events = LiberoEventsCfg()
    cfg.terminations = LiberoTerminationsCfg()
    cfg.rewards = LiberoRewardsCfg()

    cfg.decimation = 6  # 120 Hz 物理 / 6 = 20 Hz 控制
    cfg.episode_length_s = 30.0  # 600 步 × 20 Hz，与官方评测一致
    cfg.sim.dt = 1.0 / 120.0
    cfg.sim.device = "cpu"  # WSL2 上 PhysX GPU 不可用；详见 README 风险节
    cfg.sim.render_interval = 2
    return cfg
