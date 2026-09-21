"""reset 事件：把环境精确复位到官方 50 组固定初始语义状态之一。

与 LIBERO 官方评测协议对应：
- ``init_state_id=k``：装载第 k 组语义状态（机器人关节、夹爪、各物体世界位姿、
  fixture 关节角）；
- ``init_state_id=None`` 且 ``cycle=True``：每次 reset 循环取下一组（评测协议
  ``arange(n) % 50`` 的等价实现）；
- 物理沉降（settle）在 facade/评测循环里以 5 步零动作完成，与官方一致。

四元数约定：Isaac Lab 3.0 资产接口为 xyzw；语义层存储为 3x3 旋转矩阵，
这里集中转换。
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv

from libero_isaac_sim.semantics.math_utils import mat_to_quat_xyzw
from libero_isaac_sim.semantics.task_spec import load_task

_TASK_CACHE: dict[str, object] = {}
_CYCLE_COUNTER: dict[int, int] = {}


def _task(task_name: str, cache_dir: str):
    if task_name not in _TASK_CACHE:
        _TASK_CACHE[task_name] = load_task(task_name, cache_dir)
    return _TASK_CACHE[task_name]


def reset_to_libero_init_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    task_name: str,
    cache_dir: str,
    init_state_id: int | None = None,
    cycle: bool = True,
):
    """把 env_ids 指定的环境复位到 LIBERO 官方初始语义状态。

    评测语义：同一 init_state_id 在所有环境上一致（逐 episode 对齐官方协议）。
    """
    task = _task(task_name, cache_dir)

    for env_id in env_ids.tolist():
        if init_state_id is not None:
            idx = init_state_id
        elif cycle:
            idx = _CYCLE_COUNTER.get(env_id, 0) % task.num_init_states
            _CYCLE_COUNTER[env_id] = _CYCLE_COUNTER.get(env_id, 0) + 1
        else:
            idx = 0
        state = task.get_init_state(idx)
        _apply_state(env, env_id, task, state)


def _apply_state(env: ManagerBasedEnv, env_id: int, task, state) -> None:
    scene = env.scene
    origin = torch.tensor(scene.env_origins[env_id].cpu(), dtype=torch.float32)

    # --- 机器人：关节 + 夹爪 ---
    robot = scene["robot"]
    joint_pos = robot.data.joint_pos.torch[env_id].clone()
    joint_vel = torch.zeros_like(joint_pos)
    arm_ids, _ = robot.find_joints(["robot0_joint[1-7]"])
    grip_ids, _ = robot.find_joints(["gripper0_finger_joint[12]"])
    joint_pos[arm_ids] = torch.tensor(state.robot_joint_pos, dtype=torch.float32)
    joint_pos[grip_ids] = torch.tensor(state.gripper_qpos, dtype=torch.float32)
    robot.write_joint_state_to_sim(
        joint_pos.unsqueeze(0), joint_vel.unsqueeze(0), env_ids=torch.tensor([env_id])
    )

    # --- 物体与 fixture：世界位姿（+ fixture 关节角） ---
    for name, ent_state in state.entities.items():
        if name not in scene.keys():
            continue
        asset = scene[name]
        pos_w = torch.tensor(ent_state.pos, dtype=torch.float32) + origin
        quat = torch.tensor(
            mat_to_quat_xyzw(ent_state.rotmat), dtype=torch.float32
        )
        root_pose = torch.cat([pos_w, quat]).unsqueeze(0)
        root_vel = torch.zeros((1, 6), dtype=torch.float32)
        env_ids_t = torch.tensor([env_id])
        if isinstance(asset, RigidObject):
            asset.write_root_pose_to_sim(root_pose, env_ids=env_ids_t)
            asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids_t)
        else:
            # fixture articulation：根位姿 + 关节角
            asset.write_root_pose_to_sim(root_pose, env_ids=env_ids_t)
            asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids_t)
            if ent_state.joints is not None and len(ent_state.joints) > 0:
                jp = asset.data.joint_pos.torch[env_id].clone()
                jp[:] = torch.tensor(ent_state.joints, dtype=torch.float32)[
                    : jp.numel()
                ]
                asset.write_joint_state_to_sim(
                    jp.unsqueeze(0),
                    torch.zeros_like(jp).unsqueeze(0),
                    env_ids=env_ids_t,
                )
