"""LIBERO 的 OSC 动作项（修正 Isaac Lab 3.0 EA 的一个上游缺陷）。

缺陷：``OperationalSpaceControllerAction`` 在不配置 task frame 刚体时，
把内部缓冲 ``_task_frame_pose_b = torch.zeros(num_envs, 7)``（**零四元数**，
非法姿态）传给 ``OperationalSpaceController.set_command`` 的
``current_task_frame_pose_b``；而控制器只有收到 ``None`` 时才回退到单位帧。
零四元数参与 subtract_frame_transforms 的归一化，污染 pose_rel 的目标合成。

对策：子类化后仅覆写 ``process_actions``，把 task frame 显式传 ``None``
（控制器内部回退为单位帧，即基座系）。其余逻辑完全复用上游实现。
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.utils import configclass


class LiberoOscActionTerm(OperationalSpaceControllerAction):
    """pose_rel 目标合成使用单位任务帧（基座系）的 OSC 动作项。"""

    def process_actions(self, actions: torch.Tensor):
        # 与上游一致，仅把 task frame 参数从非法零四元数改为 None
        self._compute_ee_pose()
        self._preprocess_actions(actions)
        self._osc.set_command(
            command=self._processed_actions,
            current_ee_pose_b=self._ee_pose_b,
            current_task_frame_pose_b=None,
        )


@configclass
# 注意：父类的 class_type 默认是 "{DIR}.task_space_actions:OperationalSpaceControllerAction"
# 模板字符串，子类必须显式覆盖为类对象，否则 {DIR} 会解析到错误模块。
class LiberoOscActionCfg(OperationalSpaceControllerActionCfg):
    """对应 LiberoOscActionTerm 的配置。"""

    class_type: type = LiberoOscActionTerm  # noqa: A003 - 字段名沿用上游约定
