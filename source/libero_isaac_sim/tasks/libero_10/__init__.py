"""LIBERO-10 任务的 gym 注册。

注册一次即可通过 ``gym.make("Libero10-Task00-Basket-OSC")`` 创建环境。
env_cfg 在模块导入时由工厂构建（需要已导出的语义与已转换的资产缓存）。
"""

import gymnasium as gym

gym.register(
    id="Libero10-Task00-Basket-OSC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "libero_isaac_sim.tasks.libero_10.task00_basket:env_cfg"
        ),
    },
    disable_env_checker=True,
)
