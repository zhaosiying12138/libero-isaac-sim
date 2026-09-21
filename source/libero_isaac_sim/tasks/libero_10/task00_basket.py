"""任务 0：LIVING_ROOM_SCENE2 篮子任务（把 alphabet soup 和 tomato sauce 放进篮子）。

gym id: ``Libero10-Task00-Basket-OSC``
"""

import os

from libero_isaac_sim.envs.libero_env_cfg import make_libero_env_cfg

TASK_NAME = (
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
)

env_cfg = make_libero_env_cfg(TASK_NAME)
