#!/usr/bin/env python
"""双仿真同初始状态渲染对比：MuJoCo（桥）+ Isaac（env），并排保存。

用法：~/codebase/_external/venvs/libero-mujoco/bin/python experiments/render_compare.py [init_state_id]
"""

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from experiments.bridge import MujocoBridge

BDDL = (
    "/home/zhaosiying/codebase/_external/LIBERO/libero/libero/bddl_files/"
    "libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_"
    "the_tomato_sauce_in_the_basket.bddl"
)
ISAACLAB_DIR = os.path.expanduser("~/codebase/_external/IsaacLab")
UV = os.path.expanduser("~/miniforge3/bin/uv")
ISAAC_RENDER_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "render_isaac_frame.py"
)


def mujoco_frame(init_state_id: int, out_path: str) -> str:
    bridge = MujocoBridge()
    bridge.load(BDDL)
    bridge.reset(init_state_id)
    bridge.render("agentview", out_path)
    bridge.render("robot0_eye_in_hand", out_path.replace("agentview", "wrist"))
    bridge.close()
    return out_path


def isaac_frame(init_state_id: int, out_path: str) -> str:
    env = dict(os.environ)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [UV, "run", "--extra", "isaacsim", "python", "-u",
         os.path.abspath(ISAAC_RENDER_SCRIPT),
         "--init-state-id", str(init_state_id), "--out", out_path],
        cwd=ISAACLAB_DIR, env=env, capture_output=True, text=True, timeout=1800,
    )
    if not os.path.exists(out_path):
        raise RuntimeError("Isaac 渲染失败: " + proc.stderr[-1500:])
    return out_path


def main():
    init_state_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    import imageio

    mj_path = mujoco_frame(init_state_id, "/tmp/cmp_mujoco_agentview.png")
    is_path = isaac_frame(init_state_id, "/tmp/cmp_isaac_agentview.png")
    mj = imageio.imread(mj_path)
    isac = imageio.imread(is_path)
    side = np.concatenate([mj, isac], axis=1)
    imageio.imwrite("/tmp/cmp_side_by_side.png", side)
    print("[compare] mujoco:", mj_path)
    print("[compare] isaac: ", is_path)
    print("[compare] side-by-side: /tmp/cmp_side_by_side.png")


if __name__ == "__main__":
    main()
