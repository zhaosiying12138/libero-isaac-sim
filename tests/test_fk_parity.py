"""L2 验证阶梯：双仿真器前向运动学（FK）一致性探针。

流程：MuJoCo（桥）与 Isaac（子进程探针）各自把机器人置为同一组关节角，
逐 link 对比世界位姿。输出每个 link 的位置误差（mm）与姿态误差（度）。

用法：python tests/test_fk_parity.py [joint1,joint2,...]
"""

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from experiments.bridge import MujocoBridge

ISAACLAB_DIR = os.path.expanduser("~/codebase/_external/IsaacLab")
UV = os.path.expanduser("~/miniforge3/bin/uv")
PROBE = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "probe_isaac_state.py"
)
BDDL = (
    "/home/zhaosiying/codebase/_external/LIBERO/libero/libero/bddl_files/"
    "libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_"
    "the_tomato_sauce_in_the_basket.bddl"
)

# LIBERO init_qpos
INIT_QPOS = [0, -0.161037389, 0.0, -2.44459747, 0.0, 2.22675220, 0.785398]


def mujoco_fk(joint7) -> dict:
    bridge = MujocoBridge()
    bridge.load(BDDL)
    bridge.reset(0)
    s = bridge.set_joints(joint7)
    bridge.close()
    return s


def isaac_fk(joint7) -> dict:
    cmd = [
        UV, "run", "--extra", "isaacsim", "python", "-u",
        os.path.abspath(PROBE),
        "--joints", ",".join(str(v) for v in joint7),
    ]
    env = dict(os.environ)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        cmd, cwd=ISAACLAB_DIR, env=env, capture_output=True, text=True, timeout=1800
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__PROBE_JSON__"):
            return json.loads(line[len("__PROBE_JSON__"):])
    raise RuntimeError("Isaac 探针无输出；stderr 末尾：\n" + proc.stderr[-2000:])


def main():
    joint7 = INIT_QPOS
    if len(sys.argv) > 1:
        joint7 = [float(v) for v in sys.argv[1].split(",")]

    print("=== MuJoCo 侧 ===")
    mj = mujoco_fk(joint7)
    mj_eef = np.array(mj["eef_pos"])
    print("eef_pos:", np.round(mj_eef, 4))

    print("=== Isaac 侧 ===")
    isa = isaac_fk(joint7)
    isa_bodies = isa["bodies"]
    hand = isa_bodies.get("robot0_right_hand") or isa_bodies.get("robot0_link7")
    isa_hand_pos = np.array(hand["pos"])
    print("isaac hand pos:", np.round(isa_hand_pos, 4))
    err = np.linalg.norm(isa_hand_pos - mj_eef)
    print(f"[L2] EE 位置误差: {err*1000:.1f} mm")

    # 关键 link 对比
    for lname in ["robot0_link1", "robot0_link3", "robot0_link5", "robot0_link7"]:
        if lname in isa_bodies:
            print(f"  {lname} isaac_pos={np.round(isa_bodies[lname]['pos'], 3)}")
    print("[L2] 机器人 base：Isaac 侧请在探针日志中检查 base 位姿与朝向")


if __name__ == "__main__":
    main()
