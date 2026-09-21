"""L3 验证阶梯：双仿真器动作方向/幅值一致性（单轴探针序列）。

对 6 个单轴动作（±x/±y/±z 平移 + ±旋转）各施加 N 步（默认 20 步 = 1 秒），
对比 EE 位移方向与幅值。robosuite OSC_POSE 约定：输入 [-1,1] → ±0.05m/±0.5rad 增量。

用法：~/codebase/_external/venvs/libero-mujoco/bin/python tests/test_action_parity.py
"""

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from experiments.bridge import MujocoBridge

ISAACLAB_DIR = os.path.expanduser("~/codebase/_external/IsaacLab")
UV = os.path.expanduser("~/miniforge3/bin/uv")
PROBE = os.path.join(os.path.dirname(__file__), "..", "scripts", "probe_isaac_action.py")
BDDL = (
    "/home/zhaosiying/codebase/_external/LIBERO/libero/libero/bddl_files/"
    "libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_"
    "the_tomato_sauce_in_the_basket.bddl"
)

PROBES = {
    "+x": [0.5, 0, 0, 0, 0, 0, 0],
    "-x": [-0.5, 0, 0, 0, 0, 0, 0],
    "+y": [0, 0.5, 0, 0, 0, 0, 0],
    "-y": [0, -0.5, 0, 0, 0, 0, 0],
    "+z": [0, 0, 0.5, 0, 0, 0, 0],
    "-z": [0, 0, -0.5, 0, 0, 0, 0],
    "+rx": [0, 0, 0, 0.5, 0, 0, 0],
    "+ry": [0, 0, 0, 0, 0.5, 0, 0],
    "+rz": [0, 0, 0, 0, 0, 0.5, 0],
}
STEPS = 20


def mujoco_traj(action):
    bridge = MujocoBridge()
    bridge.load(BDDL)
    bridge.reset(0)
    traj = []
    for _ in range(STEPS):
        r = bridge.step(action)
        traj.append(r["state"]["eef_pos"])
    bridge.close()
    return np.array(traj)


def isaac_all(probes):
    cmd = [
        UV, "run", "--extra", "isaacsim", "python", "-u", os.path.abspath(PROBE),
        "--actions-json", json.dumps(probes), "--steps", str(STEPS),
    ]
    env = dict(os.environ)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    proc = subprocess.run(
        cmd, cwd=ISAACLAB_DIR, env=env, capture_output=True, text=True, timeout=3600
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__PROBE_JSON__"):
            d = json.loads(line[len("__PROBE_JSON__"):])
            return {k: np.array([t["pos"] for t in v]) for k, v in d["probes"].items()}
    raise RuntimeError("Isaac 动作探针无输出:\n" + proc.stderr[-1500:])


def main():
    isa_all = isaac_all(PROBES)
    rows = []
    for name, action in PROBES.items():
        mj = mujoco_traj(action)
        isa = isa_all[name]
        d_mj = mj[-1] - mj[0]
        d_isa = isa[-1] - isa[0]
        cos = float(
            np.dot(d_mj, d_isa) / (np.linalg.norm(d_mj) * np.linalg.norm(d_isa) + 1e-12)
        )
        rows.append(
            (name, np.linalg.norm(d_mj), np.linalg.norm(d_isa), cos)
        )
        print(
            f"[L3] {name:4s} mujoco位移={np.linalg.norm(d_mj)*1000:7.1f}mm "
            f"isaac位移={np.linalg.norm(d_isa)*1000:7.1f}mm 方向cos={cos:+.3f}"
        )
    print("[L3] 方向一致性（cos 应 ≈1）；幅值比用于标定 OSC scale")


if __name__ == "__main__":
    main()
