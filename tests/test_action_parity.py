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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

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
    traj, rot = [], []
    for _ in range(STEPS):
        r = bridge.step(action)
        traj.append(r["state"]["eef_pos"])
        rot.append(r["state"]["eef_rotmat"])
    bridge.close()
    return np.array(traj), np.array(rot)


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
            out = {}
            for k, v in d["probes"].items():
                pos = np.array([t["pos"] for t in v])
                quat_xyzw = np.array([t["quat_xyzw"] for t in v])
                out[k] = (pos, quat_xyzw)
            return out
    raise RuntimeError("Isaac 动作探针无输出:\n" + proc.stderr[-1500:])


def main():
    isa_all = isaac_all(PROBES)
    from libero_isaac_sim.semantics.math_utils import quat_wxyz_to_mat, quat_xyzw_to_mat

    def _rot_angle(mats):
        """首末帧间相对旋转角（度）。mats 为 wxyz 四元数或 3x3 矩阵序列。"""
        if mats.ndim == 2 and mats.shape[1] == 9:
            R0, R1 = mats[0].reshape(3, 3), mats[-1].reshape(3, 3)
        else:  # xyzw（isaac 侧）
            R0 = quat_xyzw_to_mat(mats[0])
            R1 = quat_xyzw_to_mat(mats[-1])
        dR = R1 @ R0.T
        c = np.clip((np.trace(dR) - 1) / 2, -1, 1)
        return float(np.degrees(np.arccos(c)))

    rows = []
    for name, action in PROBES.items():
        mj_pos, mj_rot = mujoco_traj(action)
        isa_pos, isa_rot = isa_all[name]
        d_mj = mj_pos[-1] - mj_pos[0]
        d_isa = isa_pos[-1] - isa_pos[0]
        cos = float(
            np.dot(d_mj, d_isa) / (np.linalg.norm(d_mj) * np.linalg.norm(d_isa) + 1e-12)
        )
        ang_mj = _rot_angle(mj_rot)
        ang_isa = _rot_angle(isa_rot)
        rows.append((name, np.linalg.norm(d_mj), np.linalg.norm(d_isa), cos, ang_mj, ang_isa))
        print(
            f"[L3] {name:4s} mujoco位移={np.linalg.norm(d_mj)*1000:7.1f}mm "
            f"isaac位移={np.linalg.norm(d_isa)*1000:7.1f}mm 方向cos={cos:+.3f} "
            f"旋转角 mujoco={ang_mj:5.1f}° isaac={ang_isa:5.1f}°"
        )
    print("[L3] 方向一致性（cos 应 ≈1）；幅值比用于标定 OSC scale")


if __name__ == "__main__":
    main()
