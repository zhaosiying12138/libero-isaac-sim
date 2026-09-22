"""对比动画生产：从 paired rollout 的渲染帧合成 side-by-side mp4/gif。

案例集（论文配图）：
1. 一致成功：分段回放 150:200（soup 提起子目标，双仿真均达成）
2. 一致失败：完整 demo_3（双仿真均未完成）
3. 翻转案例：完整 demo_0（MuJoCo 成功 / Isaac 失败）

用法：~/codebase/_external/venvs/libero-mujoco/bin/python experiments/make_animations.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MUJOCO_PY = os.path.expanduser("~/codebase/_external/venvs/libero-mujoco/bin/python")
OUT = os.path.join(REPO, "outputs", "paired_rollout")
ANIM_DIR = os.path.join(REPO, "article", "figures", "animations")

TASK0 = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"


def _run_rollout(task, demo, segment=None):
    cmd = [
        MUJOCO_PY, os.path.join(REPO, "experiments", "paired_rollout.py"),
        "--task", task, "--demos", str(demo), "--render",
    ]
    if segment:
        cmd += ["--segment", segment]
    env = dict(os.environ)
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=7200)
    print(proc.stdout[-500:])
    if proc.returncode != 0:
        print(proc.stderr[-1000:])
        raise RuntimeError("rollout 失败")


def _compose_video(task, demo, out_name, fps=10):
    d = os.path.join(OUT, task, f"demo_{demo}")
    mjs = sorted(glob.glob(os.path.join(d, "mj_*.png")))
    isas = sorted(glob.glob(os.path.join(d, "isa_*.png")))
    import imageio.v2 as imageio

    os.makedirs(ANIM_DIR, exist_ok=True)
    frames = []
    for a, b in zip(mjs, isas):
        frames.append(np.concatenate([imageio.imread(a), imageio.imread(b)], axis=1))
    mp4 = os.path.join(ANIM_DIR, out_name + ".mp4")
    writer = imageio.get_writer(mp4, fps=fps)
    for f in frames:
        writer.append_data(f)
    writer.close()
    # 同时存 gif（知乎友好）
    gif = os.path.join(ANIM_DIR, out_name + ".gif")
    small = [f[::2, ::2] for f in frames[::2]]
    imageio.mimsave(gif, small, duration=1 / fps * 2)
    print(f"[anim] {mp4} ({len(frames)} 帧) + {gif}")


def main():
    cases = [
        ("consistent_success_seg150_200", TASK0, 0, "150:200"),
        ("consistent_failure_demo3", TASK0, 3, None),
        ("flip_demo0", TASK0, 0, None),
    ]
    for name, task, demo, segment in cases:
        print(f"[anim] === {name} ===", flush=True)
        _run_rollout(task, demo, segment)
        _compose_video(task, demo, name)
    print("[anim] 完成")


if __name__ == "__main__":
    main()
