"""VLA 闭环评测（双后端）：MuJoCo 官方 LIBERO 环境 + Isaac 移植环境，同一 OpenVLA-OFT checkpoint。

三方进程架构（JSON Lines over stdio）：
- 本驱动（任意 python）：编排任务/trials、聚合指标
- vla_policy_server.py（vla venv）：OpenVLA-OFT 推理服务
- 后端 worker（mujoco venv / isaac env）：环境交互（reset/step/render）

后端 worker 协议::

    --> {"cmd": "load_task", "task": "<task_name>", "bddl": "<path>"}
    --> {"cmd": "reset", "init_state_id": i}
    <-- {"state": <semantic>, "image": "<png>", "wrist_image": "<png>"}
    --> {"cmd": "step", "action": [7]}
    <-- {"state": ..., "image": ..., "success": false}

MuJoCo 后端直接用桥接（mujoco_worker 已有）；本文件另写 isaac 后端 worker 与驱动。

评测协议（对齐 openvla-oft）：init states 用 .pruned_init（官方 50 组），
num_steps_wait=10 步 dummy（[-1 夹爪]），replan 每 8 步，libero_10 上限 520 步。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ISAACLAB_DIR = os.path.expanduser("~/codebase/_external/IsaacLab")
UV = os.path.expanduser("~/miniforge3/bin/uv")
VLA_PY = os.path.expanduser("~/codebase/_external/venvs/vla/bin/python")
MUJOCO_PY = os.path.expanduser("~/codebase/_external/venvs/libero-mujoco/bin/python")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIBERO_REPO = os.environ.get("LIBERO_REPO", "/home/zhaosiying/codebase/_external/LIBERO")

TASK_MAX_STEPS = 520
NUM_STEPS_WAIT = 10
NUM_OPEN_LOOP = 8


class JsonLineProc:
    def __init__(self, argv, cwd=None, env_extra=None):
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        self.proc = subprocess.Popen(
            argv, cwd=cwd, env=env, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=1,
        )

    def call(self, **req):
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"进程退出: {self.proc.args[0]}")
            line = line.strip()
            if not line.startswith("{"):
                continue
            out = json.loads(line)
            if not out.get("ok", False):
                raise RuntimeError(f"{self.proc.args[0]} 错误: {out.get('error')}")
            return out

    def terminate(self):
        self.proc.terminate()


def _policy_infer(server: JsonLineProc, obs_pkg: dict, prompt: str) -> list[list[float]]:
    out = server.call(
        cmd="infer",
        image=obs_pkg["image"],
        wrist_image=obs_pkg["wrist_image"],
        state=obs_pkg["state"],
        prompt=prompt,
    )
    return out["actions"]


def _state_vector(state: dict) -> list[float]:
    """observation/state = [eef_pos(3), axisangle(eef_quat)(3), gripper_qpos(2)]。"""
    eef_pos = list(state["eef_pos"])
    x, y, z, w = [float(v) for v in state["eef_quat_xyzw"]]
    # 四元数(wxyz) -> 轴角
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    if abs(angle) < 1e-6:
        aa = [0.0, 0.0, 0.0]
    else:
        s = np.sin(angle / 2) or 1e-9
        aa = (angle * np.array([x / s, y / s, z / s])).tolist()
    return eef_pos + aa + list(state["gripper_qpos"])


def eval_backend(
    backend: str,
    task_name: str,
    num_trials: int,
    policy_server: JsonLineProc,
    video_dir: str | None = None,
) -> dict:
    """对某后端跑 num_trials 个 episode，返回逐 episode 结果。"""
    if backend == "mujoco":
        worker_argv = [MUJOCO_PY, os.path.join(REPO, "experiments", "vla_eval_worker_mujoco.py")]
        env_extra = {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0"}
    else:
        worker_argv = [
            UV, "run", "--extra", "isaacsim", "python", "-u",
            os.path.join(REPO, "experiments", "vla_eval_worker_isaac.py"),
        ]
        env_extra = {"OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONUNBUFFERED": "1"}

    worker = JsonLineProc(worker_argv, cwd=ISAACLAB_DIR if backend == "isaac" else None, env_extra=env_extra)
    bddl_path = os.path.join(
        LIBERO_REPO, "libero/libero/bddl_files/libero_10", f"{task_name}.bddl"
    )
    task_desc = worker.call(cmd="load_task", task=task_name, bddl=bddl_path).get("language", task_name)

    results = []
    for trial in range(num_trials):
        # reset：装载官方 init state，走 dummy 动作稳定
        pkg = worker.call(cmd="reset", init_state_id=trial)
        policy_server.call(cmd="reset")
        for _ in range(NUM_STEPS_WAIT):
            pkg = worker.call(cmd="step", action=[0, 0, 0, 0, 0, 0, -1])

        success = False
        action_queue: list[list[float]] = []
        frames = []
        for t in range(TASK_MAX_STEPS):
            if not action_queue:
                pkg["state"] = {"eef_pos": pkg["state"]["eef_pos"],
                            "eef_quat_xyzw": pkg["state"]["eef_quat_xyzw"],
                            "gripper_qpos": pkg["state"]["gripper_qpos"]}
            pkg["state_vec"] = _state_vector(pkg["state"])
            obs_for_policy = {"image": pkg["image"], "wrist_image": pkg["wrist_image"],
                              "state": pkg["state_vec"]}
            action_queue = _policy_infer(policy_server, obs_for_policy, task_desc)[:NUM_OPEN_LOOP]
            action = action_queue.pop(0)
            pkg = worker.call(cmd="step", action=action)
            if video_dir and t % 4 == 0:
                frames.append(pkg["image"])
            if pkg.get("success", False):
                success = True
                break
        results.append({"trial": trial, "success": success, "steps": t + 1})
        status = "成功" if success else "失败"
        print(f"    [{backend}] trial {trial}: {status} ({t+1} 步)", flush=True)

        if video_dir and success or (video_dir and trial < 2):
            import imageio.v2 as imageio

            os.makedirs(video_dir, exist_ok=True)
            writer = imageio.get_writer(
                os.path.join(video_dir, f"{backend}_trial{trial}_{'succ' if success else 'fail'}.mp4"),
                fps=10,
            )
            for f in frames:
                writer.append_data(imageio.imread(f))
            writer.close()

    worker.terminate()
    sr = np.mean([r["success"] for r in results])
    return {"backend": backend, "task": task_name, "success_rate": float(sr), "episodes": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,
                        default="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--backends", type=str, default="mujoco,isaac")
    parser.add_argument("--video-dir", type=str, default=None)
    args = parser.parse_args()

    server = JsonLineProc([VLA_PY, os.path.join(REPO, "experiments", "vla_policy_server.py")])
    # 等待模型加载
    import time

    time.sleep(5)

    summaries = []
    for backend in args.backends.split(","):
        print(f"[vla-eval] 后端 {backend}，任务 {args.task[:50]}，{args.trials} trials", flush=True)
        s = eval_backend(backend, args.task, args.trials, server, args.video_dir)
        summaries.append(s)
        print(f"[vla-eval] {backend} 成功率: {s['success_rate']*100:.1f}%", flush=True)

    server.terminate()
    out_path = os.path.join(REPO, "outputs", "vla_eval", f"{args.task}_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[vla-eval] 汇总 -> {out_path}")


if __name__ == "__main__":
    main()
