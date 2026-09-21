"""paired rollout 桥的客户端侧：以子进程方式驱动 MuJoCo 工作进程。

典型用法::

    bridge = MujocoBridge()
    bridge.load(bddl_path)
    s0 = bridge.reset(init_state_id=0)
    for a in actions:
        result = bridge.step(a)   # 返回语义状态 + 官方判决
    bridge.close()

进程间通信为 stdin/stdout JSON Lines，天然锁步、零额外依赖。
渲染帧通过临时文件传递。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

MUJOCO_PYTHON = os.environ.get(
    "LIBERO_MUJOCO_PYTHON",
    "/home/zhaosiying/codebase/_external/venvs/libero-mujoco/bin/python",
)
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "mujoco_worker.py")


class MujocoBridge:
    def __init__(self, mujoco_python: str = MUJOCO_PYTHON):
        env = dict(os.environ)
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
        # worker 的协议输出走 stdout；mujoco/robosuite 的日志会污染 stdout，
        # 因此 worker 启动阶段的非协议输出由调用方按「首个 { 开始的行」过滤。
        self.proc = subprocess.Popen(
            [mujoco_python, WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            bufsize=1,
        )

    def _call(self, **req) -> dict:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("MuJoCo 工作进程意外退出")
            line = line.strip()
            if not line.startswith("{"):
                continue  # 跳过第三方日志
            out = json.loads(line)
            if not out.get("ok", False):
                raise RuntimeError(f"MuJoCo worker 错误: {out.get('error')}")
            return out

    def load(self, bddl_path: str) -> dict:
        return self._call(cmd="load", bddl=os.path.abspath(bddl_path))

    def reset(self, init_state_id: int) -> dict:
        return self._call(cmd="reset", init_state_id=int(init_state_id))["state"]

    def step(self, action) -> dict:
        return self._call(cmd="step", action=[float(a) for a in action])

    def render(self, camera: str = "agentview", path: str | None = None) -> str:
        if path is None:
            path = os.path.join(
                tempfile.mkdtemp(prefix="libero_render_"), f"{camera}.png"
            )
        self._call(cmd="render", camera=camera, path=path)
        return path

    def close(self):
        try:
            self._call(cmd="close")
        except Exception:
            pass
        self.proc.terminate()


if __name__ == "__main__":
    # 冒烟：加载任务 0，reset 到 init_state 0，走 10 步零动作
    bddl = (
        "/home/zhaosiying/codebase/_external/LIBERO/libero/libero/bddl_files/"
        "libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_"
        "the_tomato_sauce_in_the_basket.bddl"
    )
    bridge = MujocoBridge()
    info = bridge.load(bddl)
    print("[bridge] loaded:", info["task"][:60])
    s0 = bridge.reset(0)
    print("[bridge] reset: eef_pos =", [round(v, 3) for v in s0["eef_pos"]])
    print("[bridge] soup pos:", [round(v, 3) for v in s0["bodies"]["alphabet_soup_1"]["pos"]])
    for i in range(10):
        r = bridge.step([0, 0, 0, 0, 0, 0, -1])
    print("[bridge] after 10 steps, success =", r["official_success"])
    print("[bridge] contacts:", r["state"]["contacts"][:5])
    bridge.close()
    print("[bridge] OK")
