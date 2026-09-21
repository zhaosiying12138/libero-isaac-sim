"""VLA 评测 Isaac 后端 worker（isaac 环境运行，JSON Lines 协议）。

与 vla_eval_worker_mujoco.py 同协议；环境为 libero_isaac_sim 的 IsaacLiberoEnv。
图像语义与 LIBERO 原始观测一致（MuJoCo 帧为底向上的原始帧；Isaac 渲染为
顶行在前的正常帧——为与官方管线一致，本 worker 把 Isaac 帧垂直翻转回
「MuJoCo 原始帧」约定，由服务端统一做 180° 旋转与缩放）。
"""

import json
import os
import sys
import tempfile

import numpy as np

CACHE = os.path.expanduser("~/.cache/libero_isaac_sim")

_app = None


def _boot():
    global _app
    if _app is not None:
        return
    sys.path.insert(0, "/home/zhaosiying/codebase/libero-isaac-sim/source")
    from isaaclab.app import AppLauncher

    import argparse

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(args=[])
    args.headless = True
    _app = AppLauncher(args).app
    from libero_isaac_sim.compat import isaaclab_ea_fixes

    isaaclab_ea_fixes.apply()


class Worker:
    def __init__(self):
        self.env = None
        self.tmpdir = tempfile.mkdtemp(prefix="vla_isa_")

    def load_task(self, task_name: str) -> dict:
        _boot()
        from libero_isaac_sim.wrappers.libero_env import IsaacLiberoEnv

        self.env = IsaacLiberoEnv(task_name, CACHE)
        from libero_isaac_sim.semantics.task_spec import load_task

        self._task = load_task(task_name, CACHE)
        return {"language": self._task.language}

    def _pkg(self, obs: dict) -> dict:
        import imageio.v2 as imageio

        img_path = os.path.join(self.tmpdir, f"agent_{self._t}.png")
        wrist_path = os.path.join(self.tmpdir, f"wrist_{self._t}.png")
        # Isaac 渲染为正常朝向；LIBERO 原始 obs 为底向上。翻回底向上约定，
        # 与服务端对 MuJoCo 帧的处理严格对齐（服务端统一 180° 旋转）。
        imageio.imwrite(img_path, obs["agentview_image"][::-1])
        imageio.imwrite(wrist_path, obs["robot0_eye_in_hand_image"][::-1])
        state = {
            "eef_pos": [float(v) for v in obs["robot0_eef_pos"]],
            "eef_rotmat": None,
            "gripper_qpos": [float(v) for v in obs["robot0_gripper_qpos"]],
            "eef_quat_wxyz": [float(v) for v in obs["robot0_eef_quat"]],
        }
        return {"state": state, "image": img_path, "wrist_image": wrist_path}

    def serve(self):
        self._t = 0
        for line in sys.stdin:
            try:
                req = json.loads(line)
                cmd = req["cmd"]
                if cmd == "load_task":
                    out = self.load_task(req["task"])
                elif cmd == "reset":
                    obs = self.env.reset(init_state_id=req["init_state_id"])
                    out = self._pkg(obs)
                elif cmd == "step":
                    obs, _, done, info = self.env.step(req["action"])
                    out = self._pkg(obs)
                    out["success"] = bool(info["success"])
                elif cmd == "close":
                    self.env.close()
                    out = {}
                else:
                    raise ValueError(cmd)
                out["ok"] = True
            except Exception as e:  # noqa: BLE001
                import traceback

                out = {"ok": False, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"}
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
            self._t += 1


if __name__ == "__main__":
    Worker().serve()
