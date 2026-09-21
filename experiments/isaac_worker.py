"""Isaac 侧工作进程：与 mujoco_worker.py 相同的 JSON Lines 协议（stdin/stdout）。

在 isaac 环境（uv run --extra isaacsim python）中运行::

    uv run --extra isaacsim python -u experiments/isaac_worker.py

命令::

    {"cmd": "load", "task": "<task_name>"}
    {"cmd": "reset", "init_state_id": k}
    {"cmd": "step", "action": [7]}
    {"cmd": "render", "camera": "agentview", "path": "..."}
    {"cmd": "close"}

SemanticState 与 MuJoCo 侧同构（rotmat 行优先 3x3 展开，eef 为 grip_site 帧）。
"""

from __future__ import annotations

import json
import os
import sys

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


def _quat_xyzw_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    return _quat_wxyz_to_mat([w, x, y, z])


def _quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


class IsaacWorker:
    def __init__(self):
        self.env = None
        self.task = None

    def load(self, task_name: str) -> dict:
        _boot()
        from libero_isaac_sim.wrappers.libero_env import IsaacLiberoEnv

        self.env = IsaacLiberoEnv(task_name, CACHE, init_source="demo")
        self.env.reset(init_state_id=0)
        from libero_isaac_sim.semantics.task_spec import load_task

        self.task = load_task(task_name, CACHE)
        return {"task": task_name}

    # ------------------------------------------------------------------
    def _semantic_state(self) -> dict:
        from libero_isaac_sim.envs.mdp.terminations import IsaacSemanticView

        env = self.env.unwrapped
        view = IsaacSemanticView(env, 0, self.task.task_name, CACHE)

        robot = env.scene["robot"]
        arm_ids, _ = robot.find_joints(["robot0_joint[1-7]"])
        grip_ids, _ = robot.find_joints(["gripper0_finger_joint[12]"])
        origin = env.scene.env_origins[0].cpu().numpy()

        # EE = grip_site 帧（hand ∘ z+0.097）
        ee_idx = robot.find_bodies("robot0_right_hand")[0][0]
        hand = robot.data.body_pose_w.torch[0, ee_idx].cpu().numpy()
        hand_rot = _quat_xyzw_to_mat(hand[3:7])
        eef_pos = (hand[:3] - origin) + hand_rot @ np.array([0.0, 0.0, 0.097])

        bodies = {}
        for name in self.task.entities:
            if name not in env.scene.keys():
                continue
            pos, rot = view._asset_pose_world(name)
            bodies[name] = {"pos": pos.tolist(), "rotmat": rot.reshape(-1).tolist()}

        joints = {}
        for name, ent in self.task.entities.items():
            if name in env.scene.keys() and ent["joints"]:
                joints[name] = view.joint_qpos(name, 0)

        sites = {}
        for site_name in self.task.sites:
            sites[site_name] = {
                "pos": view.site_pos(site_name).tolist(),
                "rotmat": view.site_rotmat(site_name).reshape(-1).tolist(),
            }

        return {
            "robot_joint_pos": robot.data.joint_pos.torch[0, arm_ids].cpu().numpy().tolist(),
            "gripper_qpos": robot.data.joint_pos.torch[0, grip_ids].cpu().numpy().tolist(),
            "eef_pos": eef_pos.tolist(),
            "eef_rotmat": hand_rot.reshape(-1).tolist(),
            "bodies": bodies,
            "joints": {k: [v] for k, v in joints.items()},
            "sites": sites,
            "contacts": [],  # 接触对由几何谓词侧处理；精确接触在 L6 阶段补
        }

    # ------------------------------------------------------------------
    def reset(self, init_state_id: int) -> dict:
        self.env.reset(init_state_id=init_state_id)
        return {"state": self._semantic_state()}

    def step(self, action) -> dict:
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=float))
        return {
            "state": self._semantic_state(),
            "official_success": bool(info["success"]),
        }

    def render(self, camera: str, path: str) -> dict:
        img = self.env.render(camera)
        import imageio

        imageio.imwrite(path, img)
        return {"path": path, "shape": list(img.shape)}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def serve(self):
        for line in sys.stdin:
            try:
                req = json.loads(line)
                cmd = req["cmd"]
                if cmd == "load":
                    out = self.load(req["task"])
                elif cmd == "reset":
                    out = self.reset(int(req["init_state_id"]))
                elif cmd == "step":
                    out = self.step(req["action"])
                elif cmd == "render":
                    out = self.render(req.get("camera", "agentview"), req["path"])
                elif cmd == "close":
                    self.close()
                    out = {}
                else:
                    raise ValueError(f"未知命令 {cmd}")
                out["ok"] = True
            except Exception as e:  # noqa: BLE001
                out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()


def main():
    IsaacWorker().serve()


if __name__ == "__main__":
    main()
