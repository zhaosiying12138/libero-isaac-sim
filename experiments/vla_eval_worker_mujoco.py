"""VLA 评测 MuJoCo 后端 worker（libero-mujoco 环境运行，JSON Lines 协议）。

协议::

    {"cmd": "load_task", "bddl": "<path>"}   --> {"language": ...}
    {"cmd": "reset", "init_state_id": i}     --> {"state", "image", "wrist_image"}
    {"cmd": "step", "action": [7]}           --> {"state", "image", "wrist_image", "success"}

图像为 LIBERO 原始观测帧（未翻转，256×256 PNG 文件路径）；旋转/缩放由
vla_policy_server 按 openvla-oft 管线做。
"""

import json
import os
import sys
import tempfile

import numpy as np

LIBERO_REPO = os.environ.get("LIBERO_REPO", "/home/zhaosiying/codebase/_external/LIBERO")


class Worker:
    def __init__(self):
        self.env = None
        self.tmpdir = tempfile.mkdtemp(prefix="vla_mj_")

    def load_task(self, bddl: str) -> dict:
        self._bddl = bddl
        import torch
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.bddl_utils import get_problem_info

        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=256, camera_widths=256,
        )
        self.env.seed(0)
        info = get_problem_info(bddl)
        task = os.path.splitext(os.path.basename(bddl))[0]
        self._init_states = torch.load(
            os.path.join(LIBERO_REPO, "libero/libero/init_files/libero_10", f"{task}.pruned_init"),
            map_location="cpu", weights_only=False,
        )
        return {"language": info["language_instruction"]}

    def _pkg(self) -> dict:
        obs = self.env.env._get_observations()
        img_path = os.path.join(self.tmpdir, f"agent_{self._t}.png")
        wrist_path = os.path.join(self.tmpdir, f"wrist_{self._t}.png")
        import imageio.v2 as imageio

        imageio.imwrite(img_path, obs["agentview_image"])
        imageio.imwrite(wrist_path, obs["robot0_eye_in_hand_image"])
        state = {
            "eef_pos": [float(v) for v in obs["robot0_eef_pos"]],
            "eef_rotmat": None,  # 服务端只需要轴角；这里给四元数让驱动算
            "gripper_qpos": [float(v) for v in obs["robot0_gripper_qpos"]],
            "eef_quat_xyzw": [float(v) for v in obs["robot0_eef_quat"]],
        }
        return {"state": state, "image": img_path, "wrist_image": wrist_path}

    def serve(self):
        self._t = 0
        for line in sys.stdin:
            try:
                req = json.loads(line)
                cmd = req["cmd"]
                if cmd == "load_task":
                    out = self.load_task(req["bddl"])
                elif cmd == "reset_demo":
                    import h5py
                    task = os.path.splitext(os.path.basename(self._bddl))[0]
                    with h5py.File(os.path.join(LIBERO_REPO, "libero/datasets/libero_10", f"{task}_demo.hdf5"), "r") as f:
                        st = np.asarray(f["data"][f"demo_{req['init_state_id']}"].attrs["init_state"])
                    self.env.set_init_state(st)
                    self.env.sim.forward()
                    out = self._pkg()
                elif cmd == "reset":
                    # 官方协议：先 env.reset() 清 episode 状态（robosuite 内部 done 闩锁），
                    # 再装载固定初始状态覆盖随机化
                    self.env.reset()
                    st = self._init_states[req["init_state_id"] % len(self._init_states)]
                    self.env.set_init_state(st)
                    self.env.sim.forward()
                    out = self._pkg()
                elif cmd == "step":
                    _, _, _, _ = self.env.step(np.asarray(req["action"], dtype=float))
                    out = self._pkg()
                    out["success"] = bool(self.env.check_success())
                elif cmd == "close":
                    self.env.close()
                    out = {}
                else:
                    raise ValueError(cmd)
                out["ok"] = True
            except Exception as e:  # noqa: BLE001
                out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
            self._t += 1


if __name__ == "__main__":
    Worker().serve()
