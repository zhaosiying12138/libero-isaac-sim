"""MuJoCo 侧工作进程：以 JSON Lines 协议在 stdin/stdout 上提供锁步回放服务。

协议（每行一个 JSON 对象）::

    --> {"cmd": "load", "bddl": "<abs path>"}
    <-- {"ok": true, "task": "<name>"}

    --> {"cmd": "reset", "init_state_id": 17}
    <-- {"ok": true, "state": <SemanticState>}

    --> {"cmd": "step", "action": [dx,dy,dz,dax,day,daz,gripper]}
    <-- {"ok": true, "state": <SemanticState>, "official_success": false}

    --> {"cmd": "render", "camera": "agentview", "path": "/tmp/f.png"}
    <-- {"ok": true, "path": "/tmp/f.png", "shape": [128,128,3]}

    --> {"cmd": "close"}

SemanticState（旋转一律 3x3 矩阵，行优先展开；约定见 manifest）::

    {
      "robot_joint_pos": [7], "gripper_qpos": [2],
      "eef_pos": [3], "eef_rotmat": [9],
      "bodies": {"<entity>": {"pos": [3], "rotmat": [9]}},
      "joints":  {"<entity>": [q0, ...]},
      "sites":   {"<site>":   {"pos": [3], "rotmat": [9]}},
      "contacts": [["a","b"], ...]   # 当前接触对（实体名级）
    }

该进程在 libero-mujoco 环境中运行，供 experiments/paired_rollout.py 驱动。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

LIBERO_REPO = os.environ.get(
    "LIBERO_REPO", "/home/zhaosiying/codebase/_external/LIBERO"
)


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


class MujocoWorker:
    def __init__(self):
        self.env = None
        self.model = None
        self.sim = None
        self.entities = []
        self.entity_joints = {}
        self.site_names = []

    # ------------------------------------------------------------------
    def load(self, bddl: str) -> dict:
        from libero.libero.envs import OffScreenRenderEnv

        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=128,
            camera_widths=128,
        )
        self.env.reset()
        self.sim = self.env.sim
        self.model = self.sim.model
        self.entities = list(self.env.env.objects_dict.keys()) + list(
            self.env.env.fixtures_dict.keys()
        )
        for name in self.entities:
            obj = self.env.env.get_object(name)
            raw = list(obj.joints) if obj.joints else []
            # 排除 freejoint（jnt_type==0），自由关节的位姿已由根位姿覆盖
            self.entity_joints[name] = [
                j for j in raw
                if int(self.model.jnt_type[self.model.joint_name2id(j)]) != 0
            ]
        self.site_names = list(self.env.env.object_sites_dict.keys())
        return {"task": os.path.splitext(os.path.basename(bddl))[0]}

    # ------------------------------------------------------------------
    def _qpos_addr(self, jname: str) -> int:
        addr = self.model.get_joint_qpos_addr(jname)
        return int(addr[0]) if isinstance(addr, (tuple, list)) else int(addr)

    def _semantic_state(self) -> dict:
        sim, model = self.sim, self.model
        sim.forward()
        state = {}

        arm_addrs = [
            self._qpos_addr(j) for j in self.env.robots[0].robot_model.joints
        ]
        gripper_addrs = [
            self._qpos_addr(j) for j in self.env.robots[0].gripper.joints
        ]
        state["robot_joint_pos"] = [float(sim.data.qpos[a]) for a in arm_addrs]
        state["gripper_qpos"] = [float(sim.data.qpos[a]) for a in gripper_addrs]

        # EE 位姿（robosuite OSC/obs 的控制帧：gripper0_grip_site site）
        eef_site = "gripper0_grip_site"
        eef_id = model.site_name2id(eef_site)
        state["eef_pos"] = [float(v) for v in sim.data.site_xpos[eef_id]]
        state["eef_rotmat"] = np.array(sim.data.site_xmat[eef_id]).reshape(3, 3).reshape(-1).tolist()

        bodies = {}
        for name in self.entities:
            body_id = self.env.env.obj_body_id[name]
            bodies[name] = {
                "pos": [float(v) for v in sim.data.body_xpos[body_id]],
                "rotmat": _quat_wxyz_to_mat(sim.data.body_xquat[body_id])
                .reshape(-1)
                .tolist(),
            }
        state["bodies"] = bodies

        joints = {}
        for name, jnames in self.entity_joints.items():
            if jnames:
                joints[name] = [
                    float(sim.data.qpos[self._qpos_addr(j)]) for j in jnames
                ]
        state["joints"] = joints

        sites = {}
        for site_name in self.site_names:
            sid = model.site_name2id(site_name)
            sites[site_name] = {
                "pos": [float(v) for v in sim.data.site_xpos[sid]],
                "rotmat": np.array(sim.data.site_xmat[sid]).reshape(3, 3)
                .reshape(-1)
                .tolist(),
            }
        state["sites"] = sites

        # 实体级接触对（基于 geom 接触的保守近似：任一 geom 接触即记为实体接触）
        contacts = []
        geom_body = model.geom_bodyid
        body_to_entity = {v: k for k, v in self.env.env.obj_body_id.items()}
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            b1, b2 = geom_body[c.geom1], geom_body[c.geom2]
            n1 = body_to_entity.get(b1)
            n2 = body_to_entity.get(b2)
            if n1 and n2 and n1 != n2:
                contacts.append([n1, n2])
        state["contacts"] = contacts
        return state

    # ------------------------------------------------------------------
    def reset(self, init_state_id: int) -> dict:
        """回放实验：demo_i 的起始状态用 demo HDF5 的 attrs.init_state
        （.pruned_init 是评测协议用的另一套状态，与示范动作不配套）。"""
        import h5py

        task = os.path.splitext(os.path.basename(self._bddl))[0]
        hdf5 = os.path.join(
            LIBERO_REPO, "libero/datasets/libero_10", f"{task}_demo.hdf5"
        )
        with h5py.File(hdf5, "r") as f:
            key = f"demo_{init_state_id}"
            state = np.asarray(f["data"][key].attrs["init_state"])
        self.env.set_init_state(state)
        # 官方协议：稳定 5 步零动作
        for _ in range(5):
            self.env.step(np.zeros(7))
        return {"state": self._semantic_state()}

    def reset_flat(self, flat_state) -> dict:
        """从任意 MuJoCo 扁平状态复位（分段回放的段起点）。"""
        self.env.set_init_state(np.asarray(flat_state))
        self.env.sim.forward()
        for _ in range(5):
            self.env.step(np.zeros(7))
        return {"state": self._semantic_state()}

    def set_joints(self, joint7) -> dict:
        """直接写机器人 7 关节角（用于 FK 对齐测试，不经控制器）。"""
        sim, model = self.sim, self.model
        arm_addrs = [
            self._qpos_addr(j) for j in self.env.robots[0].robot_model.joints
        ]
        for a, v in zip(arm_addrs, joint7):
            sim.data.qpos[a] = float(v)
            # 同步目标缓存，避免 OSC 在下一步拉回去
        sim.forward()
        return {"state": self._semantic_state()}

    def step(self, action) -> dict:
        action = np.asarray(action, dtype=float)
        assert action.shape == (7,)
        _, _, _, _ = self.env.step(action)
        official_success = bool(self.env.check_success())
        return {
            "state": self._semantic_state(),
            "official_success": official_success,
        }

    def render(self, camera: str, path: str) -> dict:
        obs = self.env.env._get_observations()
        key = f"{camera}_image"
        img = obs[key]
        import imageio

        imageio.imwrite(path, img[::-1])  # mujoco 帧上下翻转
        return {"path": path, "shape": list(img.shape)}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    # ------------------------------------------------------------------
    def serve(self):
        for line in sys.stdin:
            try:
                req = json.loads(line)
                cmd = req["cmd"]
                if cmd == "load":
                    self._bddl = req["bddl"]
                    out = self.load(req["bddl"])
                elif cmd == "reset":
                    out = self.reset(int(req["init_state_id"]))
                elif cmd == "reset_flat":
                    out = self.reset_flat(req["flat_state"])
                elif cmd == "set_joints":
                    out = self.set_joints(req["joints"])
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
            except Exception as e:  # noqa: BLE001 - 协议层兜底
                out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()


def main():
    worker = MujocoWorker()
    worker.serve()


if __name__ == "__main__":
    main()
