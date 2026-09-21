"""paired rollout：同初始状态 + 同动作序列的双仿真锁步回放（等价性实验核心）。

流程（对每条 demo）：
1. 从 HDF5 取动作序列（7 维 OSC_POSE + 夹爪）；
2. MuJoCo 侧与 Isaac 侧都复位到同一 init_state_id（demo_i ↔ init_states[i]）；
3. 逐步锁步执行，逐步记录语义状态与判决；
4. 汇总四层指标（状态/谓词/任务/—正式性在 physics_audit）。

产出：``outputs/paired_rollout/<task>/demo_<i>.npz`` + ``summary.json``。

指标定义（论文用）：
- ee_pos_rmse_mm：EE 位置逐步 RMSE
- ee_rot_err_deg：EE 姿态逐步角误差均值
- obj_final_pos_err_mm：任务相关物体（obj_of_interest）终局位置误差
- predicate_step_agree：我方谓词在 Isaac 状态上的判定 vs MuJoCo 官方判决的逐步一致率
- verdict_agree：终局成功/失败判决一致（MuJoCo 官方 vs Isaac 我方谓词）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from bridge import MujocoBridge

ISAACLAB_DIR = os.path.expanduser("~/codebase/_external/IsaacLab")
UV = os.path.expanduser("~/miniforge3/bin/uv")
ISAAC_WORKER = os.path.join(os.path.dirname(__file__), "isaac_worker.py")
LIBERO_REPO = os.environ.get(
    "LIBERO_REPO", "/home/zhaosiying/codebase/_external/LIBERO"
)
CACHE = os.environ.get("LIBERO_ISAAC_ASSETS_DIR", os.path.expanduser("~/.cache/libero_isaac_sim"))
OUT_DIR = os.environ.get(
    "LIBERO_ISAAC_OUT", os.path.join(os.path.dirname(__file__), "..", "outputs", "paired_rollout")
)


class IsaacBridge:
    """Isaac worker 的子进程客户端（与 MujocoBridge 同协议）。"""

    def __init__(self):
        env = dict(os.environ)
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [UV, "run", "--extra", "isaacsim", "python", "-u", ISAAC_WORKER],
            cwd=ISAACLAB_DIR,
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
                raise RuntimeError("Isaac 工作进程意外退出")
            line = line.strip()
            if not line.startswith("{"):
                continue
            out = json.loads(line)
            if not out.get("ok", False):
                raise RuntimeError(f"Isaac worker 错误: {out.get('error')}")
            return out

    def load(self, task_name):
        return self._call(cmd="load", task=task_name)

    def reset(self, k):
        return self._call(cmd="reset", init_state_id=int(k))["state"]

    def step(self, action):
        return self._call(cmd="step", action=[float(a) for a in action])

    def render(self, camera, path):
        return self._call(cmd="render", camera=camera, path=path)

    def close(self):
        try:
            self._call(cmd="close")
        except Exception:
            pass
        self.proc.terminate()


def load_demo_actions(task_name: str, demo_index: int) -> np.ndarray:
    import h5py

    hdf5 = os.path.join(
        LIBERO_REPO, "libero/datasets/libero_10", f"{task_name}_demo.hdf5"
    )
    with h5py.File(hdf5, "r") as f:
        grp = f["data"][f"demo_{demo_index}"]
        actions = grp["actions"][:]
    return np.asarray(actions, dtype=float)


def _rot_err_deg(rotmat_a9, rotmat_b9) -> float:
    Ra = np.asarray(rotmat_a9).reshape(3, 3)
    Rb = np.asarray(rotmat_b9).reshape(3, 3)
    c = np.clip((np.trace(Rb @ Ra.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def paired_rollout(
    task_name: str,
    demo_index: int,
    render: bool = False,
    max_steps: int | None = None,
) -> dict:
    """执行一条 demo 的双仿真锁步回放，返回指标字典。"""
    actions = load_demo_actions(task_name, demo_index)
    if max_steps is not None:
        actions = actions[:max_steps]

    mj = MujocoBridge()
    mj.load(os.path.join(LIBERO_REPO, f"libero/libero/bddl_files/libero_10/{task_name}.bddl"))
    isa = IsaacBridge()
    isa.load(task_name)

    s_mj = mj.reset(demo_index)
    s_isa = isa.reset(demo_index)

    # 谓词层：我方谓词库在两侧语义状态上分别求值
    from libero_isaac_sim.semantics.predicates import PredicateLib
    from libero_isaac_sim.semantics.task_spec import load_task

    task = load_task(task_name, CACHE)

    class _DictView:
        """把 worker 的语义状态字典适配成 SemanticStateView。"""

        def __init__(self, state):
            self.s = state

        def body_pos(self, name):
            return np.asarray(self.s["bodies"][name]["pos"])

        def body_rotmat(self, name):
            return np.asarray(self.s["bodies"][name]["rotmat"]).reshape(3, 3)

        def site_pos(self, name):
            return np.asarray(self.s["sites"][name]["pos"])

        def site_rotmat(self, name):
            return np.asarray(self.s["sites"][name]["rotmat"]).reshape(3, 3)

        def joint_qpos(self, entity, idx):
            return float(self.s["joints"][entity][idx])

        def num_joints(self, entity):
            return len(self.s["joints"].get(entity, []))

        def in_contact(self, a, b):
            pairs = self.s.get("contacts", [])
            return [a, b] in pairs or [b, a] in pairs

    lib_mj = PredicateLib(task.sites, task.joint_thresholds())
    lib_isa = PredicateLib(task.sites, task.joint_thresholds())

    ee_pos_err, ee_rot_err = [], []
    pred_agree = []
    t = 0
    for action in actions:
        r_mj = mj.step(action)
        r_isa = isa.step(action)
        s_mj, s_isa = r_mj["state"], r_isa["state"]

        ee_pos_err.append(
            np.linalg.norm(np.array(s_mj["eef_pos"]) - np.array(s_isa["eef_pos"]))
        )
        ee_rot_err.append(_rot_err_deg(s_mj["eef_rotmat"], s_isa["eef_rotmat"]))

        v_mj = lib_mj.check_goal(_DictView(s_mj), task.goal_state)
        v_isa = lib_isa.check_goal(_DictView(s_isa), task.goal_state)
        pred_agree.append(bool(v_mj) == bool(v_isa))

        if render and t % 10 == 0:
            tmp = tempfile.mkdtemp(prefix="paired_")
            os.makedirs(os.path.join(OUT_DIR, task_name, f"demo_{demo_index}"), exist_ok=True)
            mj.render("agentview", os.path.join(OUT_DIR, task_name, f"demo_{demo_index}", f"mj_{t:04d}.png"))
            isa.render("agentview", os.path.join(OUT_DIR, task_name, f"demo_{demo_index}", f"isa_{t:04d}.png"))
        t += 1

    # 终局判决
    verdict_mj_official = bool(r_mj["official_success"])
    verdict_mj_ours = bool(lib_mj.check_goal(_DictView(s_mj), task.goal_state))
    verdict_isa_ours = bool(lib_isa.check_goal(_DictView(s_isa), task.goal_state))
    verdict_isa_env = bool(r_isa["official_success"])

    # 物体终局误差（obj_of_interest）
    obj_errs = {}
    for name in task.manifest["bddl"].get("obj_of_interest", []):
        if name in s_mj["bodies"] and name in s_isa["bodies"]:
            obj_errs[name] = float(
                np.linalg.norm(
                    np.array(s_mj["bodies"][name]["pos"]) - np.array(s_isa["bodies"][name]["pos"])
                )
            )

    result = {
        "task": task_name,
        "demo_index": demo_index,
        "num_steps": len(actions),
        "ee_pos_rmse_mm": float(np.sqrt(np.mean(np.square(ee_pos_err))) * 1000),
        "ee_pos_max_mm": float(np.max(ee_pos_err) * 1000),
        "ee_rot_err_mean_deg": float(np.mean(ee_rot_err)),
        "obj_final_pos_err": obj_errs,
        "predicate_step_agree": float(np.mean(pred_agree)),
        "verdict_mujoco_official": verdict_mj_official,
        "verdict_mujoco_ours": verdict_mj_ours,
        "verdict_isaac_ours": verdict_isa_ours,
        "verdict_isaac_env": verdict_isa_env,
        "verdict_agree": verdict_mj_official == verdict_isa_env,
        "verdict_agree_ours": verdict_mj_ours == verdict_isa_ours,
    }

    os.makedirs(os.path.join(OUT_DIR, task_name), exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT_DIR, task_name, f"demo_{demo_index}.npz"),
        ee_pos_err=np.array(ee_pos_err),
        ee_rot_err=np.array(ee_rot_err),
        pred_agree=np.array(pred_agree),
    )
    mj.close()
    isa.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,
                        default="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket")
    parser.add_argument("--demos", type=str, default="0", help="逗号分隔 demo 序号")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    results = []
    for d in [int(x) for x in args.demos.split(",")]:
        r = paired_rollout(args.task, d, render=args.render, max_steps=args.max_steps)
        results.append(r)
        print(
            f"[paired] demo_{d}: ee_rmse={r['ee_pos_rmse_mm']:.1f}mm "
            f"谓词逐步一致率={r['predicate_step_agree']:.3f} "
            f"判决: mujoco官方={r['verdict_mujoco_official']} isaac={r['verdict_isaac_env']} "
            f"一致={r['verdict_agree']}"
        )
    out = os.path.join(OUT_DIR, args.task, "summary.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[paired] summary -> {out}")


if __name__ == "__main__":
    main()
