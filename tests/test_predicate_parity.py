"""L8 谓词一致性：同一批语义状态上，我方 PredicateLib vs MuJoCo 官方判决。

方法：MuJoCo 侧沿 demo 轨迹逐步导出语义状态 + 官方 check_success，
同时用我方 PredicateLib 在**同一份 MuJoCo 状态**上求值。
若两者逐步一致 → 谓词实现忠实于官方；随后 paired_rollout 中的
「谓词逐步一致率」才能归因于物理差异而非谓词实现差异。

用法：~/codebase/_external/venvs/libero-mujoco/bin/python tests/test_predicate_parity.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from experiments.bridge import MujocoBridge
from experiments.paired_rollout import load_demo_actions

BDDL = (
    "/home/zhaosiying/codebase/_external/LIBERO/libero/libero/bddl_files/"
    "libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_"
    "the_tomato_sauce_in_the_basket.bddl"
)
TASK = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
CACHE = os.path.expanduser("~/.cache/libero_isaac_sim")


class _DictView:
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


def main():
    from libero_isaac_sim.semantics.predicates import PredicateLib
    from libero_isaac_sim.semantics.task_spec import load_task

    task = load_task(TASK, CACHE)
    lib = PredicateLib(task.sites, task.joint_thresholds())

    bridge = MujocoBridge()
    bridge.load(BDDL)
    bridge.reset(0)
    actions = load_demo_actions(TASK, 0)

    agree, total = 0, 0
    mismatches = []
    for t, a in enumerate(actions):
        r = bridge.step(a)
        official = bool(r["official_success"])
        ours = bool(lib.check_goal(_DictView(r["state"]), task.goal_state))
        agree += int(official == ours)
        total += 1
        if official != ours:
            mismatches.append(t)
    bridge.close()
    print(f"[L8] 谓词逐步一致率（我方谓词 vs 官方，同在 MuJoCo 状态上）: {agree}/{total} = {agree/total:.3f}")
    if mismatches:
        print(f"[L8] 不一致步: {mismatches[:10]}")
    assert agree / total == 1.0, "谓词实现与官方不一致"
    print("[L8] 谓词实现与官方完全一致 ✓")


if __name__ == "__main__":
    main()
