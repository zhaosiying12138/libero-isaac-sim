"""对 10 个 LIBERO-10 BDDL 文件全量验证自研解析器。

校验内容：
1. 与官方解析结果（robosuite_parse_problem）结构对齐（若 libero-mujoco 环境可用）；
2. 字段完整性：problem_name/language/objects/fixtures/regions/init/goal；
3. goal 中引用的对象/region 均已定义。
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from libero_isaac_sim.semantics.bddl_parser import parse_bddl

LIBERO_REPO = os.environ.get(
    "LIBERO_REPO", os.path.expanduser("~/codebase/_external/LIBERO")
)
BDDL_DIR = os.path.join(LIBERO_REPO, "libero/libero/bddl_files/libero_10")


def test_parse_all_libero10():
    bddl_files = sorted(glob.glob(os.path.join(BDDL_DIR, "*.bddl")))
    assert len(bddl_files) == 10, f"应有 10 个 BDDL 文件，实际 {len(bddl_files)}"
    for path in bddl_files:
        spec = parse_bddl(path)
        assert spec["problem_name"].lower().startswith("libero_"), spec["problem_name"]
        assert spec["language"], f"{path} 缺 language"
        assert spec["objects"], f"{path} 缺 objects"
        assert spec["goal_state"], f"{path} 缺 goal"
        # goal 引用的对象必须在 objects/fixtures 中定义
        instance_names = set()
        for insts in list(spec["objects"].values()) + list(spec["fixtures"].values()):
            instance_names.update(insts)
        region_names = set(spec["regions"].keys())
        for goal in spec["goal_state"]:
            for arg in goal[1:]:
                ok = arg in instance_names or arg in region_names
                assert ok, f"{os.path.basename(path)} goal 引用未定义实体: {arg}"
        # init 中 On 引用的 region 必须已定义
        for state in spec["initial_state"]:
            if state[0] in ("on", "in") and len(state) == 3:
                assert state[1] in instance_names, f"{state[1]} 未定义"
                assert state[2] in region_names, f"{state[2]} 未定义"
        print(
            f"[OK] {os.path.basename(path)[:60]:62s} "
            f"objects={sum(len(v) for v in spec['objects'].values())} "
            f"fixtures={sum(len(v) for v in spec['fixtures'].values())} "
            f"regions={len(spec['regions'])} goals={spec['goal_state']}"
        )


def test_compare_with_official_parser():
    """若 libero-mujoco 环境可用，与官方解析器逐字段比对。"""
    try:
        from libero.libero.envs import bddl_utils  # noqa: F401
    except Exception:
        print("[SKIP] 官方 libero 环境不可用，跳过比对")
        return
    for path in sorted(glob.glob(os.path.join(BDDL_DIR, "*.bddl"))):
        ours = parse_bddl(path)
        official = bddl_utils.robosuite_parse_problem(path)
        assert ours["problem_name"] == official["problem_name"]
        assert ours["language"] == official["language_instruction"]
        assert ours["objects"] == official["objects"], os.path.basename(path)
        assert ours["fixtures"] == official["fixtures"], os.path.basename(path)
        assert set(ours["regions"]) == set(official["regions"]), os.path.basename(path)
        assert len(ours["goal_state"]) == len(official["goal_state"])
        print(f"[OK] 与官方解析一致: {os.path.basename(path)[:60]}")


if __name__ == "__main__":
    test_parse_all_libero10()
    test_compare_with_official_parser()
    print("全部通过")
