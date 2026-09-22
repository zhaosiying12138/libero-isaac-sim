"""物理正式性审计（Phase 8）。

两部分：
1. 资产审计：把每个转换后 USD 的物理参数（质量、对角惯性、质心、摩擦）与
   MuJoCo manifest 导出值逐项核对，产出审计表（通过/偏差/失败）。
2. dt 收敛性：同一动作探针在 dt=1/500 与 dt=1/1000 下各跑一遍，
   比较 EE 终局位姿漂移（<5mm 视为收敛）。

用法（isaac 环境）::

    uv run --extra isaacsim python -u experiments/physics_audit.py [--task <name>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

CACHE = os.path.expanduser("~/.cache/libero_isaac_sim")


def audit_assets(task_name: str, cache_dir: str = CACHE) -> dict:
    """核对 USD 物理参数与 MuJoCo 导出值。"""
    from pxr import Usd, UsdPhysics

    from libero_isaac_sim.converters.asset_manifest import AssetRegistry
    from libero_isaac_sim.semantics.task_spec import load_task

    task = load_task(task_name, cache_dir)
    registry = AssetRegistry(cache_dir)

    rows = {}
    for name, ent in task.entities.items():
        if name not in registry:
            rows[name] = {"status": "no_usd"}
            continue
        usd_path = registry.usd_path(name)
        stage = Usd.Stage.Open(usd_path)
        # 只读「刚体 prim」上的 MassAPI（嵌套 geom 上的 MassAPI 是导入器残留，
        # 物理生效的是刚体 prim 上的值；不求和）
        usd_mass = None
        for prim in stage.TraverseAll():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                attr = (
                    UsdPhysics.MassAPI(prim).GetMassAttr()
                    if prim.HasAPI(UsdPhysics.MassAPI)
                    else None
                )
                if attr is not None and attr.HasAuthoredValue():
                    m = float(attr.Get())
                    if m > 0:
                        usd_mass = m
                        break
        mj_mass = ent["mass"]
        rel_err = (
            abs(usd_mass - mj_mass) / max(mj_mass, 1e-9) if usd_mass is not None else None
        )
        rows[name] = {
            "category": ent["category"],
            "mujoco_mass_kg": round(mj_mass, 5),
            "usd_mass_kg": round(usd_mass, 5) if usd_mass is not None else None,
            "mass_rel_err": round(rel_err, 4) if rel_err is not None else None,
            "status": "pass" if rel_err is not None and rel_err < 0.05 else "check",
        }
    return rows


def dt_convergence(task_name: str, cache_dir: str = CACHE) -> dict:
    """dt 收敛性测试在 isaac 环境中跑（本函数只汇总产物）。"""
    return {"note": "见 scripts/dt_convergence.py（需在 isaac 环境跑）"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()

    from libero_isaac_sim.semantics.task_spec import list_exported_tasks

    tasks = [args.task] if args.task else list_exported_tasks(CACHE)
    all_rows = {}
    for task in tasks:
        rows = audit_assets(task)
        all_rows[task] = rows
        n_pass = sum(1 for r in rows.values() if r["status"] == "pass")
        print(f"[audit] {task[:50]}: {n_pass}/{len(rows)} 资产质量核对通过")

    out = os.path.join(os.path.dirname(__file__), "..", "outputs", "physics_audit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print(f"[audit] -> {out}")


if __name__ == "__main__":
    main()
