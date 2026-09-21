"""等价性实验指标聚合（论文数据源）。

读取 outputs/paired_rollout/<task>/demo_*.npz 与 summary.json，
输出论文级指标表（markdown + latex 行）：

- 状态层：EE 位置 RMSE/最大误差（mm）、EE 姿态误差（度）均值±std
- 谓词层：逐步谓词一致率
- 任务层：终局判决一致率 + Cohen's κ + 双成功/双失败/翻转计数
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "paired_rollout")


def cohens_kappa(verdicts_mj: list[bool], verdicts_isa: list[bool]) -> float:
    """双 raters 的 Cohen κ。全同或全异时退化处理。"""
    n = len(verdicts_mj)
    if n == 0:
        return float("nan")
    agree = sum(1 for a, b in zip(verdicts_mj, verdicts_isa) if a == b)
    p_o = agree / n
    p_mj = sum(verdicts_mj) / n
    p_isa = sum(verdicts_isa) / n
    p_e = p_mj * p_isa + (1 - p_mj) * (1 - p_isa)
    if abs(1 - p_e) < 1e-12:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def aggregate(task: str, out_dir: str = OUT_DIR) -> dict:
    summary_path = os.path.join(out_dir, task, "summary.json")
    with open(summary_path) as f:
        results = json.load(f)

    ee_rmse = [r["ee_pos_rmse_mm"] for r in results]
    ee_max = [r["ee_pos_max_mm"] for r in results]
    ee_rot = [r["ee_rot_err_mean_deg"] for r in results]
    pred_agree = [r["predicate_step_agree"] for r in results]
    v_mj = [r["verdict_mujoco_official"] for r in results]
    v_isa = [r["verdict_isaac_env"] for r in results]
    agree_flags = [r["verdict_agree"] for r in results]

    n = len(results)
    both_succ = sum(1 for a, b in zip(v_mj, v_isa) if a and b)
    both_fail = sum(1 for a, b in zip(v_mj, v_isa) if not a and not b)
    flips = n - both_succ - both_fail
    kappa = cohens_kappa(v_mj, v_isa)

    obj_errs = {}
    for r in results:
        for k, v in r["obj_final_pos_err"].items():
            obj_errs.setdefault(k, []).append(v)

    out = {
        "task": task,
        "n_demos": n,
        "ee_pos_rmse_mm": {"mean": float(np.mean(ee_rmse)), "std": float(np.std(ee_rmse))},
        "ee_pos_max_mm_mean": float(np.mean(ee_max)),
        "ee_rot_err_deg_mean": float(np.mean(ee_rot)),
        "predicate_step_agree": {
            "mean": float(np.mean(pred_agree)),
            "std": float(np.std(pred_agree)),
        },
        "verdict_agreement_rate": sum(agree_flags) / n,
        "both_success": both_succ,
        "both_fail": both_fail,
        "flips": flips,
        "cohens_kappa": kappa,
        "mujoco_success_rate": sum(v_mj) / n,
        "isaac_success_rate": sum(v_isa) / n,
        "obj_final_pos_err_mm": {
            k: {"mean": float(np.mean(v) * 1000), "std": float(np.std(v) * 1000)}
            for k, v in obj_errs.items()
        },
    }
    return out


def to_markdown_row(agg: dict) -> str:
    """论文表格行（markdown）。"""
    return (
        f"| {agg['task'][:40]} | {agg['n_demos']} | "
        f"{agg['ee_pos_rmse_mm']['mean']:.1f}±{agg['ee_pos_rmse_mm']['std']:.1f} | "
        f"{agg['ee_rot_err_deg_mean']:.1f} | "
        f"{agg['predicate_step_agree']['mean']*100:.1f}% | "
        f"{agg['verdict_agreement_rate']*100:.0f}% | "
        f"{agg['cohens_kappa']:.2f} | "
        f"{agg['mujoco_success_rate']*100:.0f}%/{agg['isaac_success_rate']*100:.0f}% |"
    )


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else (
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
    )
    agg = aggregate(task)
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print()
    print("| 任务 | demos | EE位置RMSE(mm) | EE姿态误差(°) | 谓词一致率 | 判决一致率 | κ | 成功率 mj/isa |")
    print("|---|---|---|---|---|---|---|---|")
    print(to_markdown_row(agg))


if __name__ == "__main__":
    main()
