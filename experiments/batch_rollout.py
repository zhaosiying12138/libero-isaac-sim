"""全任务批量 paired rollout + 论文级指标汇总。

用法::

    python experiments/batch_rollout.py [--tasks all] [--demos 0,1,2,3,4] [--segment T0:T1]

产出：
- outputs/paired_rollout/<task>/demo_*.npz + summary.json
- outputs/paired_rollout/all_tasks_summary.md（论文表格）
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from metrics import aggregate, to_markdown_row
from paired_rollout import paired_rollout

LIBERO10_TASKS = [
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="all", help="all 或逗号分隔任务子串")
    parser.add_argument("--demos", type=str, default="0,1,2,3,4")
    parser.add_argument("--segment", type=str, default=None)
    args = parser.parse_args()

    if args.tasks == "all":
        tasks = LIBERO10_TASKS
    else:
        subs = args.tasks.split(",")
        tasks = [t for t in LIBERO10_TASKS if any(s in t for s in subs)]

    seg = tuple(int(v) for v in args.segment.split(":")) if args.segment else None
    demo_ids = [int(v) for v in args.demos.split(",")]

    header = (
        "| 任务 | demos | EE位置RMSE(mm) | EE姿态误差(°) | 谓词一致率 | 判决一致率 | κ | 成功率 mj/isa |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for task in tasks:
        results = []
        for d in demo_ids:
            try:
                r = paired_rollout(task, d, segment=seg)
                results.append(r)
                print(
                    f"[batch] {task[:40]} demo_{d}: ee_rmse={r['ee_pos_rmse_mm']:.1f}mm "
                    f"判决一致={r['verdict_agree']}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[batch] {task[:40]} demo_{d} FAIL: {type(e).__name__}: {e}", flush=True)
        if results:
            agg = aggregate(task)
            rows.append(to_markdown_row(agg))

    out = os.path.join(os.path.dirname(__file__), "..", "outputs", "paired_rollout", "all_tasks_summary.md")
    with open(out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[batch] 汇总表 -> {out}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
