#!/usr/bin/env python
"""批量转换 LIBERO-10 全部 10 个任务的资产（单个 Isaac 进程）。

前置：export_and_convert_all.sh 的 Stage 1（语义导出 + 网格净化）已完成。

用法（IsaacLab 仓目录）::

    OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python -u \
        /home/zhaosiying/codebase/libero-isaac-sim/scripts/convert_all_tasks.py
"""

import argparse
import os
import sys

sys.path.insert(0, "/home/zhaosiying/codebase/libero-isaac-sim/source")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from libero_isaac_sim.compat import isaaclab_ea_fixes

isaaclab_ea_fixes.apply()

from libero_isaac_sim.converters.mjcf_to_usd import convert_task_assets
from libero_isaac_sim.semantics.task_spec import list_exported_tasks

CACHE = os.path.expanduser("~/.cache/libero_isaac_sim")


def main():
    tasks = list_exported_tasks(CACHE)
    print(f"[batch] 共 {len(tasks)} 个任务待转换")
    for task in tasks:
        try:
            convert_task_assets(task, CACHE)
            print(f"[batch] OK {task[:60]}")
        except Exception as e:  # noqa: BLE001
            print(f"[batch] FAIL {task[:60]}: {type(e).__name__}: {e}")
    app.close()
    print("[batch] done")


if __name__ == "__main__":
    main()
