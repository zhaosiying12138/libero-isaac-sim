#!/usr/bin/env python
"""任务资产 MJCF→USD 转换入口（isaac 环境，需在 IsaacLab 仓目录 uv run）。

    OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python -u \
        /home/zhaosiying/codebase/libero-isaac-sim/scripts/convert_task_usd.py --task <task_name>
"""

import argparse
import os
import sys

sys.path.insert(0, "/home/zhaosiying/codebase/libero-isaac-sim/source")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--cache-dir", type=str, default=os.path.expanduser("~/.cache/libero_isaac_sim"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from libero_isaac_sim.compat import isaaclab_ea_fixes

isaaclab_ea_fixes.apply()

from libero_isaac_sim.converters.mjcf_to_usd import convert_task_assets


def main():
    convert_task_assets(args.task, args.cache_dir)
    app.close()
    print("[convert_task] OK")


if __name__ == "__main__":
    main()
