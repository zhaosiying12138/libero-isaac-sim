#!/usr/bin/env python
"""把导出的机器人 MJCF 转成 USD 并体检（isaac 环境）。

用法（在 IsaacLab 仓目录）::

    OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python -u \
        /home/zhaosiying/codebase/libero-isaac-sim/scripts/convert_robot.py
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

from libero_isaac_sim.converters.mjcf_to_usd import (audit_usd, convert_mjcf, fix_articulation_masses, strip_custom_attrs, add_world_fixed_joint)

CACHE = os.path.expanduser("~/.cache/libero_isaac_sim")


def main():
    for variant in ["panda_on_ground", "panda_mounted"]:
        xml = os.path.join(CACHE, "mjcf_robot", f"{variant}.xml")
        usd_dir = os.path.join(CACHE, "usd", "robots", variant)
        os.makedirs(usd_dir, exist_ok=True)
        usd_path = convert_mjcf(xml, usd_dir, fix_base=True, import_sites=True)
        import json as _json
        masses = _json.load(open(os.path.join(CACHE, "mjcf_robot", f"{variant}_masses.json")))
        n = fix_articulation_masses(usd_path, masses)
        print(f"[convert] {variant}: 写入 {n} 个 link 的质量")
        n2 = strip_custom_attrs(usd_path)
        print(f"[convert] {variant}: 剥除 newton 自定义属性 {n2} 个")
        report = audit_usd(usd_path)
        print(f"[convert] {variant} -> {usd_path}")
        print(f"[convert]   bodies={len(report['bodies'])} joints={len(report['joints'])}")
        print(f"[convert]   joints: {[j.split('/')[-1] for j in report['joints']][:10]}")
        print(f"[convert]   total_mass={report['total_mass']:.2f} kg, "
              f"missing_refs={len(report['missing_refs'])}")
    app.close()
    print("[convert] OK")


if __name__ == "__main__":
    main()
