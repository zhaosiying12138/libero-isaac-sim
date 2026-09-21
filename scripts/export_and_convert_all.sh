#!/usr/bin/env bash
# 任务资产一键导出+转换编排脚本。
#
# Stage 1（libero-mujoco 环境）：
#   - 导出任务语义（manifest + 50 组 init states）
#   - 导出机器人 MJCF（mounted / on_ground 两变体）
#   - 净化 .msh 网格为 OBJ
# Stage 2（isaac 环境，需在 IsaacLab 仓目录执行 uv run）：
#   - 全部资产 MJCF → USD + 体检报告
#
# 用法: bash scripts/export_and_convert_all.sh [task_bddl_path ...]
#   无参数时处理 LIBERO-10 全部 10 个任务。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_DIR="${LIBERO_REPO:-$HOME/codebase/_external/LIBERO}"
MUJOCO_PY="${LIBERO_MUJOCO_PYTHON:-$HOME/codebase/_external/venvs/libero-mujoco/bin/python}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/codebase/_external/IsaacLab}"
UV="${UV:-$HOME/miniforge3/bin/uv}"
CACHE="${LIBERO_ISAAC_ASSETS_DIR:-$HOME/.cache/libero_isaac_sim}"

export PYTHONPATH="$REPO_ROOT/source:${PYTHONPATH:-}"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMNI_KIT_ACCEPT_EULA=YES

if [ "$#" -eq 0 ]; then
    BDDLS=("$LIBERO_DIR/libero/libero/bddl_files/libero_10/"*.bddl)
else
    BDDLS=("$@")
fi

echo "==> Stage 1: 语义导出 + 机器人导出 + 网格净化（mujoco 环境）"
"$MUJOCO_PY" -m libero_isaac_sim.converters.export_robot_mjcf --out-dir "$CACHE/mjcf_robot"

for BDDL in "${BDDLS[@]}"; do
    TASK=$(basename "$BDDL" .bddl)
    echo "==> [task] $TASK"
    "$MUJOCO_PY" -m libero_isaac_sim.converters.export_libero_task --bddl "$BDDL" --out-dir "$CACHE"
done

echo "==> Stage 1.5: 净化任务相关 XML 的 .msh 网格"
"$MUJOCO_PY" - "$CACHE" << 'PYEOF'
import json, os, sys
from libero_isaac_sim.converters.sanitize_meshes import sanitize_xml
from libero_isaac_sim.converters.mjcf_to_usd import find_object_xml

cache = sys.argv[1]
xmls = set()
for fn in os.listdir(cache):
    if not fn.endswith("_manifest.json"):
        continue
    m = json.load(open(os.path.join(cache, fn)))
    for name, ent in m["entities"].items():
        xmls.add(find_object_xml(ent["category"]))
    xmls.add(m["arena"]["scene_xml"])
for x in sorted(xmls):
    try:
        sanitize_xml(x)
    except Exception as e:
        print(f"[sanitize][WARN] {x}: {type(e).__name__}: {e}")
print("[sanitize] 完成")
PYEOF

echo "==> Stage 2: MJCF → USD（isaac 环境）"
cd "$ISAACLAB_DIR"
for BDDL in "${BDDLS[@]}"; do
    TASK=$(basename "$BDDL" .bddl)
    "$UV" run --extra isaacsim python -u "$REPO_ROOT/scripts/convert_task_usd.py" --task "$TASK"
done

echo "==> 全部完成。产物在 $CACHE"
