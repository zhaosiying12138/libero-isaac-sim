#!/usr/bin/env bash
# libero-isaac-sim 一键环境安装脚本。
#
# 建两个隔离环境：
#   1. isaac 环境   —— IsaacLab v3.0.0-EA + Isaac Sim 6.1（uv，py3.12）
#   2. libero-mujoco 环境 —— robosuite 1.4 + mujoco 3.3.1（uv venv，py3.10）
#      仅用于语义状态导出与双仿真对比实验。
#
# 用法: bash scripts/setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${EXTERNAL_DIR:-$HOME/codebase/_external}"
ISAACLAB_DIR="$EXTERNAL_DIR/IsaacLab"
LIBERO_DIR="$EXTERNAL_DIR/LIBERO"
MUJOCO_VENV="$EXTERNAL_DIR/venvs/libero-mujoco"
UV="${UV:-$HOME/miniforge3/bin/uv}"

ISAACLAB_TAG="v3.0.0-EA"   # pin: ae37b028ea415c91ea2bc32609efcd759ed2b974

echo "==> [1/4] 克隆外部依赖（不入本仓）"
mkdir -p "$EXTERNAL_DIR"
if [ ! -d "$ISAACLAB_DIR" ]; then
    git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
if [ ! -d "$LIBERO_DIR" ]; then
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi

echo "==> [2/4] isaac 环境（IsaacLab $ISAACLAB_TAG + Isaac Sim 6.1，uv 解析，体积大）"
cd "$ISAACLAB_DIR"
OMNI_KIT_ACCEPT_EULA=YES "$UV" sync --extra isaacsim --python 3.12
# 把本仓包装进 isaac 环境
"$UV" pip install -e "$REPO_ROOT"

echo "==> [3/4] libero-mujoco 环境（py3.10）"
if [ ! -d "$MUJOCO_VENV" ]; then
    "$UV" venv "$MUJOCO_VENV" --python 3.10 --seed
fi
"$MUJOCO_VENV/bin/pip" install -q \
    robosuite==1.4.0 mujoco==3.3.1 bddl==1.0.1 gym==0.25.2 "numpy==1.26.4" \
    easydict cloudpickle einops future h5py imageio imageio-ffmpeg matplotlib \
    opencv-python-headless termcolor pynput numba scipy qpsolvers xmltodict pyopengl \
    huggingface_hub
"$MUJOCO_VENV/bin/pip" install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# LIBERO 通过 .pth 引用（其仓库顶层缺 __init__.py，可编辑安装不可靠）
echo "$LIBERO_DIR" > "$MUJOCO_VENV/lib/python3.10/site-packages/libero_repo.pth"
echo "$REPO_ROOT/source" > "$MUJOCO_VENV/lib/python3.10/site-packages/libero_isaac_sim.pth"

# LIBERO 首次导入需要 ~/.libero/config.yaml（非交互写入）
if [ ! -f "$HOME/.libero/config.yaml" ]; then
    mkdir -p "$HOME/.libero"
    cat > "$HOME/.libero/config.yaml" << EOF
benchmark_root: $LIBERO_DIR/libero/libero
bddl_files: $LIBERO_DIR/libero/libero/bddl_files
init_states: $LIBERO_DIR/libero/libero/init_files
datasets: $LIBERO_DIR/libero/datasets
assets: $LIBERO_DIR/libero/libero/assets
EOF
fi

echo "==> [4/4] 冒烟检查"
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 "$MUJOCO_VENV/bin/python" -c "
from libero.libero.envs import OffScreenRenderEnv
print('[setup] LIBERO MuJoCo 侧 OK')
"
cd "$ISAACLAB_DIR"
OMNI_KIT_ACCEPT_EULA=YES "$UV" run --extra isaacsim python -c "
import isaaclab, isaacsim
print('[setup] Isaac 侧 OK')
"

# ---- 可选：VLA 闭环评测环境（OpenVLA-OFT）----
if [ "${SETUP_VLA:-0}" = "1" ]; then
    echo "==> [可选] VLA 环境（OpenVLA-OFT，torch cu128）"
    "$UV" venv "$EXTERNAL_DIR/venvs/vla" --python 3.10 --seed
    "$EXTERNAL_DIR/venvs/vla/bin/pip" install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    "$EXTERNAL_DIR/venvs/vla/bin/pip" install -q -i https://pypi.org/simple \
        "transformers==4.51.3" "timm==0.9.16" tokenizers einops accelerate peft pillow \
        imageio imageio-ffmpeg h5py json_numpy draccus rich jsonlines wandb diffusers \
        "huggingface-hub==0.30.2" tensorflow-cpu
    # openvla-oft 训练链路的重依赖在推理路径不触及，用桩模块代替
    STUBD="$EXTERNAL_DIR/venvs/vla/lib/python3.10/site-packages"
    echo 'class DLataset: pass' > "$STUBD/dlimp.py"
    mkdir -p "$STUBD/tensorflow_graphics/geometry"
    echo '' > "$STUBD/tensorflow_graphics/__init__.py"
    echo '' > "$STUBD/tensorflow_graphics/geometry/__init__.py"
    echo 'def __getattr__(n): raise NotImplementedError("stub")' > "$STUBD/tensorflow_graphics/geometry/transformation.py"
    echo 'def builder(*a, **k): raise NotImplementedError("stub")' > "$STUBD/tensorflow_datasets.py"
    # checkpoint（wget 断点续传，镜像站）
    echo "    checkpoint 下载见 scripts/download_vla_ckpt.sh"
fi

echo "==> 完成。下一步：bash scripts/export_and_convert_all.sh"
