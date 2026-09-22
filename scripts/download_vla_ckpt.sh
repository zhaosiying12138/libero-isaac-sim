#!/usr/bin/env bash
# 下载 OpenVLA-OFT libero-10 checkpoint（hf-mirror，wget 断点续传）
set -euo pipefail
CKPT_DIR="${1:-$HOME/codebase/_external/checkpoints/openvla-oft-libero-10}"
mkdir -p "$CKPT_DIR/lora_adapter"
FILES=(
  .gitattributes README.md action_head--150000_checkpoint.pt added_tokens.json
  config.json configuration_prismatic.py dataset_statistics.json generation_config.json
  lora_adapter/README.md lora_adapter/adapter_config.json lora_adapter/adapter_model.safetensors
  model-00001-of-00004.safetensors model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors model-00004-of-00004.safetensors
  model.safetensors.index.json modeling_prismatic.py preprocessor_config.json
  processing_prismatic.py processor_config.json proprio_projector--150000_checkpoint.pt
  special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json
)
for F in "${FILES[@]}"; do
  mkdir -p "$CKPT_DIR/$(dirname "$F")"
  [ -f "$CKPT_DIR/$F" ] || wget -c -q --tries=20 --retry-connrefused --waitretry=5 \
    "https://hf-mirror.com/moojink/openvla-7b-oft-finetuned-libero-10/resolve/main/$F" -O "$CKPT_DIR/$F"
done
echo "checkpoint 完成: $CKPT_DIR"
