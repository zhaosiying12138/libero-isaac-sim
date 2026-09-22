"""OpenVLA-OFT 策略服务器：JSON Lines 协议（stdin/stdout），常驻加载一次模型。

协议::

    --> {"cmd": "infer", "image": "<png path>", "wrist_image": "<png path>",
         "state": [eef_pos(3), axisangle(3), gripper_qpos(2)], "prompt": "..."}
    <-- {"ok": true, "actions": [[7], ...]}   # 动作 chunk（8 步）

    --> {"cmd": "reset"}    # 清空 open-loop 队列上下文（每 episode 开始时调用）

图像约定（与 openvla-oft 训练分布一致）：调用方负责传入「180° 旋转 + 中心裁剪」
后的帧（见 vla_eval 各后端 worker 的 preprocess）。

运行环境：vla venv（torch GPU）。模型路径默认本地下载的
moojink/openvla-7b-oft-finetuned-libero-10。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

OPENVLA_OFT_REPO = os.environ.get(
    "OPENVLA_OFT_REPO", "/home/zhaosiying/codebase/_external/openvla-oft"
)
CKPT = os.environ.get(
    "OPENVLA_CKPT",
    "/home/zhaosiying/codebase/_external/checkpoints/openvla-oft-libero-10",
)


sys.path.insert(0, OPENVLA_OFT_REPO)


def _load_policy():
    import torch
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla,
    )

    # 自包含配置：与 openvla-oft 的 GenerateConfig 字段一致，但不引入
    # run_libero_eval（它会连带 LIBERO/MuJoCo 依赖链，服务器侧不需要）
    from dataclasses import dataclass

    @dataclass
    class _Cfg:
        model_family: str = "openvla"
        pretrained_checkpoint: str = CKPT
        use_l1_regression: bool = True
        use_diffusion: bool = False
        num_diffusion_steps_train: int = 50
        num_diffusion_steps_inference: int = 50
        use_film: bool = False
        num_images_in_input: int = 2
        use_proprio: bool = True
        center_crop: bool = True
        num_open_loop_steps: int = 8
        lora_rank: int = 32
        unnorm_key: str = "libero_10_no_noops"
        load_in_8bit: bool = False
        load_in_4bit: bool = False
        seed: int = 7

    cfg = _Cfg()
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    return cfg, vla, processor, action_head, proprio_projector


def _load_image(path: str) -> np.ndarray:
    import imageio.v2 as imageio

    return imageio.imread(path)


def main():
    from experiments.robot.openvla_utils import get_vla_action

    cfg, vla, processor, action_head, proprio_projector = _load_policy()
    print("[vla-server] 模型已加载", flush=True)

    for line in sys.stdin:
        try:
            req = json.loads(line)
            cmd = req["cmd"]
            if cmd == "infer":
                obs = {
                    "full_image": _load_image(req["image"]),
                    "wrist_image": _load_image(req["wrist_image"]),
                    "state": np.asarray(req["state"], dtype=float),
                }
                actions = get_vla_action(
                    cfg,
                    vla,
                    processor,
                    obs,
                    req["prompt"],
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                )
                out = {"actions": [np.asarray(a).tolist() for a in actions]}
            elif cmd == "reset":
                out = {}
            else:
                raise ValueError(f"未知命令 {cmd}")
            out["ok"] = True
        except Exception as e:  # noqa: BLE001
            import traceback

            out = {"ok": False, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
