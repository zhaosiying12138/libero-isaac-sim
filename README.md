# libero-isaac-sim

把 LIBERO-10（LIBERO-LONG）基准从 robosuite/MuJoCo 迁移到 Isaac Sim 6.1 / PhysX（Isaac Lab v3.0.0-EA），
保持任务语义与策略接口不变：相同机器人（Franka Panda）、相同物体与场景、相同初始语义状态、
相同 7 维 OSC 动作接口（20 Hz）、相同观测 schema、相同 BDDL 成功条件。

本仓只包含本项目自研的代码与脚本。Isaac Lab、LIBERO 等开源依赖通过 pip/git 引用，不入仓；
LIBERO 资产（MJCF/网格/纹理）由 `source/libero_isaac_sim/converters/` 中的转换脚本在 setup 时
离线转换为 USD，产物写入本地缓存目录（默认 `~/.cache/libero_isaac_sim/`），不入仓。

## 环境（两台隔离环境）

- **isaac 环境**：Python 3.12 + uv，Isaac Sim 6.1，Isaac Lab `v3.0.0-EA`（pin tag `ae37b02`）。
  运行仿真、转换、回放、Isaac 侧渲染。
- **libero-mujoco 环境**：Python 3.10，robosuite 1.4.0 + mujoco 3.3.1 + LIBERO 官方仓（.pth 引用）。
  用于语义状态导出与双仿真对比实验。
- **vla 环境**（可选，闭环 VLA 实验用）：Python 3.10 + torch 2.7.1(cu128) + OpenVLA-OFT。

> ⚠️ 本机为 WSL2：PhysX GPU 路径不可用（`CUDA illegal memory access`），本仓默认 PhysX **CPU**；
> RTX 渲染需要 Vulkan（WSL2 无 NVIDIA Vulkan ICD），相机渲染走 Isaac Lab 3.0 的 Newton Warp
> 渲染器（CUDA/warp 光栅化）。在原生 Linux + NVIDIA 机器上这两个限制都不存在，
> 把 `cfg.sim.device` 改回 `cuda:0` 并把相机 renderer_cfg 换回默认 RTX 即可。

## 一键安装

```bash
bash scripts/setup_env.sh
```

脚本把 IsaacLab（v3.0.0-EA）与 LIBERO 浅克隆到 `~/codebase/_external/`（不进本仓），
建好两个环境并冒烟检查。Isaac Sim 全量下载约 10+ GB，首次安装较慢。

## 资产导出与转换

```bash
# 全 10 任务：语义导出（manifest + 50 组 init states + demo 起始态）+ 网格净化 + MJCF→USD
bash scripts/export_and_convert_all.sh
```

## 验证阶梯（每个迁移任务必须全过）

```bash
python tests/test_bddl_parser.py      # BDDL 解析（10/10）
python tests/test_math_actions.py     # 旋转数学与 OSC 动作语义
python tests/test_fk_parity.py        # L2 前向运动学（EE 误差 0.5 mm）
python tests/test_action_parity.py    # L3 动作方向/幅值
python tests/test_camera_parity.py    # L7 相机（6 点像素零误差）
python tests/test_predicate_parity.py # L8 谓词（388/388 步与官方一致）
```

## 双仿真等价性实验

```bash
# 单条 demo 锁步回放（同初始状态、同动作序列）
python experiments/paired_rollout.py --task <TASK> --demos 0 --render

# 分段回放（从 demo 任意帧的语义状态重新同步起步）
python experiments/paired_rollout.py --task <TASK> --demos 0 --segment 150:235

# 全 10 任务 × 5 demo 批量
python experiments/batch_rollout.py --tasks all --demos 0,1,2,3,4

# 指标汇总（论文表格）
python experiments/metrics.py
```

## 闭环 VLA 评测（OpenVLA-OFT）

```bash
# 策略服务器（vla 环境）：OpenVLA-OFT libero-10 checkpoint
~/codebase/_external/venvs/vla/bin/python experiments/vla_policy_server.py

# 双后端闭环评测（同一 checkpoint，MuJoCo 官方环境 + Isaac 移植环境）
python experiments/vla_eval.py --task <TASK> --trials 50 --backends mujoco,isaac \
    --video-dir outputs/vla_eval/videos/<task>
```

## 目录

```
source/libero_isaac_sim/   # 核心包：converters / semantics / envs / tasks / wrappers / eval / compat
experiments/               # 双仿真等价性实验与 VLA 闭环评测
scripts/                   # 环境安装、资产导出转换、冒烟、dt 收敛性
tests/                     # 验证阶梯 L0–L8 自动化
article/                   # 论文体发表包（zhihu.md / paper.html / figures/）
outputs/                   # 实验产物（不回放入 git 的重量数据在 .gitignore）
```

## 已知上游缺陷（本项目 compat 层绕过并注释）

1. Isaac Lab v3.0.0-EA 的 warp launch cache 把 `ProxyArray` 传进 kernel 打包器导致
   articulation 初始化崩溃（`isaaclab_ea_fixes.apply()` 直发 + 解包绕过）。
2. 3.0 EA 的 `OperationalSpaceControllerAction` 在不配任务坐标系时把零四元数缓冲传给
   `set_command`，污染 pose_rel 目标合成（我们的 `LiberoOscActionTerm` 显式传 `None`）。
3. 3.0 EA 的 `fix_root_link=True` 在场景 spawn 时与 newton 的 USD 导入器冲突
   （`FixedJoint` 无法并入 D6 组）；fixture 基座固定改由转换器自身的根焊接关节承担。
4. `ArticulationCfg.InitialStateCfg.joint_pos` 对自转 USD 不生效（joints 验证
   却按它做）；我们改为 reset 事件显式写关节态。
```

## 引用

如果你用到了本仓的迁移成果，请引用 LIBERO 与 Isaac Lab 原论文，并注明本仓。
