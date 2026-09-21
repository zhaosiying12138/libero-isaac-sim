# libero-isaac-sim

将 LIBERO-10（LIBERO-LONG）基准从 robosuite/MuJoCo 迁移到 Isaac Sim 6.1 / PhysX（Isaac Lab v3.0.0-EA），
保持任务语义与策略接口不变：相同机器人（Franka Panda）、相同物体与场景、相同初始语义状态、
相同 7D OSC 动作接口（20 Hz）、相同观测 schema、相同 BDDL 成功条件。

本仓只包含本项目自研的代码与脚本。Isaac Lab、LIBERO 等开源依赖通过 pip/git 引用，不入仓；
LIBERO 资产（MJCF/网格/纹理）由 `source/libero_isaac_sim/converters/` 中的转换脚本在 setup 时
离线转换为 USD，产物写入本地缓存目录（默认 `~/.cache/libero_isaac_sim/`），不入仓。

## 目录结构

```
source/libero_isaac_sim/   # 核心包：converters / semantics / envs / tasks / wrappers / eval / rl
experiments/               # 双仿真等价性实验（paired rollout、指标、渲染对比、闭环 VLA、物理审计）
scripts/                   # 环境安装、资产导出转换、评测、遥操作、训练入口
article/                   # 论文体发表包（HTML + 知乎 markdown + figures）
tests/                     # 验证阶梯 L0-L8 自动化测试
```

## 环境

- isaac 环境：Python 3.12 + uv，Isaac Sim 6.1，Isaac Lab v3.0.0-EA（pin tag `ae37b02`）
- libero-mujoco 环境：Python 3.11，robosuite + mujoco==3.3.1，LIBERO 官方仓可编辑安装
  （仅用于语义状态导出与双仿真对比实验）

详见 `scripts/setup_env.sh`。

## 状态

实施进度见 git 历史与 tag。当前：Phase 0（环境搭建）。
