"""论文配图生成：从实验数据产出 figures。

产出（article/figures/）：
- fig_ee_error_curve.png：demo_0 的 EE 位置误差随时间曲线（双 sim）
- fig_task_metrics.png：10 任务的关键指标柱状图
- fig_dt_convergence.png：dt 收敛性
- fig_predicate_parity.png：谓词逐步一致率分布

用法：~/codebase/_external/venvs/libero-mujoco/bin/python experiments/make_figures.py
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "outputs", "paired_rollout")
FIG = os.path.join(REPO, "article", "figures")
os.makedirs(FIG, exist_ok=True)

import matplotlib.font_manager as fm

# 中文字体（论文图表要求）：注册 Noto Serif CJK
_CJK = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK)
    _CJK_NAME = fm.FontProperties(fname=_CJK).get_name()
else:
    _CJK_NAME = "sans-serif"

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": _CJK_NAME,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def fig_ee_error_curve(task: str, demo: int = 0):
    npz = os.path.join(OUT, task, f"demo_{demo}.npz")
    if not os.path.exists(npz):
        print(f"[fig] 缺数据 {npz}")
        return
    d = np.load(npz)
    err_mm = d["ee_pos_err"] * 1000
    rot_deg = d["ee_rot_err"]

    fig, ax1 = plt.subplots(figsize=(6.4, 3.2))
    ax1.plot(err_mm, color="#1f77b4", lw=1.5, label="EE 位置误差")
    ax1.set_xlabel("控制步（20 Hz）")
    ax1.set_ylabel("位置误差 (mm)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(rot_deg, color="#d62728", lw=1.2, alpha=0.8, label="EE 姿态误差")
    ax2.set_ylabel("姿态误差 (°)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.spines["top"].set_visible(False)
    ax1.set_title("同动作序列锁步回放：EE 误差随时间（demo_0）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ee_error_curve.png"), bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_ee_error_curve.png")


def fig_task_metrics():
    path = os.path.join(OUT, "all_tasks_summary.md")
    if not os.path.exists(path):
        print("[fig] 缺汇总表")
        return
    rows = []
    for line in open(path):
        line = line.strip()
        if line.startswith("|") and "RMSE" not in line and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 8:
                rows.append(cells)
    names = [r[0][:22] for r in rows]
    rmse = [float(r[2].split("±")[0]) for r in rows]
    pred = [float(r[4].rstrip("%")) for r in rows]
    agree = [float(r[5].rstrip("%")) for r in rows]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, vals, title, color in [
        (axes[0], rmse, "EE 位置 RMSE (mm)", "#1f77b4"),
        (axes[1], pred, "谓词逐步一致率 (%)", "#2ca02c"),
        (axes[2], agree, "终局判决一致率 (%)", "#d62728"),
    ]:
        ax.bar(x, vals, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_task_metrics.png"), bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_task_metrics.png")


def fig_dt_convergence():
    path = os.path.join(REPO, "outputs", "dt_convergence.json")
    if not os.path.exists(path):
        print("[fig] 缺 dt 数据")
        return
    d = json.load(open(path))
    p1 = np.array(d["dt=0.00200"]["pos"])
    p2 = np.array(d["dt=0.00100"]["pos"])
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    labels = ["x", "y", "z"]
    xx = np.arange(3)
    w = 0.35
    ax.bar(xx - w / 2, p1, w, label="dt=1/500", color="#1f77b4")
    ax.bar(xx + w / 2, p2, w, label="dt=1/1000", color="#ff7f0e")
    ax.set_xticks(xx)
    ax.set_xticklabels(labels)
    ax.set_ylabel("EE 终局位置 (m)")
    ax.set_title(f"dt 减半终局漂移 {d['drift_mm']:.1f} mm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dt_convergence.png"), bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_dt_convergence.png")


def fig_predicate_parity():
    """谓词逐步一致率直方图（跨任务）。"""
    rates = []
    for npz in glob.glob(os.path.join(OUT, "*", "demo_*.npz")):
        d = np.load(npz)
        rates.append(float(d["pred_agree"].mean()))
    if not rates:
        print("[fig] 缺谓词数据")
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.hist(np.array(rates) * 100, bins=20, color="#2ca02c", alpha=0.85)
    ax.set_xlabel("谓词逐步一致率 (%)")
    ax.set_ylabel("episode 数")
    ax.set_title(f"全部回放（{len(rates)} 条）谓词一致率分布")
    ax.axvline(np.mean(rates) * 100, color="k", ls="--", lw=1, label=f"均值 {np.mean(rates)*100:.1f}%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_predicate_parity.png"), bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_predicate_parity.png")


def main():
    task0 = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
    fig_ee_error_curve(task0, 0)
    fig_task_metrics()
    fig_dt_convergence()
    fig_predicate_parity()
    print("[fig] 全部完成")


if __name__ == "__main__":
    main()
