"""把 outputs/paired_rollout/all_tasks_summary.md 的实测表格同步进 article/zhihu.md。

zhihu.md 中用标记 ``<!-- BATCH_TABLE:BEGIN -->`` / ``<!-- BATCH_TABLE:END -->``
圈出目标表格区域；本脚本用实测数据重新生成后原位替换。
"""

import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD = os.path.join(REPO, "article", "zhihu.md")
TABLE = os.path.join(REPO, "outputs", "paired_rollout", "all_tasks_summary.md")

BEGIN = "<!-- BATCH_TABLE:BEGIN -->"
END = "<!-- BATCH_TABLE:END -->"


def main():
    md = open(MD, encoding="utf-8").read()
    table = open(TABLE, encoding="utf-8").read().strip()
    if BEGIN not in md:
        # 首次：把 §5.2 的手写表替换为标记区
        pattern = re.compile(
            r"（篇幅所限任务名截断）：\n\n\| 任务 \| demos \|.*?(?=\n\n读这张表)",
            re.S,
        )
        md = pattern.sub(
            "（篇幅所限任务名截断）：\n\n" + BEGIN + "\n\n" + table + "\n\n" + END + "\n\n读这张表",
            md,
        )
    else:
        md = re.sub(
            re.escape(BEGIN) + r".*?" + re.escape(END),
            BEGIN + "\n\n" + table + "\n\n" + END,
            md,
            flags=re.S,
        )
    with open(MD, "w", encoding="utf-8") as f:
        f.write(md)
    print("[sync] zhihu.md 表格已同步")


if __name__ == "__main__":
    main()
