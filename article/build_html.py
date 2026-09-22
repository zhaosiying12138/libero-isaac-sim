"""把 article/zhihu.md 转成自包含 article/paper.html（图表 base64 内嵌，动画用 mp4 链接）。

用法: ~/codebase/_external/venvs/libero-mujoco/bin/python article/build_html.py
"""

import base64
import html
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTICLE = os.path.join(REPO, "article")

CSS = """
body { max-width: 860px; margin: 0 auto; padding: 24px 32px; font-family: 'Noto Serif CJK SC','Songti SC','SimSun',serif; line-height: 1.85; color: #1a1a1a; background: #fdfdfd; }
h1 { font-size: 1.6em; border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { font-size: 1.3em; margin-top: 1.8em; border-left: 4px solid #666; padding-left: 10px; }
h3 { font-size: 1.1em; margin-top: 1.4em; }
table { border-collapse: collapse; width: 100%; font-size: 0.88em; margin: 12px 0; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; }
th { background: #f2f2f2; }
code { background: #f5f5f5; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
pre { background: #f7f7f7; padding: 12px 16px; border-radius: 6px; overflow-x: auto; font-size: 0.82em; line-height: 1.45; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 14px auto; border: 1px solid #ddd; }
blockquote { border-left: 4px solid #999; margin: 12px 0; padding: 2px 14px; color: #444; background: #f8f8f8; }
video { max-width: 100%; display: block; margin: 14px auto; }
"""


def _embed(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _inline_media(md: str) -> str:
    # 图片 ![alt](figures/...) → base64
    def _img(m):
        alt, rel = m.group(1), m.group(2)
        path = os.path.join(ARTICLE, rel)
        if not os.path.exists(path):
            return m.group(0)
        return f'<img src="{_embed(path)}" alt="{html.escape(alt)}"/>'

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _img, md)
    return md


def _md_to_html(md: str) -> str:
    """极简 markdown → html（覆盖本文用到的语法）。"""
    import markdown  # noqa: F811

    try:
        return markdown.markdown(md, extensions=["tables", "fenced_code"])
    except ImportError:
        pass
    # 无 markdown 库的兜底：极简单转换
    lines = md.split("\n")
    out = []
    in_code = False
    in_table = False
    for ln in lines:
        if ln.startswith("```"):
            in_code = not in_code
            out.append("<pre>" if in_code else "</pre>")
            continue
        if in_code:
            out.append(html.escape(ln))
            continue
        if ln.startswith("#"):
            level = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{level}>{html.escape(ln.strip('# '))}</h{level}>")
            continue
        if ln.strip().startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if all(re.fullmatch(r"[-:]+", c) for c in cells):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
            continue
        else:
            if in_table:
                out.append("</table>")
                in_table = False
        if not ln.strip():
            out.append("")
            continue
        out.append(f"<p>{html.escape(ln)}</p>")
    if in_table:
        out.append("</table>")
    return "\n".join(out)


def main():
    md = open(os.path.join(ARTICLE, "zhihu.md"), encoding="utf-8").read()
    md = _inline_media(md)
    body = _md_to_html(md)
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>把 LIBERO 基准从 MuJoCo 迁到 Isaac Sim</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""
    out = os.path.join(ARTICLE, "paper.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[html] -> {out}")


if __name__ == "__main__":
    main()
