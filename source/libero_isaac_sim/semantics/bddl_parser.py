"""自研 BDDL 任务定义解析器。

LIBERO 官方通过 ``bddl`` 包 + robosuite 的 ``robosuite_parse_problem`` 解析
``.bddl`` 文件（S 表达式）。本模块用约百行纯 Python 实现同样的解析结果，
不依赖 bddl/robosuite/mujoco，使运行时环境（Isaac Sim 侧）完全不需要安装
MuJoCo 技术栈。

解析产出结构（与官方解析结果字段对齐）::

    {
        "problem_name": str,          # (:domain 下的 problem 名，决定场景/arena 类型)
        "language": str,              # 自然语言指令
        "fixtures": {category: [instance_name, ...]},
        "objects":  {category: [instance_name, ...]},
        "regions":  {prefixed_name: {"target": str, "ranges": [[xmin,ymin,xmax,ymax]],
                                     "yaw_rotation": [lo, hi], "rgba": [...]}},
        "initial_state": [[pred, obj, region], ...],   # (:init ...) 合取式
        "goal_state":      [[pred, obj, region|obj], ...],  # (:goal ...) 合取式（已展平 And）
        "obj_of_interest": [instance_name, ...],
    }

region 的命名规则与官方一致：``<target>_<region_name>``（例如
``living_room_table_alphabet_soup_init_region``）；不带 ``:ranges`` 的 region
表示附着在某个物体/fixture 内嵌 site 上的区域（如 ``basket_1_contain_region``）。
"""

from __future__ import annotations

import os
import re
from typing import Any

__all__ = ["parse_bddl", "BDDFTaskSpec"]


# ---------------------------------------------------------------------------
# S 表达式词法/语法分析
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\s*([()]|[^\s()]+)")


def _tokenize(text: str) -> list[str]:
    # 去掉注释（; 到行尾）
    text = re.sub(r";[^\n]*", "", text)
    return [m.group(1) for m in _TOKEN_RE.finditer(text) if m.group(1).strip()]


def _parse_tokens(tokens: list[str]) -> Any:
    """把 token 序列解析成嵌套 list。"""
    stack: list[list] = [[]]
    for tok in tokens:
        if tok == "(":
            node: list = []
            stack[-1].append(node)
            stack.append(node)
        elif tok == ")":
            if len(stack) == 1:
                raise ValueError("BDDL 括号不匹配：多余的 ')'")
            stack.pop()
        else:
            stack[-1].append(tok)
    if len(stack) != 1:
        raise ValueError("BDDL 括号不匹配：缺少 ')'")
    return stack[0]


def _to_number(s: str) -> Any:
    """数字字符串转成 int/float；支持 pi 等简单表达式（与官方 eval 行为一致）。"""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # 官方实现用 eval 处理 yaw_rotation 中的表达式（如 "3.14159/2"）
    if re.fullmatch(r"[0-9eE\.\+\-\*/\(\)pi]+", s):
        import math

        return float(eval(s, {"__builtins__": {}}, {"pi": math.pi}))
    return s


def _parse_region(group: list) -> tuple[str, dict]:
    """解析单个 region 定义：(name (:target t) (:ranges (...)) (:yaw_rotation (...)) ...)"""
    region_name = group[0]
    region = {
        "target": None,
        "ranges": [],
        "yaw_rotation": [0.0, 0.0],
        "rgba": [0.0, 0.0, 1.0, 0.0],
    }
    for attr in group[1:]:
        key = attr[0]
        if key == ":target":
            region["target"] = attr[1]
        elif key == ":ranges":
            for rect in attr[1]:
                assert len(rect) == 4, f"region {region_name} 的 range 维度应为 4"
                region["ranges"].append([float(_to_number(x)) for x in rect])
        elif key == ":yaw_rotation":
            for value in attr[1]:
                region["yaw_rotation"] = [float(_to_number(x)) for x in value]
        elif key == ":rgba":
            region["rgba"] = [float(_to_number(x)) for x in attr[1]]
        else:
            raise NotImplementedError(f"未知 region 属性 {key}")
    return region_name, region


def _parse_typed_entity_list(group: list) -> dict[str, list[str]]:
    """解析 (:objects ...) / (:fixtures ...) 的「实例名 - 类别名」列表。

    BDDL 书写形式为 ``inst1 inst2 - category``（实例在前、类别在后，
    与官方解析器行为一致：类别为键，实例列表为值）。
    """
    result: dict[str, list[str]] = {}
    buffer: list[str] = []
    for tok in group:
        if tok == "-":
            continue
        buffer.append(tok)
        # 形如 [inst..., "-", category] 的序列在 token 流里是平坦的，
        # 约定：遇到前一个 category 后的第一个 token 起开始新一组。
    # 上面的平坦解析容易出错，改为显式状态机：
    result = {}
    instances: list[str] = []
    i = 0
    while i < len(group):
        tok = group[i]
        if tok == "-" and i + 1 < len(group):
            category = group[i + 1]
            result.setdefault(category, []).extend(instances)
            instances = []
            i += 2
        else:
            instances.append(tok)
            i += 1
    if instances:
        result.setdefault("object", []).extend(instances)
    return result


def _flatten_goal(expr: list, out: list[list]) -> None:
    """把 (:goal (And p1 p2 ...)) 展平成谓词列表。"""
    head = expr[0]
    if isinstance(head, str) and head.lower() == "and":
        for sub in expr[1:]:
            _flatten_goal(sub, out)
    else:
        out.append(expr)


class BDDFTaskSpec(dict):
    """dict 子类，仅增加一点便捷访问。"""

    @property
    def problem_name(self) -> str:
        return self["problem_name"]

    @property
    def language(self) -> str:
        return self["language"]


def parse_bddl(bddl_path: str) -> BDDFTaskSpec:
    """解析一个 .bddl 文件，返回任务语义规格。"""
    assert os.path.exists(bddl_path), f"BDDL 文件不存在: {bddl_path}"
    with open(bddl_path, "r", encoding="utf-8") as f:
        tree = _parse_tokens(_tokenize(f.read()))

    # tree 形如 [["define", ["problem", name], [":domain", ...], ...]]
    assert tree and tree[0][0] == "define", f"{bddl_path} 不是合法的 BDDL define 结构"

    spec = BDDFTaskSpec(
        problem_name="unknown",
        language="",
        fixtures={},
        objects={},
        regions={},
        initial_state=[],
        goal_state=[],
        obj_of_interest=[],
        source_file=os.path.abspath(bddl_path),
    )

    for group in tree[0][1:]:
        if not isinstance(group, list):
            continue
        tag = group[0]
        if tag == "problem":
            spec["problem_name"] = group[1]
        elif tag == ":domain":
            assert group[1] == "robosuite", f"不支持的 domain: {group[1]}"
        elif tag == ":language":
            spec["language"] = " ".join(group[1:])
        elif tag == ":objects":
            spec["objects"] = _parse_typed_entity_list(group[1:])
        elif tag == ":fixtures":
            spec["fixtures"] = _parse_typed_entity_list(group[1:])
        elif tag == ":regions":
            for region_group in group[1:]:
                region_name, region = _parse_region(region_group)
                prefixed = f"{region['target']}_{region_name}"
                spec["regions"][prefixed] = region
        elif tag == ":obj_of_interest":
            spec["obj_of_interest"] = list(group[1:])
        elif tag == ":init":
            spec["initial_state"] = [list(map(str.lower, s)) for s in group[1:]]
        elif tag == ":goal":
            goal: list[list] = []
            _flatten_goal(group[1], goal)
            spec["goal_state"] = [list(map(str.lower, g)) for g in goal]
        elif tag == ":requirements" or tag == ":scene_properties":
            pass
        else:
            raise NotImplementedError(f"未知 BDDL 顶层字段: {tag}")

    return spec
