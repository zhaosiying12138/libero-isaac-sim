"""MJCF → USD 批量转换器（在 isaac 环境中运行）。

把 LIBERO 资产按语义角色转换为 USD：

- **可动物体**（HOPE / Google Scanned / Turbosquid）：原 XML 不含自由关节
  （robosuite 运行时动态注入 ``free joint, damping=0.0005``）。本转换器先在
  MJCF 中注入相同的 ``<joint type="free" damping="0.0005"/>``，再调用
  Isaac Lab 的 ``MjcfConverter``（fix_base=False）。转换后若 USD 为单刚体
  浮动基 articulation，剥离 ArticulationRoot 使其成为普通 RigidBody，
  以便用 Isaac Lab 的 ``RigidObjectCfg`` 管理。
- **关节 fixture**（microwave / flat_stove / wooden_cabinet 等）：fixture 在
  LIBERO 中以 joints=None 实例化（根部固定），但内部关节（门铰链、旋钮、抽屉
  滑轨）保留。转换用 fix_base=True，保留 articulation，供 ``ArticulationCfg``
  使用。
- **arena 场景 XML**：静态视觉+碰撞外壳（地板/墙/桌面/背景网格），fix_base=True。

每次转换产出校验报告（prim 树、关节列表、质量、纹理引用是否存活），
写入 ``<cache>/usd/_report_<name>.json``。

用法（isaac 环境，需在 IsaacLab 仓目录或用配置好的入口）::

    OMNI_KIT_ACCEPT_EULA=YES uv run --extra isaacsim python \
        -m libero_isaac_sim.converters.mjcf_to_usd --task <task_name>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET

DEFAULT_CACHE_DIR = os.environ.get(
    "LIBERO_ISAAC_ASSETS_DIR", os.path.expanduser("~/.cache/libero_isaac_sim")
)
LIBERO_REPO = os.environ.get(
    "LIBERO_REPO", os.path.expanduser("~/codebase/_external/LIBERO")
)
LIBERO_ASSETS = os.path.join(LIBERO_REPO, "libero/libero/assets")

# 类别 → MJCF 相对路径的三类资产库
OBJECT_ASSET_DIRS = [
    "stable_hope_objects",
    "stable_scanned_objects",
    "turbosquid_objects",
]


def find_object_xml(category: str) -> str:
    """按类别名定位 LIBERO 物体 MJCF。"""
    for d in OBJECT_ASSET_DIRS:
        # 目录形式：<dir>/<category>/<category>.xml
        p = os.path.join(LIBERO_ASSETS, d, category, f"{category}.xml")
        if os.path.exists(p):
            return p
    # articulated_objects 是平铺形式：<category>.xml
    p = os.path.join(LIBERO_ASSETS, "articulated_objects", f"{category}.xml")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"找不到类别 {category} 的 MJCF")


def inject_free_joint(xml_path: str, out_path: str, damping: float = 0.0005) -> str:
    """把自由关节注入到含 geom 的第一个 body，结果写到 out_path。

    LIBERO 物体 XML 的常见结构是匿名外层 body 包裹具名内层 body；
    robosuite 运行时把自由关节加在它提取的物体 body 上。这里选择
    第一个拥有 geom 的 body 注入，语义等价。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None, f"{xml_path} 缺少 worldbody"

    def _first_body_with_geom(body):
        if body.findall("geom"):
            return body
        for child in body.findall("body"):
            found = _first_body_with_geom(child)
            if found is not None:
                return found
        return None

    target = None
    for body in worldbody.findall("body"):
        target = _first_body_with_geom(body)
        if target is not None:
            break
    assert target is not None, f"{xml_path} 找不到含 geom 的 body"
    assert not target.findall("joint") and not target.findall("freejoint"), (
        f"{xml_path} 目标 body 已有关节，按 fixture 处理即可"
    )
    ET.SubElement(target, "joint", {"type": "free", "damping": str(damping)})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # ET 会丢失 DOCTYPE 等 MJCF 不需要的声明；直接写出
    tree.write(out_path, encoding="unicode")
    return out_path


def convert_mjcf(
    xml_path: str,
    usd_dir: str,
    fix_base: bool,
    import_sites: bool = True,  # 保留参数位；3.0 新导入器默认导入 site，无独立开关
    link_density: float = 0.0,
) -> str:
    """调用 Isaac Lab MjcfConverter，返回主 USD 路径。"""
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    cfg = MjcfConverterCfg(
        asset_path=xml_path,
        usd_dir=usd_dir,
        fix_base=fix_base,
        link_density=link_density,
        force_usd_conversion=True,
    )
    converter = MjcfConverter(cfg)
    return converter.usd_path


def audit_usd(usd_path: str) -> dict:
    """转换产物体检：prim 结构、关节、质量、纹理引用。"""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    report = {
        "usd_path": usd_path,
        "root_prims": [],
        "joints": [],
        "bodies": [],
        "total_mass": 0.0,
        "missing_refs": [],
    }
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Xform" or prim.IsA(UsdPhysics.RigidBodyAPI):
            pass
        if prim.IsA(UsdPhysics.RigidBodyAPI):
            report["bodies"].append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            report["joints"].append(str(prim.GetPath()))
        mass_api = UsdPhysics.MassAPI(prim)
        if mass_api:
            mass_attr = mass_api.GetMassAttr()
            if mass_attr and mass_attr.HasAuthoredValue():
                report["total_mass"] += float(mass_attr.Get())
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        report["root_prims"] = [str(c.GetPath()) for c in default_prim.GetChildren()]
    # 纹理引用存活检查
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == "asset":
                asset_path = attr.Get()
                if asset_path is not None:
                    p = str(asset_path.path)
                    if not os.path.isabs(p):
                        p = os.path.join(os.path.dirname(usd_path), p)
                    if not os.path.exists(p):
                        report["missing_refs"].append(str(asset_path.path))
    return report


def convert_task_assets(task_name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> dict:
    """按任务 manifest 转换全部所需资产。"""
    from libero_isaac_sim.semantics.task_spec import load_task

    task = load_task(task_name, cache_dir)
    usd_root = os.path.join(cache_dir, "usd")
    tmp_dir = os.path.join(cache_dir, "mjcf_patched")
    results = {}

    # 1. 物体与 fixture
    for name, entity in task.entities.items():
        category = entity["category"]
        xml_path = find_object_xml(category)
        if entity["kind"] == "object":
            patched = os.path.join(tmp_dir, f"{category}.xml")
            # 注入自由关节的 MJCF 必须与原始资产同目录（网格/纹理相对路径），
            # 因此把 patched 文件写到原目录旁的镜像结构里并复制引用资产
            patched = _prepare_patched_tree(xml_path, tmp_dir, category)
            usd_dir = os.path.join(usd_root, "objects", category)
            usd_path = convert_mjcf(patched, usd_dir, fix_base=False)
        else:
            usd_dir = os.path.join(usd_root, "fixtures", category)
            usd_path = convert_mjcf(xml_path, usd_dir, fix_base=True)
        report = audit_usd(usd_path)
        report["category"] = category
        report["entity_name"] = name
        report["fix_base"] = entity["kind"] == "fixture"
        results[name] = report
        report_path = os.path.join(usd_root, f"_report_{category}.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[convert] {name} ({category}) -> {usd_path}; "
              f"bodies={len(report['bodies'])} joints={len(report['joints'])} "
              f"missing_refs={len(report['missing_refs'])}")

    # 2. arena 场景外壳
    scene_xml = task.arena["scene_xml"]
    arena_dir = os.path.join(usd_root, "arenas")
    os.makedirs(arena_dir, exist_ok=True)
    arena_usd = convert_mjcf(scene_xml, arena_dir, fix_base=True, import_sites=True)
    arena_report = audit_usd(arena_usd)
    arena_report["scene_xml"] = scene_xml
    with open(os.path.join(arena_dir, "_report_arena.json"), "w") as f:
        json.dump(arena_report, f, indent=2, default=str)
    results["_arena"] = arena_report
    print(f"[convert] arena -> {arena_usd}; missing_refs={len(arena_report['missing_refs'])}")
    return results


def _prepare_patched_tree(xml_path: str, tmp_root: str, category: str) -> str:
    """把注入自由关节后的 MJCF 写到资产原目录（保证 mesh/纹理相对路径有效）。

    MJCF 用相对路径引用网格与纹理，patched 文件必须和原 xml 同目录。
    文件名加 ``_freejoint`` 后缀，避免污染原资产。
    """
    out_path = xml_path.replace(".xml", "_freejoint.patched.xml")
    return inject_free_joint(xml_path, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    known = parser.parse_known_args()[0]
    known.headless = True
    app = AppLauncher(known).app

    convert_task_assets(args.task, args.cache_dir)
    app.close()


if __name__ == "__main__":
    main()
