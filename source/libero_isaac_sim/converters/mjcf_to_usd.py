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


def find_object_xml(category: str, prefer_sanitized: bool = True) -> str:
    """按类别名定位 LIBERO 物体 MJCF；优先使用净化（.msh→.obj）后的版本。"""
    for d in OBJECT_ASSET_DIRS:
        p = os.path.join(LIBERO_ASSETS, d, category, f"{category}.xml")
        if os.path.exists(p):
            sanitized = p.replace(".xml", "_sanitized.xml")
            if prefer_sanitized and os.path.exists(sanitized):
                return sanitized
            return p
    p = os.path.join(LIBERO_ASSETS, "articulated_objects", f"{category}.xml")
    if os.path.exists(p):
        sanitized = p.replace(".xml", "_sanitized.xml")
        if prefer_sanitized and os.path.exists(sanitized):
            return sanitized
        return p
    raise FileNotFoundError(f"找不到类别 {category} 的 MJCF")


def inject_free_joint(xml_path: str, out_path: str, damping: float = 0.0005) -> str:
    """在物体顶层 body 注入自由关节，结果写到 out_path。

    与 robosuite 的运行时行为一致：自由关节加在物体 XML 的顶层 body
    （即组合模型中带 ``_main`` 后缀的那个 body）上；内层 body 无关节，
    编译时刚性融合进顶层 body，sites 也随之运动。这样 USD 侧只产生
    一个浮动刚体/articulation，不会出现嵌套刚体。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None, f"{xml_path} 缺少 worldbody"

    top_bodies = worldbody.findall("body")
    assert top_bodies, f"{xml_path} 的 worldbody 为空"
    target = top_bodies[0]
    assert not target.findall("joint") and not target.findall("freejoint"), (
        f"{xml_path} 顶层 body 已有关节，按 fixture 处理即可"
    )
    ET.SubElement(target, "joint", {"type": "free", "damping": str(damping)})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tree.write(out_path, encoding="unicode")
    return out_path


def convert_mjcf(
    xml_path: str,
    usd_dir: str,
    fix_base: bool,
    import_sites: bool = True,  # 保留参数位；3.0 新导入器默认导入 site，无独立开关
    link_density: float = 0.0,
    make_instanceable: bool = False,
) -> str:
    """调用 Isaac Lab MjcfConverter，返回主 USD 路径。

    make_instanceable=False：instanceable 会把几何/关节收进 payload 层，
    导致转换后修正（fix_object_physics 的 joint/mass 编辑）无法穿透引用层。
    我们的资产规模小，不需要 instanceable。
    """
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    cfg = MjcfConverterCfg(
        asset_path=xml_path,
        usd_dir=usd_dir,
        fix_base=fix_base,
        link_density=link_density,
        force_usd_conversion=True,
        make_instanceable=make_instanceable,
    )
    converter = MjcfConverter(cfg)
    return converter.usd_path


def fix_object_physics(usd_path: str, mass: float, inertia_diag: list, com_local: list) -> dict:
    """转换后修正刚体物理参数与结构。

    Isaac 的 MJCF 导入器对 LIBERO 物体的典型输出问题：
    1. 每个 geom prim 被写 ``MassAPI mass=0.0``（不展开 MJCF density 语义）；
    2. 匿名外层 body 与具名内层 body 都带 RigidBodyAPI（嵌套刚体非法）；
    3. 无关节物体也带 ArticulationRootAPI。

    修正：清几何级零质量；只保留最外层刚体的 RigidBodyAPI（内层刚体的碰撞
    几何按 PhysX 规则归属于外层刚体）；无关节时剥离 ArticulationRootAPI；
    按 MuJoCo 导出值在最外层刚体上写总质量/质心/对角惯性。
    """
    from pxr import Gf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    cleared = 0
    body_prims = []
    articulation_roots = []
    joint_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            attr = UsdPhysics.MassAPI(prim).GetMassAttr()
            if attr and attr.HasAuthoredValue() and float(attr.Get()) == 0.0:
                attr.Clear()
                cleared += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_prims.append(prim)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim)
        if prim.IsA(UsdPhysics.Joint):
            joint_count += 1

    cleared_joints = []
    assert body_prims, f"{usd_path} 没有刚体"
    # 保留最外层刚体（遍历序最浅者），剥掉嵌套刚体的 RigidBodyAPI
    body_prims.sort(key=lambda p: len(str(p.GetPath())))
    top_body = body_prims[0]
    stripped_nested = 0
    for prim in body_prims[1:]:
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        stripped_nested += 1

    mass_api = UsdPhysics.MassAPI.Apply(top_body)
    mass_api.GetMassAttr().Set(float(mass))
    if com_local is not None:
        mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*[float(v) for v in com_local]))
    if inertia_diag is not None:
        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(v) for v in inertia_diag]))

    stripped_articulation = False
    # 单刚体物体：自由关节被导入器表达为「世界↔body 的 6DOF 关节 + ArticulationRoot」，
    # 物理上等价于自由刚体。剥掉关节 prim 与 ArticulationRootAPI，转成普通 RigidBody，
    # 以便用 RigidObjectCfg 管理（reset 写根位姿即可）。
    if articulation_roots and (len(body_prims) - stripped_nested) == 1:
        removed_joints = 0
        for prim in list(stage.Traverse()):
            # 注意：导入器生成的 PhysicsFixedJoint 等自定义类型不匹配
            # UsdPhysics.Joint 的 IsA 判定，按类型名/名字双判据兜底
            tn = prim.GetTypeName()
            if prim.IsA(UsdPhysics.Joint) or "Joint" in tn or "Joint" in prim.GetName():
                prim.GetStage().RemovePrim(prim.GetPath())
                removed_joints += 1
        for root in articulation_roots:
            root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        stripped_articulation = True
        joint_count = 0

    # 导入器把物理定义拆进 payloads/ 子层，主文件的编辑穿透不了引用层；
    # 且残留的 over 块不是 defined prim，stage.Traverse 看不到。
    # 改用 Sdf 层级 API 直接编辑 spec（能处理 over/def 一切 spec）。
    from pxr import Sdf

    payload_dir = os.path.join(os.path.dirname(usd_path), "payloads")
    if os.path.isdir(payload_dir):
        import glob as _glob

        def _strip_specs(spec, path, changed_box):
            for child in list(spec.nameChildren.values()):
                child_path = f"{path}/{child.name}"
                schemas = list(child.apiSchemas) if hasattr(child, "apiSchemas") else []
                is_jointish = (
                    "Joint" in child.name
                    or any("Joint" in a for a in schemas)
                    or "Joint" in (child.typeName or "")
                )
                if is_jointish:
                    del spec.nameChildren[child.name]
                    cleared_joints.append(child_path)
                    changed_box[0] = True
                    continue
                _strip_specs(child, child_path, changed_box)

        for sub in _glob.glob(os.path.join(payload_dir, "**", "*.usd*"), recursive=True):
            layer = Sdf.Layer.FindOrOpen(sub)
            if layer is None:
                continue
            changed_box = [False]
            for root_spec in list(layer.rootPrims):
                _strip_specs(root_spec, f"/{root_spec.name}", changed_box)
            if changed_box[0]:
                layer.Save()

    stage.GetRootLayer().Save()
    flatten_usd(usd_path)
    return {
        "cleared_zero_mass": cleared,
        "cleared_joints": cleared_joints,
        "num_rigid_bodies_before": len(body_prims),
        "nested_rigid_bodies_stripped": stripped_nested,
        "articulation_stripped": stripped_articulation,
        "mass_authored": float(mass),
        "body_prim": str(top_body.GetPath()),
    }


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
        if prim.IsA(UsdPhysics.RigidBodyAPI):
            report["bodies"].append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            report["joints"].append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr()
            if mass_attr and mass_attr.HasAuthoredValue():
                report["total_mass"] += float(mass_attr.Get())
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        report["root_prims"] = [str(c.GetPath()) for c in default_prim.GetChildren()]
    # 纹理引用存活检查
    from pxr import Sdf

    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                asset_path = attr.Get()
                if asset_path is not None:
                    p = str(asset_path.path)
                    if not os.path.isabs(p):
                        p = os.path.join(os.path.dirname(usd_path), p)
                    if not os.path.exists(p):
                        report["missing_refs"].append(str(asset_path.path))
    return report



def flatten_usd(usd_path: str) -> str:
    """把 USD 拍平成单文件（烘焙 instanceable/reference/payload 结构）。

    Isaac 的新 MJCF 导入器产物采用 payloads + instanceable 分层结构，
    Newton 渲染器与部分物理路径不穿透实例代理。拍平后所有 prim 直接可见，
    纹理引用为绝对路径（转换时已解析）。覆盖写回原路径。
    """
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    flat = stage.Flatten()
    tmp = usd_path + ".flat_tmp"
    flat.Export(tmp)
    os.replace(tmp, usd_path)
    return usd_path



def strip_all_joints(usd_path: str) -> int:
    """移除 USD 中全部关节 prim（用于纯静态 arena）。

    Newton 模型构建器拒绝「不属于任何 articulation 的孤儿关节」，
    静态外壳里的固定关节（如桌腿拼接）对我们无意义，直接删除。
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    removed = 0
    for prim in list(stage.TraverseAll()):
        tn = prim.GetTypeName()
        if prim.IsA(UsdPhysics.Joint) or "Joint" in tn or "Joint" in prim.GetName():
            stage.RemovePrim(prim.GetPath())
            removed += 1
    if removed:
        stage.GetRootLayer().Save()
    return removed



def make_static_collision_shell(usd_path: str) -> int:
    """把 USD 里的 RigidBodyAPI 全部剥掉，使其成为静态碰撞壳。

    arena（地板/墙/桌面/背景）在 LIBERO 中本来就是静态的；
    转换器 fix_base=True 会给它们加 RigidBodyAPI（负质量警告 + 网格碰撞
    不被 PhysX 动态刚体支持）。剥掉后按 USD 规则成为静态碰撞体。
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    removed = 0
    for prim in stage.TraverseAll():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            removed += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if removed:
        stage.GetRootLayer().Save()
    return removed



def fix_articulation_masses(usd_path: str, masses: dict) -> int:
    """给关节体 USD 的每个 link 写入 MuJoCo 质量/惯性/质心。

    机器人 USD 的 link prim 名与 MuJoCo body 名一致（robot0_link1 等）。
    质量以覆盖形式写在主层（payload 结构保持不变）。
    """
    from pxr import Gf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    fixed = 0
    for prim in stage.TraverseAll():
        name = prim.GetName()
        if name not in masses:
            continue
        entry = masses[name]
        m = float(entry["mass"])
        # PhysX 关节体不接受零/负质量 link（MuJoCo 的零质量虚拟 body 如
        # robot0_base/gripper0_eef 在 PhysX 里会腐蚀质量矩阵），给个下限
        if m < 1e-3:
            m = 1e-3
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(m)
        if entry.get("com_local") is not None:
            mass_api.GetCenterOfMassAttr().Set(
                Gf.Vec3f(*[float(v) for v in entry["com_local"]])
            )
        if entry.get("inertia") is not None:
            mass_api.GetDiagonalInertiaAttr().Set(
                Gf.Vec3f(*[float(v) for v in entry["inertia"]])
            )
        fixed += 1
    stage.GetRootLayer().Save()
    return fixed



def strip_custom_attrs(usd_path: str, prefixes: tuple = ("newton:",)) -> int:
    """删除 USD（含 payloads 子层）中的自定义命名空间属性。

    Isaac 的 MJCF 导入器会写入 ``newton:selfCollisionEnabled`` 等 Newton 专属
    自定义属性；Isaac Lab 3.0 EA 的 ``modify_articulation_root_properties``
    在调整关节体根属性时无法删除这些属性而直接报错（EA 缺陷）。
    """
    import glob as _glob
    import os
    from pxr import Sdf

    def _strip_layer(layer):
        removed = 0

        def _walk_spec(spec):
            nonlocal removed
            for prop_name in list(spec.properties.keys()):
                if any(prop_name.startswith(px) for px in prefixes):
                    del spec.properties[prop_name]
                    removed += 1
            for child in spec.nameChildren.values():
                _walk_spec(child)

        for root_spec in layer.rootPrims:
            _walk_spec(root_spec)
        return removed

    removed = 0
    files = [usd_path]
    payload_dir = os.path.join(os.path.dirname(usd_path), "payloads")
    if os.path.isdir(payload_dir):
        files += _glob.glob(os.path.join(payload_dir, "**", "*.usd*"), recursive=True)
    for f in files:
        layer = Sdf.Layer.FindOrOpen(f)
        if layer is None:
            continue
        n = _strip_layer(layer)
        if n:
            layer.Save()
            removed += n
    return removed



def add_world_fixed_joint(usd_path: str, child_body_path: str, joint_name: str = "world_fixed") -> str:
    """在 USD 中给关节体根 link 显式添加 世界固定关节。

    PhysX 约定：FixedJoint 不设 body0 即连接到世界。这样关节体基座固定，
    不需要 Isaac Lab 的 fix_root_link 运行时补丁（它生成的运行时关节会让
    newton 的 USD 导入器在合并关节时报错）。
    """
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    # 关节 prim 放在 child body 下（USD 惯例）
    joint_path = f"{child_body_path}/{joint_name}"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.GetBody1Rel().SetTargets([Sdf.Path(child_body_path)])
    stage.GetRootLayer().Save()
    return joint_path



def strip_joints_by_type(usd_path: str, remove_types: tuple = ("FixedJoint", "PhysicsFixedJoint")) -> int:
    """按类型删除 USD 中的关节（用于 fixture：删除焊接 FixedJoint，保留铰链/滑轨）。

    焊缝关节的两侧 body 在 MJCF 中本来就刚性相连；删除后形状仍随父体渲染，
    不影响画面与铰链动力学。
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    removed = 0
    for prim in list(stage.TraverseAll()):
        if prim.GetTypeName() in remove_types or (
            prim.IsA(UsdPhysics.FixedJoint)
        ):
            stage.RemovePrim(prim.GetPath())
            removed += 1
    if removed:
        stage.GetRootLayer().Save()
    return removed



def _add_fixture_world_joint(usd_path: str) -> None:
    """在 fixture USD 的关节体根 body 上加 世界固定关节（body0 缺省=世界）。"""
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    root_body = None
    for prim in stage.TraverseAll():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            root_body = prim
            break
    if root_body is None:
        # 没有 articulation：纯静态 fixture，无需关节
        return
    joint_path = f"{root_body.GetPath()}/WorldFixedJoint"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.GetBody1Rel().SetTargets([Sdf.Path(str(root_body.GetPath()))])
    stage.GetRootLayer().Save()


def convert_task_assets(task_name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> dict:
    """按任务 manifest 转换全部所需资产。"""
    from libero_isaac_sim.converters.asset_manifest import AssetRegistry
    from libero_isaac_sim.semantics.task_spec import load_task

    task = load_task(task_name, cache_dir)
    usd_root = os.path.join(cache_dir, "usd")
    tmp_dir = os.path.join(cache_dir, "mjcf_patched")
    registry = AssetRegistry(cache_dir)
    results = {}

    # 1. 物体与 fixture
    for name, entity in task.entities.items():
        category = entity["category"]
        xml_path = find_object_xml(category)
        if entity["kind"] == "object":
            patched = _prepare_patched_tree(xml_path, tmp_dir, category)
            usd_dir = os.path.join(usd_root, "objects", category)
            usd_path = convert_mjcf(patched, usd_dir, fix_base=False)
            src_xml = patched
            fix_info = fix_object_physics(
                usd_path,
                mass=entity["mass"],
                inertia_diag=entity.get("inertia"),
                com_local=entity.get("com_local"),
            )
            print(f"[fix] {name}: 质量 {fix_info['mass_authored']:.4f} kg，"
                  f"嵌套刚体剥离 {fix_info['nested_rigid_bodies_stripped']}，"
                  f"articulation_stripped={fix_info['articulation_stripped']}")
        else:
            usd_dir = os.path.join(usd_root, "fixtures", category)
            usd_path = convert_mjcf(xml_path, usd_dir, fix_base=True)
            src_xml = xml_path
            fix_info = {}
            n_attrs = strip_custom_attrs(usd_path)
            flatten_usd(usd_path)  # fixture 拍平但保留关节（渲染需要穿透 payload）
            # fixture 基座固定由转换器自身的根焊接关节承担（fix_base=True 产物），
            # 场景侧与 USD 侧都不再额外加关节（多加会触发 newton 的合并缺陷）。
            if n_attrs:
                print(f"[fix] {name}: 剥除 newton 自定义属性 {n_attrs} 个")
        report = audit_usd(usd_path)
        report.update(fix_info)
        report["category"] = category
        report["entity_name"] = name
        report["fix_base"] = entity["kind"] == "fixture"
        results[name] = report
        registry.register(
            entity_name=name,
            category=category,
            kind=entity["kind"],
            usd_path=usd_path,
            source_mjcf=src_xml,
            fix_base=entity["kind"] == "fixture",
            report=report,
        )
        report_path = os.path.join(usd_root, f"_report_{category}.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[convert] {name} ({category}) -> {usd_path}; "
              f"bodies={len(report['bodies'])} joints={len(report['joints'])} "
              f"missing_refs={len(report['missing_refs'])}")

    # 2. arena 场景外壳
    scene_xml = task.arena["scene_xml"]
    sanitized_scene = scene_xml.replace(".xml", "_sanitized.xml")
    if os.path.exists(sanitized_scene):
        scene_xml = sanitized_scene
    arena_dir = os.path.join(usd_root, "arenas", os.path.basename(os.path.dirname(scene_xml)))
    os.makedirs(arena_dir, exist_ok=True)
    arena_usd = convert_mjcf(scene_xml, arena_dir, fix_base=True)
    n_joints_removed = strip_all_joints(arena_usd)
    n_rb_removed = make_static_collision_shell(arena_usd)
    flatten_usd(arena_usd)
    arena_report = audit_usd(arena_usd)
    arena_report["joints_removed"] = n_joints_removed
    arena_report["rigid_bodies_removed"] = n_rb_removed
    arena_report["scene_xml"] = scene_xml
    with open(os.path.join(arena_dir, "_report_arena.json"), "w") as f:
        json.dump(arena_report, f, indent=2, default=str)
    results["_arena"] = arena_report
    registry.register(
        entity_name=f"_arena/{task_name}",
        category="arena",
        kind="arena",
        usd_path=arena_usd,
        source_mjcf=scene_xml,
        fix_base=True,
        report=arena_report,
    )
    registry.save()
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
