"""从活体 LIBERO 环境导出「机器人本体」MJCF（libero-mujoco 环境运行）。

动机：Isaac Nucleus 资产源在本机不可达，且任务等价性要求 Isaac 侧的 Panda
与 LIBERO 使用**同一份模型**（robosuite 的 panda robot.xml + PandaGripper
+ 底座 + LIBERO 的 damping 覆写）。因此直接从组装完毕的 MuJoCo 模型导出
机器人子树，作为 MJCF→USD 转换的输入。

做法：
1. 按 arena 类型（kitchen → MountedPanda + RethinkMount；living_room/study →
   OnTheGroundPanda 无底座）实例化一个最小 LIBERO 环境；
2. ``mujoco.mj_saveLastXML`` 导出组装后的完整 MJCF；
3. 裁掉非机器人 body（桌子/物体/fixture/相机保持默认），保留机器人子树、
   其执行器与网格资产引用，写成独立 robot MJCF。

产出：``<cache>/mjcf_robot/panda_mounted.xml`` / ``panda_on_ground.xml``
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

LIBERO_REPO = os.environ.get(
    "LIBERO_REPO", "/home/zhaosiying/codebase/_external/LIBERO"
)
DEFAULT_CACHE_DIR = os.environ.get(
    "LIBERO_ISAAC_ASSETS_DIR", os.path.expanduser("~/.cache/libero_isaac_sim")
)

# 各 arena 类型的代表 BDDL（kitchen 用带 mount 的变体）
VARIANT_BDDL = {
    "panda_mounted": "libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl",
    "panda_on_ground": "libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl",
}

# 非机器人顶层 body 的名字前缀（桌子/物体/fixture），裁剪用
_ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")


def _is_robot_body(body: ET.Element) -> bool:
    name = body.get("name") or ""
    return name.startswith(_ROBOT_PREFIXES)


def export_robot(variant: str, out_dir: str) -> str:
    import mujoco
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(
        LIBERO_REPO, "libero/libero/bddl_files", VARIANT_BDDL[variant]
    )
    env = OffScreenRenderEnv(bddl_file_name=bddl, use_camera_obs=False)
    env.reset()

    tmp_xml = os.path.join(out_dir, f"{variant}_full_scene.xml")
    os.makedirs(out_dir, exist_ok=True)
    mujoco.mj_saveLastXML(tmp_xml, env.sim.model._model)

    # 先取质量/惯性/质心（close 之前），供 USD 转换后修正
    import json as _json

    masses = {}
    m = env.sim.model
    for i in range(m.nbody):
        bname = m.body_id2name(i)
        if not bname or not bname.startswith(_ROBOT_PREFIXES):
            continue
        masses[bname] = {
            "mass": float(m.body_mass[i]),
            "inertia": [float(v) for v in m.body_inertia[i]],
            "com_local": [float(v) for v in m.body_ipos[i]],
            "inertia_quat_wxyz": [float(v) for v in m.body_iquat[i]],
        }
    env.close()

    tree = ET.parse(tmp_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None

    removed = []
    for body in list(worldbody.findall("body")):
        if not _is_robot_body(body):
            removed.append(body.get("name"))
            worldbody.remove(body)
    # 相机/灯光挂在 worldbody 上的非 body 元素保留与否均可；删掉避免语义混淆
    for tag in ["camera", "light"]:
        for elem in list(worldbody.findall(tag)):
            worldbody.remove(elem)
    # 背景网格常以 worldbody 直属 geom（或 site）形式存在，也一并裁掉
    for tag in ["geom", "site"]:
        for elem in list(worldbody.findall(tag)):
            worldbody.remove(elem)

    # ------------------------------------------------------------------
    # 裁掉不再被引用的资产：mj_saveLastXML 会保留全场景的 texture/material/mesh
    # 声明，其中含 Isaac 导入器不支持的 .msh，必须按存活 geom 反向引用清理。
    # ------------------------------------------------------------------
    used_materials = set()
    used_textures = set()
    used_meshes = set()
    for geom in worldbody.iter("geom"):
        if geom.get("material"):
            used_materials.add(geom.get("material"))
        if geom.get("mesh"):
            used_meshes.add(geom.get("mesh"))
    asset = root.find("asset")
    if asset is not None:
        # material 可能引用 texture；先收集存活 material 依赖的 texture
        for mat in asset.findall("material"):
            if mat.get("name") in used_materials and mat.get("texture"):
                used_textures.add(mat.get("texture"))
        for tex in asset.findall("texture"):
            # skybox 等 builtin 无名的保留按默认规则处理：无名字或被引用则留
            if tex.get("name") and tex.get("name") not in used_textures and tex.get("name") is not None:
                if tex.get("builtin") is None:
                    asset.remove(tex)
        for mat in list(asset.findall("material")):
            if mat.get("name") and mat.get("name") not in used_materials:
                asset.remove(mat)
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") and mesh.get("name") not in used_meshes:
                asset.remove(mesh)

    # 归零机器人根 body 的位姿：mj_saveLastXML 会把场景中的基座偏移烘进
    # body pos（如 living_room 的 (-0.51, 0, 0.42)）。机器人 USD 应在原点，
    # 摆放位姿由 Isaac 环境的 InitialStateCfg 负责，否则会双重叠加。
    for body in worldbody.findall("body"):
        if _is_robot_body(body):
            if body.get("pos"):
                body.set("pos", "0 0 0")
            if body.get("quat"):
                body.set("quat", "1 0 0 0")

    out_path = os.path.join(out_dir, f"{variant}.xml")
    tree.write(out_path, encoding="unicode")

    mass_path = os.path.join(out_dir, f"{variant}_masses.json")
    with open(mass_path, "w") as f:
        _json.dump(masses, f, indent=2)

    print(f"[robot-export] {variant}: 裁剪掉非机器人 body: {removed}")
    print(f"[robot-export] {variant} -> {out_path}; 质量表 -> {mass_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.join(DEFAULT_CACHE_DIR, "mjcf_robot"))
    parser.add_argument("--variants", nargs="*", default=list(VARIANT_BDDL.keys()))
    args = parser.parse_args()
    for variant in args.variants:
        export_robot(variant, args.out_dir)


if __name__ == "__main__":
    main()
