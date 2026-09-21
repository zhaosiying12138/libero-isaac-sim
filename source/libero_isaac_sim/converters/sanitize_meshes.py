"""MJCF 资产网格净化器：把 Isaac 导入器不支持的 .msh 网格转成 OBJ。

背景：LIBERO 资产的视觉网格大量使用 MuJoCo 自有二进制 ``.msh`` 格式，
Isaac Sim 6.1 的 MJCF 导入器（mujoco_usd_converter）不支持该扩展。
本净化器在 libero-mujoco 环境中用 mujoco 本体读取网格（顶点/三角面/UV），
导出为带 UV 的 OBJ，并把 MJCF 中的 ``file`` 引用改写指向 OBJ。

纹理贴图（.png）无需处理，材质/UV 保留后贴图可正常工作。

用法::

    python -m libero_isaac_sim.converters.sanitize_meshes <xml_path> [<xml_path> ...]

每个输入生成同目录 ``<name>_sanitized.xml`` 与若干 ``<meshname>_sanitized.obj``。
"""

from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np


def _export_mesh_to_obj(model, mesh_id: int, out_path: str) -> None:
    """从编译后的 MjModel 中提取单个网格写为 OBJ（含 UV）。"""
    v_adr = model.mesh_vertadr[mesh_id]
    v_num = model.mesh_vertnum[mesh_id]
    f_adr = model.mesh_faceadr[mesh_id]
    f_num = model.mesh_facenum[mesh_id]

    verts = np.asarray(model.mesh_vert[v_adr : v_adr + v_num])
    faces = np.asarray(model.mesh_face[f_adr : f_adr + f_num])

    # UV（可能不存在）
    t_num = int(model.mesh_texcoordnum[mesh_id]) if hasattr(model, "mesh_texcoordnum") else 0
    texcoords = None
    face_tc = None
    if t_num > 0:
        t_adr = model.mesh_texcoordadr[mesh_id]
        texcoords = np.asarray(model.mesh_texcoord[t_adr : t_adr + t_num])
        if hasattr(model, "mesh_facetexcoord") and model.mesh_facetexcoord is not None:
            face_tc = np.asarray(model.mesh_facetexcoord[f_adr : f_adr + f_num])

    with open(out_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        if texcoords is not None:
            for t in texcoords:
                # mujoco UV 的 v 轴与 OBJ 约定相反
                f.write(f"vt {t[0]:.8f} {1.0 - t[1]:.8f}\n")
        for i, tri in enumerate(faces):
            if texcoords is not None and face_tc is not None:
                a, b, c = tri + 1
                ta, tb, tc = face_tc[i] + 1
                f.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")
            else:
                a, b, c = tri + 1
                f.write(f"f {a} {b} {c}\n")


def sanitize_xml(xml_path: str) -> str:
    """净化一个 MJCF：.msh → .obj。返回净化后的 xml 路径。"""
    import mujoco

    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    model = mujoco.MjModel.from_xml_path(xml_path)

    # mesh 名 → 文件路径 映射来自 xml；网格数据来自编译后的 model
    tree = ET.parse(xml_path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        return xml_path

    meshdir = ""
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = compiler.get("meshdir")

    name_to_id = {}
    for i in range(model.nmesh):
        name_to_id[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)] = i

    changed = False
    for mesh_elem in asset.findall("mesh"):
        file_attr = mesh_elem.get("file")
        if not file_attr or not file_attr.lower().endswith(".msh"):
            continue
        name = mesh_elem.get("name") or os.path.splitext(os.path.basename(file_attr))[0]
        mesh_id = name_to_id.get(name)
        if mesh_id is None:
            # 未编译进模型（可能未被引用），跳过
            continue
        obj_name = os.path.splitext(os.path.basename(file_attr))[0] + "_sanitized.obj"
        obj_path = os.path.join(xml_dir, os.path.dirname(file_attr), obj_name)
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)
        _export_mesh_to_obj(model, mesh_id, obj_path)
        new_ref = os.path.join(os.path.dirname(file_attr), obj_name)
        mesh_elem.set("file", new_ref)
        changed = True
        print(f"[sanitize] {name}: {file_attr} -> {new_ref}")

    if not changed:
        return xml_path

    out_path = xml_path.replace(".xml", "_sanitized.xml")
    tree.write(out_path, encoding="unicode")
    print(f"[sanitize] {xml_path} -> {out_path}")
    return out_path


def main():
    out_paths = []
    for xml_path in sys.argv[1:]:
        out_paths.append(sanitize_xml(os.path.abspath(xml_path)))
    print(f"[sanitize] 共处理 {len(out_paths)} 个文件")


if __name__ == "__main__":
    main()
