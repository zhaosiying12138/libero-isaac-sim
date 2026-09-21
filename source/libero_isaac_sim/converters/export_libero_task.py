"""语义状态导出器：把 LIBERO 任务从 MuJoCo 私有表示转成仿真器无关的语义描述。

在 libero-mujoco 环境中运行（仅此脚本依赖 robosuite/mujoco），对每个任务导出两份产物：

1. ``<task>_manifest.json`` —— 静态语义：
   - BDDL 解析结果（objects/fixtures/regions/init/goal/language）
   - 每个实体的类别、MJCF xml 路径、网格文件、关节名与 open/close/turnon/turnoff 阈值
   - 每个 site region 的局部位姿与尺寸（相对父 body）
   - 相机内外参（agentview / robot0_eye_in_hand 的 pos/quat/fovy）
   - 物理参数：每个 body 的质量、惯性、geom 摩擦
   - arena/桌子尺寸、机器人基座偏移、初始关节角、动作缩放约定

2. ``<task>_init_states.npz`` —— 50 组官方固定初始状态（从 .pruned_init 逐个加载后按名提取）：
   - 机器人 7 关节角 + 夹爪 2 关节角
   - 每个物体/fixture 的世界位姿（位置 + 3x3 旋转矩阵，显式 convention）
   - 每个关节物体的关节角

约定（写入 manifest["convention"]）：
- 位置单位：米；世界上轴：+Z
- 旋转存储：3x3 旋转矩阵（避免四元数 wxyz/xyzw 约定歧义），另附 quat_wxyz 冗余字段
- MuJoCo body_xquat 为 wxyz

用法（libero-mujoco 环境）::

    python -m libero_isaac_sim.converters.export_libero_task \
        --bddl <task.bddl> --out-dir <cache_dir>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# 路径约定
# ---------------------------------------------------------------------------
LIBERO_REPO = os.environ.get(
    "LIBERO_REPO", os.path.expanduser("~/codebase/_external/LIBERO")
)
DEFAULT_CACHE_DIR = os.environ.get(
    "LIBERO_ISAAC_ASSETS_DIR", os.path.expanduser("~/.cache/libero_isaac_sim")
)


def _quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 四元数转 3x3 旋转矩阵（显式实现，不依赖第三方变换库）。"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def _mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵转 wxyz 四元数（Shepperd 方法，数值稳定分支）。"""
    m = np.asarray(mat, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _name2id(model, objtype, name: str) -> int:
    """兼容不同 mujoco 版本的 name→id 查询。"""
    import mujoco

    try:
        return int(mujoco.mj_name2id(model, objtype, name))
    except Exception:
        pass
    # 部分版本 MjModel 挂有 *_name2id 便捷方法
    method = {
        mujoco.mjtObj.mjOBJ_BODY: "body_name2id",
        mujoco.mjtObj.mjOBJ_CAMERA: "camera_name2id",
    }.get(objtype)
    if method and hasattr(model, method):
        return int(getattr(model, method)(name))
    raise KeyError(f"无法解析名称 {name} (type={objtype})")


def _body_id(model, name: str) -> int:
    import mujoco

    return _name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _cam_id(model, name: str) -> int:
    import mujoco

    return _name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)


def _qpos_addr(model, joint_name: str) -> int:
    """不同 mujoco 版本返回值不同：老版本 int，3.3.x 起返回 (qpos_adr, dof_adr)。"""
    addr = model.get_joint_qpos_addr(joint_name)
    if isinstance(addr, (tuple, list)):
        return int(addr[0])
    return int(addr)


def export_task(bddl_file: str, out_dir: str, num_init_states: int | None = None) -> dict:
    """导出一个任务，返回 manifest dict。"""
    # 延迟导入：仅在 libero-mujoco 环境中可用
    import torch
    from libero.libero.envs import OffScreenRenderEnv

    from libero_isaac_sim.semantics.bddl_parser import parse_bddl

    task_name = os.path.splitext(os.path.basename(bddl_file))[0]
    spec = parse_bddl(bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=128,
        camera_widths=128,
    )
    env.reset()

    sim = env.sim
    model = sim.model

    # ------------------------------------------------------------------
    # 1. 静态实体清单：类别、关节、阈值、物理参数
    # ------------------------------------------------------------------
    def _classify_joints(obj):
        """把实体关节分成 articulation 关节与自由关节（freejoint 由根位姿覆盖）"""
        joints_all = list(obj.joints) if obj.joints is not None else []
        articulation_joints, free_joints = [], []
        for jn in joints_all:
            jid = model.joint_name2id(jn)
            jtype = int(model.jnt_type[jid]) if jid >= 0 else -1
            # 0=free 1=ball 2=slide 3=hinge
            if jtype == 0:
                free_joints.append(jn)
            else:
                articulation_joints.append(jn)
        return articulation_joints, free_joints

    entities = {}
    for name, obj in list(env.env.objects_dict.items()):
        aj, fj = _classify_joints(obj)
        entry = {
            "kind": "object",
            "category": obj.category_name,
            "root_body": obj.root_body,
            "joints": aj,
            "free_joints": fj,
            "articulation_ranges": dict(
                obj.object_properties.get("articulation", {})
            ),
        }
        body_id = env.env.obj_body_id[name]
        entry["mass"] = float(model.body_mass[body_id])
        entry["inertia"] = [float(v) for v in model.body_inertia[body_id]]
        entry["com_local"] = [float(v) for v in model.body_ipos[body_id]]
        entry["inertia_quat_wxyz"] = [float(v) for v in model.body_iquat[body_id]]
        # 收集该 body 树（含子 body）内所有 geom 的摩擦均值，供物理审计参考
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        if geoms:
            entry["geom_friction_mean"] = [
                float(np.mean([model.geom_friction[g][i] for g in geoms]))
                for i in range(3)
            ]
        entities[name] = entry
    for name, obj in list(env.env.fixtures_dict.items()):
        aj, fj = _classify_joints(obj)
        entry = {
            "kind": "fixture",
            "category": obj.category_name,
            "root_body": obj.root_body,
            "joints": aj,
            "free_joints": fj,
            "articulation_ranges": dict(
                obj.object_properties.get("articulation", {})
            ),
        }
        body_id = env.env.obj_body_id[name]
        entry["mass"] = float(model.body_mass[body_id])
        entry["inertia"] = [float(v) for v in model.body_inertia[body_id]]
        entities[name] = entry

    # 关节阈值（open/close/turnon/turnoff）提取
    for name, entry in entities.items():
        obj = env.env.get_object(name)
        props = obj.object_properties.get("articulation", {})
        entry["joint_thresholds"] = {
            k: [float(v) for v in val] for k, val in props.items()
        }

    # ------------------------------------------------------------------
    # 2. site region 定义：局部位姿 + 尺寸（相对父 body）
    #    局部变换用「site 世界位姿 @ 父 body 世界位姿⁻¹」构造，与 XML 坐标系无关。
    # ------------------------------------------------------------------
    sites = {}
    sim.forward()
    for site_name, site_obj in env.env.object_sites_dict.items():
        size = site_obj.size
        if isinstance(size, str):
            size = [float(v) for v in size.split()]
        sid = model.site_name2id(site_name)
        site_world_pos = np.array(sim.data.site_xpos[sid])
        site_world_mat = np.array(sim.data.site_xmat[sid]).reshape(3, 3)
        parent_bid = int(model.site_bodyid[sid])
        parent_body = model.body_id2name(parent_bid)
        if parent_bid > 0 and parent_body:
            pb_pos = np.array(sim.data.body_xpos[parent_bid])
            pb_mat = _quat_wxyz_to_mat(sim.data.body_xquat[parent_bid])
            local_mat = pb_mat.T @ site_world_mat
            local_pos = pb_mat.T @ (site_world_pos - pb_pos)
        else:
            local_mat = site_world_mat
            local_pos = site_world_pos
        sites[site_name] = {
            "parent_name": site_obj.parent_name,
            "parent_body": parent_body,
            "size": [float(v) for v in np.atleast_1d(size)],
            "local_pos": [float(v) for v in local_pos],
            "local_rotmat": local_mat.reshape(-1).tolist(),
            "world_pos_ref": [float(v) for v in site_world_pos],
            "site_type": site_obj.site_type,
            "joints": list(site_obj.joints) if site_obj.joints else [],
        }

    # ------------------------------------------------------------------
    # 3. 相机内外参
    # ------------------------------------------------------------------
    cameras = {}
    for cam_name in ["agentview", "robot0_eye_in_hand"]:
        cam_id = model.camera_name2id(cam_name)
        cameras[cam_name] = {
            "pos": [float(v) for v in model.cam_pos[cam_id]],
            "quat_wxyz": [float(v) for v in model.cam_quat[cam_id]],
            "rotmat": _quat_wxyz_to_mat(model.cam_quat[cam_id]).reshape(-1).tolist(),
            "fovy_deg": float(model.cam_fovy[cam_id]),
            "width": 128,
            "height": 128,
        }
        # 腕部相机记录挂载 body
        cam_body = model.cam_bodyid[cam_id]
        cameras[cam_name]["body"] = model.body_id2name(cam_body)

    # ------------------------------------------------------------------
    # 4. arena 与机器人静态配置
    # ------------------------------------------------------------------
    arena_info = {
        "arena_type": env.env._arena_type,
        "scene_xml": env.env._arena_xml,
        "workspace_offset": [float(v) for v in env.env.workspace_offset],
        "z_offset": float(env.env.z_offset),
    }
    for attr in [
        "table_full_size",
        "kitchen_table_full_size",
        "living_room_table_full_size",
        "study_table_full_size",
        "coffee_table_full_size",
    ]:
        if hasattr(env.env, attr):
            arena_info[attr] = [float(v) for v in getattr(env.env, attr)]

    robot_info = {
        "name": type(env.robots[0].robot_model).__name__,
        "init_qpos": [float(v) for v in env.robots[0].robot_model.init_qpos],
        "control_freq": float(env.env.control_freq),
        # robosuite OSC_POSE 默认输出限幅（与 suite.load_controller_config("OSC_POSE") 一致）
        "action_scale": {"pos_delta_m": 0.05, "rot_delta_rad": 0.5},
        "action_dim": 7,
    }
    # 机器人基座与 EE 帧位姿（动作/观测帧对齐的依据）
    try:
        base_bid = _body_id(model, "robot0_base")
        robot_info["base_pos"] = [float(v) for v in sim.data.body_xpos[base_bid]]
        robot_info["base_rotmat"] = (
            _quat_wxyz_to_mat(sim.data.body_xquat[base_bid]).reshape(-1).tolist()
        )
    except Exception:
        robot_info["base_pos"] = None
        robot_info["base_rotmat"] = None
    try:
        eef_name = env.robots[0].robot_model.eef_name
        eef_bid = _body_id(model, eef_name)
        robot_info["eef_body_name"] = eef_name
        robot_info["eef_pos_at_init"] = [float(v) for v in sim.data.body_xpos[eef_bid]]
        robot_info["eef_rotmat_at_init"] = (
            _quat_wxyz_to_mat(sim.data.body_xquat[eef_bid]).reshape(-1).tolist()
        )
    except Exception as e:
        robot_info["eef_body_name"] = None
        robot_info["eef_error"] = str(e)

    # ------------------------------------------------------------------
    # 5. 50 组固定初始状态
    # ------------------------------------------------------------------
    init_file = os.path.join(
        LIBERO_REPO, "libero/libero/init_files/libero_10", f"{task_name}.pruned_init"
    )
    assert os.path.exists(init_file), f"init 文件不存在: {init_file}"
    init_states = torch.load(init_file, map_location="cpu", weights_only=False)
    if not isinstance(init_states, (np.ndarray, torch.Tensor)):
        init_states = np.asarray(init_states)
    if num_init_states is not None:
        init_states = init_states[:num_init_states]
    n_states = len(init_states)

    entity_names = list(entities.keys())
    robot_joint_names = [f"robot0_joint{i+1}" for i in range(7)]
    gripper_joint_names = [f"robot0_gripper_qpos_{i}" for i in range(2)]

    # 预分配
    init_data = {
        "robot_joint_pos": np.zeros((n_states, 7)),
        "gripper_qpos": np.zeros((n_states, 2)),
    }
    for name in entity_names:
        init_data[f"pos::{name}"] = np.zeros((n_states, 3))
        init_data[f"rotmat::{name}"] = np.zeros((n_states, 9))
        n_joints = len(entities[name]["joints"])
        if n_joints > 0:
            init_data[f"joint::{name}"] = np.zeros((n_states, n_joints))

    # 机器人关节地址
    arm_addrs = []
    for jname in env.robots[0].robot_model.joints:
        arm_addrs.append(_qpos_addr(model, jname))
    gripper_addrs = [
        _qpos_addr(model, j) for j in env.robots[0].gripper.joints
    ]

    for i in range(n_states):
        env.set_init_state(init_states[i])
        env.sim.forward()
        qpos = sim.data.qpos
        init_data["robot_joint_pos"][i] = [qpos[a] for a in arm_addrs[:7]]
        init_data["gripper_qpos"][i] = [qpos[a] for a in gripper_addrs[:2]]
        for name in entity_names:
            body_id = env.env.obj_body_id[name]
            pos = np.array(sim.data.body_xpos[body_id])
            quat_wxyz = np.array(sim.data.body_xquat[body_id])
            init_data[f"pos::{name}"][i] = pos
            init_data[f"rotmat::{name}"][i] = _quat_wxyz_to_mat(quat_wxyz).reshape(-1)
            joints = entities[name]["joints"]
            if joints:
                init_data[f"joint::{name}"][i] = [
                    qpos[_qpos_addr(model, j)] for j in joints
                ]

    # ------------------------------------------------------------------
    # 6. 写盘
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "task_name": task_name,
        "bddl": dict(spec),
        "entities": entities,
        "sites": sites,
        "cameras": cameras,
        "arena": arena_info,
        "robot": robot_info,
        "num_init_states": n_states,
        "convention": {
            "position_unit": "meter",
            "world_up_axis": "z",
            "rotation_storage": "rotation_matrix_row_major_3x3",
            "external_quaternion_convention": "wxyz (MuJoCo body_xquat)",
            "action_space": "7D: [dpos(3) daxis_angle(3) gripper(1)], clipped to [-1,1]",
        },
    }
    manifest_path = os.path.join(out_dir, f"{task_name}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    npz_path = os.path.join(out_dir, f"{task_name}_init_states.npz")
    np.savez_compressed(npz_path, **init_data)

    env.close()
    print(f"[export] {task_name}: manifest -> {manifest_path}")
    print(f"[export] {task_name}: {n_states} init states -> {npz_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", type=str, default=None, help="单个 bddl 文件路径")
    parser.add_argument("--all-libero10", action="store_true", help="导出全部 10 个任务")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--num-init-states", type=int, default=None)
    args = parser.parse_args()

    if args.all_libero10:
        import glob

        bddl_dir = os.path.join(
            LIBERO_REPO, "libero/libero/bddl_files/libero_10"
        )
        bddl_files = sorted(glob.glob(os.path.join(bddl_dir, "*.bddl")))
    else:
        assert args.bddl, "需要 --bddl 或 --all-libero10"
        bddl_files = [args.bddl]

    for bddl_file in bddl_files:
        export_task(bddl_file, args.out_dir, args.num_init_states)


if __name__ == "__main__":
    main()
