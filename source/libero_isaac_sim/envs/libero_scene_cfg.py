"""从任务语义 + 资产注册表动态构建 Isaac Lab 场景配置。

构成（对照 LIBERO 的 MJCF 组合链）：
- arena：LIBERO 场景 XML 转出的 USD（地板/墙/桌面/背景，静态外壳）
- fixtures：关节体 USD（fix_base articulation，如 microwave/flat_stove/cabinet）
- objects：单刚体 USD（注入自由关节转换 + 质量修正）
- robot：自转的 Panda USD（on_ground / mounted 两变体，与 LIBERO 同一模型来源）
- cameras：agentview（世界固定，位姿/内参来自 manifest）、
  robot0_eye_in_hand（挂在 robot0_right_hand 下，局部变换来自 manifest）

坐标约定：Isaac Lab 3.0 四元数为 xyzw；相机 OffsetCfg 的 rot 同样 xyzw。
MuJoCo 相机与 USD 相机同为「-Z 朝前、+Y 朝上」，因此姿态矩阵可直接复用，
只需 wxyz→xyzw 的重排。
"""

from __future__ import annotations

import os

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from libero_isaac_sim.converters.asset_manifest import AssetRegistry
from libero_isaac_sim.semantics.math_utils import mat_to_quat_xyzw
from libero_isaac_sim.semantics.task_spec import TaskSemantics

# 机器人 USD 变体选择：kitchen 用带 RethinkMount 的，living_room/study 用落地式
ARENA_ROBOT_VARIANT = {
    "kitchen": "panda_mounted",
    "living_room": "panda_on_ground",
    "study": "panda_on_ground",
    "table": "panda_on_ground",
}


def _quat_xyzw_from_rotmat(rotmat9) -> tuple[float, float, float, float]:
    m = np.asarray(rotmat9, dtype=float).reshape(3, 3)
    q = mat_to_quat_xyzw(m)
    return tuple(float(v) for v in q)


def _camera_cfg_from_manifest(name: str, cam: dict, parent_prim: str | None) -> CameraCfg:
    """按 manifest 的相机参数构建 CameraCfg。

    MuJoCo fovy（纵向视场角，度）→ Isaac PinholeCameraCfg 焦距/光圈换算：
    方形像素下 horizontal_aperture = 2 * focal * tan(fov_x / 2)，
    对等效方形画幅 fov_x == fovy。焦距沿用 Isaac 默认 24mm。
    """
    focal = 24.0
    fovy_rad = np.deg2rad(cam["fovy_deg"])
    aperture = 2.0 * focal * np.tan(fovy_rad / 2.0)

    if parent_prim is None:
        offset = CameraCfg.OffsetCfg(
            pos=tuple(float(v) for v in cam["pos"]),
            rot=_quat_xyzw_from_rotmat(cam["rotmat"]),
            convention="opengl",
        )
    else:
        offset = CameraCfg.OffsetCfg(
            pos=tuple(float(v) for v in cam["pos"]),
            rot=_quat_xyzw_from_rotmat(cam["rotmat"]),
            convention="opengl",
        )
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/" + name if parent_prim is None else parent_prim + "/" + name,
        update_period=0,
        height=cam["height"],
        width=cam["width"],
        data_types=["rgb", "normals"],
        renderer_cfg=NewtonWarpRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal,
            horizontal_aperture=float(aperture),
            clipping_range=(0.01, 10.0),
        ),
        offset=offset,
    )


def _find_prim_path_by_name(usd_path: str, name: str) -> str:
    """在 USD 中按 prim 名找完整路径（机器人 USD 的 body 嵌套很深）。"""
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    for prim in stage.TraverseAll():
        if prim.GetName() == name:
            return str(prim.GetPath())
    raise KeyError(f"{usd_path} 中找不到 prim {name}")


def build_scene_cfg(
    task: TaskSemantics,
    registry: AssetRegistry,
    num_envs: int = 1,
    env_spacing: float = 2.5,
) -> InteractiveSceneCfg:
    """按任务语义构建 InteractiveSceneCfg。"""
    cache_dir = registry.cache_dir
    arena_type = task.arena["arena_type"]
    robot_variant = ARENA_ROBOT_VARIANT[arena_type]
    robot_usd = os.path.join(
        cache_dir, "usd", "robots", robot_variant, robot_variant, f"{robot_variant}.usda"
    )
    assert os.path.exists(robot_usd), f"机器人 USD 不存在: {robot_usd}，先跑 scripts/convert_robot.py"

    attrs = {}

    # --- arena 静态外壳 ---
    arena_usd = registry.usd_path(f"_arena/{task.task_name}")
    attrs["arena"] = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Arena",
        spawn=sim_utils.UsdFileCfg(usd_path=arena_usd),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # --- 灯光 ---
    attrs["dome_light"] = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # --- 机器人 ---
    base_pos = task.robot.get("base_pos") or [0.0, 0.0, 0.0]
    base_rotmat = task.robot.get("base_rotmat")
    base_quat = (
        _quat_xyzw_from_rotmat(base_rotmat) if base_rotmat else (0.0, 0.0, 0.0, 1.0)
    )
    init_qpos = task.robot["init_qpos"]
    joint_pos_init = {f"robot0_joint{i+1}": float(init_qpos[i]) for i in range(7)}
    # 镜像手指关节：joint2 限位为负向（[-0.04, 0]），张开 = joint1 +0.04 / joint2 -0.04
    joint_pos_init["gripper0_finger_joint1"] = 0.04
    joint_pos_init["gripper0_finger_joint2"] = -0.04
    attrs["robot"] = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,  # LIBERO 机器人基座固定（运行时 FixedJoint 由渲染补丁忽略）
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(v) for v in base_pos),
            rot=base_quat,
            joint_pos=joint_pos_init,
        ),
        actuators={
            # 手臂由 OSC 力矩控制（stiffness=0 的隐式执行器占位）
            "panda_arm": ImplicitActuatorCfg(
                joint_names_expr=["robot0_joint[1-7]"], stiffness=0.0, damping=0.0
            ),
            # 夹爪位置控制（二值开合）
            "panda_gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper0_finger_joint[12]"],
                stiffness=float(os.environ.get("LIBERO_GRIPPER_STIFFNESS", "2000")),
                damping=float(os.environ.get("LIBERO_GRIPPER_DAMPING", "100")),
            ),
        },
    )

    # --- 可动物体（单刚体） ---
    for name in task.movable_objects():
        entry = registry.get(name)
        attrs[name] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            spawn=sim_utils.UsdFileCfg(
                usd_path=entry["usd_path"],
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005, rest_offset=0.0
                ),
                # MuJoCo 摩擦三元组的滑动分量 → PhysX 静/动摩擦
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=float(
                        task.entities[name].get("geom_friction_mean", [0.95])[0]
                    ),
                    dynamic_friction=float(
                        task.entities[name].get("geom_friction_mean", [0.95])[0]
                    ),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

    # --- fixtures（关节体或静态） ---
    for name in task.fixtures():
        entry = registry.get(name)
        n_joints = len(task.entities[name].get("joints", []))
        if n_joints > 0:
            attrs[name] = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=entry["usd_path"],
                    activate_contact_sensors=True,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        # 基座固定由 USD 内的世界固定关节负责（避免运行时补丁
                        # 与 newton 导入器的关节合并缺陷冲突）
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
                actuators={
                    "fixture_joints": ImplicitActuatorCfg(
                        joint_names_expr=[".*"], stiffness=0.0, damping=10.0
                    ),
                },
            )
        else:
            # 无关节 fixture（如 desk_caddy）：每个 episode 仍需按初始状态重摆位，
            # 用运动学刚体（kinematic rigid body）：物理中静止、reset 时可写位姿
            attrs[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=entry["usd_path"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            )

    # --- 相机 ---
    if os.environ.get("LIBERO_TEST_NO_CAM") == "1":
        SceneCfg = configclass(
            type(f"LiberoSceneCfg_{task.task_name[:32]}", (InteractiveSceneCfg,), {})
        )
        cfg = SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        for k, v in attrs.items():
            setattr(cfg, k, v)
        cfg.replicate_physics = False
        return cfg
    cams = task.cameras
    attrs["agentview"] = _camera_cfg_from_manifest("agentview", cams["agentview"], None)
    wrist = cams["robot0_eye_in_hand"]
    wrist_body = wrist["body"]  # robot0_right_hand
    # 挂接点在机器人 USD 内的真实深层路径；spawn 到该路径下
    hand_path_in_usd = _find_prim_path_by_name(robot_usd, wrist_body)
    # USD 内路径以 defaultPrim（如 /base）开头；spawn 后挂在 {ENV_REGEX_NS}/Robot 下的相对部分
    rel = hand_path_in_usd.split("/", 2)[-1]  # 去掉 "/<defaultPrim>/" 前缀
    attrs["robot0_eye_in_hand"] = _camera_cfg_from_manifest(
        "robot0_eye_in_hand",
        wrist,
        parent_prim="{ENV_REGEX_NS}/Robot/" + rel,
    )

    # 动态构造 configclass
    SceneCfg = configclass(
        type(
            f"LiberoSceneCfg_{task.task_name[:32]}",
            (InteractiveSceneCfg,),
            {},
        )
    )
    cfg = SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    for k, v in attrs.items():
        setattr(cfg, k, v)
    cfg.replicate_physics = False  # CPU 后端下保持 False，语义最直观
    return cfg
