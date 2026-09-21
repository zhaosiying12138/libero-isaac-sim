#!/usr/bin/env python
"""相机朝向探针：同一世界位姿下渲染 4 种候选旋转约定，与 MuJoCo 参考帧对比。

候选（MuJoCo agentview quat wxyz = (0.6182, 0.3432, 0.3432, 0.6182)）：
  A. xyzw 直排
  B. xyzw 后再绕相机 X 轴转 180°（USD/MuJoCo 上下镜像差）
  C. xyzw 后再绕相机 Z 轴转 180°
  D. wxyz 原样（错误对照）
"""

import argparse
import os
import sys

sys.path.insert(0, "/home/zhaosiying/codebase/libero-isaac-sim/source")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from libero_isaac_sim.compat import isaaclab_ea_fixes

isaaclab_ea_fixes.apply()

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass
from isaaclab_newton.renderers import NewtonWarpRendererCfg

from libero_isaac_sim.semantics.math_utils import mat_to_quat_xyzw, quat_wxyz_to_mat

CACHE = os.path.expanduser("~/.cache/libero_isaac_sim")
SOUP_USD = os.path.join(
    CACHE,
    "usd/objects/alphabet_soup/alphabet_soup_sanitized_freejoint.patched/"
    "alphabet_soup_sanitized_freejoint.patched.usda",
)

CAM_POS = (0.607, 0.0, 0.96)
CAM_QUAT_WXYZ = (0.6182, 0.3432, 0.3432, 0.6182)
R = quat_wxyz_to_mat(CAM_QUAT_WXYZ)

flip_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
flip_z = np.diag([-1.0, -1.0, 1.0])

CANDIDATES = {
    "A_xyzw": R,
    "B_flipx": R @ flip_x,
    "C_flipz": R @ flip_z,
    "D_wxyz_raw": None,  # 特殊标记：直接传 wxyz
}


@configclass
class ProbeCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.4)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.01)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    soup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/soup",
        spawn=sim_utils.UsdFileCfg(usd_path=SOUP_USD),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.102, -0.152, 0.46)),
    )


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
    scene = InteractiveScene(ProbeCfg(num_envs=1, env_spacing=2.0))

    cameras = {}
    for tag, rot in CANDIDATES.items():
        quat = (
            tuple(float(v) for v in mat_to_quat_xyzw(rot))
            if rot is not None
            else (0.6182, 0.3432, 0.3432, 0.6182)  # D: 原样 wxyz
        )
        cameras[tag] = Camera(
            CameraCfg(
                prim_path=f"/World/cam_{tag}",
                update_period=0,
                height=128,
                width=128,
                data_types=["rgb"],
                renderer_cfg=NewtonWarpRendererCfg(),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, horizontal_aperture=19.9,
                    clipping_range=(0.01, 10.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=CAM_POS, rot=quat, convention="world"
                ),
            )
        )

    sim.reset()
    scene.reset()
    for _ in range(5):
        sim.step(render=True)
        scene.update(1.0 / 120.0)
    for c in cameras.values():
        c.update(1.0 / 120.0)

    import imageio

    imgs = []
    for tag, cam in cameras.items():
        rgb = cam.data.output["rgb"][0].cpu().numpy()
        imageio.imwrite(f"/tmp/camprobe_{tag}.png", rgb)
        imgs.append(rgb)
    grid = np.concatenate([np.concatenate(imgs[:2], axis=1), np.concatenate(imgs[2:], axis=1)], axis=0)
    imageio.imwrite("/tmp/camprobe_grid.png", grid)
    print("[camprobe] saved /tmp/camprobe_{A_xyzw,B_flipx,C_flipz,D_wxyz_raw}.png + grid")
    app.close()


if __name__ == "__main__":
    main()
