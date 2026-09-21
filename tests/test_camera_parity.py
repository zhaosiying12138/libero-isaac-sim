"""L7 相机对齐：把同一世界点投影到两侧图像坐标系，求像素级对应关系。

做法：
- MuJoCo 侧：用 mujoco 的相机模型投影（-Z 朝前、+Y 朝上、渲染帧底向上，LIBERO 翻转存图）
- Isaac 侧：用相机世界位姿 + 内参投影
- 世界点集：机器人基座、桌面四角、几个物体中心（init state 0）

输出每点的两侧像素坐标与差异模式（x 镜像 / y 镜像 / 旋转），据此修正相机偏移约定。
"""

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from experiments.bridge import MujocoBridge

ISAACLAB_DIR = os.path.expanduser("~/codebase/_external/IsaacLab")
UV = os.path.expanduser("~/miniforge3/bin/uv")
BDDL = (
    "/home/zhaosiying/codebase/_external/LIBERO/libero/libero/bddl_files/"
    "libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_"
    "the_tomato_sauce_in_the_basket.bddl"
)
ISAAC_PROJ_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "project_isaac_camera.py"
)

# 世界点集（init state 0 的语义状态：物体中心 + 机器人基座 + 桌角）
POINTS = {
    "robot_base": [-0.51, 0.0, 0.42],
    "soup": [-0.102, -0.152, 0.46],
    "basket": [-0.002, 0.269, 0.48],
    "table_center": [0.0, 0.0, 0.434],
    "table_corner_--": [-0.35, -0.8, 0.434],
    "table_corner_-+": [-0.35, 0.8, 0.434],
}


def project_mujoco(point):
    """mujoco agentview 投影（翻转后图像坐标）。"""
    import mujoco

    cam_pos = np.array([0.6065773716836134, 0.0, 0.96])
    cam_quat_wxyz = np.array([0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979])
    # wxyz -> rotmat
    w, x, y, z = cam_quat_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    p_cam = R.T @ (np.array(point) - cam_pos)
    # mujoco 相机：-z 朝前，x 右，y 上
    W = H = 128
    fovy = np.deg2rad(45.0)
    f_px = H / (2 * np.tan(fovy / 2))
    u = W / 2 + f_px * p_cam[0] / (-p_cam[2])
    v_up = H / 2 - f_px * p_cam[1] / (-p_cam[2])
    # LIBERO 存储帧 = 顶行在前（翻转后）= v_up 本身就是正确目标
    return u, v_up, p_cam


def project_isaac(points):
    env = dict(os.environ)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    proc = subprocess.run(
        [UV, "run", "--extra", "isaacsim", "python", "-u",
         os.path.abspath(ISAAC_PROJ_SCRIPT), "--points", json.dumps(points)],
        cwd=ISAACLAB_DIR, env=env, capture_output=True, text=True, timeout=1800,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__PROJ_JSON__"):
            return json.loads(line[len("__PROJ_JSON__"):])
    raise RuntimeError("Isaac 投影探针无输出:\n" + proc.stderr[-1500:])


def main():
    isa = project_isaac(POINTS)
    print("isaac cam pos_w:", isa["_cam_pos_w"])
    import numpy as _np
    print("isaac cam 世界矩阵:\n", _np.round(_np.array(isa["_cam_matrix"]), 3))
    print(f"{'点':16s} {'mujoco(u,v)':>18s} {'isaac(u,v)':>18s} {'差':>14s}")
    for name, pt in POINTS.items():
        u_mj, v_mj, p_cam = project_mujoco(pt)
        u_is, v_is = isa[name]
        print(
            f"{name:16s} ({u_mj:7.1f},{v_mj:7.1f}) ({u_is:7.1f},{v_is:7.1f}) "
            f"({u_is - u_mj:+7.1f},{v_is - v_mj:+7.1f})"
        )
        if p_cam[2] > 0:
            print(f"    [警告] {name} 在 mujoco 相机后方 (z_cam={p_cam[2]:.3f})")


if __name__ == "__main__":
    main()
