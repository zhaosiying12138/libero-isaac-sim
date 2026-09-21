"""Isaac 侧状态探针：固定关节角，输出各 link/物体的世界位姿（JSON 到 stdout）。

供 tests/test_fk_parity.py 的子进程调用。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/home/zhaosiying/codebase/libero-isaac-sim/source")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--joints", type=str, default="")  # 逗号分隔 7 关节角；空=默认 init_qpos
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from libero_isaac_sim.compat import isaaclab_ea_fixes

isaaclab_ea_fixes.apply()

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from libero_isaac_sim.envs.libero_env_cfg import make_libero_env_cfg

CACHE = os.path.expanduser("~/.cache/libero_isaac_sim")
TASK = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"


def main():
    env_cfg = make_libero_env_cfg(TASK, CACHE)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos.torch[0].clone()
    arm_ids, _ = robot.find_joints(["robot0_joint[1-7]"])
    if args.joints:
        vals = [float(v) for v in args.joints.split(",")]
        assert len(vals) == 7
        joint_pos[arm_ids] = torch.tensor(vals)
    robot.write_joint_state_to_sim(
        joint_pos.unsqueeze(0), torch.zeros_like(joint_pos).unsqueeze(0),
        env_ids=torch.tensor([0]),
    )
    # 0 步：纯 FK（写完关节态直接读，不经控制器步进）
    import argparse as _ap
    pass

    origin = env.scene.env_origins[0].cpu().numpy()
    out = {"joints": robot.data.joint_pos.torch[0, arm_ids].cpu().numpy().tolist()}
    bodies = {}
    for i, bname in enumerate(robot.data.body_names):
        pose = robot.data.body_pose_w.torch[0, i].cpu().numpy()
        bodies[bname] = {
            "pos": (pose[:3] - origin).tolist(),
            "quat_xyzw": pose[3:].tolist(),
        }
    out["bodies"] = bodies
    print("__PROBE_JSON__" + json.dumps(out))
    env.close()
    app.close()


if __name__ == "__main__":
    main()
