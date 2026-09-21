"""Isaac Lab v3.0.0-EA（tag ae37b02）已知缺陷的兼容补丁。

背景
----
3.0 EA 的 PhysX 资产数据路径（``isaaclab_physx`` 的 ``articulation_data.py``）把
``ProxyArray`` 包装器直接传入 warp kernel launch（如
``get_body_com_pose_from_body_link_pose``）。warp 1.16.0 的参数打包器不识别
``ProxyArray``（它不是 ``wp.array`` 子类），报::

    RuntimeError: Error launching kernel 'get_body_com_pose_from_body_link_pose',
    argument 'body_link_pose' expects an array of type array(ndim=2, dtype=transformf),
    but passed value has type ProxyArray.

这使得任何 Articulation 初始化都会失败（WrenchComposer 构建期触发）。
另外 ``_WarpLaunchCache`` 的 ``record_cmd=True`` 录制回放在该版本同样不稳定。

对策
----
将 ``_WarpLaunchCache.launch`` 替换为「解包 ProxyArray → 直接 wp.launch」，
跳过 record_cmd 录制路径。性能差异对本项目（评测/回放场景）可忽略。

本补丁只在本项目代码显式导入时生效，不改动 IsaacLab 源文件。
"""

from __future__ import annotations

import warp as wp

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    from isaaclab.utils.warp import launch_cache
    from isaaclab.utils.warp.proxy_array import ProxyArray

    def _unwrap(v):
        return v.warp if isinstance(v, ProxyArray) else v

    def _direct_launch(self, key: object, kernel: "wp.Kernel", *, dim, inputs, outputs) -> None:
        wp.launch(
            kernel,
            dim=dim,
            inputs=[_unwrap(v) for v in inputs],
            outputs=[_unwrap(v) for v in outputs],
            device=self._device,
        )

    launch_cache._WarpLaunchCache.launch = _direct_launch

    # ------------------------------------------------------------------
    # 补丁 2：让 newton 的 USD 导入忽略 PhysX fix_root_link 运行时生成的
    # 基座 FixedJoint（newton 导入器无法把 FixedJoint 并入 D6 关节组）。
    # 渲染模型的 body 位姿每帧由 USD 变换推送，忽略该关节不影响画面。
    # 注入点选在 ModelBuilder.add_usd，覆盖所有调用方（含可视化模型构建）。
    # ------------------------------------------------------------------
    try:
        from newton._src.sim.builder import ModelBuilder

        _orig_add_usd = ModelBuilder.add_usd
        _ROBOT_FIXED_JOINT_RE = r"/World/envs/env_\d+/Robot/Geometry/robot0_base/FixedJoint"

        def _add_usd_ignore_robot_fixed(self, *args, **kwargs):
            existing = kwargs.get("ignore_paths")
            if existing is None:
                kwargs["ignore_paths"] = [_ROBOT_FIXED_JOINT_RE]
            else:
                kwargs["ignore_paths"] = [*existing, _ROBOT_FIXED_JOINT_RE]
            return _orig_add_usd(self, *args, **kwargs)

        ModelBuilder.add_usd = _add_usd_ignore_robot_fixed
        print("[compat] newton 忽略基座 FixedJoint 补丁已应用")
    except Exception as e:  # noqa: BLE001
        print(f"[compat] newton 补丁跳过（{e}）")

    _APPLIED = True
    print("[compat] isaaclab 3.0 EA ProxyArray/launch_cache 补丁已应用")
