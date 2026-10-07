"""Bounded, read-only-state diagnosis of the Cable V2 action pipeline."""

from __future__ import annotations

import numpy as np

from tools.cable_demo import make_cable_training_env


ACTIONS = (
    ("ZERO", [0.0, 0.0, 0.0]),
    ("X_POS", [0.05, 0.0, 0.0]),
    ("X_NEG", [-0.05, 0.0, 0.0]),
    ("Y_POS", [0.0, 0.05, 0.0]),
    ("Y_NEG", [0.0, -0.05, 0.0]),
    ("Z_POS", [0.0, 0.0, 0.05]),
    ("Z_NEG", [0.0, 0.0, -0.05]),
)
BOUNDS = np.asarray([[0.2, -0.3, 0.0], [0.6, 0.3, 0.5]], dtype=np.float64)


def array_text(value) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    return np.array2string(
        np.asarray(value), precision=10, suppress_small=False, separator=", "
    )


def robot_qpos(base) -> np.ndarray:
    addresses = [
        int(base.model.jnt_qposadr[base.model.joint(f"joint{i}").id])
        for i in range(1, 8)
    ]
    return base.data.qpos[addresses].copy()


def snapshot(base) -> dict[str, np.ndarray | None]:
    try:
        tcp_sensor = base.data.sensor("2f85/pinch_pos").data.copy()
    except Exception:
        tcp_sensor = None
    try:
        tcp_site = base.data.site_xpos[base._pinch_site_id].copy()
    except Exception:
        tcp_site = None
    mocap = base.data.mocap_pos[0].copy() if base.data.mocap_pos.size else None
    return {
        "tcp_sensor": tcp_sensor,
        "tcp_site": tcp_site,
        # In this environment the controller target x_des is read directly
        # from mocap_pos[0]; there is no independent persistent target_pos.
        "target_pos": None if mocap is None else mocap.copy(),
        "mocap_pos": mocap,
        "ctrl": base.data.ctrl.copy(),
        "robot_qpos": robot_qpos(base),
    }


def difference(after, before):
    if after is None or before is None:
        return None
    return np.asarray(after) - np.asarray(before)


def print_step(label: str, action: np.ndarray, base, before, after, step_result) -> None:
    reward, terminated, truncated, info = step_result
    clipped = np.clip(action, base.action_space.low, base.action_space.high)
    scaled = clipped * float(base._action_scale)
    expected_target = np.clip(before["mocap_pos"] + scaled, *BOUNDS)
    print(f"\nCASE: {label}")
    print(f"ACTION_RAW: {array_text(action)}")
    print(f"ACTION_AFTER_SCALE: {array_text(scaled)}")
    print(f"TCP_POS_BEFORE: {array_text(before['tcp_sensor'])}")
    print(f"TCP_POS_AFTER: {array_text(after['tcp_sensor'])}")
    print(f"TCP_DELTA: {array_text(difference(after['tcp_sensor'], before['tcp_sensor']))}")
    print(f"TCP_SITE_BEFORE: {array_text(before['tcp_site'])}")
    print(f"TCP_SITE_AFTER: {array_text(after['tcp_site'])}")
    print(f"TCP_SITE_DELTA: {array_text(difference(after['tcp_site'], before['tcp_site']))}")
    print(f"TARGET_POS_BEFORE: {array_text(before['target_pos'])}")
    print(f"TARGET_POS_EXPECTED: {array_text(expected_target)}")
    print(f"TARGET_POS_AFTER: {array_text(after['target_pos'])}")
    print(f"TARGET_DELTA: {array_text(difference(after['target_pos'], before['target_pos']))}")
    print(f"MOCAP_POS_BEFORE: {array_text(before['mocap_pos'])}")
    print(f"MOCAP_POS_AFTER: {array_text(after['mocap_pos'])}")
    print(f"MOCAP_DELTA: {array_text(difference(after['mocap_pos'], before['mocap_pos']))}")
    print(f"DATA_CTRL_BEFORE: {array_text(before['ctrl'])}")
    print(f"DATA_CTRL_AFTER: {array_text(after['ctrl'])}")
    print(f"CTRL_DELTA: {array_text(difference(after['ctrl'], before['ctrl']))}")
    print(f"ROBOT_QPOS_BEFORE: {array_text(before['robot_qpos'])}")
    print(f"ROBOT_QPOS_AFTER: {array_text(after['robot_qpos'])}")
    print(f"QPOS_DELTA: {array_text(difference(after['robot_qpos'], before['robot_qpos']))}")
    print(f"REWARD: {reward}")
    print(f"TERMINATED: {terminated}")
    print(f"TRUNCATED: {truncated}")
    print(f"SUCCESS: {bool(info.get('success', False))}")


def run_held(env, base, label: str, action: np.ndarray, steps: int = 20) -> None:
    env.reset()
    initial = snapshot(base)
    final_info = {}
    executed = 0
    for _ in range(steps):
        _, _, terminated, truncated, final_info = env.step(action)
        executed += 1
        if terminated or truncated:
            break
    final = snapshot(base)
    scaled = action * float(base._action_scale)
    print(f"\nHELD_ACTION: {label}")
    print(f"ACTION_RAW: {array_text(action)}")
    print(f"ACTION_AFTER_SCALE_PER_STEP: {array_text(scaled)}")
    print(f"STEPS_EXECUTED: {executed}")
    print(f"THEORETICAL_UNCLIPPED_TARGET_CHANGE: {array_text(scaled * executed)}")
    print(f"INITIAL_TARGET: {array_text(initial['target_pos'])}")
    print(f"FINAL_TARGET: {array_text(final['target_pos'])}")
    print(f"TOTAL_TARGET_DELTA: {array_text(difference(final['target_pos'], initial['target_pos']))}")
    print(f"INITIAL_MOCAP: {array_text(initial['mocap_pos'])}")
    print(f"FINAL_MOCAP: {array_text(final['mocap_pos'])}")
    print(f"TOTAL_MOCAP_DELTA: {array_text(difference(final['mocap_pos'], initial['mocap_pos']))}")
    print(f"INITIAL_TCP: {array_text(initial['tcp_sensor'])}")
    print(f"FINAL_TCP: {array_text(final['tcp_sensor'])}")
    print(f"TOTAL_TCP_DELTA: {array_text(difference(final['tcp_sensor'], initial['tcp_sensor']))}")
    print(f"INITIAL_TCP_SITE: {array_text(initial['tcp_site'])}")
    print(f"FINAL_TCP_SITE: {array_text(final['tcp_site'])}")
    print(f"TOTAL_TCP_SITE_DELTA: {array_text(difference(final['tcp_site'], initial['tcp_site']))}")
    print(f"TOTAL_QPOS_DELTA: {array_text(difference(final['robot_qpos'], initial['robot_qpos']))}")
    print(f"FINAL_DISTANCE_TO_SLOT: {final_info.get('distance_to_slot', 'NOT_AVAILABLE')}")


def main() -> int:
    env = make_cable_training_env()
    try:
        base = env.unwrapped
        print("PROCESS: Cable Routing V2 action pipeline diagnosis")
        print(f"ACTION_SCALE: {base._action_scale}")
        print(f"CONTROL_DT: {base.control_dt}")
        print(f"PHYSICS_DT: {base.physics_dt}")
        print(f"PHYSICS_SUBSTEPS_PER_CONTROL_STEP: {base._n_substeps}")
        print("USE_TARGET_AS_CTRL: NOT_AVAILABLE")
        print("TARGET_SOURCE: data.mocap_pos[0]")
        print("CONTROLLER: franka_sim.controllers.opspace")
        print(f"PANDA_DOF_IDS: {array_text(base._panda_dof_ids)}")
        print(f"PANDA_CTRL_IDS: {array_text(base._panda_ctrl_ids)}")
        print(
            "PANDA_ACTUATOR_FORCE_RANGES: "
            f"{array_text(base.model.actuator_forcerange[base._panda_ctrl_ids])}"
        )
        print(
            "PANDA_ACTUATOR_CTRL_LIMITED: "
            f"{array_text(base.model.actuator_ctrllimited[base._panda_ctrl_ids])}"
        )
        print(
            "PANDA_ACTUATOR_CTRL_RANGES: "
            f"{array_text(base.model.actuator_ctrlrange[base._panda_ctrl_ids])}"
        )
        print(
            "PANDA_ACTUATOR_GAINPRM: "
            f"{array_text(base.model.actuator_gainprm[base._panda_ctrl_ids])}"
        )

        env.reset()
        for label, values in ACTIONS:
            action = np.asarray(values, dtype=np.float32)
            before = snapshot(base)
            _, reward, terminated, truncated, info = env.step(action)
            after = snapshot(base)
            print_step(
                label,
                action,
                base,
                before,
                after,
                (reward, terminated, truncated, info),
            )
            if terminated or truncated:
                print("SEQUENCE_RESET_AFTER_EPISODE_BOUNDARY: TRUE")
                env.reset()

        run_held(env, base, "X_POS_20", np.asarray([0.05, 0.0, 0.0], np.float32))
        run_held(env, base, "X_NEG_20", np.asarray([-0.05, 0.0, 0.0], np.float32))
        print("\nFORMAL_DEMO_SAVED: NO")
        print("RESULT: COMPLETED")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
