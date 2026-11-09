"""Validate offline dual-arm IK/smoothing without contacting robot hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_joint_trajectory(document: dict[str, Any], expected_frames: int) -> tuple[np.ndarray, np.ndarray]:
    samples = document.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_frames:
        raise ValueError("trajectory sample count differs from plan")
    times = np.asarray([sample["t_s"] for sample in samples], dtype=float)
    joints = np.asarray(
        [
            sample["left_arm_joint_positions_rad"]
            + sample["right_arm_joint_positions_rad"]
            for sample in samples
        ],
        dtype=float,
    )
    if joints.shape != (expected_frames, 14) or not np.all(np.isfinite(joints)):
        raise ValueError("trajectory must contain finite [frames, 14] arm joints")
    return times, joints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("trajectory", type=Path, help="PyRoki-smoothed trajectory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--ik", type=Path, help="optional pre-smoothing Pinocchio IK artifact")
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--max-translation-error", type=float, default=0.005)
    parser.add_argument("--max-rotation-error", type=float, default=0.02)
    parser.add_argument("--branch-jump-threshold", type=float, default=0.35)
    args = parser.parse_args()

    import pinocchio as pin

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    smoothed = json.loads(args.trajectory.read_text(encoding="utf-8"))
    if plan.get("execution_enabled") is not False or smoothed.get("execution_enabled") is not False:
        raise ValueError("inputs must be offline artifacts")
    plan_samples = plan["samples"]
    plan_times = np.asarray([sample["t_s"] for sample in plan_samples], dtype=float)
    if not np.all(np.diff(plan_times) > 0.0):
        raise ValueError("plan timestamps must be strictly increasing")

    model = pin.buildModelFromUrdf(str(args.urdf))
    data = model.createData()
    expected_names = tuple(
        f"{side}_fr3v2_joint{index}"
        for side in ("left", "right")
        for index in range(1, 8)
    )
    if tuple(str(name) for name in model.names[1:]) != expected_names:
        raise ValueError("unexpected URDF joint order")
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    mounts = {
        side: data.oMf[model.getFrameId(f"{side}_fr3v2_link0")].copy()
        for side in ("left", "right")
    }
    eef_ids = {
        side: model.getFrameId(f"{side}_fr3v2_link8")
        for side in ("left", "right")
    }

    def evaluate(label: str, document: dict[str, Any]) -> dict[str, Any]:
        times, q_values = load_joint_trajectory(document, len(plan_samples))
        if not np.array_equal(times, plan_times):
            raise ValueError(f"{label} did not preserve the original timestamps exactly")
        errors: dict[str, dict[str, list[float]]] = {
            side: {"translation_m": [], "rotation_rad": []}
            for side in ("left", "right")
        }
        for plan_sample, q in zip(plan_samples, q_values, strict=True):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            for side in ("left", "right"):
                base_target = np.asarray(plan_sample[side]["base_T_link8_target"]["matrix"], dtype=float)
                target = mounts[side] * pin.SE3(base_target[:3, :3], base_target[:3, 3])
                error = pin.log6(data.oMf[eef_ids[side]].actInv(target)).vector
                errors[side]["translation_m"].append(float(np.linalg.norm(error[:3])))
                errors[side]["rotation_rad"].append(float(np.linalg.norm(error[3:])))

        lower = np.asarray(model.lowerPositionLimit, dtype=float)
        upper = np.asarray(model.upperPositionLimit, dtype=float)
        below = q_values < lower[None, :] - 1e-9
        above = q_values > upper[None, :] + 1e-9
        position_excess = np.maximum(lower[None, :] - q_values, q_values - upper[None, :])
        position_excess = np.maximum(position_excess, 0.0)
        dt = np.diff(times)
        velocity = np.diff(q_values, axis=0) / dt[:, None]
        velocity_limits = np.asarray(model.velocityLimit, dtype=float)
        velocity_excess = np.abs(velocity) - velocity_limits[None, :]
        velocity_violations = velocity_excess > 1e-9
        interval_mid_dt = (dt[:-1] + dt[1:]) / 2.0
        acceleration = np.diff(velocity, axis=0) / interval_mid_dt[:, None]
        adjacent = np.abs(np.diff(q_values, axis=0))

        arm_reports: dict[str, Any] = {}
        for side_index, side in enumerate(("left", "right")):
            translation = np.asarray(errors[side]["translation_m"])
            rotation = np.asarray(errors[side]["rotation_rad"])
            joint_slice = slice(side_index * 7, side_index * 7 + 7)
            arm_reports[side] = {
                "eef_within_thresholds": bool(
                    np.max(translation) <= args.max_translation_error
                    and np.max(rotation) <= args.max_rotation_error
                ),
                "max_translation_error_m": float(np.max(translation)),
                "p95_translation_error_m": float(np.percentile(translation, 95)),
                "max_rotation_error_rad": float(np.max(rotation)),
                "p95_rotation_error_rad": float(np.percentile(rotation, 95)),
                "max_adjacent_joint_change_rad": float(np.max(adjacent[:, joint_slice])),
                "max_velocity_rad_s": float(np.max(np.abs(velocity[:, joint_slice]))),
                "max_acceleration_rad_s2": float(np.max(np.abs(acceleration[:, joint_slice]))),
            }

        max_adjacent_flat = int(np.argmax(adjacent))
        adjacent_interval, adjacent_joint = np.unravel_index(max_adjacent_flat, adjacent.shape)
        max_velocity_flat = int(np.argmax(np.abs(velocity)))
        velocity_interval, velocity_joint = np.unravel_index(max_velocity_flat, velocity.shape)
        max_acceleration_flat = int(np.argmax(np.abs(acceleration)))
        acceleration_interval, acceleration_joint = np.unravel_index(max_acceleration_flat, acceleration.shape)
        return {
            "label": label,
            "frame_count": int(len(times)),
            "timestamps_preserved_exactly": True,
            "duration_s": float(times[-1] - times[0]),
            "arms": arm_reports,
            "joint_position_limits": {
                "passed": not bool(np.any(below | above)),
                "violation_frame_count": int(np.count_nonzero(np.any(below | above, axis=1))),
                "violation_element_count": int(np.count_nonzero(below | above)),
                "maximum_violation_rad": float(np.max(position_excess)),
                "minimum_margin_to_limit_rad": float(np.min(np.minimum(q_values - lower[None, :], upper[None, :] - q_values))),
            },
            "joint_velocity_limits": {
                "source": "URDF velocity attributes",
                "passed": not bool(np.any(velocity_violations)),
                "violation_interval_count": int(np.count_nonzero(np.any(velocity_violations, axis=1))),
                "violation_element_count": int(np.count_nonzero(velocity_violations)),
                "maximum_excess_rad_s": float(np.max(np.maximum(velocity_excess, 0.0))),
                "maximum_abs_velocity_rad_s": float(np.max(np.abs(velocity))),
                "maximum_joint": expected_names[velocity_joint],
                "maximum_ending_frame_index": int(velocity_interval + 1),
            },
            "joint_acceleration": {
                "limit_available": False,
                "limit_check_passed": None,
                "reason": "the supplied kinematics URDF contains no acceleration limits",
                "maximum_abs_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
                "maximum_joint": expected_names[acceleration_joint],
                "maximum_center_frame_index": int(acceleration_interval + 1),
            },
            "ik_branch_continuity": {
                "threshold_rad_per_frame": args.branch_jump_threshold,
                "passed": not bool(np.any(adjacent > args.branch_jump_threshold)),
                "intervals_over_threshold": int(np.count_nonzero(np.any(adjacent > args.branch_jump_threshold, axis=1))),
                "maximum_adjacent_change_rad": float(np.max(adjacent)),
                "maximum_joint": expected_names[adjacent_joint],
                "maximum_ending_frame_index": int(adjacent_interval + 1),
            },
        }

    evaluations: dict[str, Any] = {"pyroki_smoothed": evaluate("pyroki_smoothed", smoothed)}
    if args.ik is not None:
        ik = json.loads(args.ik.read_text(encoding="utf-8"))
        if ik.get("execution_enabled") is not False:
            raise ValueError("IK input must be an offline artifact")
        evaluations["pinocchio_ik"] = evaluate("pinocchio_ik", ik)

    smoothed_report = evaluations["pyroki_smoothed"]
    eef_passed = all(report["eef_within_thresholds"] for report in smoothed_report["arms"].values())
    kinematics_passed = bool(
        eef_passed
        and smoothed_report["joint_position_limits"]["passed"]
        and smoothed_report["joint_velocity_limits"]["passed"]
        and smoothed_report["ik_branch_continuity"]["passed"]
    )
    collision_elements = args.urdf.read_text(encoding="utf-8").count("<collision")
    collision_check = {
        "performed": False,
        "passed": None,
        "urdf_collision_element_count": collision_elements,
        "environment_collision_geometry_available": False,
        "reason": (
            "the supplied dual-FR3 kinematics URDF has no collision geometry and the offline solver does not run collision checking"
            if collision_elements == 0
            else "collision geometry exists but no collision checker is implemented in this offline pipeline"
        ),
    }
    start_plan = {
        "performed": False,
        "passed": None,
        "reason": "no current robot state was read and no collision-capable plan-only service was contacted",
    }
    hardware_gate_passed = bool(kinematics_passed and collision_check["performed"] and start_plan["performed"])
    report: dict[str, Any] = {
        "artifact_type": "offline_smoothed_replay_validation",
        "execution_enabled": False,
        "thresholds": {
            "max_translation_error_m": args.max_translation_error,
            "max_rotation_error_rad": args.max_rotation_error,
        },
        # execute_real_replay.py requires this exact gate.  Keep it false until
        # collision and start-to-frame0 checks are actually completed.
        "passed": hardware_gate_passed,
        "offline_kinematics_passed": kinematics_passed,
        "ready_for_manual_hardware_check": hardware_gate_passed,
        "arms": smoothed_report["arms"],
        "trajectories": evaluations,
        "collision_check": collision_check,
        "current_state_to_frame0_plan": start_plan,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not hardware_gate_passed:
        raise RuntimeError("offline artifact remains hardware-gated; see collision/start-plan fields")


if __name__ == "__main__":
    main()
