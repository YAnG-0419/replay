"""Offline sequential dual-arm Pinocchio IK for a prepared link8 replay plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pinocchio_ik import PinocchioIK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--left-frame", default="left_fr3v2_link8")
    parser.add_argument("--right-frame", default="right_fr3v2_link8")
    parser.add_argument("--left-base-frame", default="left_fr3v2_link0")
    parser.add_argument("--right-base-frame", default="right_fr3v2_link0")
    parser.add_argument("--max-joint-step", type=float, default=0.35)
    parser.add_argument(
        "--initial-q",
        type=float,
        nargs=14,
        metavar="Q",
        help=(
            "explicit 14-joint numerical seed when the plan has no source arm joints; "
            "this seed is not treated as a commanded/current robot pose"
        ),
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("execution_enabled") is not False or plan.get("output_controlled_frame") != "fr3_link8":
        raise ValueError("plan must be an offline fr3_link8 plan")
    samples = plan["samples"]
    first = samples[0]
    has_source_seed = all(
        "source_arm_joint_positions_rad" in first[side]
        for side in ("left", "right")
    )
    if has_source_seed:
        if args.initial_q is not None:
            raise ValueError("do not combine --initial-q with source arm joints from the plan")
        q = np.asarray(
            first["left"]["source_arm_joint_positions_rad"]
            + first["right"]["source_arm_joint_positions_rad"],
            dtype=float,
        )
        seed_source = "plan_first_sample_source_arm_joint_positions_rad"
    else:
        if args.initial_q is None:
            raise ValueError("plan has no source arm joints; pass an explicit 14-value --initial-q numerical seed")
        q = np.asarray(args.initial_q, dtype=float)
        seed_source = "explicit_numerical_ik_seed_not_robot_state"
    left = PinocchioIK(args.urdf, args.left_frame)
    right = PinocchioIK(args.urdf, args.right_frame)
    expected_names = tuple(
        f"{side}_fr3v2_joint{index}"
        for side in ("left", "right")
        for index in range(1, 8)
    )
    if tuple(str(name) for name in left.model.names[1:]) != expected_names:
        raise ValueError(f"unexpected URDF joint order: {tuple(left.model.names[1:])}")

    # The prepared replay plan stores each EEF target in its own arm's link0
    # frame. Pinocchio's frame placements are expressed in the dual-arm URDF's
    # common root (lychee_root), so prepend the fixed root_T_link0 mount pose.
    def root_T_frame(solver: PinocchioIK, frame_name: str) -> np.ndarray:
        if not solver.model.existFrame(frame_name):
            raise ValueError(f"base frame {frame_name!r} not found in URDF")
        neutral = solver.pin.neutral(solver.model)
        solver.pin.forwardKinematics(solver.model, solver.data, neutral)
        solver.pin.updateFramePlacements(solver.model, solver.data)
        return np.asarray(solver.data.oMf[solver.model.getFrameId(frame_name)].homogeneous)

    root_T_left_base = root_T_frame(left, args.left_base_frame)
    root_T_right_base = root_T_frame(right, args.right_base_frame)

    solved: list[dict[str, object]] = []
    previous = q.copy()
    initial_seed = q.copy()
    max_steps = {"left": (0.0, None), "right": (0.0, None)}
    for index, sample in enumerate(samples):
        left_target = root_T_left_base @ np.asarray(
            sample["left"]["base_T_link8_target"]["matrix"], dtype=float
        )
        right_target = root_T_right_base @ np.asarray(
            sample["right"]["base_T_link8_target"]["matrix"], dtype=float
        )
        q_left, left_result = left.solve(left_target, previous)
        q_both, right_result = right.solve(right_target, q_left)
        if not left_result.converged or not right_result.converged:
            raise RuntimeError(f"IK did not converge at frame {index}: left={left_result}, right={right_result}")
        left_step = float(np.max(np.abs(q_both[:7] - previous[:7])))
        right_step = float(np.max(np.abs(q_both[7:] - previous[7:])))
        # With an explicit numerical seed, frame zero is the solution of a
        # static IK problem; its distance from that seed is not a trajectory
        # discontinuity.  Every actual frame-to-frame transition remains
        # strictly checked.
        if index > 0 and max(left_step, right_step) > args.max_joint_step:
            raise RuntimeError(f"joint jump at frame {index}: left={left_step:.4f}, right={right_step:.4f} rad")
        if index > 0:
            if left_step > max_steps["left"][0]:
                max_steps["left"] = (left_step, index)
            if right_step > max_steps["right"][0]:
                max_steps["right"] = (right_step, index)
        solved.append(
            {
                "t_s": float(sample["t_s"]),
                "left_arm_joint_positions_rad": q_both[:7].tolist(),
                "left_hand_joint_positions_rad": sample["left"]["hand_joint_positions_rad"],
                "right_arm_joint_positions_rad": q_both[7:].tolist(),
                "right_hand_joint_positions_rad": sample["right"]["hand_joint_positions_rad"],
                "left_ik": left_result.to_dict(),
                "right_ik": right_result.to_dict(),
            }
        )
        previous = q_both

    payload = {
        "artifact_type": "offline_bimanual_pinocchio_ik",
        "execution_enabled": False,
        "joint_names": list(expected_names),
        "source_plan": str(args.plan),
        "ik_summary": {
            "frame_count": len(solved),
            "left_converged_frames": len(solved),
            "right_converged_frames": len(solved),
            "unreachable_frames": 0,
            "initial_seed_source": seed_source,
            "initial_seed_joint_positions_rad": initial_seed.tolist(),
            "frame0_distance_from_seed_rad": {
                "left_max_abs": float(np.max(np.abs(np.asarray(solved[0]["left_arm_joint_positions_rad"]) - initial_seed[:7]))),
                "right_max_abs": float(np.max(np.abs(np.asarray(solved[0]["right_arm_joint_positions_rad"]) - initial_seed[7:]))),
            },
            "max_adjacent_joint_change_rad": {
                side: {"value": value, "frame_index": frame_index}
                for side, (value, frame_index) in max_steps.items()
            },
        },
        "samples": solved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(solved)} OFFLINE bimanual IK samples to {args.output}")


if __name__ == "__main__":
    main()

