"""Smooth an offline Pinocchio replay result with PyRoki and preserve hands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyroki_smoothing import smooth_joint_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--fidelity-weight", type=float, default=10.0)
    parser.add_argument("--smoothness-weight", type=float, default=1.0)
    parser.add_argument("--acceleration-weight", type=float, default=0.1)
    parser.add_argument(
        "--max-timestamp-jitter-s",
        type=float,
        default=0.002,
        help="maximum deviation from the median sample period accepted by the nominal-dt smoother",
    )
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("execution_enabled") is not False:
        raise ValueError("input must be an offline IK artifact")
    samples = source.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("input must contain at least two samples")
    times = np.asarray([sample["t_s"] for sample in samples], dtype=float)
    dt = float(np.median(np.diff(times)))
    timestamp_jitter = float(np.max(np.abs(np.diff(times) - dt)))
    if timestamp_jitter > args.max_timestamp_jitter_s:
        raise ValueError(
            "PyRoki smoother requires a near-uniform trajectory; "
            f"max timestamp jitter {timestamp_jitter:.9f}s exceeds {args.max_timestamp_jitter_s:.9f}s"
        )
    reference = np.asarray(
        [
            sample["left_arm_joint_positions_rad"]
            + sample["right_arm_joint_positions_rad"]
            for sample in samples
        ],
        dtype=float,
    )
    smoothed = smooth_joint_trajectory(
        args.urdf,
        reference,
        dt_s=dt,
        fidelity_weight=args.fidelity_weight,
        smoothness_weight=args.smoothness_weight,
        acceleration_weight=args.acceleration_weight,
    )
    # The optimizer enforces limits through an augmented-Lagrangian
    # constraint, whose numerical tolerance can leave microradian residuals.
    # Project those residuals to the exact URDF bounds before an artifact is
    # written; this never relaxes a limit.
    from yourdfpy import URDF

    urdf = URDF.load(str(args.urdf))
    if list(urdf.actuated_joint_names) != list(source["joint_names"]):
        raise ValueError("URDF actuated-joint order differs from the IK artifact")
    lower = np.asarray([joint.limit.lower for joint in urdf.actuated_joints], dtype=float)
    upper = np.asarray([joint.limit.upper for joint in urdf.actuated_joints], dtype=float)
    before_projection = smoothed.copy()
    smoothed = np.clip(smoothed, lower[None, :], upper[None, :])
    limit_projection = smoothed - before_projection
    output_samples = []
    for sample, q in zip(samples, smoothed, strict=True):
        output_samples.append(
            {
                "t_s": float(sample["t_s"]),
                "left_arm_joint_positions_rad": q[:7].tolist(),
                "left_hand_joint_positions_rad": sample[
                    "left_hand_joint_positions_rad"
                ],
                "right_arm_joint_positions_rad": q[7:].tolist(),
                "right_hand_joint_positions_rad": sample[
                    "right_hand_joint_positions_rad"
                ],
            }
        )
    delta = smoothed - reference
    payload = {
        "artifact_type": "offline_bimanual_pyroki_smoothed_replay",
        "execution_enabled": False,
        "warning": "OFFLINE RESULT ONLY; no controller command is implemented",
        "source_ik": str(args.input),
        "joint_names": source["joint_names"],
        "dt_s": dt,
        "timestamps_preserved": True,
        "maximum_timestamp_jitter_from_nominal_s": timestamp_jitter,
        "smoothing": {
            "fidelity_weight": args.fidelity_weight,
            "smoothness_weight": args.smoothness_weight,
            "acceleration_weight": args.acceleration_weight,
            "maximum_change_from_ik_rad": float(np.max(np.abs(delta))),
            "rms_change_from_ik_rad": float(np.sqrt(np.mean(delta**2))),
            "endpoints_preserved": True,
            "exact_urdf_limit_projection_applied": True,
            "maximum_limit_projection_rad": float(np.max(np.abs(limit_projection))),
        },
        "samples": output_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(output_samples)} PyRoki-smoothed OFFLINE samples to {args.output}")


if __name__ == "__main__":
    main()
