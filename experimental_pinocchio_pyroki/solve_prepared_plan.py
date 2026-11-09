"""CLI for offline Pinocchio IK over one arm of a prepared replay plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pinocchio_ik import PinocchioIK


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Pinocchio IK; never sends robot commands")
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arm", required=True, choices=("left", "right"))
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--eef-frame", required=True)
    parser.add_argument("--q0", required=True, help="comma-separated model.nq joint values in radians")
    parser.add_argument("--max-joint-step", type=float, default=0.35)
    args = parser.parse_args()

    with args.plan.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("execution_enabled") is not False:
        raise ValueError("refusing input that is not explicitly marked execution_enabled=false")
    targets = [
        np.asarray(sample[args.arm]["base_T_link8_target"]["matrix"], dtype=float)
        for sample in plan["samples"]
        if args.arm in sample
    ]
    q0 = np.asarray([float(value) for value in args.q0.split(",")], dtype=float)
    solver = PinocchioIK(args.urdf, args.eef_frame)
    joints, results = solver.solve_trajectory(targets, q0, max_joint_step_rad=args.max_joint_step)
    payload = {
        "artifact_type": "offline_pinocchio_ik",
        "execution_enabled": False,
        "arm": args.arm,
        "urdf": str(args.urdf),
        "eef_frame": args.eef_frame,
        "joint_positions_rad": joints.tolist(),
        "ik_results": [result.to_dict() for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote OFFLINE IK artifact only: {args.output}")


if __name__ == "__main__":
    main()

