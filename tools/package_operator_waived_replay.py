#!/usr/bin/env python3
"""Package a dynamics-checked replay that requires explicit runtime waivers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("dynamics_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    dynamics = json.loads(args.dynamics_report.read_text(encoding="utf-8"))
    if trajectory.get("execution_enabled") is not False:
        raise ValueError("trajectory must be an offline artifact")
    samples = trajectory.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("trajectory must contain at least two samples")
    times = [float(sample["t_s"]) for sample in samples]
    duration = times[-1] - times[0]
    trajectory_sha = sha256(args.trajectory)
    if dynamics.get("sha256") != trajectory_sha:
        raise ValueError("dynamics report SHA256 does not match trajectory")
    if int(dynamics.get("frames", -1)) != len(samples):
        raise ValueError("dynamics report frame count differs")
    if abs(float(dynamics.get("duration_s", -1.0)) - duration) > 1e-9:
        raise ValueError("dynamics report duration differs")
    if dynamics.get("position_limits_passed") is not True:
        raise ValueError("position limits did not pass")
    if int(dynamics.get("velocity_violation_intervals", -1)) != 0:
        raise ValueError("velocity limits did not pass")
    if int(dynamics.get("acceleration_violation_centers", -1)) != 0:
        raise ValueError("acceleration limits did not pass")
    if dynamics.get("eef_error_validation_performed") is not False:
        raise ValueError("this packager expects an explicit unvalidated-EEF artifact")
    if float(trajectory.get("retiming", {}).get("speed_scale", -1.0)) != 0.5:
        raise ValueError("trajectory is not the expected 0.5x retimed artifact")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    packaged_trajectory = args.output_dir / "pyroki_smoothed.json"
    shutil.copyfile(args.trajectory, packaged_trajectory)
    packaged_sha = sha256(packaged_trajectory)
    if packaged_sha != trajectory_sha:
        raise RuntimeError("packaged trajectory hash changed during copy")

    validation = {
        "artifact_type": "offline_operator_waived_replay_gate",
        "execution_enabled": False,
        "passed": False,
        "execution_waiver_eligible": True,
        "dynamics_passed": True,
        "warning": (
            "NOT A FULL VALIDATION: execution requires explicit command-line acknowledgement "
            "of unvalidated EEF error and skipped replay collision checking"
        ),
        "trajectory": {
            "sha256": packaged_sha,
            "frame_count": len(samples),
            "duration_s": duration,
            "speed_scale": 0.5,
            "source": str(args.trajectory),
        },
        "joint_dynamics": dynamics,
        "eef_validation": {
            "performed": False,
            "passed": None,
            "waived_by_operator": True,
            "reason": "operator explicitly requested no EEF-error validation for this PyRoki comparison trajectory",
        },
        "collision_check": {
            "performed": False,
            "passed": None,
            "waived_by_operator": True,
            "reason": "operator explicitly requested replay collision checking be skipped",
        },
        "current_state_to_frame0_plan": {
            "performed": False,
            "deferred_to_execute_time_moveit_plan": True,
            "planner": "MoveIt both_arms RRTConnect",
            "note": "execute_real_replay.py must obtain a successful plan before sending any replay goal",
        },
        "required_runtime_acknowledgements": [
            "--acknowledge-unvalidated-eef",
            "--acknowledge-skipped-collision",
        ],
    }
    validation_path = args.output_dir / "pyroki_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(
        f"Packaged explicit-waiver replay: {args.output_dir} "
        f"(sha256={packaged_sha}, frames={len(samples)}, duration={duration:.9f}s)"
    )


if __name__ == "__main__":
    main()
