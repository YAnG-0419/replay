"""Build a minimum-jerk, offline joint transition to replay frame zero."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-velocity", type=float, default=0.25)
    parser.add_argument("--max-acceleration", type=float, default=0.5)
    parser.add_argument("--minimum-duration", type=float, default=2.0)
    args = parser.parse_args()
    if args.fps <= 0 or args.max_velocity <= 0 or args.max_acceleration <= 0:
        raise ValueError("fps and limits must be positive")

    current = json.loads(args.current.read_text(encoding="utf-8"))
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    if current.get("execution_enabled") is not False or replay.get("execution_enabled") is not False:
        raise ValueError("both inputs must be offline artifacts")
    names = replay["joint_names"]
    if current["joint_names"] != names:
        raise ValueError("current-state and replay joint orders differ")
    q0 = np.asarray(current["joint_positions_rad"], dtype=float)
    first = replay["samples"][0]
    q1 = np.asarray(
        first["left_arm_joint_positions_rad"]
        + first["right_arm_joint_positions_rad"],
        dtype=float,
    )
    delta = q1 - q0
    # For s(u)=10u^3-15u^4+6u^5, max s'=1.875 and max |s''|≈5.7735.
    duration = max(
        args.minimum_duration,
        1.875 * float(np.max(np.abs(delta))) / args.max_velocity,
        math.sqrt(5.7735 * float(np.max(np.abs(delta))) / args.max_acceleration),
    )
    frames = max(2, int(math.ceil(duration * args.fps)) + 1)
    times = np.linspace(0.0, duration, frames)
    u = times / duration
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    ds = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    dds = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / duration**2
    q = q0[None, :] + s[:, None] * delta[None, :]
    dq = ds[:, None] * delta[None, :]
    ddq = dds[:, None] * delta[None, :]
    payload = {
        "artifact_type": "offline_minimum_jerk_transition_to_replay_frame0",
        "execution_enabled": False,
        "warning": "JOINT-LIMITED OFFLINE TRANSITION; collision checking is not included",
        "joint_names": names,
        "duration_s": duration,
        "fps": args.fps,
        "limits": {
            "requested_max_velocity_rad_s": args.max_velocity,
            "requested_max_acceleration_rad_s2": args.max_acceleration,
            "actual_max_velocity_rad_s": float(np.max(np.abs(dq))),
            "actual_max_acceleration_rad_s2": float(np.max(np.abs(ddq))),
        },
        "samples": [
            {
                "t_s": float(t),
                "joint_positions_rad": position.tolist(),
                "joint_velocities_rad_s": velocity.tolist(),
                "joint_accelerations_rad_s2": acceleration.tolist(),
            }
            for t, position, velocity, acceleration in zip(times, q, dq, ddq, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {frames}-frame OFFLINE transition ({duration:.2f} s) to {args.output}")


if __name__ == "__main__":
    main()
