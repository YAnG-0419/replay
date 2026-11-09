"""Read the current dual-FR3 joint state and save it; never publishes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time


NAMES = tuple(
    f"{side}_fr3v2_joint{index}"
    for side in ("left", "right")
    for index in range(1, 8)
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    import rclpy
    from sensor_msgs.msg import JointState

    rclpy.init(args=None)
    node = rclpy.create_node("eef_replay_read_current_joint_state")
    positions: dict[str, float] = {}

    def callback(message: JointState) -> None:
        for name, value in zip(message.name, message.position, strict=True):
            if name in NAMES and math.isfinite(float(value)):
                positions[str(name)] = float(value)

    subscriptions = [
        node.create_subscription(JointState, "/left/franka/joint_states", callback, 10),
        node.create_subscription(JointState, "/right/franka/joint_states", callback, 10),
        node.create_subscription(JointState, "/joint_states", callback, 10),
    ]
    del subscriptions
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline and any(name not in positions for name in NAMES):
            rclpy.spin_once(node, timeout_sec=0.05)
        missing = [name for name in NAMES if name not in positions]
        if missing:
            raise RuntimeError(f"timed out waiting for joints: {missing}")
        payload = {
            "artifact_type": "read_only_dual_arm_joint_snapshot",
            "execution_enabled": False,
            "joint_names": list(NAMES),
            "joint_positions_rad": [positions[name] for name in NAMES],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved read-only joint snapshot to {args.output}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
