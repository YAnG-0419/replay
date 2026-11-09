"""Replay only the two Sharpa hands; this module has no FR3/MoveIt clients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any


SIDES = ("left", "right")
HAND_NAMES = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(
    trajectory_path: Path,
    trajectory: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[int, float]:
    if trajectory.get("execution_enabled") is not False:
        raise ValueError("trajectory must be an offline artifact")
    samples = trajectory.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("trajectory must contain at least two samples")
    times = [float(sample["t_s"]) for sample in samples]
    if any(right <= left for left, right in zip(times[:-1], times[1:], strict=True)):
        raise ValueError("trajectory timestamps must be strictly increasing")
    for index, sample in enumerate(samples):
        for side in SIDES:
            values = sample.get(f"{side}_hand_joint_positions_rad")
            if not isinstance(values, list) or len(values) != len(HAND_NAMES):
                raise ValueError(f"sample {index} {side} hand must contain 22 joints")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"sample {index} {side} hand contains non-finite values")

    expected = validation.get("trajectory", {})
    actual_sha = sha256_file(trajectory_path)
    if actual_sha != expected.get("sha256"):
        raise ValueError("trajectory SHA256 differs from its packaged validation artifact")
    if len(samples) != int(expected.get("frame_count", -1)):
        raise ValueError("trajectory frame count differs from validation artifact")
    duration = times[-1] - times[0]
    if not math.isclose(duration, float(expected.get("duration_s", -1.0)), abs_tol=1e-9):
        raise ValueError("trajectory duration differs from validation artifact")
    return len(samples), duration


def minimum_jerk(alpha: float) -> float:
    value = min(1.0, max(0.0, alpha))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


class HandsOnlyExecutor:
    def __init__(self, trajectory: dict[str, Any]) -> None:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState

        self.rclpy = rclpy
        self.JointState = JointState
        self.trajectory = trajectory
        self.measured: dict[str, list[float]] = {}
        self._measurement_lock = threading.Lock()

        rclpy.init(args=None)
        self.node = rclpy.create_node("eef_replay_hands_only")
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.publishers = {
            side: self.node.create_publisher(
                JointState, f"/sharpa/{side}/command", 10
            )
            for side in SIDES
        }

        def state_callback(side: str, message: Any) -> None:
            names = [str(name) for name in message.name]
            values = [float(value) for value in message.position]
            expected_prefixed = [f"{side}_{name}" for name in HAND_NAMES]
            if names == expected_prefixed:
                ordered = values
            elif names == list(HAND_NAMES):
                ordered = values
            else:
                mapping = dict(zip(names, values, strict=True))
                try:
                    ordered = [
                        mapping[f"{side}_{name}"]
                        if f"{side}_{name}" in mapping
                        else mapping[name]
                        for name in HAND_NAMES
                    ]
                except KeyError:
                    return
            if len(ordered) == 22 and all(math.isfinite(value) for value in ordered):
                with self._measurement_lock:
                    self.measured[side] = ordered

        self.subscriptions = [
            self.node.create_subscription(
                JointState,
                f"/sharpa/{side}/joint_states",
                lambda message, side=side: state_callback(side, message),
                qos_profile_sensor_data,
            )
            for side in SIDES
        ]

    def close(self) -> None:
        self.executor.shutdown(timeout_sec=2.0)
        self.node.destroy_node()
        self.rclpy.shutdown()
        self.spin_thread.join(timeout=2.0)

    def require_live(self, timeout: float = 10.0) -> dict[str, list[float]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            subscribers_ready = all(
                self.publishers[side].get_subscription_count() == 1
                for side in SIDES
            )
            with self._measurement_lock:
                state_ready = all(side in self.measured for side in SIDES)
                measured = {side: list(self.measured[side]) for side in SIDES if side in self.measured}
            if subscribers_ready and state_ready:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(
                "both Sharpa command subscribers and both measured joint states are required"
            )
        for side in SIDES:
            other_publishers = [
                endpoint
                for endpoint in self.node.get_publishers_info_by_topic(
                    f"/sharpa/{side}/command"
                )
                if endpoint.node_name != self.node.get_name()
            ]
            if other_publishers:
                names = sorted({endpoint.node_name for endpoint in other_publishers})
                raise RuntimeError(
                    f"refusing duplicate {side} hand command publishers: {names}"
                )
        return measured

    def publish(self, values: dict[str, list[float]]) -> None:
        stamp = self.node.get_clock().now().to_msg()
        for side in SIDES:
            message = self.JointState()
            message.header.stamp = stamp
            message.header.frame_id = "eef_replay_hands_only"
            message.name = list(HAND_NAMES)
            message.position = [float(value) for value in values[side]]
            self.publishers[side].publish(message)

    def transition_to_first(
        self,
        current: dict[str, list[float]],
        target: dict[str, list[float]],
        duration_s: float,
        rate_hz: float = 30.0,
    ) -> None:
        frames = max(2, int(math.ceil(duration_s * rate_hz)) + 1)
        start = time.monotonic()
        for index in range(frames):
            elapsed = min(duration_s, index / rate_hz)
            blend = minimum_jerk(elapsed / duration_s)
            values = {
                side: [
                    before + blend * (after - before)
                    for before, after in zip(current[side], target[side], strict=True)
                ]
                for side in SIDES
            }
            deadline = start + elapsed
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            self.publish(values)
        self.publish(target)

    def execute(self, start_transition_s: float) -> None:
        current = self.require_live()
        samples = self.trajectory["samples"]
        first = {
            side: list(samples[0][f"{side}_hand_joint_positions_rad"])
            for side in SIDES
        }
        print(
            f"Executing HANDS-ONLY transition to frame 0 over {start_transition_s:.3f}s...",
            flush=True,
        )
        self.transition_to_first(current, first, start_transition_s)
        print("Executing HANDS-ONLY replay; no FR3 client exists in this process...", flush=True)
        start = time.monotonic() + 0.25
        t0 = float(samples[0]["t_s"])
        for sample in samples:
            deadline = start + float(sample["t_s"]) - t0
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            self.publish(
                {
                    side: list(sample[f"{side}_hand_joint_positions_rad"])
                    for side in SIDES
                }
            )
        print("Hands-only replay complete; both hands hold the final frame.")

    def check_live(self) -> None:
        self.require_live()
        print(
            "HANDS LIVE READY: one driver subscriber and measured feedback for each hand; "
            "no hand command was sent and no FR3 client was created."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--check-live", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--start-transition-s", type=float, default=3.0)
    args = parser.parse_args()
    if args.check_live == args.execute:
        raise ValueError("choose exactly one of --check-live or --execute")
    if not math.isfinite(args.start_transition_s) or args.start_transition_s <= 0.0:
        raise ValueError("--start-transition-s must be finite and positive")

    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    frames, duration = validate_artifacts(
        args.trajectory, trajectory, validation
    )
    print(
        f"HANDS-ONLY artifact accepted: sha256={sha256_file(args.trajectory)}, "
        f"frames={frames}, duration={duration:.9f}s"
    )
    executor = HandsOnlyExecutor(trajectory)
    try:
        if args.check_live:
            executor.check_live()
        else:
            executor.execute(args.start_transition_s)
        return 0
    except KeyboardInterrupt:
        print("Interrupted: hand publishing stopped; no FR3 client was present.")
        return 130
    finally:
        executor.close()


if __name__ == "__main__":
    raise SystemExit(main())
