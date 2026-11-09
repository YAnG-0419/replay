"""Convert one completed rosbag2 episode into a 30 Hz camera-frame link8 plan.

This is an offline reader. It filters the bag to measured arm/hand JointState
topics and has no publishers, controller clients, or robot network access.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import json
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin

from eef_replay.calibration import matrix_to_pose
from eef_replay.hand_mapping import SHARPA_HAND_JOINT_NAMES


ARM_TOPICS = {
    "left": "/left/franka/joint_states",
    "right": "/right/franka/joint_states",
}
HAND_TOPICS = {
    "left": "/sharpa/left/joint_states",
    "right": "/sharpa/right/joint_states",
}


def _nearest(samples: list[tuple[int, dict[str, float]]], timestamp: int, topic: str) -> dict[str, float]:
    times = [item[0] for item in samples]
    index = bisect_left(times, timestamp)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(samples)]
    selected = min(candidates, key=lambda i: abs(times[i] - timestamp))
    age_ns = abs(times[selected] - timestamp)
    if age_ns > 50_000_000:
        raise RuntimeError(f"{topic} nearest sample is {age_ns / 1e6:.1f} ms away")
    return samples[selected][1]


def _ordered_arm(mapping: dict[str, float], side: str) -> list[float]:
    names = [f"{side}_fr3_joint{index}" for index in range(1, 8)]
    try:
        return [mapping[name] for name in names]
    except KeyError as exc:
        raise ValueError(f"{side} arm message lacks {exc.args[0]}") from exc


def _ordered_hand(mapping: dict[str, float], side: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in SHARPA_HAND_JOINT_NAMES:
        prefixed = f"{side}_{name}"
        if prefixed not in mapping:
            raise ValueError(f"{side} hand message lacks {prefixed}")
        result[name] = mapping[prefixed]
    return result


def _read_joint_topics(episode: Path) -> dict[str, list[tuple[int, dict[str, float]]]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise RuntimeError("Source /opt/ros/jazzy/setup.zsh and run with /usr/bin/python3") from exc

    topics = set(ARM_TOPICS.values()) | set(HAND_TOPICS.values())
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(episode), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(topics)))
    samples: dict[str, list[tuple[int, dict[str, float]]]] = {topic: [] for topic in topics}
    while reader.has_next():
        topic, payload, timestamp = reader.read_next()
        message = deserialize_message(payload, JointState)
        if len(message.name) != len(message.position):
            raise ValueError(f"{topic} name/position length mismatch")
        mapping = {str(name): float(value) for name, value in zip(message.name, message.position, strict=True)}
        if not all(np.isfinite(value) for value in mapping.values()):
            raise ValueError(f"{topic} contains non-finite joint values")
        samples[topic].append((int(timestamp), mapping))
    missing = [topic for topic, values in samples.items() if not values]
    if missing:
        raise RuntimeError(f"episode lacks required samples: {missing}")
    return samples


def _effective_window(episode: Path, samples: dict[str, list[tuple[int, dict[str, float]]]]) -> tuple[int, int]:
    state_path = episode / "collection_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        report = state.get("validation_report", {})
        start = report.get("effective_bag_start_time_ns")
        end = report.get("effective_bag_end_time_ns")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            return start, end
    start = max(values[0][0] for values in samples.values())
    end = min(values[-1][0] for values in samples.values())
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    camera_T_base = {
        side: np.asarray(calibration["transforms"][f"camera_T_{side}_base"], dtype=float)
        for side in ("left", "right")
    }
    streams = _read_joint_topics(args.episode)
    start_ns, end_ns = _effective_window(args.episode, streams)
    period_ns = round(1_000_000_000 / args.fps)

    model = pin.buildModelFromUrdf(str(args.urdf))
    data = model.createData()
    expected_model_names = tuple(
        f"{side}_fr3v2_joint{index}"
        for side in ("left", "right")
        for index in range(1, 8)
    )
    if tuple(str(name) for name in model.names[1:]) != expected_model_names:
        raise ValueError(f"unexpected URDF joint order: {tuple(model.names[1:])}")
    frame_ids = {
        side: (
            model.getFrameId(f"{side}_fr3v2_link0"),
            model.getFrameId(f"{side}_fr3v2_link8"),
        )
        for side in ("left", "right")
    }

    output_samples: list[dict[str, Any]] = []
    timestamp = start_ns
    while timestamp <= end_ns:
        arm = {
            side: _ordered_arm(_nearest(streams[ARM_TOPICS[side]], timestamp, ARM_TOPICS[side]), side)
            for side in ("left", "right")
        }
        hand = {
            side: _ordered_hand(_nearest(streams[HAND_TOPICS[side]], timestamp, HAND_TOPICS[side]), side)
            for side in ("left", "right")
        }
        q = np.asarray(arm["left"] + arm["right"], dtype=float)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        sample: dict[str, Any] = {"t_s": (timestamp - start_ns) / 1e9}
        for side in ("left", "right"):
            base_id, link8_id = frame_ids[side]
            base_T_link8 = data.oMf[base_id].actInv(data.oMf[link8_id]).homogeneous
            camera_T_link8 = camera_T_base[side] @ base_T_link8
            sample[side] = {
                "eef_pose": matrix_to_pose(camera_T_link8),
                "hand_joints_rad": hand[side],
                "source_arm_joint_positions_rad": arm[side],
            }
        output_samples.append(sample)
        timestamp += period_ns

    document = {
        "schema_version": 1,
        "reference_frame": "camera_color_optical_frame",
        "translation_unit": "metre",
        "pose_mode": "absolute",
        "controlled_frame": "fr3_link8",
        "source": {
            "episode": str(args.episode),
            "arm_topics": ARM_TOPICS,
            "hand_topics": HAND_TOPICS,
            "fps": args.fps,
            "effective_start_ns": start_ns,
            "effective_end_ns": end_ns,
        },
        "samples": output_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(output_samples)} OFFLINE camera-frame link8 samples to {args.output}")


if __name__ == "__main__":
    main()

