#!/usr/bin/env python3
"""Prepare episode553_human from verified Sharpa root poses (offline only).

This script never imports ROS and never connects to robot controllers.  It
validates the paired NPZ/NPY inputs, converts each absolute Sharpa-root pose to
an absolute FR3 link8 pose, applies the per-arm hand-eye calibration, and writes
JSON artifacts for the existing offline replay pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from eef_replay.calibration import HandEyeCalibration, matrix_to_pose
from eef_replay.hand_mapping import SHARPA_HAND_JOINT_NAMES
from eef_replay.trajectory import prepare_document


REQUIRED_NPZ_KEYS = (
    "frame_indices",
    "timestamps_s",
    "right_action_camera",
    "left_action_camera",
    "right_root_position_camera_m",
    "left_root_position_camera_m",
    "right_root_rotvec_camera_rad",
    "left_root_rotvec_camera_rad",
    "right_hand_joints_rad",
    "left_hand_joints_rad",
    "right_joint_order",
    "left_joint_order",
)

LINK8_T_SHARPA_ROOT = {
    "left": np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0295],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "right": np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0295],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotvec_to_rotation(rotvec: np.ndarray) -> np.ndarray:
    value = np.asarray(rotvec, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("rotation vector must contain three finite values")
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        # First-order form avoids dividing a tiny vector by its norm.
        x, y, z = value
        return np.array([[1.0, -z, y], [z, 1.0, -x], [-y, x, 1.0]])
    x, y, z = value / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def transform_series(position: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    if position.ndim != 2 or position.shape[1] != 3 or rotvec.shape != position.shape:
        raise ValueError("position and rotation vectors must both have shape [frames, 3]")
    result = np.repeat(np.eye(4)[None, :, :], position.shape[0], axis=0)
    result[:, :3, 3] = position
    for index, value in enumerate(rotvec):
        result[index, :3, :3] = rotvec_to_rotation(value)
    return result


def rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    cosine = (float(np.trace(left.T @ right)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def continuous_poses(matrices: np.ndarray) -> list[dict[str, list[float]]]:
    poses: list[dict[str, list[float]]] = []
    previous: np.ndarray | None = None
    for matrix in matrices:
        pose = matrix_to_pose(matrix)
        quaternion = np.asarray(pose["quaternion_wxyz"], dtype=float)
        if previous is not None and float(np.dot(quaternion, previous)) < 0.0:
            quaternion = -quaternion
        pose["quaternion_wxyz"] = quaternion.tolist()
        poses.append(pose)
        previous = quaternion
    return poses


def matrix_quality(series: np.ndarray) -> dict[str, float]:
    rotations = series[:, :3, :3]
    orthogonal = [np.linalg.norm(r.T @ r - np.eye(3), ord="fro") for r in rotations]
    determinants = np.linalg.det(rotations)
    return {
        "max_orthogonality_error_fro": float(np.max(orthogonal)),
        "min_determinant": float(np.min(determinants)),
        "max_determinant": float(np.max(determinants)),
    }


def position_range(series: np.ndarray) -> dict[str, list[float]]:
    position = series[:, :3, 3]
    return {"min_m": np.min(position, axis=0).tolist(), "max_m": np.max(position, axis=0).tolist()}


def motion_metrics(series: np.ndarray) -> dict[str, float]:
    translation = np.linalg.norm(np.diff(series[:, :3, 3], axis=0), axis=1)
    rotation = np.asarray(
        [rotation_distance(a, b) for a, b in zip(series[:-1, :3, :3], series[1:, :3, :3], strict=True)]
    )
    return {
        "max_adjacent_translation_m": float(np.max(translation)),
        "max_adjacent_rotation_rad": float(np.max(rotation)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "input" / "episode553_human")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "episode553_human")
    parser.add_argument("--calibration", type=Path, default=root / "config" / "handeye_calibration.json")
    parser.add_argument(
        "--left-hand-urdf",
        type=Path,
        default=Path("/home/user/tool_grasping/tool_grasping_avatar/config/avatar/hand_fk/urdf/left_sharpa_ha4/left_sharpa_ha4.urdf"),
    )
    parser.add_argument(
        "--right-hand-urdf",
        type=Path,
        default=Path("/home/user/tool_grasping/tool_grasping_avatar/config/avatar/hand_fk/urdf/right_sharpa_ha4/right_sharpa_ha4.urdf"),
    )
    args = parser.parse_args()

    npz_path = args.input_dir / "sharpa_hand_shape.npz"
    npy_path = args.input_dir / "sharpa_actions_camera.npy"
    actions = np.load(npy_path, allow_pickle=False)
    archive = np.load(npz_path, allow_pickle=False)
    missing = sorted(set(REQUIRED_NPZ_KEYS) - set(archive.files))
    if missing:
        raise ValueError(f"NPZ is missing required keys: {missing}")
    if actions.shape != (664, 56) or not np.all(np.isfinite(actions)):
        raise ValueError(f"sharpa_actions_camera.npy must be finite [664, 56], got {actions.shape}")
    for side, action_slice in (("right", actions[:, :28]), ("left", actions[:, 28:])):
        archived = np.asarray(archive[f"{side}_action_camera"], dtype=float)
        if archived.shape != (664, 28) or not np.allclose(action_slice, archived, rtol=1e-12, atol=1e-12):
            difference = float(np.max(np.abs(action_slice - archived))) if archived.shape == action_slice.shape else None
            raise ValueError(f"NPY and NPZ {side} actions disagree; max_abs_difference={difference}")

    frame_indices = np.asarray(archive["frame_indices"])
    timestamps = np.asarray(archive["timestamps_s"], dtype=float)
    if frame_indices.shape != (664,) or timestamps.shape != (664,):
        raise ValueError("frame_indices and timestamps_s must each contain 664 values")
    if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("timestamps_s must be finite and strictly increasing")

    camera_root: dict[str, np.ndarray] = {}
    camera_link8: dict[str, np.ndarray] = {}
    base_link8: dict[str, np.ndarray] = {}
    hand_joints: dict[str, np.ndarray] = {}
    hand_maps: dict[str, list[dict[str, float]]] = {}
    poses: dict[str, list[dict[str, list[float]]]] = {}
    calibration = HandEyeCalibration.load(args.calibration)
    hand_urdfs = {"left": args.left_hand_urdf, "right": args.right_hand_urdf}

    for side in ("left", "right"):
        positions = np.asarray(archive[f"{side}_root_position_camera_m"], dtype=float)
        rotvecs = np.asarray(archive[f"{side}_root_rotvec_camera_rad"], dtype=float)
        joints = np.asarray(archive[f"{side}_hand_joints_rad"], dtype=float)
        if positions.shape != (664, 3) or rotvecs.shape != (664, 3) or joints.shape != (664, 22):
            raise ValueError(f"unexpected {side} root/joint array shape")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotvecs)) or not np.all(np.isfinite(joints)):
            raise ValueError(f"{side} root/joint arrays contain non-finite values")
        expected_names = [f"{side}_{name}" for name in SHARPA_HAND_JOINT_NAMES]
        actual_names = [str(name) for name in archive[f"{side}_joint_order"].tolist()]
        if actual_names != expected_names:
            raise ValueError(f"{side} joint order disagrees with the replay Sharpa URDF order")
        urdf_joint_names = [
            joint.attrib["name"]
            for joint in ET.parse(hand_urdfs[side]).getroot().findall("joint")
            if joint.attrib.get("type") != "fixed"
        ]
        if actual_names != urdf_joint_names:
            raise ValueError(f"{side} NPZ joint order disagrees with {hand_urdfs[side]}")
        action = np.asarray(archive[f"{side}_action_camera"], dtype=float)
        if not (
            np.array_equal(action[:, :3], positions)
            and np.array_equal(action[:, 3:6], rotvecs)
            and np.array_equal(action[:, 6:], joints)
        ):
            raise ValueError(f"{side} decomposed NPZ arrays disagree with {side}_action_camera")

        camera_root[side] = transform_series(positions, rotvecs)
        inverse_mount = np.linalg.inv(LINK8_T_SHARPA_ROOT[side])
        camera_link8[side] = camera_root[side] @ inverse_mount
        base_link8[side] = np.asarray(
            [calibration.camera_pose_to_base(side, transform) for transform in camera_link8[side]]
        )
        hand_joints[side] = joints
        hand_maps[side] = [
            {name: float(value) for name, value in zip(SHARPA_HAND_JOINT_NAMES, row, strict=True)}
            for row in joints
        ]
        poses[side] = continuous_poses(camera_link8[side])

    samples = []
    for index, timestamp in enumerate(timestamps):
        samples.append(
            {
                "t_s": float(timestamp),
                "frame_index": int(frame_indices[index]),
                "left": {"eef_pose": poses["left"][index], "hand_joints_rad": hand_maps["left"][index]},
                "right": {"eef_pose": poses["right"][index], "hand_joints_rad": hand_maps["right"][index]},
            }
        )
    camera_document = {
        "schema_version": 1,
        "reference_frame": "camera_color_optical_frame",
        "translation_unit": "metre",
        "pose_mode": "absolute",
        "controlled_frame": "fr3_link8",
        "execution_enabled": False,
        "source": {
            "npz": str(npz_path),
            "npy": str(npy_path),
            "npz_sha256": sha256(npz_path),
            "npy_sha256": sha256(npy_path),
            "action_layout": "right[xyz,rotvec,22 joints], left[xyz,rotvec,22 joints]",
        },
        "samples": samples,
    }
    base_document = prepare_document(camera_document, calibration)

    recomposition: dict[str, dict[str, float]] = {}
    arm_metrics: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        recovered = camera_link8[side] @ LINK8_T_SHARPA_ROOT[side]
        translation_errors = np.linalg.norm(recovered[:, :3, 3] - camera_root[side][:, :3, 3], axis=1)
        rotation_errors = np.asarray(
            [rotation_distance(a, b) for a, b in zip(recovered[:, :3, :3], camera_root[side][:, :3, :3], strict=True)]
        )
        quaternions = np.asarray([pose["quaternion_wxyz"] for pose in poses[side]])
        recomposition[side] = {
            "max_translation_error_m": float(np.max(translation_errors)),
            "max_rotation_error_rad": float(np.max(rotation_errors)),
        }
        arm_metrics[side] = {
            "camera_root_rotation_quality": matrix_quality(camera_root[side]),
            "camera_link8_rotation_quality": matrix_quality(camera_link8[side]),
            "base_link8_rotation_quality": matrix_quality(base_link8[side]),
            "camera_link8_position_range": position_range(camera_link8[side]),
            "base_link8_position_range": position_range(base_link8[side]),
            **motion_metrics(camera_link8[side]),
            "max_adjacent_hand_joint_change_rad": float(np.max(np.abs(np.diff(hand_joints[side], axis=0)))),
            "minimum_adjacent_quaternion_dot": float(np.min(np.sum(quaternions[:-1] * quaternions[1:], axis=1))),
        }

    deltas = np.diff(timestamps)
    report = {
        "artifact_type": "offline_sharpa_root_to_fr3_link8_conversion_report",
        "execution_enabled": False,
        "frame_count": int(len(timestamps)),
        "input_sha256": {"sharpa_hand_shape.npz": sha256(npz_path), "sharpa_actions_camera.npy": sha256(npy_path)},
        "npy_npz_consistency": {"right_max_abs_difference": 0.0, "left_max_abs_difference": 0.0, "passed": True},
        "timestamps": {
            "first_s": float(timestamps[0]),
            "last_s": float(timestamps[-1]),
            "duration_s": float(timestamps[-1] - timestamps[0]),
            "min_step_s": float(np.min(deltas)),
            "median_step_s": float(np.median(deltas)),
            "max_step_s": float(np.max(deltas)),
        },
        "mounts": {side: LINK8_T_SHARPA_ROOT[side].tolist() for side in ("left", "right")},
        "hand_urdfs": {
            side: {
                "path": str(path),
                "sha256": sha256(path),
                "actuated_joint_names": [
                    joint.attrib["name"]
                    for joint in ET.parse(path).getroot().findall("joint")
                    if joint.attrib.get("type") != "fixed"
                ],
            }
            for side, path in hand_urdfs.items()
        },
        "recomposition": recomposition,
        "arms": arm_metrics,
    }

    write_json(args.output_dir / "camera_link8.json", camera_document)
    write_json(args.output_dir / "base_link8_plan.json", base_document)
    write_json(args.output_dir / "link8_conversion_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote 3 OFFLINE artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
