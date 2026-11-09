from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Arm = Literal["left", "right"]


def _matrix4(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant is not +1")
    return matrix


def quaternion_wxyz_to_rotation(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion_wxyz must contain four finite numbers")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion_wxyz must not be zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotation_to_quaternion_wxyz(rotation: Any) -> np.ndarray:
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def pose_to_matrix(pose: dict[str, Any], name: str = "pose") -> np.ndarray:
    if "matrix" in pose:
        return _matrix4(pose["matrix"], f"{name}.matrix")
    try:
        position = np.asarray(pose["position_m"], dtype=float)
        rotation = quaternion_wxyz_to_rotation(pose["quaternion_wxyz"])
    except KeyError as exc:
        raise ValueError(f"{name} needs matrix or position_m + quaternion_wxyz") from exc
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"{name}.position_m must contain three finite numbers")
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def matrix_to_pose(matrix: Any) -> dict[str, list[float]]:
    transform = _matrix4(matrix, "pose matrix")
    return {
        "position_m": transform[:3, 3].tolist(),
        "quaternion_wxyz": rotation_to_quaternion_wxyz(transform[:3, :3]).tolist(),
        "matrix": transform.tolist(),
    }


@dataclass(frozen=True)
class HandEyeCalibration:
    source: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "HandEyeCalibration":
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("conventions", {}).get("camera_frame") != "camera_color_optical_frame":
            raise ValueError("calibration camera frame is not camera_color_optical_frame")
        calibration = cls(source=source, raw=raw)
        for arm in ("left", "right"):
            direct = calibration.base_T_camera(arm)
            inverse = calibration.camera_T_base(arm)
            if not np.allclose(direct @ inverse, np.eye(4), atol=1e-6):
                raise ValueError(f"{arm} calibration transform and inverse disagree")
        return calibration

    @property
    def created_at(self) -> str:
        return str(self.raw["created_at"])

    @property
    def camera_frame(self) -> str:
        return str(self.raw["conventions"]["camera_frame"])

    def base_T_camera(self, arm: Arm) -> np.ndarray:
        return _matrix4(self.raw["transforms"][f"{arm}_base_T_camera"], f"{arm}_base_T_camera")

    def camera_T_base(self, arm: Arm) -> np.ndarray:
        return _matrix4(self.raw["transforms"][f"camera_T_{arm}_base"], f"camera_T_{arm}_base")

    def camera_pose_to_base(self, arm: Arm, camera_T_controlled: Any) -> np.ndarray:
        return self.base_T_camera(arm) @ _matrix4(camera_T_controlled, "camera_T_controlled")

