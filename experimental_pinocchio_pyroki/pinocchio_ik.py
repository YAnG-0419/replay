"""Offline fixed-base Pinocchio IK. This module cannot send robot commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class IKResult:
    converged: bool
    iterations: int
    translation_error_m: float
    rotation_error_rad: float
    step_norm: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


class PinocchioIK:
    def __init__(
        self,
        urdf_path: str | Path,
        end_effector_frame: str,
        *,
        max_iterations: int = 200,
        translation_tolerance_m: float = 1e-4,
        rotation_tolerance_rad: float = 1e-3,
        damping: float = 1e-6,
        integration_step: float = 0.25,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError("Pinocchio is required; use the gello-upper-body-teleop Conda environment") from exc
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        if not self.model.existFrame(end_effector_frame):
            names = [frame.name for frame in self.model.frames]
            raise ValueError(f"EEF frame {end_effector_frame!r} not found; available frames include {names[:20]}")
        self.frame_id = self.model.getFrameId(end_effector_frame)
        self.max_iterations = max_iterations
        self.translation_tolerance_m = translation_tolerance_m
        self.rotation_tolerance_rad = rotation_tolerance_rad
        self.damping = damping
        self.integration_step = integration_step

    def _check_q(self, q: Iterable[float], name: str) -> np.ndarray:
        value = np.asarray(q, dtype=float)
        if value.shape != (self.model.nq,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain model.nq={self.model.nq} finite values")
        below = np.isfinite(self.model.lowerPositionLimit) & (value < self.model.lowerPositionLimit - 1e-9)
        above = np.isfinite(self.model.upperPositionLimit) & (value > self.model.upperPositionLimit + 1e-9)
        if np.any(below | above):
            indices = np.flatnonzero(below | above).tolist()
            raise ValueError(f"{name} violates model position limits at q indices {indices}")
        return value

    def solve(self, target_matrix: np.ndarray, q_seed: Iterable[float]) -> tuple[np.ndarray, IKResult]:
        pin = self.pin
        target_array = np.asarray(target_matrix, dtype=float)
        if target_array.shape != (4, 4) or not np.all(np.isfinite(target_array)):
            raise ValueError("target_matrix must be a finite 4x4 matrix")
        target = pin.SE3(target_array[:3, :3], target_array[:3, 3])
        q_start = self._check_q(q_seed, "q_seed")
        q = q_start.copy()
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit

        translation_error = rotation_error = np.inf
        for iteration in range(1, self.max_iterations + 1):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.frame_id]
            current_T_target = current.actInv(target)
            local_error = pin.log6(current_T_target).vector
            translation_error = float(np.linalg.norm(local_error[:3]))
            rotation_error = float(np.linalg.norm(local_error[3:]))
            if translation_error <= self.translation_tolerance_m and rotation_error <= self.rotation_tolerance_rad:
                return q, IKResult(True, iteration - 1, translation_error, rotation_error, float(np.linalg.norm(q - q_start)))

            frame_jacobian = pin.computeFrameJacobian(
                self.model, self.data, q, self.frame_id, pin.ReferenceFrame.LOCAL
            )
            # Pinocchio's frame Jacobian is a velocity Jacobian, while the
            # derivative of log(current^-1 * target) needs the SE(3) log-map
            # Jacobian and the opposite sign (same construction as the
            # official Pinocchio inverse-kinematics example).
            error_jacobian = -pin.Jlog6(current_T_target.inverse()) @ frame_jacobian
            system = error_jacobian @ error_jacobian.T + self.damping * np.eye(6)
            velocity = -error_jacobian.T @ np.linalg.solve(system, local_error)
            q = pin.integrate(self.model, q, velocity * self.integration_step)
            finite_lower = np.isfinite(lower)
            finite_upper = np.isfinite(upper)
            q[finite_lower] = np.maximum(q[finite_lower], lower[finite_lower])
            q[finite_upper] = np.minimum(q[finite_upper], upper[finite_upper])

        return q, IKResult(False, self.max_iterations, translation_error, rotation_error, float(np.linalg.norm(q - q_start)))

    def solve_trajectory(
        self,
        target_matrices: Iterable[np.ndarray],
        q0: Iterable[float],
        *,
        max_joint_step_rad: float = 0.35,
    ) -> tuple[np.ndarray, list[IKResult]]:
        previous = self._check_q(q0, "q0")
        trajectory: list[np.ndarray] = []
        results: list[IKResult] = []
        for frame_index, target in enumerate(target_matrices):
            q, result = self.solve(target, previous)
            if not result.converged:
                raise RuntimeError(f"IK failed at frame {frame_index}: {result}")
            largest_step = float(np.max(np.abs(q - previous)))
            if largest_step > max_joint_step_rad:
                raise RuntimeError(
                    f"IK frame {frame_index} joint jump {largest_step:.4f} rad exceeds {max_joint_step_rad:.4f} rad"
                )
            trajectory.append(q.copy())
            results.append(result)
            previous = q
        return np.asarray(trajectory), results

