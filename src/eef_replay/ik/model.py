"""Name-mapped fixed-base kinematics, with Pinocchio imported only on use."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np


def positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def checked_transform(value) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-10, rtol=0)):
        raise ValueError("Expected a finite homogeneous 4x4 target")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or abs(np.linalg.det(rotation) - 1) > 1e-7):
        raise ValueError("Target rotation must belong to SO(3)")
    return transform


@dataclass(frozen=True)
class IKResult:
    converged: bool
    iterations: int
    translation_error_m: float
    rotation_error_rad: float
    step_norm: float

    def to_dict(self) -> dict:
        return asdict(self)


class RobotModel:
    """Arm-only URDF with bounded scalar revolute joints; q uses profile order."""

    def __init__(self, urdf: Path, joint_names: tuple[str, ...] | None = None):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError("Install optional IK dependencies with pip install '.[ik]'") from exc
        self.pin = pin
        self.urdf = Path(urdf)
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        actual = tuple(str(name) for name in self.model.names[1:])
        self.joint_names = actual if joint_names is None else tuple(joint_names)
        if len(set(self.joint_names)) != len(self.joint_names) or set(actual) != set(self.joint_names):
            raise ValueError("URDF actuated joints must match robot profile by name")
        if self.model.nq != self.model.nv or self.model.nq != len(actual):
            raise ValueError("Only fixed-base scalar-joint URDFs are supported")
        for joint in self.model.joints[1:]:
            if joint.nq != 1 or joint.nv != 1 or not joint.shortname().startswith("JointModelR"):
                raise ValueError("Only bounded scalar revolute joints are supported")
        self.model_indices = np.array([
            self.model.joints[self.model.getJointId(name)].idx_q for name in self.joint_names
        ], dtype=int)
        self.lower = np.array(self.model.lowerPositionLimit[self.model_indices], copy=True)
        self.upper = np.array(self.model.upperPositionLimit[self.model_indices], copy=True)
        self.velocity = np.array(self.model.velocityLimit[self.model_indices], copy=True)
        if not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper):
            raise ValueError("Finite, nonempty URDF joint position limits are required")
        if not np.isfinite(self.velocity).all() or np.any(self.velocity <= 0):
            raise ValueError("Finite positive URDF joint velocity limits are required")

    def check_q(self, q, name="q") -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != self.lower.shape or not np.isfinite(q).all():
            raise ValueError(f"{name} must contain {len(self.lower)} finite joint angles")
        if np.any(q < self.lower - 1e-9) or np.any(q > self.upper + 1e-9):
            raise ValueError(f"{name} violates joint position limits")
        return q

    def restrict_limits(self, lower, upper) -> None:
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if lower.shape != self.lower.shape or upper.shape != self.upper.shape or not np.isfinite([lower, upper]).all():
            raise ValueError("Supplied limits must match the joint-name order")
        restricted_lower = np.maximum(self.lower, lower)
        restricted_upper = np.minimum(self.upper, upper)
        if np.any(restricted_lower >= restricted_upper):
            raise ValueError("Supplied and URDF limits have no common joint range")
        self.lower, self.upper = restricted_lower, restricted_upper

    def frame_id(self, name: str) -> int:
        if not self.model.existFrame(name):
            raise ValueError(f"URDF frame {name!r} was not found")
        return int(self.model.getFrameId(name))

    def pin_q(self, q) -> np.ndarray:
        result = np.empty(self.model.nq)
        result[self.model_indices] = self.check_q(q)
        return result

    def fk(self, frame: str, q) -> np.ndarray:
        self.pin.framesForwardKinematics(self.model, self.data, self.pin_q(q))
        return np.array(self.data.oMf[self.frame_id(frame)].homogeneous, copy=True)

    def fixed_base(self, frame: str) -> np.ndarray:
        frame_id = self.frame_id(frame)
        frame_model = self.model.frames[frame_id]
        parent = getattr(frame_model, "parentJoint", None)
        if parent is None:  # Pinocchio 2 compatibility.
            parent = frame_model.parent
        if parent != 0:
            raise ValueError("The configured arm base must be fixed to the model root")
        # Fixed frames do not depend on q; neutral may lie outside arm limits.
        self.pin.framesForwardKinematics(self.model, self.data, self.pin.neutral(self.model))
        return np.array(self.data.oMf[frame_id].homogeneous, copy=True)


class ArmKinematics:
    def __init__(self, robot: RobotModel, frame: str, active_indices):
        self.robot = robot
        self.frame = frame
        self.frame_id = robot.frame_id(frame)
        self.active = np.asarray(active_indices, dtype=int)
        if (self.active.ndim != 1 or not self.active.size
                or len(set(self.active)) != len(self.active)
                or np.any(self.active < 0) or np.any(self.active >= len(robot.joint_names))):
            raise ValueError("Active joint indices must be distinct and in range")

    def fk(self, q) -> np.ndarray:
        return self.robot.fk(self.frame, q)

    def error(self, target, q, *, jacobian=False) -> np.ndarray:
        pin = self.robot.pin
        target = checked_transform(target)
        q_pin = self.robot.pin_q(q)
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q_pin)
        desired = pin.SE3(target[:3, :3], target[:3, 3])
        delta = self.robot.data.oMf[self.frame_id].actInv(desired)
        if jacobian:
            jac = pin.computeFrameJacobian(
                self.robot.model, self.robot.data, q_pin, self.frame_id, pin.ReferenceFrame.LOCAL
            )
            return (-pin.Jlog6(delta.inverse()) @ jac)[:, self.robot.model_indices[self.active]]
        return np.array(pin.log6(delta).vector, copy=True)

    def bounds(self, seed, max_step):
        seed = self.robot.check_q(seed, "seed").copy()
        radius = np.broadcast_to(np.asarray(max_step, dtype=float), self.active.shape)
        if not np.isfinite(radius).all() or np.any(radius <= 0):
            raise ValueError("max_step must contain finite positive bounds")
        lower = np.maximum(self.robot.lower[self.active], seed[self.active] - radius)
        upper = np.minimum(self.robot.upper[self.active], seed[self.active] + radius)
        if np.any(lower >= upper):
            raise RuntimeError("Empty IK trust region")
        return seed, lower, upper

    def validate(self, target, q, seed, lower, upper, iterations) -> IKResult:
        error = self.error(target, q)
        translation, rotation = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
        self.robot.check_q(q, "solution")
        bounded = np.all(q[self.active] >= lower - 1e-9) and np.all(q[self.active] <= upper + 1e-9)
        inactive = np.ones(len(q), dtype=bool)
        inactive[self.active] = False
        bounded = bounded and np.array_equal(q[inactive], seed[inactive])
        result = IKResult(bool(bounded and translation <= 1e-5 and rotation <= 1e-4),
                          iterations, translation, rotation, float(np.linalg.norm(q - seed)))
        if not result.converged:
            raise RuntimeError(f"IK constraints or pose tolerance not satisfied: {result}")
        return result
