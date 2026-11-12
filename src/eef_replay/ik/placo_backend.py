"""Optional PlaCo QP adapter, sharing the same continuity/validation contract."""
from __future__ import annotations

import numpy as np

from .model import ArmKinematics, checked_transform


class PlacoIK(ArmKinematics):
    def __init__(self, robot, frame, active_indices, *, max_iterations=200):
        super().__init__(robot, frame, active_indices)
        try:
            import placo
        except ImportError as exc:
            raise RuntimeError("PlaCo backend is optional; install '.[ik,placo]' in a separate environment") from exc
        self.max_iterations = max_iterations
        self.wrapper = placo.RobotWrapper(str(robot.urdf), placo.Flags.ignore_collisions)
        if set(self.wrapper.joint_names()) != set(robot.joint_names):
            raise ValueError("PlaCo and the robot profile disagree on joint names")
        self.wrapper.set_T_world_fbase(np.eye(4))
        self.qp = placo.KinematicsSolver(self.wrapper)
        self.qp.mask_fbase(True)
        self.qp.enable_joint_limits(True)
        self.qp.enable_velocity_limits(True)
        # Internal iterations are not replay timesteps. The CLI already turns
        # real dt and velocity limits into a bound on the TOTAL frame change.
        self.qp.dt = 1.
        active_names = {robot.joint_names[i] for i in self.active}
        for name in robot.joint_names:
            if name not in active_names:
                self.qp.mask_dof(name)
            self.wrapper.set_velocity_limit(name, .1)
        self.target_task = self.qp.add_frame_task(frame, np.eye(4))
        self.target_task.configure("eef_target", "scaled", 1., 1.)
        self.seed_task = self.qp.add_joints_task()
        self.seed_task.configure("previous_frame", "soft", 1e-4)
        self.reference_task = self.qp.add_joints_task()
        self.limit_task = self.qp.add_joints_task()
        self.qp.add_regularization_task(1e-6)
        self.velocity_indices = np.array([self.wrapper.get_joint_v_offset(name) for name in robot.joint_names])

    def set_q(self, q):
        for name, value in zip(self.robot.joint_names, q):
            self.wrapper.set_joint(name, float(value))
        self.wrapper.update_kinematics()

    def active_values(self, q):
        return {self.robot.joint_names[i]: float(q[i]) for i in self.active}

    def solve_near(self, target, q_seed, *, reference=None, reference_weight=.05,
                   limit_weight=.001, max_step=.12):
        target = checked_transform(target)
        seed, lower, upper = self.bounds(q_seed, max_step)
        reference = seed if reference is None else self.robot.check_q(reference, "reference")
        for weight in [reference_weight, limit_weight]:
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("Posture and limit weights must be finite and nonnegative")
        self.set_q(seed)
        if not np.allclose(self.wrapper.get_T_world_frame(self.frame), self.fk(seed), atol=1e-8, rtol=0):
            raise ValueError("PlaCo and Pinocchio FK disagree; check model/base conventions")
        for index, lo, hi in zip(self.active, lower, upper):
            self.wrapper.set_joint_limits(self.robot.joint_names[index], float(lo), float(hi))
        self.target_task.T_world_frame = target
        self.seed_task.set_joints(self.active_values(seed))
        self.reference_task.configure("reference_posture", "soft", 1e-4 * reference_weight)
        self.reference_task.set_joints(self.active_values(reference))
        self.limit_task.configure("limit_margin", "soft", 1e-4)
        q = seed.copy()
        for iteration in range(self.max_iterations):
            error = self.error(target, q)
            if np.linalg.norm(error[:3]) <= 1e-5 and np.linalg.norm(error[3:]) <= 1e-4:
                return q, self.validate(target, q, seed, lower, upper, iteration)
            self.set_q(q)
            a = q[self.active] - self.robot.lower[self.active] + 1e-6
            b = self.robot.upper[self.active] - q[self.active] + 1e-6
            limit_target = q.copy()
            limit_target[self.active] += np.clip(limit_weight * (1 / a - 1 / b), -.1, .1)
            self.limit_task.set_joints(self.active_values(limit_target))
            try:
                delta = np.asarray(self.qp.solve(False))[self.velocity_indices]
            except RuntimeError as exc:
                raise RuntimeError(f"PlaCo QP failed within the fixed joint bounds: {exc}") from exc
            if not np.isfinite(delta).all():
                raise RuntimeError("PlaCo returned a non-finite joint increment")
            inactive = np.ones(len(q), dtype=bool)
            inactive[self.active] = False
            if np.any(np.abs(delta[inactive]) > 1e-9):
                raise RuntimeError("PlaCo moved a masked joint")
            moved = False
            for scale in (1., .5, .25, .125, .0625):
                candidate = q.copy()
                candidate[self.active] += scale * delta[self.active]
                if np.any(candidate[self.active] < lower - 1e-9) or np.any(candidate[self.active] > upper + 1e-9):
                    continue
                if np.linalg.norm(self.error(target, candidate)) < np.linalg.norm(error) - 1e-13:
                    q, moved = candidate, True
                    break
            if not moved:
                break
        raise RuntimeError(f"PlaCo IK did not converge inside the fixed step bounds; pose residual={self.error(target, q).tolist()}")
