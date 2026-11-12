"""Local differential IK: bounded Jacobian steps from the previous solution.

This is velocity/increment-level IK, not an automatic-differentiation layer.
All inner iterations remain inside the same frame-to-frame trust region.
"""
from __future__ import annotations

import numpy as np

from .model import ArmKinematics, checked_transform


class DifferentialIK(ArmKinematics):
    def __init__(self, robot, frame, active_indices, *, max_iterations=200):
        super().__init__(robot, frame, active_indices)
        self.max_iterations = max_iterations

    def solve_near(self, target, q_seed, *, reference=None, reference_weight=.05,
                   limit_weight=.001, max_step=.12):
        from scipy.optimize import lsq_linear

        target = checked_transform(target)
        seed, lower, upper = self.bounds(q_seed, max_step)
        reference = seed if reference is None else self.robot.check_q(reference, "reference")
        if not np.isfinite(reference_weight) or reference_weight < 0:
            raise ValueError("reference_weight must be finite and nonnegative")
        if not np.isfinite(limit_weight) or limit_weight < 0:
            raise ValueError("limit_weight must be finite and nonnegative")
        q = seed.copy()
        damping = 1e-6
        active = self.active
        for iteration in range(self.max_iterations):
            error = self.error(target, q)
            if np.linalg.norm(error[:3]) <= 1e-5 and np.linalg.norm(error[3:]) <= 1e-4:
                return q, self.validate(target, q, seed, lower, upper, iteration)
            jacobian = self.error(target, q, jacobian=True)
            x = q[active]
            posture = reference_weight * (reference[active] - x)
            a, b = x - self.robot.lower[active] + 1e-6, self.robot.upper[active] - x + 1e-6
            posture += np.clip(limit_weight * (1 / a - 1 / b), -.1, .1)
            # Solve for a local increment; clamp its bounds rather than clipping
            # a previously unconstrained IK solution.
            delta_lower = np.maximum(lower - x, -.1)
            delta_upper = np.minimum(upper - x, .1)
            if np.any(delta_lower >= delta_upper):
                break
            matrix = np.vstack([jacobian, np.sqrt(damping) * np.eye(len(active))])
            rhs = np.r_[-error, np.sqrt(damping) * posture]
            result = lsq_linear(matrix, rhs, bounds=(delta_lower, delta_upper), tol=1e-12)
            if not result.success or not np.isfinite(result.x).all():
                break
            moved = False
            for scale in (1., .5, .25, .125, .0625):
                candidate = q.copy()
                candidate[active] += scale * result.x
                if np.linalg.norm(self.error(target, candidate)) < np.linalg.norm(error) - 1e-13:
                    q, moved = candidate, True
                    break
            if not moved:
                break
        raise RuntimeError(
            f"Differential IK did not converge inside the fixed step bounds; "
            f"pose residual={self.error(target, q).tolist()}"
        )

