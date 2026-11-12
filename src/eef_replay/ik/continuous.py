"""Seed-aware constrained IK. The seed anchors every hard step bound."""
from __future__ import annotations

import numpy as np

from .model import ArmKinematics, checked_transform, positive


class ContinuousIK(ArmKinematics):
    def __init__(self, robot, frame, active_indices, *, max_iterations=150):
        super().__init__(robot, frame, active_indices)
        self.max_iterations = max_iterations

    def objective(self, seed, reference, reference_weight, limit_weight):
        for weight, name in [(reference_weight, "reference_weight"), (limit_weight, "limit_weight")]:
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        active = self.active
        lo, hi = self.robot.lower[active], self.robot.upper[active]

        def value(x):
            a, b = x - lo + 1e-8, hi - x + 1e-8
            return (.5 * np.sum((x - seed[active]) ** 2)
                    + .5 * reference_weight * np.sum((x - reference[active]) ** 2)
                    - limit_weight * np.sum(np.log(a) + np.log(b)))

        def gradient(x):
            a, b = x - lo + 1e-8, hi - x + 1e-8
            return (x - seed[active] + reference_weight * (x - reference[active])
                    + limit_weight * (-1 / a + 1 / b))

        return value, gradient

    def solve_near(self, target, q_seed, *, reference=None, reference_weight=.05,
                   limit_weight=.001, max_step=.12, home=None, max_home_excursion=None):
        from scipy.optimize import minimize

        target = checked_transform(target)
        seed, lower, upper = self.bounds(q_seed, max_step)
        reference = seed if reference is None else self.robot.check_q(reference, "reference")
        if home is not None:
            home = self.robot.check_q(home, "home")
            radius = positive(max_home_excursion, "max_home_excursion")
            lower = np.maximum(lower, home[self.active] - radius)
            upper = np.minimum(upper, home[self.active] + radius)
            if np.any(lower >= upper):
                raise RuntimeError("Empty home-constrained IK trust region")
        objective, gradient = self.objective(seed, reference, reference_weight, limit_weight)

        def q_at(x):
            q = seed.copy()
            q[self.active] = x
            return q

        result = minimize(
            objective, seed[self.active], jac=gradient, method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=[{"type": "eq", "fun": lambda x: self.error(target, q_at(x)),
                          "jac": lambda x: self.error(target, q_at(x), jacobian=True)}],
            options={"maxiter": self.max_iterations, "ftol": 1e-11},
        )
        q = q_at(result.x)
        try:
            details = self.validate(target, q, seed, lower, upper, int(result.nit))
        except RuntimeError as exc:
            raise RuntimeError(f"Constrained IK failed ({result.message}): {exc}") from exc
        return q, details

