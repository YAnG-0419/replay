"""Experimental differentiable IK interface using local KKT sensitivity.

Returns dq/dxi for T_target @ exp6(xi), holding seed, posture and bounds fixed.
It is valid only on a smooth, interior, stationary IK branch. It intentionally
rejects active bounds and singular KKT systems instead of inventing a gradient.
No PyTorch/JAX autograd integration is implied. Hessians are central differences
of the analytic Pinocchio constraint Jacobian.
"""
from __future__ import annotations

import numpy as np

from .model import checked_transform, positive


def solution_jacobian(solver, target, seed, solution, *, reference=None,
                      reference_weight=.05, limit_weight=.001, max_step=.12,
                      epsilon=1e-5) -> np.ndarray:
    """Local [all joints, 6] sensitivity; inactive joint rows are exactly zero."""
    epsilon = positive(epsilon, "epsilon")
    target = checked_transform(target)
    seed, lower, upper = solver.bounds(seed, max_step)
    q = solver.robot.check_q(solution, "solution").copy()
    reference = seed if reference is None else solver.robot.check_q(reference, "reference")
    solver.validate(target, q, seed, lower, upper, 0)
    active, pin = solver.active, solver.robot.pin
    x = q[active]
    if np.any(np.minimum(x - lower, upper - x) <= 10 * epsilon):
        raise ValueError("IK sensitivity is undefined by this interface at an active bound")
    _, gradient = solver.objective(seed, reference, reference_weight, limit_weight)
    jac = solver.error(target, q, jacobian=True)
    if np.linalg.matrix_rank(jac, tol=1e-8) != 6:
        raise ValueError("IK sensitivity requires a full-rank task Jacobian")
    multipliers = np.linalg.lstsq(jac.T, -gradient(x), rcond=None)[0]
    if np.linalg.norm(gradient(x) + jac.T @ multipliers, ord=np.inf) > 1e-4:
        raise ValueError("IK solution is not sufficiently stationary for implicit differentiation")

    def stationarity(q_at, target_at):
        return gradient(q_at[active]) + solver.error(target_at, q_at, jacobian=True).T @ multipliers

    hessian = np.empty((len(active), len(active)))
    for column, index in enumerate(active):
        plus, minus = q.copy(), q.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        hessian[:, column] = (stationarity(plus, target) - stationarity(minus, target)) / (2 * epsilon)
    hessian = .5 * (hessian + hessian.T)
    target_derivative = np.empty((len(active) + 6, 6))
    target_se3 = pin.SE3(target[:3, :3], target[:3, 3])
    for column in range(6):
        delta = np.zeros(6)
        delta[column] = epsilon
        plus = (target_se3 * pin.exp6(delta)).homogeneous
        minus = (target_se3 * pin.exp6(-delta)).homogeneous
        target_derivative[:, column] = np.r_[
            stationarity(q, plus) - stationarity(q, minus),
            solver.error(plus, q) - solver.error(minus, q),
        ] / (2 * epsilon)
    kkt = np.block([[hessian, jac.T], [jac, np.zeros((6, 6))]])
    if not np.isfinite(kkt).all() or np.linalg.cond(kkt) > 1e12:
        raise ValueError("IK sensitivity KKT system is singular or ill-conditioned")
    derivative = np.zeros((len(q), 6))
    derivative[active] = np.linalg.solve(kkt, -target_derivative)[:len(active)]
    return derivative

