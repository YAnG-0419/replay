"""Offline PyRoki joint-trajectory smoother; imports PyRoki only when called."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def smooth_joint_trajectory(
    urdf_path: str | Path,
    joint_trajectory: np.ndarray,
    *,
    dt_s: float,
    fidelity_weight: float = 10.0,
    smoothness_weight: float = 1.0,
    acceleration_weight: float = 0.1,
    fixed_indices: tuple[int, ...] = (0, -1),
) -> np.ndarray:
    """Smooth a full actuated-joint trajectory while preserving endpoints.

    Joint order must match PyRoki's actuated-joint order for ``urdf_path``.
    This is optimization only: no visualization, ROS, or hardware connection.
    """
    if dt_s <= 0 or not np.isfinite(dt_s):
        raise ValueError("dt_s must be finite and positive")
    reference = np.asarray(joint_trajectory, dtype=float)
    if reference.ndim != 2 or reference.shape[0] < 2 or not np.all(np.isfinite(reference)):
        raise ValueError("joint_trajectory must be a finite [frames, joints] array with at least two frames")

    try:
        import jax
        import jax.numpy as jnp
        import jaxls
        import pyroki as pk
        from yourdfpy import URDF
    except ImportError as exc:
        raise RuntimeError(
            "PyRoki is not installed. Install it only in a separate offline environment; "
            "do not alter the active data-collection environment."
        ) from exc

    urdf = URDF.load(str(urdf_path))
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=reference[0])
    frames, joints = reference.shape
    if robot.joints.num_actuated_joints != joints:
        raise ValueError(
            f"trajectory has {joints} joints but URDF/PyRoki has {robot.joints.num_actuated_joints} actuated joints"
        )

    trajectory_var = robot.joint_var_cls(jnp.arange(frames))
    costs: list[jaxls.Cost] = []

    @jaxls.Cost.factory(name="ReplayFidelityCost")
    def replay_fidelity_cost(vals, var, target, weight):
        return ((vals[var] - target) * weight).flatten()

    costs.append(
        replay_fidelity_cost(
            trajectory_var,
            jnp.asarray(reference),
            jnp.asarray(fidelity_weight),
        )
    )
    costs.append(
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, frames)),
            robot.joint_var_cls(jnp.arange(0, frames - 1)),
            jnp.asarray([smoothness_weight])[None],
        )
    )
    if frames >= 5 and acceleration_weight > 0:
        costs.append(
            pk.costs.five_point_acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, frames - 2)),
                robot.joint_var_cls(jnp.arange(4, frames)),
                robot.joint_var_cls(jnp.arange(3, frames - 1)),
                robot.joint_var_cls(jnp.arange(1, frames - 3)),
                robot.joint_var_cls(jnp.arange(0, frames - 4)),
                dt_s,
                jnp.asarray([acceleration_weight])[None],
            )
        )

    @jaxls.Cost.factory(kind="constraint_eq_zero", name="ReplayEndpointConstraint")
    def endpoint_constraint(vals, var, target):
        return (vals[var] - target).flatten()

    normalized_fixed = tuple(sorted({index % frames for index in fixed_indices}))
    for index in normalized_fixed:
        costs.append(
            endpoint_constraint(
                robot.joint_var_cls(index), jnp.asarray(reference[index])
            )
        )
    batched_robot = jax.tree.map(lambda value: value[None], robot)
    costs.append(pk.costs.limit_constraint(batched_robot, trajectory_var))
    costs.append(
        pk.costs.limit_velocity_constraint(
            batched_robot,
            robot.joint_var_cls(jnp.arange(1, frames)),
            robot.joint_var_cls(jnp.arange(0, frames - 1)),
            dt_s,
        )
    )

    solution = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[trajectory_var])
        .analyze()
        .solve(initial_vals=jaxls.VarValues.make((trajectory_var.with_value(jnp.asarray(reference)),)))
    )
    smoothed = np.asarray(solution[trajectory_var], dtype=float)
    if smoothed.shape != reference.shape or not np.all(np.isfinite(smoothed)):
        raise RuntimeError("PyRoki returned an invalid trajectory")
    return smoothed

