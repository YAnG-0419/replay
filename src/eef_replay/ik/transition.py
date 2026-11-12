"""Separate quintic entry path, preserving start/end positions exactly."""
import numpy as np

from .model import positive


def minimum_jerk_transition(start, goal, lower, upper, *, velocity=.09,
                            acceleration=.108, jerk=.432, dt=.01):
    """Analytic extrema bound this monotonic joint path; no collision claim."""
    for value, name in [(velocity, "velocity"), (acceleration, "acceleration"),
                        (jerk, "jerk"), (dt, "dt")]:
        positive(value, name)
    start, goal, lower, upper = [np.asarray(value, dtype=float) for value in (start, goal, lower, upper)]
    if start.ndim != 1 or start.size == 0 or any(
        value.shape != start.shape or not np.isfinite(value).all()
        for value in (start, goal, lower, upper)
    ):
        raise ValueError("Transition vectors must have identical finite nonempty shapes")
    if (np.any(lower >= upper) or np.any(np.minimum(start, goal) < lower)
            or np.any(np.maximum(start, goal) > upper)):
        raise ValueError("Transition endpoints exceed joint limits")
    delta = goal - start
    excursion = float(np.max(abs(delta)))
    # Exact maxima for s(u) = 10u^3 - 15u^4 + 6u^5.
    duration = max(
        2., 1.875 * excursion / velocity,
        np.sqrt((10 / np.sqrt(3)) * excursion / acceleration),
        np.cbrt(60 * excursion / jerk),
    ) * (1 + 1e-8)
    times = np.linspace(0, duration, int(np.ceil(duration / dt)) + 1)
    u = times / duration
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
    dds = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
    q = start + s[:, None] * delta
    dq, ddq = ds[:, None] * delta, dds[:, None] * delta
    q[0], q[-1] = start, goal
    report = {
        "duration_s": duration, "interpolation": "analytic_quintic_smoothstep",
        "peak_velocity_rad_s": 1.875 * excursion / duration,
        "peak_acceleration_rad_s2": (10 / np.sqrt(3)) * excursion / duration**2,
        "peak_jerk_rad_s3": 60 * excursion / duration**3,
        "max_joint_excursion_rad": excursion, "collision_checked": False,
        "execution_enabled": False,
    }
    return times, q, dq, ddq, report
