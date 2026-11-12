"""Lazy backend registration. Optional dependencies never affect legacy IK."""
from __future__ import annotations

from importlib import import_module


BACKENDS = {
    "continuous": ("eef_replay.ik.continuous", "ContinuousIK"),
    "differential": ("eef_replay.ik.differential", "DifferentialIK"),
    "placo": ("eef_replay.ik.placo_backend", "PlacoIK"),
}


def create_solver(name, robot, frame, active_indices):
    if name not in BACKENDS:
        raise ValueError(f"Unknown IK backend {name!r}; choices: {', '.join(BACKENDS)}")
    module, class_name = BACKENDS[name]
    return getattr(import_module(module), class_name)(robot, frame, active_indices)
