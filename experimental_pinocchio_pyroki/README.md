# Experimental Pinocchio IK + PyRoki smoothing

This directory is deliberately separate from the normal replay preparation code.
Everything here is **offline only**: it reads JSON/URDF files and writes JSON/NPY
files. It contains no ROS imports, robot IP addresses, controller clients, Docker
commands, or Safety Gateway calls.

## Current machine status

- Pinocchio 4.1.0 is already available in the Conda environment
  `gello-upper-body-teleop`.
- PyRoki was not installed when this directory was created. Do not install it into
  a data-collection environment. Create a separate environment after collection.

## Pinocchio IK

`pinocchio_ik.py` provides a fixed-base damped least-squares solver. It loads a
URDF, solves each target sequentially using the previous solution as its seed,
clips to model limits, and rejects excessive per-frame joint jumps. The caller
must explicitly provide the URDF, EEF frame name, initial joints and target poses.

The target matrices must be `base_T_link8_target` from the prepared replay plan.
For the current FR3 convention, use a single-arm fixed-base URDF whose base is the
matching libfranka `O` frame. Do not silently use `lychee_root`; the delivered
calibration has separate left and right base frames.

Example import:

```python
from pinocchio_ik import PinocchioIK

solver = PinocchioIK("/path/to/single_fr3.urdf", "fr3_link8")
q, result = solver.solve(target_matrix, q_seed)
```

## PyRoki smoothing

`pyroki_smoothing.py` follows the upstream trajectory-optimization API and adds:

- fidelity to the Pinocchio IK result;
- adjacent-frame smoothness;
- optional five-point acceleration cost;
- hard start/end preservation;
- hard URDF joint limits.

It is intentionally lazy-loaded, so importing the file does not require JAX or
PyRoki. Once a separate environment is available, call `smooth_joint_trajectory`.
The input joint order must exactly equal PyRoki's actuated-joint order from the
same URDF.

Neither solver performs collision checking in this first version. A trajectory
must pass offline visualization, joint-limit/velocity checks, collision checks,
TCP verification, and a separate operator-approved execution layer before any
real-robot use.

