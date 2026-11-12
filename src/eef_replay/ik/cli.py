"""Offline profile-based IK, kept separate from the original Pinocchio CLI."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np

from .backends import BACKENDS, create_solver
from .model import RobotModel, checked_transform, positive
from .profiles import load_profile
from .transition import minimum_jerk_transition


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new_documents(documents):
    """Never overwrite an earlier successful run; roll back this run on IO error."""
    created = []
    try:
        for path, document in documents.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                created.append(path)
                json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
    except BaseException:
        for path in created:
            path.unlink()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="continuous")
    parser.add_argument("--frame0-backend", choices=tuple(BACKENDS),
                        help="Explicit optional initializer for frame zero; later frames use --backend")
    parser.add_argument("--robot", default="franka-sharpa")
    parser.add_argument("--robot-config", type=Path, help="Explicit profile for another body")
    parser.add_argument("--start-state", type=Path, help="Offline named arm-joint state")
    parser.add_argument("--reference", choices=("home", "previous", "source"), default="home")
    parser.add_argument("--reference-weight", type=float, default=.05)
    parser.add_argument("--limit-weight", type=float, default=.001)
    parser.add_argument("--max-home-excursion", type=float, default=1.2)
    parser.add_argument("--continuous-max-step", "--max-joint-step", dest="max_step", type=float, default=.12)
    parser.add_argument("--velocity-scale", type=float, default=1., help="Scale URDF frame-to-frame velocity limits")
    parser.add_argument("--first-frame-only", action="store_true")
    parser.add_argument("--transition-output", type=Path)
    parser.add_argument("--transition-max-velocity", type=float, default=.09)
    parser.add_argument("--transition-max-acceleration", type=float, default=.108)
    parser.add_argument("--transition-max-jerk", type=float, default=.432)
    return parser.parse_args(argv)


def solve(args):
    profile = load_profile(args.robot, args.robot_config)
    transition_path = args.transition_output or args.output.with_name(args.output.stem + "_start_transition.json")
    failure_path = args.output.with_suffix(".failure.json")
    outputs = [args.output, transition_path, failure_path]
    inputs = [args.plan, args.urdf, args.start_state, args.robot_config]
    if (len({p.resolve() for p in outputs}) != len(outputs)
            or {p.resolve() for p in outputs} & {p.resolve() for p in inputs if p is not None}):
        raise ValueError("Output paths must be distinct and must not overwrite input files")
    for output in [args.output, transition_path]:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output already exists; use a new run path: {output}")
    if failure_path.is_symlink():
        raise ValueError("Failure output must not be a symlink")
    for value, name in [(args.max_home_excursion, "max-home-excursion"), (args.max_step, "max-joint-step"),
                        (args.velocity_scale, "velocity-scale"), (args.transition_max_velocity, "transition-max-velocity"),
                        (args.transition_max_acceleration, "transition-max-acceleration"),
                        (args.transition_max_jerk, "transition-max-jerk")]:
        positive(value, name)
    if args.velocity_scale > 1:
        raise ValueError("velocity-scale must not exceed 1")
    for value in [args.reference_weight, args.limit_weight]:
        if not np.isfinite(value) or value < 0:
            raise ValueError("Reference and limit weights must be finite and nonnegative")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("execution_enabled") is not False or plan.get("output_controlled_frame") != profile.controlled_frame:
        raise ValueError(f"Plan must be an offline {profile.controlled_frame} plan")
    if plan.get("robot_profile", profile.name) != profile.name:
        raise ValueError("Plan and robot profile do not match")
    samples = plan["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("Plan must contain samples")
    times = np.array([sample["t_s"] for sample in samples], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Replay timestamps must be finite and strictly increasing")
    input_count = len(samples)
    if args.first_frame_only:
        samples, times = samples[:1], times[:1]
    # Validate every selected target before starting an expensive solve.
    for sample in samples:
        for arm in profile.arms:
            checked_transform(sample[arm.name][arm.target_key]["matrix"])
            for key in arm.payload_keys:
                if key not in sample[arm.name]:
                    raise ValueError(f"Missing payload {arm.name}.{key}")
    robot = RobotModel(args.urdf, profile.joint_names)

    def source_q(sample):
        values = []
        for arm in profile.arms:
            source = sample[arm.name].get("source_arm_joint_positions_rad")
            if source is None or len(source) != len(arm.joint_names):
                raise ValueError("Source reference requires every arm's named-order source joints at every frame")
            values.extend(source)
        return robot.check_q(values, "source joint reference").copy()

    start_document = None
    if args.start_state is not None:
        start_document = json.loads(args.start_state.read_text(encoding="utf-8"))
        if start_document.get("execution_enabled") is not False:
            raise ValueError("Start state must be an offline artifact")
        if start_document.get("robot_profile", profile.name) != profile.name:
            raise ValueError("Start state and robot profile do not match")
        if start_document.get("joint_names") != list(profile.joint_names):
            raise ValueError("Start-state joint names/order differ from the robot profile")
        if "joint_limits_rad" in start_document:
            limits = start_document["joint_limits_rad"]
            robot.restrict_limits(limits["lower"], limits["upper"])
        initial = robot.check_q(start_document["joint_positions_rad"], "start state").copy()
        seed_source = "explicit_start_state"
    else:
        initial = source_q(samples[0])
        seed_source = "first_frame_source_joints"
    if args.reference == "source":
        references = [source_q(sample) for sample in samples]
    else:
        references = None
    solvers = {arm.name: create_solver(args.backend, robot, arm.eef_frame, profile.indices(arm)) for arm in profile.arms}
    frame0_backend = args.frame0_backend or args.backend
    initializers = (solvers if frame0_backend == args.backend else
                    {arm.name: create_solver(frame0_backend, robot, arm.eef_frame, profile.indices(arm)) for arm in profile.arms})
    mounts = {arm.name: robot.fixed_base(arm.base_frame) for arm in profile.arms}
    solved, q_values = [], []
    previous = initial.copy()
    for index, sample in enumerate(samples):
        bounds = (np.full(len(initial), args.max_home_excursion) if index == 0 else
                  np.minimum(args.max_step, robot.velocity * args.velocity_scale * (times[index] - times[index - 1])))
        reference = (references[index] if references is not None else
                     previous if args.reference == "previous" else initial)
        current = previous.copy()
        frame_backend = frame0_backend if index == 0 else args.backend
        frame_solvers = initializers if index == 0 else solvers
        result_sample = {"t_s": float(times[index]), "backend": frame_backend}
        try:
            for arm in profile.arms:
                indices = profile.indices(arm)
                target = mounts[arm.name] @ np.array(sample[arm.name][arm.target_key]["matrix"], dtype=float)
                arm_seed = current.copy()
                current, result = frame_solvers[arm.name].solve_near(
                    target, current, reference=reference, reference_weight=args.reference_weight,
                    limit_weight=args.limit_weight, max_step=bounds[list(indices)],
                )
                current = np.asarray(current, dtype=float)
                if current.shape != initial.shape or not np.isfinite(current).all():
                    raise RuntimeError("Backend returned an invalid joint vector")
                inactive = np.ones(len(current), dtype=bool)
                inactive[list(indices)] = False
                if not np.array_equal(current[inactive], arm_seed[inactive]):
                    raise RuntimeError("Backend moved joints outside the current arm")
                if np.any(current < robot.lower - 1e-9) or np.any(current > robot.upper + 1e-9):
                    raise RuntimeError("Backend exceeded configured joint limits")
                result_sample[f"{arm.name}_arm_joint_positions_rad"] = current[list(indices)].tolist()
                result_sample[f"{arm.name}_ik"] = result.to_dict()
                for key in arm.payload_keys:
                    result_sample[f"{arm.name}_{key}"] = sample[arm.name][key]
            # Independent verification, including cross-arm effects and name mappings.
            if np.any(np.abs(current - previous) > bounds + 1e-8):
                raise RuntimeError("Backend exceeded its joint step bounds")
            for arm in profile.arms:
                target = mounts[arm.name] @ np.array(sample[arm.name][arm.target_key]["matrix"], dtype=float)
                error = solvers[arm.name].error(target, current)
                if np.linalg.norm(error[:3]) > 1e-5 or np.linalg.norm(error[3:]) > 1e-4:
                    raise RuntimeError(f"Final combined state violates {arm.name} target")
        except RuntimeError as exc:
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure = {"execution_enabled": False, "completed": False, "output_written": False,
                       "backend": frame_backend, "trajectory_backend": args.backend,
                       "robot_profile": profile.name, "failed_frame": index,
                       "source_time_s": float(times[index]), "reason": str(exc),
                       "joint_step_bounds_rad": bounds.tolist(),
                       "note": "No partial replay exported; failure does not prove global infeasibility"}
            failure_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            raise RuntimeError(f"{frame_backend} failed at frame {index}, t={times[index]:.6f}s: {exc}") from exc
        solved.append(result_sample)
        q_values.append(current.copy())
        previous = current
    joints = np.array(q_values)
    adjacent = np.diff(joints, axis=0)
    velocity = adjacent / np.diff(times)[:, None]
    summary = {
        "frame_count": len(solved), "initial_seed_source": seed_source,
        "initial_seed_joint_positions_rad": initial.tolist(),
        "frame0_distance_from_seed_rad": float(np.max(np.abs(joints[0] - initial))),
        "max_adjacent_joint_change_rad": float(np.max(np.abs(adjacent))) if adjacent.size else 0.,
        "total_joint_travel_rad": np.sum(np.abs(adjacent), axis=0).tolist(),
        "max_velocity_rad_s": float(np.max(np.abs(velocity))) if velocity.size else 0.,
        "minimum_joint_limit_margin_rad": float(np.min(np.minimum(joints - robot.lower, robot.upper - joints))),
    }
    t, positions, velocities, accelerations, transition_summary = minimum_jerk_transition(
        initial, joints[0], robot.lower, robot.upper,
        velocity=min(args.transition_max_velocity, float(np.min(robot.velocity)) * args.velocity_scale),
        acceleration=args.transition_max_acceleration, jerk=args.transition_max_jerk,
    )
    payload = {
        "artifact_type": "offline_home_aware_frame0_ik" if args.first_frame_only else "offline_profile_ik",
        "execution_enabled": False, "completed": True, "backend": args.backend,
        "frame0_backend": frame0_backend,
        "robot_profile": profile.name, "profile": asdict(profile), "joint_names": list(profile.joint_names),
        "source_plan": str(args.plan), "source_plan_sha256": fingerprint(args.plan),
        "source_urdf": str(args.urdf), "source_urdf_sha256": fingerprint(args.urdf),
        "source_start_state": str(args.start_state) if args.start_state else None,
        "source_start_state_sha256": fingerprint(args.start_state) if args.start_state else None,
        "input_frame_count": input_count, "first_frame_only": args.first_frame_only,
        "reference_strategy": args.reference, "seed_strategy": "previous_solved_frame",
        "constraints": {"max_home_excursion_rad": args.max_home_excursion, "max_joint_step_rad": args.max_step,
                        "velocity_scale": args.velocity_scale, "translation_tolerance_m": 1e-5,
                        "rotation_tolerance_rad": 1e-4, "collision_checked": False,
                        "acceleration_and_jerk_checked_for_replay": False},
        "ik_summary": summary, "transition": str(transition_path), "samples": solved,
    }
    transition = {
        "artifact_type": "offline_home_aware_transition", "execution_enabled": False,
        "robot_profile": profile.name, "joint_names": list(profile.joint_names),
        "start_state_kind": (start_document or {}).get("state_kind", "offline_source_seed_not_live_measurement"),
        "source_replay": str(args.output), "collision_checked": False, "summary": transition_summary,
        "samples": [{"t_s": float(time), "joint_positions_rad": pos.tolist(),
                     "joint_velocities_rad_s": vel.tolist(), "joint_accelerations_rad_s2": acc.tolist()}
                    for time, pos, vel, acc in zip(t, positions, velocities, accelerations)],
    }
    write_new_documents({args.output: payload, transition_path: transition})
    if failure_path.exists():
        failure_path.unlink()
    return payload


def main(argv=None):
    args = parse_args(argv)
    payload = solve(args)
    print(f"Wrote {payload['ik_summary']['frame_count']} OFFLINE {args.backend} frames for {payload['robot_profile']} to {args.output}")


if __name__ == "__main__":
    main()

