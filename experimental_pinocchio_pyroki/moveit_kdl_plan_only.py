"""MoveIt/KDL replay IK plus a collision-aware plan to frame 0.

This program is deliberately plan-only: it creates no action clients, no
controller-manager clients, and no command publishers.  It only subscribes to
joint states and calls MoveIt's /compute_ik and /plan_kinematic_path services.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np


SIDES = ("left", "right")
SUCCESS = 1


def joint_names(side: str) -> tuple[str, ...]:
    return tuple(f"{side}_fr3v2_joint{index}" for index in range(1, 8))


ALL_JOINT_NAMES = tuple(name for side in SIDES for name in joint_names(side))


def rotation_matrix_to_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a normalized ROS quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
            w = (rotation[2, 1] - rotation[1, 2]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    quaternion = np.asarray([x, y, z, w], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def root_T_bases(urdf_path: Path) -> dict[str, np.ndarray]:
    """Read the fixed dual-arm mount transforms from the same URDF as replay."""
    try:
        import pinocchio as pin
    except ImportError as error:
        raise RuntimeError("Pinocchio is required to read the fixed arm mounts") from error
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    result: dict[str, np.ndarray] = {}
    for side in SIDES:
        frame = f"{side}_fr3v2_link0"
        if not model.existFrame(frame):
            raise ValueError(f"URDF has no frame {frame!r}")
        result[side] = np.asarray(data.oMf[model.getFrameId(frame)].homogeneous)
    return result


def duration_to_seconds(duration: Any) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def serialize_trajectory(trajectory: Any) -> dict[str, Any]:
    joint = trajectory.joint_trajectory
    return {
        "joint_names": list(joint.joint_names),
        "points": [
            {
                "time_from_start_s": duration_to_seconds(point.time_from_start),
                "positions": list(point.positions),
                "velocities": list(point.velocities),
                "accelerations": list(point.accelerations),
                "effort": list(point.effort),
            }
            for point in joint.points
        ],
    }


def wait_for_future(node: Any, future: Any, timeout_sec: float, label: str) -> Any:
    import rclpy

    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done():
        raise TimeoutError(f"timed out waiting for {label}")
    error = future.exception()
    if error is not None:
        raise RuntimeError(f"{label} failed: {error}")
    response = future.result()
    if response is None:
        raise RuntimeError(f"{label} returned no response")
    return response


class JointStateCollector:
    def __init__(self, node: Any, joint_state_type: Any) -> None:
        self.positions: dict[str, float] = {}
        self.received_at = 0.0

        def callback(message: Any) -> None:
            incoming = {
                str(name): float(value)
                for name, value in zip(message.name, message.position, strict=True)
                if math.isfinite(float(value))
            }
            if incoming:
                self.positions.update(incoming)
                self.received_at = time.monotonic()

        self._subscriptions = [
            node.create_subscription(joint_state_type, "/left/franka/joint_states", callback, 10),
            node.create_subscription(joint_state_type, "/right/franka/joint_states", callback, 10),
            node.create_subscription(joint_state_type, "/joint_states", callback, 10),
        ]

    def ordered(self) -> np.ndarray | None:
        if any(name not in self.positions for name in ALL_JOINT_NAMES):
            return None
        return np.asarray([self.positions[name] for name in ALL_JOINT_NAMES], dtype=float)


def wait_until_ready(
    node: Any,
    collector: JointStateCollector,
    compute_ik: Any,
    plan_path: Any,
    timeout_sec: float,
) -> np.ndarray:
    import rclpy

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        positions = collector.ordered()
        if (
            positions is not None
            and compute_ik.service_is_ready()
            and plan_path.service_is_ready()
        ):
            return positions
    missing = []
    if collector.ordered() is None:
        missing.append("fresh dual-arm joint states")
    if not compute_ik.service_is_ready():
        missing.append("/compute_ik")
    if not plan_path.service_is_ready():
        missing.append("/plan_kinematic_path")
    raise RuntimeError("not ready: " + ", ".join(missing))


def fill_robot_state(robot_state: Any, positions: np.ndarray) -> None:
    robot_state.is_diff = True
    robot_state.joint_state.name = list(ALL_JOINT_NAMES)
    robot_state.joint_state.position = positions.tolist()


def solve_one(
    *,
    node: Any,
    client: Any,
    request_type: Any,
    pose_type: Any,
    side: str,
    root_T_link8: np.ndarray,
    seed: np.ndarray,
    timeout_sec: float,
) -> np.ndarray:
    request = request_type.Request()
    request.ik_request.group_name = f"{side}_arm"
    request.ik_request.ik_link_name = f"{side}_fr3v2_link8"
    request.ik_request.avoid_collisions = True
    request.ik_request.timeout.sec = 0
    request.ik_request.timeout.nanosec = 100_000_000
    fill_robot_state(request.ik_request.robot_state, seed)
    pose = pose_type()
    pose.header.frame_id = "lychee_root"
    pose.header.stamp = node.get_clock().now().to_msg()
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
        float(value) for value in root_T_link8[:3, 3]
    )
    (
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z,
        pose.pose.orientation.w,
    ) = rotation_matrix_to_xyzw(root_T_link8[:3, :3])
    request.ik_request.pose_stamped = pose
    response = wait_for_future(
        node,
        client.call_async(request),
        timeout_sec,
        f"compute_ik {side}",
    )
    if int(response.error_code.val) != SUCCESS:
        raise RuntimeError(
            f"MoveIt KDL IK failed for {side}: error_code={response.error_code.val}"
        )
    by_name = {
        str(name): float(value)
        for name, value in zip(
            response.solution.joint_state.name,
            response.solution.joint_state.position,
            strict=True,
        )
    }
    solved = seed.copy()
    for name in joint_names(side):
        if name not in by_name:
            raise RuntimeError(f"MoveIt IK response is missing joint {name}")
        solved[ALL_JOINT_NAMES.index(name)] = by_name[name]
    return solved


def plan_to_first_frame(
    *,
    node: Any,
    client: Any,
    request_type: Any,
    constraints_type: Any,
    joint_constraint_type: Any,
    current: np.ndarray,
    first: np.ndarray,
    timeout_sec: float,
    velocity_scale: float,
    acceleration_scale: float,
) -> dict[str, Any]:
    request = request_type.Request()
    motion = request.motion_plan_request
    motion.group_name = "both_arms"
    motion.pipeline_id = "ompl"
    motion.planner_id = "RRTConnectkConfigDefault"
    motion.num_planning_attempts = 3
    motion.allowed_planning_time = min(timeout_sec, 20.0)
    motion.max_velocity_scaling_factor = velocity_scale
    motion.max_acceleration_scaling_factor = acceleration_scale
    fill_robot_state(motion.start_state, current)
    constraints = constraints_type()
    constraints.name = "episode_first_frame"
    for name, value in zip(ALL_JOINT_NAMES, first, strict=True):
        constraint = joint_constraint_type()
        constraint.joint_name = name
        constraint.position = float(value)
        constraint.tolerance_above = 1e-4
        constraint.tolerance_below = 1e-4
        constraint.weight = 1.0
        constraints.joint_constraints.append(constraint)
    motion.goal_constraints = [constraints]
    response = wait_for_future(
        node,
        client.call_async(request),
        timeout_sec + 5.0,
        "plan_kinematic_path to frame 0",
    ).motion_plan_response
    if int(response.error_code.val) != SUCCESS:
        raise RuntimeError(
            "MoveIt could not plan from current state to frame 0: "
            f"error_code={response.error_code.val}"
        )
    return {
        "artifact_type": "moveit_plan_to_replay_first_frame",
        "execution_enabled": False,
        "warning": "PLAN ONLY; this file is never submitted for execution",
        "planner": "MoveIt OMPL RRTConnect",
        "group_name": str(response.group_name),
        "planning_time_s": float(response.planning_time),
        "current_joint_positions_rad": current.tolist(),
        "first_frame_joint_positions_rad": first.tolist(),
        "trajectory": serialize_trajectory(response.trajectory),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path, help="MoveIt/KDL IK result JSON")
    parser.add_argument("--start-plan-output", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--ready-timeout", type=float, default=20.0)
    parser.add_argument("--service-timeout", type=float, default=8.0)
    parser.add_argument("--max-joint-step", type=float, default=0.35)
    parser.add_argument("--velocity-scale", type=float, default=0.1)
    parser.add_argument("--acceleration-scale", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.velocity_scale <= 1.0:
        raise ValueError("--velocity-scale must be in (0, 1]")
    if not 0.0 < args.acceleration_scale <= 1.0:
        raise ValueError("--acceleration-scale must be in (0, 1]")
    document = json.loads(args.plan.read_text(encoding="utf-8"))
    if (
        document.get("execution_enabled") is not False
        or document.get("output_controlled_frame") != "fr3_link8"
    ):
        raise ValueError("input must be an offline fr3_link8 replay plan")
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("input replay plan contains no samples")
    mounts = root_T_bases(args.urdf)

    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.msg import Constraints, JointConstraint
        from moveit_msgs.srv import GetMotionPlan, GetPositionIK
        from sensor_msgs.msg import JointState
    except ImportError as error:
        raise RuntimeError("source ROS 2 and the MoveIt workspace before running") from error

    rclpy.init(args=None)
    node = rclpy.create_node("eef_replay_moveit_plan_only")
    try:
        collector = JointStateCollector(node, JointState)
        compute_ik = node.create_client(GetPositionIK, "/compute_ik")
        plan_path = node.create_client(GetMotionPlan, "/plan_kinematic_path")
        current = wait_until_ready(
            node, collector, compute_ik, plan_path, args.ready_timeout
        )
        seed = current.copy()
        solved_samples: list[dict[str, Any]] = []
        maximum_step = 0.0
        for index, sample in enumerate(samples):
            previous = seed.copy()
            for side in SIDES:
                base_T_link8 = np.asarray(
                    sample[side]["base_T_link8_target"]["matrix"], dtype=float
                )
                seed = solve_one(
                    node=node,
                    client=compute_ik,
                    request_type=GetPositionIK,
                    pose_type=PoseStamped,
                    side=side,
                    root_T_link8=mounts[side] @ base_T_link8,
                    seed=seed,
                    timeout_sec=args.service_timeout,
                )
            step = float(np.max(np.abs(seed - previous)))
            maximum_step = max(maximum_step, step)
            # Frame zero is deliberately reached through a separate collision-
            # aware plan. Subsequent frames must stay on a continuous IK branch.
            if index > 0 and step > args.max_joint_step:
                raise RuntimeError(
                    f"MoveIt IK joint jump at frame {index}: {step:.4f} rad"
                )
            solved_samples.append(
                {
                    "t_s": float(sample["t_s"]),
                    "left_arm_joint_positions_rad": seed[:7].tolist(),
                    "left_hand_joint_positions_rad": sample["left"][
                        "hand_joint_positions_rad"
                    ],
                    "right_arm_joint_positions_rad": seed[7:].tolist(),
                    "right_hand_joint_positions_rad": sample["right"][
                        "hand_joint_positions_rad"
                    ],
                }
            )
            if index == 0:
                first_solution = seed.copy()

        start_plan = plan_to_first_frame(
            node=node,
            client=plan_path,
            request_type=GetMotionPlan,
            constraints_type=Constraints,
            joint_constraint_type=JointConstraint,
            current=current,
            first=first_solution,
            timeout_sec=args.service_timeout,
            velocity_scale=args.velocity_scale,
            acceleration_scale=args.acceleration_scale,
        )
        payload = {
            "artifact_type": "offline_bimanual_moveit_kdl_ik",
            "execution_enabled": False,
            "warning": "IK RESULT ONLY; no controller command is implemented",
            "solver": "kdl_kinematics_plugin/KDLKinematicsPlugin via /compute_ik",
            "source_plan": str(args.plan),
            "joint_names": list(ALL_JOINT_NAMES),
            "maximum_joint_step_rad": maximum_step,
            "samples": solved_samples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        args.start_plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.start_plan_output.write_text(
            json.dumps(start_plan, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {len(solved_samples)} MoveIt/KDL IK frames to {args.output}")
        print(f"Wrote PLAN-ONLY transition to frame 0 to {args.start_plan_output}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
