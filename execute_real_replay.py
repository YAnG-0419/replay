"""Execute a validated bimanual replay on real FR3 + Sharpa hardware.

The MoveIt transition to frame zero uses an explicit velocity/acceleration scale.
The replay itself uses the recorded timestamps without speed scaling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any


SIDES = ("left", "right")
# The offline IK URDF uses the fr3v2 prefix, while the deployed ros2_control
# stack exposes fr3 joint names.  Joint indices are identical; keep the two
# contracts explicit so an artifact name is never sent to a hardware action.
ARTIFACT_ARM_NAMES = {
    side: tuple(f"{side}_fr3v2_joint{index}" for index in range(1, 8))
    for side in SIDES
}
ARM_NAMES = {
    side: tuple(f"{side}_fr3_joint{index}" for index in range(1, 8))
    for side in SIDES
}
ALL_ARTIFACT_ARM_NAMES = tuple(
    name for side in SIDES for name in ARTIFACT_ARM_NAMES[side]
)
ALL_ARM_NAMES = tuple(name for side in SIDES for name in ARM_NAMES[side])
HAND_NAMES = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_execution_gate(
    trajectory_path: Path,
    trajectory: dict[str, Any],
    validation: dict[str, Any],
    *,
    acknowledge_unvalidated_eef: bool,
    acknowledge_skipped_collision: bool,
) -> None:
    samples = trajectory.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("trajectory must contain at least two samples")
    times = [float(sample["t_s"]) for sample in samples]
    if any(right <= left for left, right in zip(times[:-1], times[1:], strict=True)):
        raise ValueError("trajectory timestamps are not strictly increasing")

    expected = validation.get("trajectory", {})
    expected_sha = expected.get("sha256")
    actual_sha = sha256_file(trajectory_path)
    if expected_sha is not None and actual_sha != expected_sha:
        raise ValueError(
            f"trajectory SHA256 mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    expected_frames = expected.get("frame_count")
    if expected_frames is not None and len(samples) != int(expected_frames):
        raise ValueError("trajectory frame count differs from validation artifact")
    duration = times[-1] - times[0]
    expected_duration = expected.get("duration_s")
    if expected_duration is not None and not math.isclose(
        duration, float(expected_duration), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("trajectory duration differs from validation artifact")
    expected_speed_scale = expected.get("speed_scale")
    actual_speed_scale = trajectory.get("retiming", {}).get("speed_scale")
    if expected_speed_scale is not None and not math.isclose(
        float(actual_speed_scale),
        float(expected_speed_scale),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("trajectory speed scale differs from validation artifact")

    if validation.get("passed") is True:
        return
    if validation.get("execution_waiver_eligible") is not True:
        raise ValueError("validation did not pass and is not eligible for explicit waivers")
    if validation.get("dynamics_passed") is not True:
        raise ValueError("joint position/velocity/acceleration dynamics did not pass")

    eef = validation.get("eef_validation", {})
    if eef.get("performed") is not False or eef.get("waived_by_operator") is not True:
        raise ValueError("EEF validation state is not an explicit operator waiver")
    if not acknowledge_unvalidated_eef:
        raise ValueError("add --acknowledge-unvalidated-eef to accept the recorded EEF waiver")

    collision = validation.get("collision_check", {})
    if (
        collision.get("performed") is not False
        or collision.get("waived_by_operator") is not True
    ):
        raise ValueError("collision-check state is not an explicit operator waiver")
    if not acknowledge_skipped_collision:
        raise ValueError("add --acknowledge-skipped-collision to accept the recorded collision waiver")

    start_plan = validation.get("current_state_to_frame0_plan", {})
    if start_plan.get("deferred_to_execute_time_moveit_plan") is not True:
        raise ValueError("current-state to frame-zero planning is not safely deferred")

    print(
        "EXPLICIT WAIVER GATE ACCEPTED: "
        f"sha256={actual_sha}, frames={len(samples)}, duration={duration:.9f}s; "
        "EEF validation and replay collision checking were waived by the operator."
    )


def wait_future(future: Any, timeout: float, label: str) -> Any:
    event = threading.Event()
    future.add_done_callback(lambda _future: event.set())
    if not event.wait(timeout):
        raise TimeoutError(f"{label} timed out")
    error = future.exception()
    if error is not None:
        raise RuntimeError(f"{label} failed: {error}")
    value = future.result()
    if value is None:
        raise RuntimeError(f"{label} returned no result")
    return value


class RealReplayExecutor:
    def __init__(self, trajectory: dict[str, Any], *, start_speed_scale: float) -> None:
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from controller_manager_msgs.srv import (
            ConfigureController,
            ListControllers,
            LoadController,
            SwitchController,
        )
        from moveit_msgs.srv import GetMotionPlan
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import JointState

        self.rclpy = rclpy
        self.FollowJointTrajectory = FollowJointTrajectory
        self.GetMotionPlan = GetMotionPlan
        self.JointState = JointState
        self.ConfigureController = ConfigureController
        self.ListControllers = ListControllers
        self.LoadController = LoadController
        self.SwitchController = SwitchController
        self.Time = Time
        self.trajectory = trajectory
        self.start_speed_scale = start_speed_scale
        self.positions: dict[str, float] = {}
        self._positions_lock = threading.Lock()
        self._active_handles: list[Any] = []

        rclpy.init(args=None)
        self.node = rclpy.create_node("eef_replay_real_executor")
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        def arm_state(message: Any) -> None:
            with self._positions_lock:
                for name, value in zip(message.name, message.position, strict=True):
                    if name in ALL_ARM_NAMES and math.isfinite(float(value)):
                        self.positions[str(name)] = float(value)

        self.subscriptions = [
            self.node.create_subscription(
                JointState,
                "/left/franka/joint_states",
                arm_state,
                qos_profile_sensor_data,
            ),
            self.node.create_subscription(
                JointState,
                "/right/franka/joint_states",
                arm_state,
                qos_profile_sensor_data,
            ),
            self.node.create_subscription(
                JointState, "/joint_states", arm_state, qos_profile_sensor_data
            ),
        ]
        self.hand_publishers = {
            side: self.node.create_publisher(JointState, f"/sharpa/{side}/command", 10)
            for side in SIDES
        }
        self.plan_client = self.node.create_client(
            GetMotionPlan, "/plan_kinematic_path"
        )
        self.controller_clients: dict[str, dict[str, Any]] = {}
        for side in SIDES:
            manager = f"/{side}/controller_manager"
            self.controller_clients[side] = {
                "list": self.node.create_client(
                    ListControllers, f"{manager}/list_controllers"
                ),
                "load": self.node.create_client(
                    LoadController, f"{manager}/load_controller"
                ),
                "configure": self.node.create_client(
                    ConfigureController, f"{manager}/configure_controller"
                ),
                "switch": self.node.create_client(
                    SwitchController, f"{manager}/switch_controller"
                ),
            }
        self.action_clients: dict[str, list[Any]] = {
            side: [
                ActionClient(
                    self.node,
                    FollowJointTrajectory,
                    f"/{side}/{side}_arm_controller/follow_joint_trajectory",
                ),
                ActionClient(
                    self.node,
                    FollowJointTrajectory,
                    f"/{side}_arm_controller/follow_joint_trajectory",
                ),
            ]
            for side in SIDES
        }

    def close(self) -> None:
        self.cancel_active()
        self.executor.shutdown(timeout_sec=2.0)
        self.node.destroy_node()
        self.rclpy.shutdown()
        self.spin_thread.join(timeout=2.0)

    def current_q(self, timeout: float = 10.0) -> list[float]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._positions_lock:
                if all(name in self.positions for name in ALL_ARM_NAMES):
                    return [self.positions[name] for name in ALL_ARM_NAMES]
            time.sleep(0.05)
        raise RuntimeError("dual-arm joint states are not available")

    def call(self, client: Any, request: Any, label: str, timeout: float = 15.0) -> Any:
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service unavailable: {label}")
        return wait_future(client.call_async(request), timeout, label)

    def controller_list(self, side: str) -> dict[str, Any]:
        response = self.call(
            self.controller_clients[side]["list"],
            self.ListControllers.Request(),
            f"{side} list_controllers",
        )
        return {controller.name: controller for controller in response.controller}

    def switch_to_trajectory(self, side: str) -> None:
        wanted = f"{side}_arm_controller"
        teach = f"{side}_teach_controller"
        clients = self.controller_clients[side]
        controllers = self.controller_list(side)
        if wanted not in controllers:
            request = self.LoadController.Request()
            request.name = wanted
            response = self.call(clients["load"], request, f"load {wanted}")
            if not response.ok:
                raise RuntimeError(f"could not load {wanted}")
            controllers = self.controller_list(side)
        if controllers[wanted].state == "unconfigured":
            request = self.ConfigureController.Request()
            request.name = wanted
            response = self.call(clients["configure"], request, f"configure {wanted}")
            if not response.ok:
                raise RuntimeError(f"could not configure {wanted}")
            controllers = self.controller_list(side)
        activate = [] if controllers[wanted].state == "active" else [wanted]
        deactivate = (
            [teach]
            if teach in controllers and controllers[teach].state == "active"
            else []
        )
        if activate or deactivate:
            request = self.SwitchController.Request()
            request.activate_controllers = activate
            request.deactivate_controllers = deactivate
            request.strictness = self.SwitchController.Request.STRICT
            request.activate_asap = True
            request.timeout.sec = 10
            response = self.call(clients["switch"], request, f"switch {side} trajectory")
            if not response.ok:
                raise RuntimeError(f"could not switch {side} to trajectory controller")

    def plan_to_first(self, current: list[float], target: list[float]) -> Any:
        from moveit_msgs.msg import Constraints, JointConstraint

        request = self.GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = "both_arms"
        motion.pipeline_id = "ompl"
        motion.planner_id = "RRTConnectkConfigDefault"
        motion.num_planning_attempts = 3
        motion.allowed_planning_time = 10.0
        motion.max_velocity_scaling_factor = self.start_speed_scale
        motion.max_acceleration_scaling_factor = self.start_speed_scale
        motion.start_state.is_diff = True
        motion.start_state.joint_state.name = list(ALL_ARM_NAMES)
        motion.start_state.joint_state.position = list(current)
        constraints = Constraints()
        constraints.name = "replay_frame_zero"
        for name, value in zip(ALL_ARM_NAMES, target, strict=True):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = float(value)
            joint.tolerance_above = 1e-4
            joint.tolerance_below = 1e-4
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        motion.goal_constraints = [constraints]
        response = self.call(
            self.plan_client,
            request,
            "MoveIt plan current state to replay frame zero",
            timeout=20.0,
        ).motion_plan_response
        if int(response.error_code.val) != 1:
            raise RuntimeError(
                f"MoveIt start plan failed: error_code={response.error_code.val}"
            )
        if not response.trajectory.joint_trajectory.points:
            raise RuntimeError("MoveIt start plan is empty")
        return response.trajectory.joint_trajectory

    def action_client(self, side: str) -> Any:
        for client in self.action_clients[side]:
            if client.wait_for_server(timeout_sec=0.2):
                return client
        raise RuntimeError(f"{side} FollowJointTrajectory action is unavailable")

    def split_moveit_goal(self, trajectory: Any, side: str) -> Any:
        from trajectory_msgs.msg import JointTrajectoryPoint

        source_names = tuple(str(name) for name in trajectory.joint_names)
        indices = [source_names.index(name) for name in ARM_NAMES[side]]
        goal = self.FollowJointTrajectory.Goal()
        goal.trajectory.header = trajectory.header
        goal.trajectory.joint_names = list(ARM_NAMES[side])
        for source in trajectory.points:
            point = JointTrajectoryPoint()
            point.time_from_start = source.time_from_start
            point.positions = [source.positions[index] for index in indices]
            if source.velocities:
                point.velocities = [source.velocities[index] for index in indices]
            if source.accelerations:
                point.accelerations = [source.accelerations[index] for index in indices]
            goal.trajectory.points.append(point)
        return goal

    def replay_goal(self, side: str, start_stamp: Any) -> Any:
        from trajectory_msgs.msg import JointTrajectoryPoint

        samples = self.trajectory["samples"]
        t0 = float(samples[0]["t_s"])
        key = f"{side}_arm_joint_positions_rad"
        goal = self.FollowJointTrajectory.Goal()
        goal.trajectory.header.stamp = start_stamp
        goal.trajectory.joint_names = list(ARM_NAMES[side])
        for sample in samples:
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in sample[key]]
            elapsed = float(sample["t_s"]) - t0 + 0.05
            total_ns = int(round(elapsed * 1_000_000_000))
            point.time_from_start.sec, point.time_from_start.nanosec = divmod(
                total_ns, 1_000_000_000
            )
            goal.trajectory.points.append(point)
        return goal

    def send_pair(self, goals: dict[str, Any], timeout: float) -> list[Any]:
        futures = {
            side: self.action_client(side).send_goal_async(goals[side])
            for side in SIDES
        }
        handles: list[Any] = []
        try:
            for side in SIDES:
                handle = wait_future(futures[side], 10.0, f"send {side} trajectory")
                if not handle.accepted:
                    raise RuntimeError(f"{side} controller rejected trajectory")
                handles.append(handle)
            self._active_handles = handles
            result_futures = [handle.get_result_async() for handle in handles]
            return [
                wait_future(future, timeout, f"wait {side} trajectory")
                for side, future in zip(SIDES, result_futures, strict=True)
            ]
        except BaseException:
            for handle in handles:
                handle.cancel_goal_async()
            raise
        finally:
            self._active_handles = []

    def send_pair_async(self, goals: dict[str, Any]) -> tuple[list[Any], list[Any]]:
        handles: list[Any] = []
        try:
            for side in SIDES:
                future = self.action_client(side).send_goal_async(goals[side])
                handle = wait_future(future, 10.0, f"send {side} replay")
                if not handle.accepted:
                    raise RuntimeError(f"{side} controller rejected replay")
                handles.append(handle)
            self._active_handles = handles
            return handles, [handle.get_result_async() for handle in handles]
        except BaseException:
            for handle in handles:
                handle.cancel_goal_async()
            raise

    def cancel_active(self) -> None:
        for handle in self._active_handles:
            try:
                handle.cancel_goal_async()
            except Exception:
                pass

    def publish_hand_sample(self, sample: dict[str, Any]) -> None:
        stamp = self.node.get_clock().now().to_msg()
        for side in SIDES:
            message = self.JointState()
            message.header.stamp = stamp
            message.header.frame_id = "eef_replay_episode395"
            message.name = list(HAND_NAMES)
            message.position = [
                float(value)
                for value in sample[f"{side}_hand_joint_positions_rad"]
            ]
            self.hand_publishers[side].publish(message)

    def require_hand_drivers(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(
                self.hand_publishers[side].get_subscription_count() > 0
                for side in SIDES
            ):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("both Sharpa command subscribers are required")
        for side in SIDES:
            other_publishers = [
                endpoint
                for endpoint in self.node.get_publishers_info_by_topic(
                    f"/sharpa/{side}/command"
                )
                if endpoint.node_name != self.node.get_name()
            ]
            if other_publishers:
                names = sorted({endpoint.node_name for endpoint in other_publishers})
                raise RuntimeError(
                    f"refusing duplicate {side} hand command publishers: {names}"
                )

    def execute(self) -> None:
        samples = self.trajectory["samples"]
        first = samples[0]
        first_q = (
            first["left_arm_joint_positions_rad"]
            + first["right_arm_joint_positions_rad"]
        )
        current = self.current_q()
        self.require_hand_drivers()
        start_plan = self.plan_to_first(current, first_q)
        for side in SIDES:
            self.switch_to_trajectory(side)
        self.publish_hand_sample(first)
        print(
            "Executing MoveIt transition to replay frame 0 at "
            f"{self.start_speed_scale:.6g}x velocity/acceleration scaling...",
            flush=True,
        )
        start_stamp_ns = self.node.get_clock().now().nanoseconds + 1_000_000_000
        start_plan.header.stamp = self.Time(nanoseconds=start_stamp_ns).to_msg()
        start_goals = {
            side: self.split_moveit_goal(start_plan, side) for side in SIDES
        }
        start_duration = (
            start_plan.points[-1].time_from_start.sec
            + start_plan.points[-1].time_from_start.nanosec * 1e-9
        )
        start_results = self.send_pair(start_goals, timeout=start_duration + 20.0)
        self._check_results(start_results, "start transition")

        start_ns = self.node.get_clock().now().nanoseconds + 1_000_000_000
        start_stamp = self.Time(nanoseconds=start_ns).to_msg()
        replay_goals = {
            side: self.replay_goal(side, start_stamp) for side in SIDES
        }
        handles, futures = self.send_pair_async(replay_goals)
        del handles
        speed_scale = float(self.trajectory.get("retiming", {}).get("speed_scale", 1.0))
        print(
            f"Executing replay artifact timing (speed_scale={speed_scale:.6g}x)...",
            flush=True,
        )
        monotonic_start = time.monotonic() + 1.0
        t0 = float(samples[0]["t_s"])
        for sample in samples:
            deadline = monotonic_start + float(sample["t_s"]) - t0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.005))
            self.publish_hand_sample(sample)
        results = [
            wait_future(future, 20.0, f"wait {side} replay result")
            for side, future in zip(SIDES, futures, strict=True)
        ]
        self._active_handles = []
        self._check_results(results, "replay")
        print("Replay complete; both arm controllers and hands hold the final frame.")

    def check_live(self) -> None:
        self.current_q()
        self.require_hand_drivers()
        for side in SIDES:
            wanted = f"{side}_arm_controller"
            controllers = self.controller_list(side)
            if wanted not in controllers or controllers[wanted].state != "active":
                raise RuntimeError(f"{wanted} is not active")
            self.action_client(side)
        if not self.plan_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("MoveIt planning service is unavailable")
        print(
            "LIVE READY: dual FR3 state/controllers/actions, MoveIt, and both "
            "Sharpa command subscribers are available; no command was sent."
        )

    def _check_results(self, results: list[Any], label: str) -> None:
        success = int(self.FollowJointTrajectory.Result.SUCCESSFUL)
        for side, wrapped in zip(SIDES, results, strict=True):
            if int(wrapped.result.error_code) != success:
                raise RuntimeError(
                    f"{label}: {side} controller error={wrapped.result.error_code} "
                    f"{wrapped.result.error_string}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check-live", action="store_true")
    parser.add_argument("--acknowledge-unvalidated-eef", action="store_true")
    parser.add_argument("--acknowledge-skipped-collision", action="store_true")
    parser.add_argument("--start-speed-scale", type=float, default=1.0)
    args = parser.parse_args()
    if not math.isfinite(args.start_speed_scale) or not 0.0 < args.start_speed_scale <= 1.0:
        raise ValueError("--start-speed-scale must be finite and in (0, 1]")
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if trajectory.get("execution_enabled") is not False:
        raise ValueError("trajectory is not an offline replay artifact")
    validate_execution_gate(
        args.trajectory,
        trajectory,
        validation,
        acknowledge_unvalidated_eef=args.acknowledge_unvalidated_eef,
        acknowledge_skipped_collision=args.acknowledge_skipped_collision,
    )
    if tuple(trajectory.get("joint_names", ())) != ALL_ARTIFACT_ARM_NAMES:
        raise ValueError("unexpected dual-arm joint order")
    if args.execute and args.check_live:
        raise ValueError("choose either --check-live or --execute")
    if not args.execute and not args.check_live:
        print("READY: trajectory and validation loaded; add --execute to command real hardware")
        return 0
    executor = RealReplayExecutor(
        trajectory,
        start_speed_scale=args.start_speed_scale,
    )
    try:
        if args.check_live:
            executor.check_live()
        else:
            executor.execute()
        return 0
    except KeyboardInterrupt:
        executor.cancel_active()
        print("Interrupted: cancel requested for both arm trajectories.")
        return 130
    finally:
        executor.close()


if __name__ == "__main__":
    raise SystemExit(main())
