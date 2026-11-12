"""Behavioral regression for optional solvers, profile mapping and IK gradients."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from eef_replay.ik.profiles import RobotProfile, load_profile
from test_continuous_ik import make_model

ROOT = Path(__file__).resolve().parents[1]
HAS_IK = bool(importlib.util.find_spec("pinocchio") and importlib.util.find_spec("scipy"))


class ProfileTests(unittest.TestCase):
    def test_builtin_profile_and_template_validation(self):
        profile = load_profile()
        self.assertEqual(profile.name, "franka-sharpa")
        self.assertEqual(len(profile.joint_names), 14)
        template = json.loads((ROOT / "config/robots/template.json").read_text())
        with self.assertRaisesRegex(ValueError, "template"):
            RobotProfile.from_dict(template)
        document = asdict(profile)
        document["schema_version"] = 1
        document["arms"][1]["joint_names"] = document["arms"][0]["joint_names"]
        with self.assertRaisesRegex(ValueError, "unique"):
            RobotProfile.from_dict(document)
        with self.assertRaisesRegex(ValueError, "Unknown robot"):
            load_profile("unconfigured-robot")

    def test_importing_profiles_does_not_load_optional_solvers(self):
        code = """
import sys
from eef_replay.ik.profiles import load_profile
from eef_replay.ik.backends import BACKENDS
assert load_profile().name == 'franka-sharpa'
assert not {'pinocchio', 'scipy', 'placo'} & set(sys.modules)
"""
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


@unittest.skipUnless(HAS_IK, "Optional IK dependencies unavailable")
class BackendTests(unittest.TestCase):
    def setUp(self):
        from eef_replay.ik.model import RobotModel
        from eef_replay.ik.continuous import ContinuousIK
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.urdf = self.root / "robot.urdf"
        make_model(self.urdf)
        self.profile = load_profile()
        self.robot = RobotModel(self.urdf, self.profile.joint_names)
        self.left = ContinuousIK(self.robot, self.profile.arms[0].eef_frame, range(7))
        self.home = np.tile([.2, .4, -.3, -1.8, .3, 2.5, .2], 2)

    def fixture(self, count=6):
        samples = []
        for index in range(count):
            q = self.home.copy()
            q[[0, 7]] += .01 + .002 * index
            sample = {"t_s": .05 * index}
            for arm in self.profile.arms:
                base = self.robot.fixed_base(arm.base_frame)
                target = np.linalg.inv(base) @ self.robot.fk(arm.eef_frame, q)
                sample[arm.name] = {
                    arm.target_key: {"matrix": target.tolist()},
                    "source_arm_joint_positions_rad": q[list(self.profile.indices(arm))].tolist(),
                    "hand_joint_positions_rad": [index * .001] * 22,
                }
            samples.append(sample)
        plan = {"execution_enabled": False, "output_controlled_frame": "fr3_link8", "samples": samples}
        state = {"execution_enabled": False, "joint_names": list(self.profile.joint_names),
                 "joint_positions_rad": self.home.tolist()}
        source, start = self.root / "plan.json", self.root / "state.json"
        source.write_text(json.dumps(plan))
        start.write_text(json.dumps(state))
        return source, start, plan

    def test_both_backends_track_with_velocity_caps_and_preserve_payloads(self):
        from eef_replay.ik.cli import parse_args, solve
        source, start, original = self.fixture()
        backends = ["continuous", "differential"]
        if importlib.util.find_spec("placo"):
            backends.append("placo")
        for backend in backends:
            with self.subTest(backend=backend):
                output = self.root / (backend + ".json")
                args = parse_args([str(source), str(output), "--urdf", str(self.urdf),
                                   "--start-state", str(start), "--backend", backend,
                                   "--velocity-scale", ".05"])
                payload = solve(args)
                self.assertEqual(payload["seed_strategy"], "previous_solved_frame")
                joints = np.array([s["left_arm_joint_positions_rad"] + s["right_arm_joint_positions_rad"]
                                   for s in payload["samples"]])
                self.assertLessEqual(np.max(abs(np.diff(joints, axis=0))), .005 + 1e-8)
                self.assertGreater(np.linalg.norm(joints[-1] - joints[0]), .001)
                for expected, actual, q in zip(original["samples"], payload["samples"], joints):
                    self.assertEqual(expected["t_s"], actual["t_s"])
                    for arm in self.profile.arms:
                        self.assertEqual(expected[arm.name]["hand_joint_positions_rad"], actual[arm.name + "_hand_joint_positions_rad"])
                        goal = self.robot.fixed_base(arm.base_frame) @ np.array(expected[arm.name][arm.target_key]["matrix"])
                        np.testing.assert_allclose(self.robot.fk(arm.eef_frame, q), goal, atol=1e-4)
                with self.assertRaises(FileExistsError):
                    solve(args)

    def test_differential_failure_does_not_export_partial_or_reuse_legacy_files(self):
        from eef_replay.ik.cli import parse_args, solve
        source, start, plan = self.fixture()
        plan["samples"][2]["right"]["base_T_link8_target"]["matrix"][0][3] = 10.
        source.write_text(json.dumps(plan))
        output = self.root / "failed.json"
        with self.assertRaisesRegex(RuntimeError, "frame 2"):
            solve(parse_args([str(source), str(output), "--urdf", str(self.urdf),
                              "--start-state", str(start), "--backend", "differential"]))
        self.assertFalse(output.exists())
        self.assertFalse((self.root / "failed_start_transition.json").exists())
        failure = json.loads(output.with_suffix(".failure.json").read_text())
        self.assertEqual(failure["failed_frame"], 2)
        self.assertFalse(failure["output_written"])

    def test_source_reference_is_explicit_and_missing_source_fails(self):
        from eef_replay.ik.cli import parse_args, solve
        source, start, plan = self.fixture()
        del plan["samples"][2]["left"]["source_arm_joint_positions_rad"]
        source.write_text(json.dumps(plan))
        output = self.root / "missing_source.json"
        with self.assertRaisesRegex(ValueError, "Source reference"):
            solve(parse_args([str(source), str(output), "--urdf", str(self.urdf),
                              "--start-state", str(start), "--reference", "source"]))
        self.assertFalse(output.exists())

    def test_placo_missing_dependency_does_not_fall_back_to_another_solver(self):
        from unittest.mock import patch
        from eef_replay.ik.backends import create_solver
        with patch.dict(sys.modules, {"placo": None}):
            with self.assertRaisesRegex(RuntimeError, "optional"):
                create_solver("placo", self.robot, "left_fr3v2_link8", range(7))

    def test_explicit_frame0_initializer_and_later_backend_are_recorded(self):
        from eef_replay.ik.cli import parse_args, solve
        source, start, _ = self.fixture(count=3)
        payload = solve(parse_args([str(source), str(self.root / "initialized.json"),
                                    "--urdf", str(self.urdf), "--start-state", str(start),
                                    "--backend", "differential", "--frame0-backend", "continuous"]))
        self.assertEqual(payload["frame0_backend"], "continuous")
        self.assertEqual([s["backend"] for s in payload["samples"]],
                         ["continuous", "differential", "differential"])

    def test_common_runner_rejects_a_backend_moving_other_arm_joints(self):
        from unittest.mock import patch
        from eef_replay.ik.cli import parse_args, solve
        from eef_replay.ik.model import IKResult
        class BadSolver:
            def solve_near(self, target, seed, **kwargs):
                changed = seed.copy()
                changed[7] += .001
                return changed, IKResult(True, 0, 0., 0., .001)
        source, start, _ = self.fixture(count=2)
        output = self.root / "bad_backend.json"
        with patch("eef_replay.ik.cli.create_solver", return_value=BadSolver()):
            with self.assertRaisesRegex(RuntimeError, "outside the current arm"):
                solve(parse_args([str(source), str(output), "--urdf", str(self.urdf), "--start-state", str(start)]))
        self.assertFalse(output.exists())

    def test_profile_joint_order_and_renamed_robot_are_respected(self):
        from eef_replay.ik.cli import parse_args, solve
        source, start, plan = self.fixture(count=2)
        document = asdict(self.profile)
        document.update(name="test-other-body", schema_version=1, controlled_frame="test-tool")
        order = []
        for arm in document["arms"]:
            arm["joint_names"] = list(reversed(arm["joint_names"]))
            order.extend(self.profile.joint_names.index(name) for name in arm["joint_names"])
        profile_path = self.root / "other.json"
        profile_path.write_text(json.dumps(document))
        plan["output_controlled_frame"] = "test-tool"
        source.write_text(json.dumps(plan))
        start.write_text(json.dumps({"execution_enabled": False,
                                    "joint_names": [self.profile.joint_names[i] for i in order],
                                    "joint_positions_rad": self.home[order].tolist()}))
        payload = solve(parse_args([str(source), str(self.root / "other_ik.json"), "--urdf", str(self.urdf),
                                    "--robot-config", str(profile_path), "--start-state", str(start)]))
        self.assertEqual(payload["robot_profile"], "test-other-body")
        for sample, target in zip(payload["samples"], plan["samples"]):
            reordered = np.array(sample["left_arm_joint_positions_rad"] + sample["right_arm_joint_positions_rad"])
            q = np.empty(14)
            q[order] = reordered
            desired = self.robot.fixed_base("left_fr3v2_link0") @ np.array(target["left"]["base_T_link8_target"]["matrix"])
            np.testing.assert_allclose(self.left.fk(q), desired, atol=1e-4)

    def test_six_active_joints_do_not_require_a_seven_joint_arm(self):
        from eef_replay.ik.continuous import ContinuousIK
        solver = ContinuousIK(self.robot, "left_fr3v2_link8", range(6))
        desired = self.home.copy()
        desired[:6] += [.002, -.001, .001, 0, .001, -.001]
        q, result = solver.solve_near(solver.fk(desired), self.home, max_step=.03, limit_weight=0)
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(q[6:], self.home[6:])

    def test_local_ik_sensitivity_matches_perturbed_ik_solutions(self):
        from eef_replay.ik.sensitivity import solution_jacobian
        desired = self.home.copy()
        desired[:7] += [.01, -.005, .005, 0, .005, -.003, .005]
        target = self.left.fk(desired)
        q, _ = self.left.solve_near(target, self.home, limit_weight=0, max_step=.2)
        derivative = solution_jacobian(self.left, target, self.home, q, limit_weight=0, max_step=.2)
        pin = self.robot.pin
        target_se3 = pin.SE3(target[:3, :3], target[:3, 3])
        epsilon = 2e-5
        for index in (0, 2, 4):
            delta = np.zeros(6)
            delta[index] = epsilon
            plus, _ = self.left.solve_near((target_se3 * pin.exp6(delta)).homogeneous, self.home, limit_weight=0, max_step=.2)
            minus, _ = self.left.solve_near((target_se3 * pin.exp6(-delta)).homogeneous, self.home, limit_weight=0, max_step=.2)
            np.testing.assert_allclose(derivative[:, index], (plus - minus) / (2 * epsilon), atol=2e-3, rtol=2e-3)
        np.testing.assert_array_equal(derivative[7:], 0)
        seed = q.copy()
        seed[0] -= .1
        with self.assertRaisesRegex(ValueError, "active bound"):
            solution_jacobian(self.left, target, seed, q, max_step=.1)

    def test_legacy_cli_works_when_optional_backend_imports_are_blocked(self):
        source, _, _ = self.fixture(count=2)
        output = self.root / "legacy.json"
        code = """
import importlib.abc, runpy, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scipy' or fullname == 'placo' or fullname.startswith('eef_replay.ik'):
            raise AssertionError('Legacy IK loaded optional dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
entry = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(__import__('pathlib').Path(entry).parent))
runpy.run_path(entry, run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", code,
                                 str(ROOT / "experimental_pinocchio_pyroki/solve_bimanual_plan.py"),
                                 str(source), str(output), "--urdf", str(self.urdf)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())["artifact_type"], "offline_bimanual_pinocchio_ik")


if __name__ == "__main__":
    unittest.main()
