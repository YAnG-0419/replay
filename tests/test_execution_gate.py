from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from execute_real_replay import sha256_file, validate_execution_gate


class ExplicitWaiverGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "trajectory.json"
        self.trajectory = {
            "execution_enabled": False,
            "samples": [{"t_s": 0.0}, {"t_s": 2.0}],
        }
        self.path.write_text(json.dumps(self.trajectory), encoding="utf-8")
        self.validation = {
            "passed": False,
            "execution_waiver_eligible": True,
            "dynamics_passed": True,
            "trajectory": {
                "sha256": sha256_file(self.path),
                "frame_count": 2,
                "duration_s": 2.0,
            },
            "eef_validation": {"performed": False, "waived_by_operator": True},
            "collision_check": {"performed": False, "waived_by_operator": True},
            "current_state_to_frame0_plan": {
                "deferred_to_execute_time_moveit_plan": True
            },
        }

    def test_both_explicit_acknowledgements_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "acknowledge-unvalidated-eef"):
            validate_execution_gate(
                self.path,
                self.trajectory,
                self.validation,
                acknowledge_unvalidated_eef=False,
                acknowledge_skipped_collision=True,
            )
        with self.assertRaisesRegex(ValueError, "acknowledge-skipped-collision"):
            validate_execution_gate(
                self.path,
                self.trajectory,
                self.validation,
                acknowledge_unvalidated_eef=True,
                acknowledge_skipped_collision=False,
            )

    def test_matching_artifact_and_both_acknowledgements_pass(self) -> None:
        validate_execution_gate(
            self.path,
            self.trajectory,
            self.validation,
            acknowledge_unvalidated_eef=True,
            acknowledge_skipped_collision=True,
        )

    def test_hash_mismatch_is_rejected(self) -> None:
        self.validation["trajectory"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_execution_gate(
                self.path,
                self.trajectory,
                self.validation,
                acknowledge_unvalidated_eef=True,
                acknowledge_skipped_collision=True,
            )


if __name__ == "__main__":
    unittest.main()
