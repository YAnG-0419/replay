from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from eef_replay.calibration import HandEyeCalibration, matrix_to_pose, pose_to_matrix
from eef_replay.hand_mapping import SHARPA_HAND_JOINT_NAMES, canonical_hand_vector
from eef_replay.trajectory import prepare_document

ROOT = Path(__file__).resolve().parents[1]


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = HandEyeCalibration.load(ROOT / "config" / "handeye_calibration.json")

    def test_quaternion_round_trip(self) -> None:
        pose = {"position_m": [0.1, -0.2, 0.3], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]}
        matrix = pose_to_matrix(pose)
        rebuilt = pose_to_matrix(matrix_to_pose(matrix))
        np.testing.assert_allclose(rebuilt, matrix, atol=1e-9)

    def test_calibration_inverse_round_trip(self) -> None:
        for arm in ("left", "right"):
            np.testing.assert_allclose(
                self.calibration.base_T_camera(arm) @ self.calibration.camera_T_base(arm),
                np.eye(4),
                atol=1e-6,
            )

    def test_hand_mapping_is_name_based(self) -> None:
        values = {name: float(index) for index, name in reversed(list(enumerate(SHARPA_HAND_JOINT_NAMES)))}
        self.assertEqual(canonical_hand_vector(values), list(map(float, range(22))))

    def test_prepare_synthetic_document(self) -> None:
        with (ROOT / "examples" / "synthetic_camera_eef.json").open(encoding="utf-8") as handle:
            source = json.load(handle)
        plan = prepare_document(source, self.calibration)
        self.assertIs(plan["execution_enabled"], False)
        self.assertEqual(len(plan["samples"]), 2)
        for arm in ("left", "right"):
            first = plan["samples"][0][arm]
            self.assertEqual(len(first["hand_joint_positions_rad"]), 22)
            self.assertEqual(first["controlled_frame"], "fr3_link8")
            self.assertEqual(np.asarray(first["base_T_link8_target"]["matrix"]).shape, (4, 4))

    def test_non_link8_frame_is_rejected(self) -> None:
        hand = {name: 0.0 for name in SHARPA_HAND_JOINT_NAMES}
        source = {
            "schema_version": 1,
            "reference_frame": "camera_color_optical_frame",
            "controlled_frame": "flange",
            "samples": [{
                "t_s": 0.0,
                "left": {
                    "eef_pose": {"position_m": [0, 0, 1], "quaternion_wxyz": [1, 0, 0, 0]},
                    "hand_joints_rad": hand,
                },
            }],
        }
        with self.assertRaisesRegex(ValueError, "controlled_frame must be 'fr3_link8'"):
            prepare_document(source, self.calibration)


if __name__ == "__main__":
    unittest.main()

