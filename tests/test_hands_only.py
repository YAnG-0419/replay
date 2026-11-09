from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from execute_hands_only import HAND_NAMES, minimum_jerk, sha256_file, validate_artifacts


class HandsOnlyTests(unittest.TestCase):
    def test_minimum_jerk_endpoints(self) -> None:
        self.assertEqual(minimum_jerk(-1.0), 0.0)
        self.assertEqual(minimum_jerk(0.0), 0.0)
        self.assertAlmostEqual(minimum_jerk(0.5), 0.5)
        self.assertEqual(minimum_jerk(1.0), 1.0)
        self.assertEqual(minimum_jerk(2.0), 1.0)

    def test_artifact_hash_shape_and_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.json"
            hand = [0.0] * len(HAND_NAMES)
            trajectory = {
                "execution_enabled": False,
                "samples": [
                    {
                        "t_s": 0.0,
                        "left_hand_joint_positions_rad": hand,
                        "right_hand_joint_positions_rad": hand,
                    },
                    {
                        "t_s": 2.0,
                        "left_hand_joint_positions_rad": hand,
                        "right_hand_joint_positions_rad": hand,
                    },
                ],
            }
            path.write_text(json.dumps(trajectory), encoding="utf-8")
            validation = {
                "trajectory": {
                    "sha256": sha256_file(path),
                    "frame_count": 2,
                    "duration_s": 2.0,
                }
            }
            self.assertEqual(validate_artifacts(path, trajectory, validation), (2, 2.0))
            validation["trajectory"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_artifacts(path, trajectory, validation)


if __name__ == "__main__":
    unittest.main()
