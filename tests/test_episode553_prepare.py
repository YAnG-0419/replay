from __future__ import annotations

import unittest

import numpy as np

from tools.prepare_episode553_human_replay import (
    LINK8_T_SHARPA_ROOT,
    continuous_poses,
    rotvec_to_rotation,
)


class Episode553PreparationTests(unittest.TestCase):
    def test_link8_mount_round_trip_uses_full_transform(self) -> None:
        camera_T_root = np.eye(4)
        camera_T_root[:3, :3] = rotvec_to_rotation(np.array([0.4, -0.2, 0.7]))
        camera_T_root[:3, 3] = [0.2, -0.1, 0.6]
        for side in ("left", "right"):
            camera_T_link8 = camera_T_root @ np.linalg.inv(LINK8_T_SHARPA_ROOT[side])
            recovered = camera_T_link8 @ LINK8_T_SHARPA_ROOT[side]
            np.testing.assert_allclose(recovered, camera_T_root, atol=1e-12)

    def test_output_quaternion_sign_is_continuous_across_pi(self) -> None:
        matrices = np.repeat(np.eye(4)[None, :, :], 3, axis=0)
        for index, degrees in enumerate((179.0, 180.0, 181.0)):
            matrices[index, :3, :3] = rotvec_to_rotation(
                np.array([0.0, 0.0, np.deg2rad(degrees)])
            )
        quaternions = np.asarray(
            [pose["quaternion_wxyz"] for pose in continuous_poses(matrices)]
        )
        self.assertTrue(np.all(np.sum(quaternions[:-1] * quaternions[1:], axis=1) >= 0.0))


if __name__ == "__main__":
    unittest.main()
