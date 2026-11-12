"""PlaCo is optional; vendor integrity is always checked, numeric tests opt in."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
import numpy as np

import test_ik_backends as fixtures

ROOT = Path(__file__).resolve().parents[1]
HAS_PLACO = bool(fixtures.HAS_IK and importlib.util.find_spec("placo"))


class VendorTests(unittest.TestCase):
    def test_pinned_source_and_licenses_are_intact(self):
        vendor = ROOT / "third_party/placo"
        metadata = json.loads((vendor / "UPSTREAM.json").read_text())
        self.assertEqual(metadata["repository"], "https://github.com/Rhoban/placo")
        self.assertEqual(len(metadata["revision"]), 40)
        self.assertIn("LICENSE", metadata["files_sha256"])
        self.assertIn("thirdparty/qpmad/LICENSE", metadata["files_sha256"])
        for name, digest in metadata["files_sha256"].items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((vendor / name).read_bytes()).hexdigest(), digest)
        for name, target in metadata["symlinks"].items():
            self.assertTrue((vendor / name).is_symlink())
            self.assertEqual(os.readlink(vendor / name), target)
        self.assertFalse((vendor / ".git").exists())


@unittest.skipUnless(HAS_PLACO, "Optional PlaCo dependency unavailable")
class PlacoTests(unittest.TestCase):
    setUp = fixtures.BackendTests.setUp

    def test_seed_inactive_arm_failure_and_recovery(self):
        from eef_replay.ik.placo_backend import PlacoIK
        solver = PlacoIK(self.robot, "left_fr3v2_link8", range(7))
        desired = self.home.copy()
        desired[:7] += [.02, -.01, .01, 0, .02, -.01, .01]
        target = self.left.fk(desired)
        q, result = solver.solve_near(target, self.home, max_step=.08)
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(q[7:], self.home[7:])
        self.assertLessEqual(np.max(abs(q - self.home)), .08 + 1e-9)
        with self.assertRaises(RuntimeError):
            solver.solve_near(target, self.home, max_step=1e-5)
        recovered, _ = solver.solve_near(target, self.home, max_step=.08)
        np.testing.assert_allclose(recovered, q, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
