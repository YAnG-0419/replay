"""Compatibility imports for the previously uncommitted continuity experiment.

New programs should import eef_replay.ik. The legacy pinocchio_ik.py is untouched.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eef_replay.ik.continuous import ContinuousIK as _ContinuousIK
from eef_replay.ik.model import RobotModel, checked_transform, positive
from eef_replay.ik.transition import minimum_jerk_transition


class ContinuousIK(_ContinuousIK):
    def __init__(self, urdf_path, frame, active_indices):
        super().__init__(RobotModel(urdf_path), frame, active_indices)
        self.model = self.robot.model
        self.pin = self.robot.pin
        self.data = self.robot.data

    def _check_q(self, q, name):
        return self.robot.check_q(q, name)
