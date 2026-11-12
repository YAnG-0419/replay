"""Robot descriptions, independent of the numerical IK implementation."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path


@dataclass(frozen=True)
class ArmProfile:
    name: str
    joint_names: tuple[str, ...]
    base_frame: str
    eef_frame: str
    target_key: str
    payload_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class RobotProfile:
    name: str
    controlled_frame: str
    arms: tuple[ArmProfile, ...]

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint for arm in self.arms for joint in arm.joint_names)

    def indices(self, arm: ArmProfile) -> tuple[int, ...]:
        return tuple(self.joint_names.index(name) for name in arm.joint_names)

    @classmethod
    def from_dict(cls, document: dict) -> "RobotProfile":
        if document.get("schema_version") != 1:
            raise ValueError("Robot profile requires schema_version=1")
        if document.get("template", False):
            raise ValueError("Fill in the robot template and set template=false first")
        arms = tuple(
            ArmProfile(
                name=entry["name"], joint_names=tuple(entry["joint_names"]),
                base_frame=entry["base_frame"], eef_frame=entry["eef_frame"],
                target_key=entry.get("target_key", "base_T_link8_target"),
                payload_keys=tuple(entry.get("payload_keys", ())),
            ) for entry in document["arms"]
        )
        profile = cls(document["name"], document["controlled_frame"], arms)
        fields = [profile.name, profile.controlled_frame]
        for arm in arms:
            fields.extend([arm.name, arm.base_frame, arm.eef_frame, arm.target_key])
            fields.extend(arm.joint_names)
            fields.extend(arm.payload_keys)
        if not arms or any(not isinstance(s, str) or not s.strip() for s in fields):
            raise ValueError("Robot profile fields must be nonempty strings")
        if any(not arm.joint_names for arm in arms):
            raise ValueError("Each arm needs at least one joint")
        if len(set(a.name for a in arms)) != len(arms):
            raise ValueError("Arm names must be unique")
        if len(set(profile.joint_names)) != len(profile.joint_names):
            raise ValueError("Joint names must be unique across arms")
        return profile


def load_profile(name: str = "franka-sharpa", path: Path | None = None) -> RobotProfile:
    if path is not None:
        return RobotProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
    # Only installed, explicitly registered profiles are resolved by name.
    if name != "franka-sharpa":
        raise ValueError(f"Unknown robot {name!r}; supply --robot-config for another body")
    resource = files("eef_replay.ik").joinpath("profiles").joinpath(name + ".json")
    return RobotProfile.from_dict(json.loads(resource.read_text(encoding="utf-8")))

