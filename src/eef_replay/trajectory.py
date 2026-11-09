from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .calibration import HandEyeCalibration, matrix_to_pose, pose_to_matrix
from .hand_mapping import SHARPA_HAND_JOINT_NAMES, canonical_hand_vector


def prepare_document(document: dict[str, Any], calibration: HandEyeCalibration) -> dict[str, Any]:
    if document.get("schema_version") != 1:
        raise ValueError("input schema_version must be 1")
    if document.get("reference_frame") != calibration.camera_frame:
        raise ValueError(f"reference_frame must be {calibration.camera_frame!r}")
    if document.get("translation_unit", "metre") != "metre":
        raise ValueError("translation_unit must be 'metre'")
    if document.get("pose_mode", "absolute") != "absolute":
        raise ValueError("only absolute EEF poses are supported")
    if document.get("controlled_frame") != "fr3_link8":
        raise ValueError("controlled_frame must be 'fr3_link8'; link7, flange and task TCP inputs are rejected")
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a non-empty list")

    output_samples: list[dict[str, Any]] = []
    previous_t = -np.inf
    for index, sample in enumerate(samples):
        t = float(sample["t_s"])
        if not np.isfinite(t) or t <= previous_t:
            raise ValueError(f"samples[{index}].t_s must be finite and strictly increasing")
        previous_t = t
        prepared: dict[str, Any] = {"t_s": t}
        for arm in ("left", "right"):
            arm_sample = sample.get(arm)
            if arm_sample is None:
                continue
            camera_T_link8 = pose_to_matrix(arm_sample["eef_pose"], f"samples[{index}].{arm}.eef_pose")
            base_T_link8 = calibration.camera_pose_to_base(arm, camera_T_link8)
            hand_map = arm_sample.get("hand_joints_rad")
            if not isinstance(hand_map, dict):
                raise ValueError(f"samples[{index}].{arm}.hand_joints_rad must be a name-to-value object")
            prepared[arm] = {
                "base_frame": f"{arm}_libfranka_O",
                "controlled_frame": "fr3_link8",
                "base_T_link8_target": matrix_to_pose(base_T_link8),
                "hand_joint_names": list(SHARPA_HAND_JOINT_NAMES),
                "hand_joint_positions_rad": canonical_hand_vector(hand_map),
            }
            source_arm = arm_sample.get("source_arm_joint_positions_rad")
            if source_arm is not None:
                source_arm_array = np.asarray(source_arm, dtype=float)
                if source_arm_array.shape != (7,) or not np.all(np.isfinite(source_arm_array)):
                    raise ValueError(f"samples[{index}].{arm}.source_arm_joint_positions_rad must have 7 finite values")
                prepared[arm]["source_arm_joint_positions_rad"] = source_arm_array.tolist()
        if len(prepared) == 1:
            raise ValueError(f"samples[{index}] contains neither left nor right arm")
        output_samples.append(prepared)

    return {
        "schema_version": 1,
        "artifact_type": "offline_replay_plan",
        "execution_enabled": False,
        "warning": "OFFLINE ARTIFACT ONLY; no controller or ROS command is implemented",
        "calibration": {
            "created_at": calibration.created_at,
            "source": calibration.source.name,
            "camera_frame": calibration.camera_frame,
        },
        "input_controlled_frame": "fr3_link8",
        "output_controlled_frame": "fr3_link8",
        "samples": output_samples,
    }


def detached_copy(document: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy for callers that want to annotate a plan safely."""
    return deepcopy(document)

