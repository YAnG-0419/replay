from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

SHARPA_HAND_JOINT_NAMES = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)


def canonical_hand_vector(values: Mapping[str, float]) -> list[float]:
    missing = sorted(set(SHARPA_HAND_JOINT_NAMES) - set(values))
    extra = sorted(set(values) - set(SHARPA_HAND_JOINT_NAMES))
    if missing or extra:
        raise ValueError(f"hand joint names mismatch; missing={missing}, extra={extra}")
    vector = np.asarray([values[name] for name in SHARPA_HAND_JOINT_NAMES], dtype=float)
    if not np.all(np.isfinite(vector)):
        raise ValueError("hand joints contain non-finite values")
    return vector.tolist()


def named_hand_joints(values: Sequence[float]) -> dict[str, float]:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (len(SHARPA_HAND_JOINT_NAMES),) or not np.all(np.isfinite(vector)):
        raise ValueError("hand joint vector must contain 22 finite values")
    return dict(zip(SHARPA_HAND_JOINT_NAMES, map(float, vector), strict=True))

