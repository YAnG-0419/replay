#!/usr/bin/env python3
"""Slice an offline replay artifact by timestamp without changing its samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--end-s", type=float, required=True)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("execution_enabled") is not False:
        raise ValueError("input must be an offline artifact")
    samples = source.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("input must contain samples")
    selected = [sample for sample in samples if float(sample["t_s"]) <= args.end_s]
    if not selected:
        raise ValueError("requested interval contains no samples")
    times = [float(sample["t_s"]) for sample in selected]
    if any(right <= left for left, right in zip(times[:-1], times[1:], strict=True)):
        raise ValueError("selected timestamps are not strictly increasing")

    output = dict(source)
    output["artifact_type"] = "offline_bimanual_pinocchio_ik_time_prefix"
    output["execution_enabled"] = False
    output["source_artifact"] = str(args.input)
    output["time_slice"] = {
        "requested_end_s": float(args.end_s),
        "actual_first_s": times[0],
        "actual_last_s": times[-1],
        "duration_s": times[-1] - times[0],
        "frame_count": len(selected),
        "timestamps_and_joint_positions_unchanged": True,
    }
    output["samples"] = selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(selected)} unchanged OFFLINE samples through "
        f"t={times[-1]:.9f}s to {args.output}"
    )


if __name__ == "__main__":
    main()
