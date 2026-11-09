#!/usr/bin/env python3
"""Package only Sharpa hand samples, with no arm fields, at a chosen speed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


HAND_NAMES = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--speed-scale", type=float, default=0.5)
    args = parser.parse_args()
    if not math.isfinite(args.speed_scale) or not 0.0 < args.speed_scale <= 1.0:
        raise ValueError("--speed-scale must be finite and in (0, 1]")

    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("execution_enabled") is not False:
        raise ValueError("input must be an offline artifact")
    source_samples = source.get("samples")
    if not isinstance(source_samples, list) or len(source_samples) < 2:
        raise ValueError("input must contain at least two samples")
    source_times = [float(sample["t_s"]) for sample in source_samples]
    if any(
        right <= left
        for left, right in zip(source_times[:-1], source_times[1:], strict=True)
    ):
        raise ValueError("input timestamps must be strictly increasing")

    t0 = source_times[0]
    samples = []
    for index, (sample, source_t) in enumerate(
        zip(source_samples, source_times, strict=True)
    ):
        output_sample: dict[str, object] = {
            "t_s": t0 + (source_t - t0) / args.speed_scale
        }
        for side in ("left", "right"):
            key = f"{side}_hand_joint_positions_rad"
            values = [float(value) for value in sample[key]]
            if len(values) != len(HAND_NAMES) or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError(f"sample {index} {side} hand is not 22 finite joints")
            output_sample[key] = values
        samples.append(output_sample)

    duration = float(samples[-1]["t_s"]) - float(samples[0]["t_s"])
    payload = {
        "artifact_type": "offline_sharpa_hands_only_replay",
        "execution_enabled": False,
        "warning": "HANDS ONLY: this artifact contains no FR3 arm joint fields",
        "source_artifact": str(args.input),
        "hand_joint_names": list(HAND_NAMES),
        "retiming": {
            "speed_scale": args.speed_scale,
            "time_scale": 1.0 / args.speed_scale,
            "source_duration_s": source_times[-1] - source_times[0],
            "output_duration_s": duration,
        },
        "samples": samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_dir / "pyroki_smoothed.json"
    trajectory_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    trajectory_sha = sha256(trajectory_path)
    validation = {
        "artifact_type": "offline_hands_only_package_manifest",
        "execution_enabled": False,
        "passed": True,
        "arms_present": False,
        "trajectory": {
            "sha256": trajectory_sha,
            "frame_count": len(samples),
            "duration_s": duration,
            "speed_scale": args.speed_scale,
        },
    }
    (args.output_dir / "pyroki_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Packaged HANDS-ONLY replay: frames={len(samples)}, "
        f"duration={duration:.9f}s, speed={args.speed_scale:.6g}x, "
        f"sha256={trajectory_sha}"
    )


if __name__ == "__main__":
    main()
