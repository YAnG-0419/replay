#!/usr/bin/env python3
"""Apply a uniform speed scale to an offline replay by retiming samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speed-scale", type=float, required=True)
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
    output_samples = []
    for sample, source_t in zip(source_samples, source_times, strict=True):
        retimed = dict(sample)
        retimed["t_s"] = t0 + (source_t - t0) / args.speed_scale
        output_samples.append(retimed)

    output = dict(source)
    output["artifact_type"] = "offline_bimanual_uniformly_retimed_replay"
    output["execution_enabled"] = False
    output["source_artifact"] = str(args.input)
    output["retiming"] = {
        "speed_scale": args.speed_scale,
        "time_scale": 1.0 / args.speed_scale,
        "source_duration_s": source_times[-1] - source_times[0],
        "output_duration_s": output_samples[-1]["t_s"] - output_samples[0]["t_s"],
        "joint_positions_unchanged": True,
        "hand_positions_unchanged": True,
    }
    output["samples"] = output_samples
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(output_samples)} OFFLINE samples at {args.speed_scale:.6g}x "
        f"speed ({output['retiming']['output_duration_s']:.9f}s) to {args.output}"
    )


if __name__ == "__main__":
    main()
