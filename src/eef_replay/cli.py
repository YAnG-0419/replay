from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import HandEyeCalibration
from .trajectory import prepare_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an OFFLINE EEF replay plan; never sends robot commands")
    parser.add_argument("input", type=Path, help="camera-frame EEF trajectory JSON")
    parser.add_argument("output", type=Path, help="offline base-frame plan JSON")
    parser.add_argument("--calibration", type=Path, default=Path("config/handeye_calibration.json"))
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    result = prepare_document(source, HandEyeCalibration.load(args.calibration))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote OFFLINE plan only: {args.output}")


if __name__ == "__main__":
    main()

