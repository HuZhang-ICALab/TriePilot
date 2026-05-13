from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triepilot.workloads.materialize import materialize_workload_to_sharegpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize TriePilot workload rows into a ShareGPT-style JSON file."
    )
    parser.add_argument("--workload", required=True)
    parser.add_argument("--normalized-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload_path = Path(args.workload)
    rows = [
        json.loads(line)
        for line in workload_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    materialized = materialize_workload_to_sharegpt(
        rows,
        normalized_dir=args.normalized_dir,
        output_path=args.output,
        max_rows=args.max_rows,
    )
    print(f"materialized {len(materialized)} requests to {args.output}")


if __name__ == "__main__":
    main()
