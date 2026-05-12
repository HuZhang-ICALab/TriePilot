from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triepilot.workloads.builder import (
    build_binary_mix,
    build_homogeneous,
    build_shift_schedule,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create TriePilot workload request plans.")
    sub = parser.add_subparsers(dest="kind", required=True)

    hom = sub.add_parser("homogeneous")
    hom.add_argument("--dataset", required=True)
    hom.add_argument("--count", type=int, default=1000)
    hom.add_argument("--seed", type=int, default=20260512)
    hom.add_argument("--output", required=True)

    binary = sub.add_parser("binary")
    binary.add_argument("--left-dataset", required=True)
    binary.add_argument("--right-dataset", required=True)
    binary.add_argument("--left-ratio", type=float, required=True)
    binary.add_argument("--count", type=int, default=1200)
    binary.add_argument("--seed", type=int, default=20260512)
    binary.add_argument("--output", required=True)

    shift = sub.add_parser("shift")
    shift.add_argument("--phases", nargs="+", required=True)
    shift.add_argument("--per-phase", type=int, default=400)
    shift.add_argument("--seed", type=int, default=20260512)
    shift.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind == "homogeneous":
        rows = build_homogeneous(args.dataset, args.count, args.seed)
    elif args.kind == "binary":
        rows = build_binary_mix(
            args.left_dataset,
            args.right_dataset,
            args.left_ratio,
            args.count,
            args.seed,
        )
    else:
        rows = build_shift_schedule(args.phases, args.per_phase, args.seed)
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
