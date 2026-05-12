from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _request_row(dataset: str, seed: int, index: int, workload_type: str) -> dict[str, object]:
    return {
        "request_id": f"{dataset}-{seed}-{index:06d}",
        "dataset": dataset,
        "sample_index": index,
        "seed": seed,
        "workload_type": workload_type,
    }


def build_homogeneous(dataset: str, count: int, seed: int) -> list[dict[str, object]]:
    return [_request_row(dataset, seed, index, "homogeneous") for index in range(count)]


def build_binary_mix(
    left_dataset: str,
    right_dataset: str,
    left_ratio: float,
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    if not 0.0 <= left_ratio <= 1.0:
        raise ValueError("left_ratio must be between 0 and 1")

    left_count = round(count * left_ratio)
    right_count = count - left_count
    left_emitted = 0
    right_emitted = 0
    rows: list[dict[str, object]] = []
    for index in range(count):
        target_left = round((index + 1) * left_ratio)
        if left_emitted < left_count and target_left > left_emitted:
            dataset = left_dataset
            left_emitted += 1
        elif right_emitted < right_count:
            dataset = right_dataset
            right_emitted += 1
        else:
            dataset = left_dataset
            left_emitted += 1
        rows.append(_request_row(dataset, seed, index, "binary_mixed"))
    return rows


def build_shift_schedule(
    phases: list[str],
    per_phase: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for phase_index, dataset in enumerate(phases, start=1):
        for local_index in range(per_phase):
            global_index = len(rows)
            row = _request_row(dataset, seed, global_index, "workload_shift")
            row["phase"] = phase_index
            row["phase_sample_index"] = local_index
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TriePilot workload request lists.")
    parser.add_argument("--kind", choices=["homogeneous", "binary", "shift"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--dataset")
    parser.add_argument("--left-dataset")
    parser.add_argument("--right-dataset")
    parser.add_argument("--left-ratio", type=float, default=0.5)
    parser.add_argument("--phases", nargs="+")
    parser.add_argument("--per-phase", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.kind == "homogeneous":
        if not args.dataset:
            raise SystemExit("--dataset is required for homogeneous workloads")
        rows = build_homogeneous(args.dataset, args.count, args.seed)
    elif args.kind == "binary":
        if not args.left_dataset or not args.right_dataset:
            raise SystemExit("--left-dataset and --right-dataset are required for binary workloads")
        rows = build_binary_mix(
            args.left_dataset,
            args.right_dataset,
            args.left_ratio,
            args.count,
            args.seed,
        )
    else:
        if not args.phases:
            raise SystemExit("--phases is required for shift workloads")
        rows = build_shift_schedule(args.phases, args.per_phase, args.seed)
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
