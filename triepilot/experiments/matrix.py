from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .registry import BASELINE_METHODS


def _slug(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def build_baseline_matrix(
    models: list[str],
    workloads: list[str],
    methods: list[str] | None = None,
    seed: int = 20260512,
    framework: str = "sglang",
) -> list[dict[str, object]]:
    method_names = methods or list(BASELINE_METHODS)
    rows: list[dict[str, object]] = []
    for model in models:
        for workload in workloads:
            for method in method_names:
                method_profile = BASELINE_METHODS[method]
                deployable = bool(method_profile["deployable"])
                rows.append(
                    {
                        "run_id": f"{_slug(model)}__{_slug(workload)}__{method}__seed{seed}",
                        "model": model,
                        "workload": workload,
                        "method": method,
                        "framework": framework,
                        "seed": seed,
                        "deployable": deployable,
                        "execution_mode": "online_serving" if deployable else "offline_replay",
                    }
                )
    return rows


def write_matrix_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TriePilot baseline run matrix.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--workloads", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--framework", default="sglang")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = build_baseline_matrix(
        models=args.models,
        workloads=args.workloads,
        methods=args.methods,
        seed=args.seed,
        framework=args.framework,
    )
    write_matrix_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
