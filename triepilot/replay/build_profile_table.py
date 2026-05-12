from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def build_profile_table(path: str | Path) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(path):
        buckets[str(row.get("tier", "unknown"))].append(row)

    table: dict[str, dict[str, Any]] = {}
    for tier, rows in buckets.items():
        table[tier] = {
            "count": len(rows),
            "accepted_drafts_mean": mean(
                float(row.get("accepted_drafts_mean", 0.0)) for row in rows
            ),
            "step_latency_us_mean": mean(
                float(row.get("step_latency_us", 0.0)) for row in rows
            ),
            "controller_overhead_ns_mean": mean(
                float(row.get("controller_overhead_ns", 0.0)) for row in rows
            ),
        }
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--output")
    args = parser.parse_args()

    table = build_profile_table(args.trace)
    text = json.dumps(table, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

