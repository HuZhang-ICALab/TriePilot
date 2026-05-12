from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def _number(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * percentile)
    return ordered[index]


def aggregate_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    materialized = list(rows)
    if not materialized:
        return {
            "count": 0,
            "mean_TPOT": 0.0,
            "p95_TPOT": 0.0,
            "p99_TPOT": 0.0,
            "accepted_per_verified_node": 0.0,
            "wasted_node_ratio": 0.0,
            "negative_speedup_ratio": 0.0,
            "controller_overhead_p50_us": 0.0,
            "controller_overhead_p99_us": 0.0,
            "strategy_bank_hit_rate": 0.0,
        }

    tpot = [_number(row, "TPOT", "tpot_ms", "tpot") for row in materialized]
    verified = sum(_number(row, "verified_nodes") for row in materialized)
    accepted = sum(_number(row, "accepted_tokens", "accepted_drafts_sum") for row in materialized)
    wasted = sum(_number(row, "wasted_nodes") for row in materialized)
    controller = [_number(row, "controller_time_us", "controller_overhead_us") for row in materialized]
    negative = [
        row
        for row in materialized
        if bool(row.get("negative_speedup")) or _number(row, "speedup") < 0
    ]
    hits = [row for row in materialized if bool(row.get("strategy_bank_hit"))]

    return {
        "count": len(materialized),
        "mean_TPOT": mean(tpot),
        "p95_TPOT": _percentile(tpot, 0.95),
        "p99_TPOT": _percentile(tpot, 0.99),
        "accepted_per_verified_node": accepted / verified if verified else 0.0,
        "wasted_node_ratio": wasted / verified if verified else 0.0,
        "negative_speedup_ratio": len(negative) / len(materialized),
        "controller_overhead_p50_us": _percentile(controller, 0.50),
        "controller_overhead_p99_us": _percentile(controller, 0.99),
        "strategy_bank_hit_rate": len(hits) / len(materialized),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]
