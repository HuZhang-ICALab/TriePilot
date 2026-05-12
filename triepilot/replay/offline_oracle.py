from __future__ import annotations

from typing import Any

from triepilot.controller.features import ControllerFeatures


def label_event(row: dict[str, Any]) -> str:
    if "label" in row:
        return str(row["label"])

    candidate_rewards = row.get("candidate_rewards")
    if isinstance(candidate_rewards, dict) and candidate_rewards:
        return str(max(candidate_rewards, key=candidate_rewards.get))

    f = ControllerFeatures.from_mapping(row)
    if f.kv_usage >= 0.88:
        return "off"
    if f.recent_accept_ema < 0.25 and f.mean_match_depth < 2:
        return "off"
    if f.recent_accept_ema >= 1.5 and f.mean_match_depth >= 5:
        return "medium"
    if f.recent_accept_ema >= 0.8 or f.mean_match_depth >= 3:
        return "small"
    return "tiny"


def reward_from_event(row: dict[str, Any], latency_scale_us: float = 1000.0) -> float:
    accepted = float(row.get("accepted_drafts_mean", 0.0))
    latency = float(row.get("step_latency_us", 0.0))
    overhead = float(row.get("controller_overhead_ns", 0.0)) / 1000.0
    return accepted - (latency + overhead) / latency_scale_us

