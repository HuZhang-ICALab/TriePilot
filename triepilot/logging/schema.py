from __future__ import annotations

from typing import Any

REQUIRED_EVENT_KEYS = {
    "run_id",
    "step_id",
    "model",
    "device",
    "tier",
    "features",
    "draft_tokens",
    "ngram_match_window",
    "ngram_bfs_breadth",
    "accepted_drafts_sum",
    "accepted_drafts_mean",
    "step_latency_us",
    "controller_overhead_ns",
}

REQUIRED_FEATURE_KEYS = {
    "batch_size",
    "queue_len",
    "kv_usage",
    "seq_len_mean",
    "recent_accept_ema",
    "mean_match_depth",
}


def validate_trace_event(event: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_EVENT_KEYS - set(event))
    if missing:
        errors.append(f"missing event keys: {', '.join(missing)}")

    features = event.get("features")
    if not isinstance(features, dict):
        errors.append("features must be an object")
        return errors

    missing_features = sorted(REQUIRED_FEATURE_KEYS - set(features))
    if missing_features:
        errors.append(f"missing feature keys: {', '.join(missing_features)}")

    if event.get("tier") == "off" and int(event.get("draft_tokens", 0)) != 0:
        errors.append("off tier must have draft_tokens == 0")

    return errors

