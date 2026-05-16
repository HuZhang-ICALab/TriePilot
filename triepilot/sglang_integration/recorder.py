from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _to_plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    return [value]


def _feature_vector(
    structural_features: list[dict[str, Any]], key: str, batch_size: int
) -> list[float]:
    vector = []
    for index in range(batch_size):
        feature = (
            structural_features[index]
            if index < len(structural_features)
            and isinstance(structural_features[index], dict)
            else {}
        )
        value = feature.get(key, 0)
        vector.append(float(value) if isinstance(value, float) else int(value))
    return vector


def _mean(values: list[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _float_vector(values: Any) -> list[float]:
    result = []
    for value in _to_plain_list(values):
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            result.append(0.0)
    return result


def _int_vector(values: Any) -> list[int]:
    result = []
    for value in _to_plain_list(values):
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            result.append(0)
    return result


def _bool_vector(values: Any) -> list[bool]:
    return [bool(value) for value in _to_plain_list(values)]


class TriePilotNgramRecorder:
    def __init__(self, path: str | Path | None, run_id: str | None = None):
        self.path = Path(path).expanduser() if path else None
        self.run_id = run_id or "sglang-ngram"
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8", buffering=1)

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def record_step(
        self,
        *,
        step_id: int,
        batch_size: int,
        request_ids: Any,
        seq_lens: Any,
        draft_token_num: int,
        verify_draft_token_num: int | None = None,
        requested_draft_budgets: Any = None,
        bucketed_draft_budgets: Any = None,
        active_draft_lengths: Any = None,
        bucket_ids: Any = None,
        bucket_padding_nodes: Any = None,
        shape_padding_tokens: Any = None,
        accept_lens: Any,
        num_accepted_tokens: int,
        can_run_cuda_graph: bool,
        timings_ns: dict[str, int],
        structural_features: list[dict[str, Any]] | None = None,
        allocation_policy: str = "custom",
        batch_budget: int | None = None,
        accept_len_emas: Any = None,
        allocation_metadata: dict[str, Any] | None = None,
        controller_time_ns: int = 0,
    ) -> None:
        if self._fh is None:
            return

        budget_list = _to_plain_list(requested_draft_budgets)
        bucketed_budget_list = _to_plain_list(bucketed_draft_budgets)
        active_length_list = _to_plain_list(active_draft_lengths)
        if not budget_list:
            budget_list = [int(draft_token_num)] * int(batch_size)
        if not bucketed_budget_list:
            bucketed_budget_list = list(budget_list)
        if not active_length_list:
            active_length_list = [max(int(budget), 1) for budget in bucketed_budget_list]

        has_explicit_verify_shape = verify_draft_token_num is not None
        verify_draft_token_num = int(verify_draft_token_num or draft_token_num)
        allocated_budget = sum(int(budget) for budget in budget_list)
        actual_draft_nodes = sum(int(budget) for budget in bucketed_budget_list)
        if has_explicit_verify_shape and verify_draft_token_num > 0:
            verify_input_tokens = int(batch_size) * verify_draft_token_num
        else:
            verify_input_tokens = sum(int(length) for length in active_length_list)
        cuda_graph_expected_tokens = int(batch_size) * verify_draft_token_num
        accepted_tokens = int(num_accepted_tokens)
        structural_features = structural_features or []
        accept_len_ema_list = _to_plain_list(accept_len_emas)
        bucket_padding_node_list = _int_vector(bucket_padding_nodes)
        shape_padding_token_list = _int_vector(shape_padding_tokens)
        match_depths = _feature_vector(
            structural_features, "match_depth", int(batch_size)
        )
        candidate_counts = _feature_vector(
            structural_features, "candidate_count", int(batch_size)
        )
        branch_entropies = _feature_vector(
            structural_features, "branch_entropy", int(batch_size)
        )
        top_branch_ratios = _feature_vector(
            structural_features, "top_branch_ratio", int(batch_size)
        )
        filled_node_counts = _feature_vector(
            structural_features, "filled_nodes", int(batch_size)
        )
        allocation_metadata = allocation_metadata or {}
        strategy_bank_hits = _bool_vector(
            allocation_metadata.get("strategy_bank_hits", [])
        )
        request_local_probe_flags = _bool_vector(
            allocation_metadata.get("request_local_probe_flags", [])
        )
        event = {
            "time_ns": time.time_ns(),
            "event": "ngram_step",
            "run_id": self.run_id,
            "method": "sglang_ngram",
            "allocation_policy": allocation_policy,
            "step_id": int(step_id),
            "batch_size": int(batch_size),
            "request_ids": _to_plain_list(request_ids),
            "seq_lens": _to_plain_list(seq_lens),
            "batch_budget": batch_budget,
            "draft_token_num": int(draft_token_num),
            "verify_draft_token_num": verify_draft_token_num,
            "allocated_budget": allocated_budget,
            "allocated_budgets": budget_list,
            "bucketed_budgets": bucketed_budget_list,
            "active_draft_lengths": active_length_list,
            "bucket_ids": _to_plain_list(bucket_ids),
            "bucket_padding_nodes": bucket_padding_node_list,
            "bucket_padding_nodes_total": sum(bucket_padding_node_list),
            "shape_padding_tokens": shape_padding_token_list,
            "shape_padding_tokens_total": sum(shape_padding_token_list),
            "cuda_graph_expected_tokens": cuda_graph_expected_tokens,
            "cuda_graph_actual_tokens": verify_input_tokens,
            "cuda_graph_token_shape_ok": bool(
                verify_input_tokens == cuda_graph_expected_tokens
            ),
            "actual_draft_nodes": actual_draft_nodes,
            "verify_input_tokens": verify_input_tokens,
            "verified_nodes": actual_draft_nodes,
            "accepted_tokens": accepted_tokens,
            "wasted_nodes": max(actual_draft_nodes - accepted_tokens, 0),
            "accept_lens": _to_plain_list(accept_lens),
            "accept_len_emas": accept_len_ema_list,
            "accept_len_ema": _mean(
                [
                    float(value)
                    for value in accept_len_ema_list
                    if isinstance(value, (int, float))
                ]
            ),
            "match_depths": match_depths,
            "candidate_counts": candidate_counts,
            "branch_entropies": branch_entropies,
            "top_branch_ratios": top_branch_ratios,
            "filled_nodes": filled_node_counts,
            "match_depth": _mean(match_depths),
            "candidate_count": _mean(candidate_counts),
            "branch_entropy": _mean(branch_entropies),
            "top_branch_ratio": _mean(top_branch_ratios),
            "filled_nodes_mean": _mean(filled_node_counts),
            "can_run_cuda_graph": bool(can_run_cuda_graph),
            "regime_ids": _to_plain_list(allocation_metadata.get("regime_ids", [])),
            "ablation_modes": _to_plain_list(
                allocation_metadata.get("ablation_modes", [])
            ),
            "strategy_bank_hits": strategy_bank_hits,
            "strategy_bank_hit_rate": _mean(
                [1.0 if hit else 0.0 for hit in strategy_bank_hits]
            ),
            "expected_gain_per_node": _float_vector(
                allocation_metadata.get("expected_gain_per_node", [])
            ),
            "preferred_budgets": _int_vector(
                allocation_metadata.get("preferred_budgets", [])
            ),
            "budget_caps": _int_vector(allocation_metadata.get("budget_caps", [])),
            "marginal_upgrade_steps": _int_vector(
                allocation_metadata.get("marginal_upgrade_steps", [])
            ),
            "marginal_upgrade_gains": _float_vector(
                allocation_metadata.get("marginal_upgrade_gains", [])
            ),
            "recovery_probe_flags": _bool_vector(
                allocation_metadata.get("recovery_probe_flags", [])
            ),
            "recovery_probe_count": sum(
                1
                for flag in _bool_vector(
                    allocation_metadata.get("recovery_probe_flags", [])
                )
                if flag
            ),
            "request_local_probe_flags": request_local_probe_flags,
            "request_local_probe_count": sum(
                1 for flag in request_local_probe_flags if flag
            ),
            "request_local_probe_counts": _int_vector(
                allocation_metadata.get("request_local_probe_counts", [])
            ),
            "positive_observations": _int_vector(
                allocation_metadata.get("positive_observations", [])
            ),
            "max_observed_gain_per_node": _float_vector(
                allocation_metadata.get("max_observed_gain_per_node", [])
            ),
            "zero_gain_streaks": _int_vector(
                allocation_metadata.get("zero_gain_streaks", [])
            ),
            "strategy_confidences": _float_vector(
                allocation_metadata.get("strategy_confidences", [])
            ),
            "exploration_flags": _bool_vector(
                allocation_metadata.get("exploration_flags", [])
            ),
            "controller_time_us": int(controller_time_ns) / 1000.0,
            "ngram_query_time_us": timings_ns.get("ngram_query", 0) / 1000.0,
            "target_forward_time_us": timings_ns.get("target_forward", 0) / 1000.0,
            "verify_time_us": timings_ns.get("verify", 0) / 1000.0,
            "step_latency_us": timings_ns.get("step", 0) / 1000.0,
        }

        try:
            self._fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            logger.exception("Failed to write TriePilot NGRAM telemetry event.")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
