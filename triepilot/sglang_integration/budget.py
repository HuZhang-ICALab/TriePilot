from dataclasses import dataclass
import random
from typing import Any, Iterable

TRIEPILOT_DRAFT_BUDGET_KEY = "triepilot_draft_budget"
TRIEPILOT_ACCEPT_EMA_ATTR = "triepilot_accept_len_ema"
TRIEPILOT_NEGATIVE_GAIN_ATTR = "triepilot_negative_gain_count"
TRIEPILOT_DEFAULT_SHAPE_BUCKETS = (2, 4, 8, 16)

CUSTOM_POLICY_NAMES = {"", "custom", "custom_params", "per_request_custom"}
REQUEST_ALLOCATION_POLICIES = {
    "equal_budget_allocation",
    "random_budget_allocation",
    "match_depth_greedy",
    "accept_ema_greedy",
    "triepilot_allocation",
}


@dataclass
class StrategyRecord:
    regime_id: str
    preferred_budget: int
    safe_budget_set: tuple[int, ...]
    tree_shape_preference: str
    expected_gain_per_node: float
    fallback_budget: int
    confidence: float
    exploration_allowed: bool
    last_update_step: int = 0


def _safe_budget_set(max_budget: int) -> tuple[int, ...]:
    candidates = [0, 2, 4, 8, 16, 24, 32]
    values = sorted({min(max(int(value), 0), int(max_budget)) for value in candidates})
    if int(max_budget) not in values:
        values.append(int(max_budget))
    return tuple(values)


def _budget_floor(max_budget: int, value: int) -> int:
    safe_values = [budget for budget in _safe_budget_set(max_budget) if budget <= value]
    return max(safe_values) if safe_values else 0


def _infer_strategy_record(regime_id: str, max_budget: int) -> StrategyRecord:
    max_budget = max(int(max_budget), 1)
    safe_budget_set = _safe_budget_set(max_budget)
    fallback_budget = min(max_budget, 2)
    preferred_budget = fallback_budget
    expected_gain = 0.15
    confidence = 0.20
    exploration_allowed = True
    shape = "compact"

    if "low_match" in regime_id and "low_accept" in regime_id:
        preferred_budget = 0
        fallback_budget = 0
        expected_gain = 0.02
        shape = "off"
    elif (
        "high_match" in regime_id
        and "low_entropy" in regime_id
        and "high_accept" in regime_id
    ):
        preferred_budget = max_budget
        fallback_budget = _budget_floor(max_budget, max_budget // 2)
        expected_gain = 0.75
        confidence = 0.45
        exploration_allowed = False
        shape = "deep"
    elif "high_branch" in regime_id:
        preferred_budget = _budget_floor(max_budget, max(max_budget // 2, 2))
        fallback_budget = min(max_budget, 2)
        expected_gain = 0.35
        shape = "wide"
    elif "medium_match" in regime_id or "medium_accept" in regime_id:
        preferred_budget = _budget_floor(max_budget, max(max_budget // 2, 2))
        expected_gain = 0.25
        shape = "balanced"

    return StrategyRecord(
        regime_id=regime_id,
        preferred_budget=preferred_budget,
        safe_budget_set=safe_budget_set,
        tree_shape_preference=shape,
        expected_gain_per_node=expected_gain,
        fallback_budget=fallback_budget,
        confidence=confidence,
        exploration_allowed=exploration_allowed,
    )


class TriePilotStrategyBank:
    """Small online Strategy Bank for probability-free symbolic allocation."""

    def __init__(self, max_budget: int):
        self.max_budget = max(int(max_budget), 1)
        self.records: dict[str, StrategyRecord] = {}

    def lookup(self, regime_id: str) -> tuple[StrategyRecord, bool]:
        hit = regime_id in self.records
        if not hit:
            self.records[regime_id] = _infer_strategy_record(
                regime_id, self.max_budget
            )
        return self.records[regime_id], hit

    def observe(
        self,
        regime_id: str,
        *,
        allocated_budget: int,
        accepted_tokens: float,
        alpha: float,
        step_id: int = 0,
    ) -> StrategyRecord:
        record, _ = self.lookup(regime_id)
        allocated_budget = max(int(allocated_budget), 0)
        observed_gain = (
            float(accepted_tokens) / float(allocated_budget)
            if allocated_budget > 0
            else 0.0
        )
        alpha = min(1.0, max(0.0, float(alpha)))
        record.expected_gain_per_node = (
            alpha * observed_gain + (1.0 - alpha) * record.expected_gain_per_node
        )
        if allocated_budget > 0 and observed_gain >= 0.20:
            record.preferred_budget = min(self.max_budget, max(record.preferred_budget, allocated_budget))
        elif allocated_budget > 0 and observed_gain <= 0.01:
            record.preferred_budget = max(0, min(record.preferred_budget, allocated_budget // 2))
        record.confidence = min(1.0, record.confidence + 0.10)
        record.last_update_step = int(step_id)
        return record


def _get_custom_params(req: Any) -> dict[str, Any]:
    sampling_params = getattr(req, "sampling_params", None)
    custom_params = getattr(sampling_params, "custom_params", None)
    return custom_params if isinstance(custom_params, dict) else {}


def _coerce_budget(value: Any, default_budget: int) -> int:
    if value is None:
        return default_budget
    try:
        return int(value)
    except (TypeError, ValueError):
        return default_budget


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


def _effective_batch_budget(
    *,
    batch_budget: int | None,
    batch_size: int,
    default_budget: int,
) -> int:
    max_total = max(int(batch_size), 0) * max(int(default_budget), 0)
    if batch_budget is None or int(batch_budget) < 0:
        return max_total
    return min(max(int(batch_budget), 0), max_total)


def _active_lengths_from_budgets(budgets: list[int]) -> list[int]:
    return [max(int(budget), 1) for budget in budgets]


def parse_triepilot_shape_buckets(
    bucket_spec: str | Iterable[int] | None,
    *,
    max_budget: int,
) -> tuple[int, ...]:
    """Parse graph-compatible active lengths for shape-bucket verification."""
    max_budget = max(int(max_budget), 1)
    if bucket_spec is None:
        raw_values: list[Any] = list(TRIEPILOT_DEFAULT_SHAPE_BUCKETS)
    elif isinstance(bucket_spec, str):
        stripped = bucket_spec.strip()
        if not stripped or stripped.lower() in {"off", "none", "disabled", "false"}:
            return ()
        if stripped.lower() == "default":
            raw_values = list(TRIEPILOT_DEFAULT_SHAPE_BUCKETS)
        else:
            raw_values = [
                item.strip()
                for item in stripped.replace(";", ",").replace(" ", ",").split(",")
                if item.strip()
            ]
    else:
        raw_values = list(bucket_spec)

    values: set[int] = set()
    for raw_value in raw_values:
        if isinstance(raw_value, str) and raw_value in {"0/1", "0"}:
            continue
        else:
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
        value = min(max(value, 1), max_budget)
        values.add(value)

    return tuple(sorted(values))


def _ceil_to_bucket(value: int, buckets: tuple[int, ...], max_budget: int) -> int:
    if value <= 0:
        return 0
    for bucket in buckets:
        if bucket >= value:
            return min(int(bucket), int(max_budget))
    return int(max_budget)


def bucketize_triepilot_draft_budgets(
    requested_budgets: Iterable[Any],
    *,
    default_budget: int,
    bucket_spec: str | Iterable[int] | None = None,
) -> dict[str, list[Any]]:
    """Round request budgets to a small graph-compatible bucket set.

    Budget 0 keeps one active target token for normal decoding but contributes
    zero logical draft nodes.
    """
    max_budget = max(int(default_budget), 1)
    buckets = parse_triepilot_shape_buckets(bucket_spec, max_budget=max_budget)
    if not buckets:
        requested = [
            min(max(_coerce_budget(value, max_budget), 0), max_budget)
            for value in requested_budgets
        ]
        return {
            "requested_budgets": requested,
            "bucketed_budgets": requested,
            "active_draft_lengths": _active_lengths_from_budgets(requested),
            "bucket_ids": ["0/1" if budget <= 0 else str(budget) for budget in requested],
            "bucket_padding_nodes": [0] * len(requested),
        }

    requested_budgets_list: list[int] = []
    bucketed_budgets: list[int] = []
    active_draft_lengths: list[int] = []
    bucket_ids: list[str] = []
    bucket_padding_nodes: list[int] = []

    for raw_budget in requested_budgets:
        requested = min(max(_coerce_budget(raw_budget, max_budget), 0), max_budget)
        bucketed = _ceil_to_bucket(requested, buckets, max_budget)
        active_length = max(bucketed, 1)
        requested_budgets_list.append(requested)
        bucketed_budgets.append(bucketed)
        active_draft_lengths.append(active_length)
        bucket_ids.append("0/1" if bucketed <= 0 else str(bucketed))
        bucket_padding_nodes.append(max(bucketed - requested, 0))

    return {
        "requested_budgets": requested_budgets_list,
        "bucketed_budgets": bucketed_budgets,
        "active_draft_lengths": active_draft_lengths,
        "bucket_ids": bucket_ids,
        "bucket_padding_nodes": bucket_padding_nodes,
    }


def _allocate_equal(batch_size: int, max_budget: int, total_budget: int) -> list[int]:
    if batch_size <= 0:
        return []
    base = min(max_budget, total_budget // batch_size)
    budgets = [base] * batch_size
    remaining = total_budget - base * batch_size
    index = 0
    while remaining > 0 and index < batch_size:
        room = max_budget - budgets[index]
        take = 1 if room > 0 else 0
        budgets[index] += take
        remaining -= take
        index += 1
    return budgets


def _allocate_greedy(
    scores: list[float],
    *,
    max_budget: int,
    total_budget: int,
) -> list[int]:
    budgets = [0] * len(scores)
    remaining = total_budget
    ranked = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    for idx in ranked:
        if remaining <= 0:
            break
        take = min(max_budget, remaining)
        budgets[idx] = take
        remaining -= take
    return budgets


def _allocate_random(
    batch_size: int,
    *,
    max_budget: int,
    total_budget: int,
    random_seed: int,
    step_id: int,
) -> list[int]:
    budgets = [0] * batch_size
    remaining = total_budget
    rng = random.Random(int(random_seed) + int(step_id))
    indices = list(range(batch_size))
    rng.shuffle(indices)
    while remaining > 0 and any(budget < max_budget for budget in budgets):
        progressed = False
        for idx in indices:
            room = max_budget - budgets[idx]
            if room <= 0:
                continue
            take = rng.randint(1, min(room, remaining))
            budgets[idx] += take
            remaining -= take
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
        rng.shuffle(indices)
    return budgets


def _feature_score(
    structural_features: list[dict[str, Any]] | None,
    index: int,
    key: str,
) -> float:
    if not structural_features or index >= len(structural_features):
        return 0.0
    feature = structural_features[index]
    if not isinstance(feature, dict):
        return 0.0
    try:
        return float(feature.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _accept_ema(req: Any) -> float:
    try:
        return float(getattr(req, TRIEPILOT_ACCEPT_EMA_ATTR, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _negative_gain_count(req: Any) -> int:
    try:
        return int(getattr(req, TRIEPILOT_NEGATIVE_GAIN_ATTR, 0))
    except (TypeError, ValueError):
        return 0


def _bucket_match_depth(match_depth: float) -> str:
    if match_depth >= 4:
        return "high_match"
    if match_depth >= 2:
        return "medium_match"
    return "low_match"


def _bucket_branch(
    *,
    candidate_count: float,
    branch_entropy: float,
    top_branch_ratio: float,
) -> str:
    if candidate_count <= 1 or branch_entropy <= 0.25 or top_branch_ratio >= 0.75:
        return "low_entropy"
    if candidate_count >= 3 or branch_entropy >= 1.0:
        return "high_branch"
    return "medium_branch"


def _bucket_accept(accept_ema: float) -> str:
    if accept_ema >= 2.0:
        return "high_accept"
    if accept_ema >= 0.75:
        return "medium_accept"
    return "low_accept"


def _bucket_load(batch_size: int) -> str:
    if batch_size >= 3:
        return "high_load"
    if batch_size >= 2:
        return "medium_load"
    return "low_load"


def _encode_regime_id(
    *,
    req: Any,
    structural_features: list[dict[str, Any]] | None,
    index: int,
    batch_size: int,
) -> str:
    match_bucket = _bucket_match_depth(
        _feature_score(structural_features, index, "match_depth")
    )
    branch_bucket = _bucket_branch(
        candidate_count=_feature_score(structural_features, index, "candidate_count"),
        branch_entropy=_feature_score(structural_features, index, "branch_entropy"),
        top_branch_ratio=_feature_score(structural_features, index, "top_branch_ratio"),
    )
    accept_bucket = _bucket_accept(_accept_ema(req))
    load_bucket = _bucket_load(batch_size)
    return f"R_{match_bucket}_{branch_bucket}_{accept_bucket}_{load_bucket}"


def _blank_allocation_metadata(batch_size: int) -> dict[str, list[Any]]:
    return {
        "regime_ids": [""] * batch_size,
        "strategy_bank_hits": [False] * batch_size,
        "expected_gain_per_node": [0.0] * batch_size,
        "preferred_budgets": [0] * batch_size,
        "strategy_confidences": [0.0] * batch_size,
        "exploration_flags": [False] * batch_size,
    }


def _return_budget_result(
    budgets: list[int],
    active_lengths: list[int],
    metadata: dict[str, Any] | None,
    return_metadata: bool,
):
    if return_metadata:
        return budgets, active_lengths, metadata or _blank_allocation_metadata(len(budgets))
    return budgets, active_lengths


def _allocate_triepilot(
    reqs: list[Any],
    *,
    structural_features: list[dict[str, Any]] | None,
    strategy_bank: TriePilotStrategyBank,
    max_budget: int,
    total_budget: int,
) -> tuple[list[int], dict[str, Any]]:
    records: list[StrategyRecord] = []
    regime_ids: list[str] = []
    hits: list[bool] = []
    scores: list[float] = []
    preferred_budgets: list[int] = []

    for index, req in enumerate(reqs):
        regime_id = _encode_regime_id(
            req=req,
            structural_features=structural_features,
            index=index,
            batch_size=len(reqs),
        )
        record, hit = strategy_bank.lookup(regime_id)
        negative_penalty = min(0.25, 0.05 * _negative_gain_count(req))
        score = max(0.0, record.expected_gain_per_node - negative_penalty)
        regime_ids.append(regime_id)
        records.append(record)
        hits.append(hit)
        scores.append(score)
        preferred_budgets.append(min(max(int(record.preferred_budget), 0), max_budget))

    budgets = [0] * len(reqs)
    remaining = max(int(total_budget), 0)
    ranked = sorted(range(len(reqs)), key=lambda idx: (-scores[idx], idx))
    for idx in ranked:
        if remaining <= 0:
            break
        preferred = preferred_budgets[idx]
        if preferred <= 0 or scores[idx] <= 0.0:
            continue
        take = min(preferred, remaining, max_budget)
        budgets[idx] = take
        remaining -= take

    metadata = {
        "regime_ids": regime_ids,
        "strategy_bank_hits": hits,
        "expected_gain_per_node": scores,
        "preferred_budgets": preferred_budgets,
        "strategy_confidences": [record.confidence for record in records],
        "exploration_flags": [record.exploration_allowed for record in records],
    }
    return budgets, metadata


def resolve_triepilot_draft_budgets(
    reqs: Iterable[Any],
    default_budget: int,
    *,
    allocation_policy: str | None = "custom",
    batch_budget: int | None = None,
    structural_features: list[dict[str, Any]] | None = None,
    step_id: int = 0,
    random_seed: int = 0,
    strategy_bank: TriePilotStrategyBank | None = None,
    return_metadata: bool = False,
) -> tuple[list[int], list[int]] | tuple[list[int], list[int], dict[str, Any]]:
    """Resolve request-level NGRAM budgets.

    A requested budget of 0 means "no speculative draft nodes" but the target
    still needs one normal decode token, so the active verify length is 1.
    """
    req_list = list(reqs)
    max_budget = max(int(default_budget), 1)
    policy = (allocation_policy or "custom").strip()

    if policy not in CUSTOM_POLICY_NAMES and policy not in REQUEST_ALLOCATION_POLICIES:
        raise ValueError(f"Unsupported TriePilot allocation policy: {policy}")

    if policy in REQUEST_ALLOCATION_POLICIES:
        total_budget = _effective_batch_budget(
            batch_budget=batch_budget,
            batch_size=len(req_list),
            default_budget=max_budget,
        )
        if policy == "equal_budget_allocation":
            requested_budgets = _allocate_equal(len(req_list), max_budget, total_budget)
            metadata = _blank_allocation_metadata(len(req_list))
        elif policy == "random_budget_allocation":
            requested_budgets = _allocate_random(
                len(req_list),
                max_budget=max_budget,
                total_budget=total_budget,
                random_seed=random_seed,
                step_id=step_id,
            )
            metadata = _blank_allocation_metadata(len(req_list))
        elif policy == "match_depth_greedy":
            scores = [
                _feature_score(structural_features, index, "match_depth")
                for index in range(len(req_list))
            ]
            requested_budgets = _allocate_greedy(
                scores, max_budget=max_budget, total_budget=total_budget
            )
            metadata = _blank_allocation_metadata(len(req_list))
            metadata["expected_gain_per_node"] = scores
        elif policy == "accept_ema_greedy":
            scores = [_accept_ema(req) for req in req_list]
            requested_budgets = _allocate_greedy(
                scores, max_budget=max_budget, total_budget=total_budget
            )
            metadata = _blank_allocation_metadata(len(req_list))
            metadata["expected_gain_per_node"] = scores
        else:
            bank = strategy_bank or TriePilotStrategyBank(max_budget=max_budget)
            requested_budgets, metadata = _allocate_triepilot(
                req_list,
                structural_features=structural_features,
                strategy_bank=bank,
                max_budget=max_budget,
                total_budget=total_budget,
            )
        return _return_budget_result(
            requested_budgets,
            _active_lengths_from_budgets(requested_budgets),
            metadata,
            return_metadata,
        )

    requested_budgets: list[int] = []
    active_lengths: list[int] = []

    for req in req_list:
        custom_params = _get_custom_params(req)
        raw_budget = custom_params.get(TRIEPILOT_DRAFT_BUDGET_KEY)
        budget = _coerce_budget(raw_budget, max_budget)
        budget = min(max(budget, 0), max_budget)
        requested_budgets.append(budget)
        active_lengths.append(max(budget, 1))

    return _return_budget_result(
        requested_budgets,
        active_lengths,
        _blank_allocation_metadata(len(req_list)),
        return_metadata,
    )


def observe_triepilot_accept_lengths(
    reqs: Iterable[Any],
    accept_lens: Any,
    *,
    alpha: float = 0.20,
) -> list[float]:
    values = _to_plain_list(accept_lens)
    alpha = min(1.0, max(0.0, float(alpha)))
    updated: list[float] = []
    for index, req in enumerate(reqs):
        accepted = values[index] if index < len(values) else 0
        try:
            accepted_value = float(accepted)
        except (TypeError, ValueError):
            accepted_value = 0.0
        previous = _accept_ema(req)
        current = alpha * accepted_value + (1.0 - alpha) * previous
        setattr(req, TRIEPILOT_ACCEPT_EMA_ATTR, current)
        updated.append(current)
    return updated


def observe_triepilot_strategy_feedback(
    reqs: Iterable[Any],
    *,
    strategy_bank: TriePilotStrategyBank,
    allocation_metadata: dict[str, Any] | None,
    requested_draft_budgets: Any,
    accept_lens: Any,
    alpha: float = 0.20,
    step_id: int = 0,
) -> None:
    req_list = list(reqs)
    metadata = allocation_metadata or {}
    regime_ids = _to_plain_list(metadata.get("regime_ids"))
    budgets = _to_plain_list(requested_draft_budgets)
    accepted_values = _to_plain_list(accept_lens)

    for index, req in enumerate(req_list):
        regime_id = regime_ids[index] if index < len(regime_ids) else ""
        if not regime_id:
            continue
        try:
            allocated_budget = int(budgets[index]) if index < len(budgets) else 0
        except (TypeError, ValueError):
            allocated_budget = 0
        try:
            accepted = float(accepted_values[index]) if index < len(accepted_values) else 0.0
        except (TypeError, ValueError):
            accepted = 0.0
        strategy_bank.observe(
            str(regime_id),
            allocated_budget=allocated_budget,
            accepted_tokens=accepted,
            alpha=alpha,
            step_id=step_id,
        )
        previous_negative = _negative_gain_count(req)
        if allocated_budget > 0 and accepted <= 0:
            setattr(req, TRIEPILOT_NEGATIVE_GAIN_ATTR, previous_negative + 1)
        elif accepted > 0:
            setattr(req, TRIEPILOT_NEGATIVE_GAIN_ATTR, 0)
