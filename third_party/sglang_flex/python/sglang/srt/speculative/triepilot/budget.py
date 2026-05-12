from __future__ import annotations

from typing import Any, Iterable

TRIEPILOT_DRAFT_BUDGET_KEY = "triepilot_draft_budget"


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


def resolve_triepilot_draft_budgets(
    reqs: Iterable[Any],
    default_budget: int,
) -> tuple[list[int], list[int]]:
    """Resolve request-level NGRAM budgets from SamplingParams.custom_params.

    A requested budget of 0 means "no speculative draft nodes" but the target
    still needs one normal decode token, so the active verify length is 1.
    """
    max_budget = max(int(default_budget), 1)
    requested_budgets: list[int] = []
    active_lengths: list[int] = []

    for req in reqs:
        custom_params = _get_custom_params(req)
        raw_budget = custom_params.get(TRIEPILOT_DRAFT_BUDGET_KEY)
        budget = _coerce_budget(raw_budget, max_budget)
        budget = min(max(budget, 0), max_budget)
        requested_budgets.append(budget)
        active_lengths.append(max(budget, 1))

    return requested_budgets, active_lengths
