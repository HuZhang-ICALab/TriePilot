from __future__ import annotations

import math
from typing import Any


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


def _matrix_slice(mask_values: list[Any], start: int, width: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for row_idx in range(width):
        row_start = start + row_idx * width
        row = mask_values[row_start : row_start + width]
        rows.append([1 if bool(value) else 0 for value in row])
    return rows


def _root_child_for_row(row: list[int], active_len: int) -> int | None:
    for idx in range(1, active_len):
        if row[idx]:
            return idx
    return None


def _feature_for_request(rows: list[list[int]], active_len: int) -> dict[str, float | int]:
    active_len = max(min(int(active_len), len(rows)), 0)
    if active_len <= 1:
        return {
            "match_depth": 0,
            "candidate_count": 0,
            "branch_entropy": 0.0,
            "top_branch_ratio": 0.0,
            "filled_nodes": active_len,
        }

    branch_counts: dict[int, int] = {}
    max_depth = 0
    for row_idx in range(active_len):
        row = rows[row_idx][:active_len]
        depth = max(sum(row) - 1, 0)
        max_depth = max(max_depth, depth)
        if row_idx == 0:
            continue
        root_child = _root_child_for_row(row, active_len)
        if root_child is not None:
            branch_counts[root_child] = branch_counts.get(root_child, 0) + 1

    branch_total = sum(branch_counts.values())
    if branch_total == 0:
        branch_entropy = 0.0
        top_branch_ratio = 0.0
    else:
        branch_entropy = -sum(
            (count / branch_total) * math.log2(count / branch_total)
            for count in branch_counts.values()
            if count > 0
        )
        top_branch_ratio = max(branch_counts.values()) / branch_total

    return {
        "match_depth": int(max_depth),
        "candidate_count": int(len(branch_counts)),
        "branch_entropy": float(branch_entropy),
        "top_branch_ratio": float(top_branch_ratio),
        "filled_nodes": int(active_len),
    }


def compute_ngram_tree_features(
    *,
    req_drafts: Any,
    mask: Any,
    draft_token_num: int,
    active_draft_lengths: Any = None,
) -> list[dict[str, float | int]]:
    """Summarize per-request NGRAM draft-tree shape from SGLang tree masks."""
    draft_values = _to_plain_list(req_drafts)
    mask_values = _to_plain_list(mask)
    width = int(draft_token_num)
    if width <= 0:
        return []

    batch_size = len(draft_values) // width
    active_lengths = _to_plain_list(active_draft_lengths)
    if not active_lengths:
        active_lengths = [width] * batch_size

    features: list[dict[str, float | int]] = []
    for req_idx in range(batch_size):
        mask_start = req_idx * width * width
        rows = _matrix_slice(mask_values, mask_start, width)
        active_len = active_lengths[req_idx] if req_idx < len(active_lengths) else width
        features.append(_feature_for_request(rows, int(active_len)))
    return features
