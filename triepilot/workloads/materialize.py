from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Iterable, Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _message_content(messages: Any, role: str) -> str:
    for message in messages or []:
        if message.get("role") == role or message.get("from") == role:
            content = message.get("content", message.get("value", ""))
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _prompt_from_row(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    content = _message_content(row.get("messages"), "user")
    if content:
        return content
    conversations = row.get("conversations") or row.get("conversation") or []
    if conversations:
        content = conversations[0].get("content", conversations[0].get("value", ""))
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _reference_from_row(row: dict[str, Any]) -> str:
    for key in ("reference", "completion", "answer", "output"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    content = _message_content(row.get("messages"), "assistant")
    if content:
        return content
    conversations = row.get("conversations") or row.get("conversation") or []
    if len(conversations) >= 2:
        content = conversations[1].get("content", conversations[1].get("value", ""))
        if isinstance(content, str) and content.strip():
            return content
    return "OK"


def materialize_workload_to_sharegpt(
    workload_rows: Iterable[dict[str, Any]],
    *,
    normalized_dir: str | Path,
    output_path: str | Path,
    max_rows: int | None = None,
    prompt_token_counter: Callable[[str], int] | None = None,
    max_prompt_tokens: int | None = None,
    fill_filtered: bool = False,
) -> list[dict[str, Any]]:
    if max_prompt_tokens is not None and prompt_token_counter is None:
        raise ValueError("prompt_token_counter is required with max_prompt_tokens")

    normalized_root = Path(normalized_dir)
    output = Path(output_path)
    cache: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []

    for workload_index, workload_row in enumerate(workload_rows):
        if max_rows is not None and len(rows) >= max_rows:
            break
        dataset = str(workload_row["dataset"])
        if dataset not in cache:
            source_path = normalized_root / f"{dataset}.jsonl"
            cache[dataset] = _read_jsonl(source_path)
            if not cache[dataset]:
                raise ValueError(f"No rows found in {source_path}")
        source_rows = cache[dataset]
        sample_index = int(workload_row.get("sample_index", workload_index))

        selected_index: int | None = None
        selected_prompt = ""
        selected_reference = ""
        candidate_offsets = range(len(source_rows)) if fill_filtered else range(1)
        for offset in candidate_offsets:
            candidate_index = (sample_index + offset) % len(source_rows)
            source_row = source_rows[candidate_index]
            prompt = _prompt_from_row(source_row)
            if not prompt:
                continue
            if (
                max_prompt_tokens is not None
                and prompt_token_counter is not None
                and prompt_token_counter(prompt) > max_prompt_tokens
            ):
                continue
            selected_index = candidate_index
            selected_prompt = prompt
            selected_reference = _reference_from_row(source_row)
            break

        if selected_index is None:
            continue

        rows.append(
            {
                "conversations": [
                    {"from": "human", "value": selected_prompt},
                    {"from": "gpt", "value": selected_reference},
                ],
                "triepilot_dataset": dataset,
                "triepilot_request_id": workload_row.get("request_id", ""),
                "triepilot_original_sample_index": sample_index,
                "triepilot_sample_index": selected_index,
            }
        )

    if not rows:
        raise ValueError("No usable prompts were materialized from the workload")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows
