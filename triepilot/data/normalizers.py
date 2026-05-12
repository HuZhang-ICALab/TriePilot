from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable


NormalizedRow = dict[str, Any]
Normalizer = Callable[[dict[str, Any], str, int, int], NormalizedRow]


def _sample_id(dataset: str, seed: int, index: int) -> str:
    return f"{dataset}-{seed}-{index:06d}"


def _base_row(dataset: str, index: int, seed: int, prompt: str, reference: str) -> NormalizedRow:
    return {
        "dataset": dataset,
        "sample_id": _sample_id(dataset, seed, index),
        "sample_index": index,
        "seed": seed,
        "prompt": prompt.strip(),
        "reference": reference.strip(),
    }


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def normalize_sharegpt_record(
    record: dict[str, Any],
    dataset: str,
    index: int,
    seed: int,
) -> NormalizedRow:
    conversations = record.get("conversations") or []
    messages: list[dict[str, str]] = []
    first_user = ""
    first_assistant = ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        raw_role = str(turn.get("from") or turn.get("role") or "")
        content = str(turn.get("value") or turn.get("content") or "")
        role = "assistant" if raw_role in {"gpt", "assistant"} else "user"
        messages.append({"role": role, "content": content})
        if role == "user" and not first_user:
            first_user = content
        if role == "assistant" and not first_assistant:
            first_assistant = content

    row = _base_row(dataset, index, seed, first_user, first_assistant)
    row["messages"] = messages
    return row


def normalize_gsm8k_record(
    record: dict[str, Any],
    dataset: str,
    index: int,
    seed: int,
) -> NormalizedRow:
    question = str(record.get("question") or record.get("problem") or "")
    answer = str(record.get("answer") or record.get("solution") or "")
    prompt = f"Solve the math problem. Show the reasoning and final answer.\n\nProblem:\n{question}"
    return _base_row(dataset, index, seed, prompt, answer)


def normalize_cnn_dailymail_record(
    record: dict[str, Any],
    dataset: str,
    index: int,
    seed: int,
) -> NormalizedRow:
    article = str(record.get("article") or record.get("document") or record.get("text") or "")
    highlights = str(record.get("highlights") or record.get("summary") or "")
    prompt = f"Summarize the following article in a concise paragraph.\n\nArticle:\n{article}"
    return _base_row(dataset, index, seed, prompt, highlights)


def normalize_json_tool_record(
    record: dict[str, Any],
    dataset: str,
    index: int,
    seed: int,
) -> NormalizedRow:
    schema = record.get("target_schema") or record.get("schema") or record.get("json_schema")
    prompt = str(record.get("prompt") or record.get("question") or record.get("instruction") or "")
    if not prompt and schema is not None:
        prompt = (
            "Generate one valid JSON object that satisfies the following JSON schema. "
            "Return only JSON.\n\nSchema:\n"
            f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
        )
    reference = str(record.get("answer") or record.get("response") or record.get("output") or "")
    row = _base_row(dataset, index, seed, prompt, reference)
    row["metadata"] = {
        ("target_schema" if key == "json_schema" else key): record[key]
        for key in ("target_schema", "schema", "tools", "function", "ground_truth")
        if key in record
    }
    if "json_schema" in record:
        row["metadata"]["target_schema"] = record["json_schema"]
    return row


def normalize_code_edit_record(
    record: dict[str, Any],
    dataset: str,
    index: int,
    seed: int,
) -> NormalizedRow:
    instruction = str(record.get("instruction") or record.get("prompt") or record.get("question") or "")
    code_input = str(record.get("input") or record.get("code") or record.get("before") or "")
    reference = str(record.get("output") or record.get("answer") or record.get("after") or "")
    prompt = f"{instruction}\n\nCode:\n{code_input}".strip()
    return _base_row(dataset, index, seed, prompt, reference)


def normalize_qa_with_docs_record(
    record: dict[str, Any],
    dataset: str,
    index: int,
    seed: int,
) -> NormalizedRow:
    question = str(record.get("question") or "")
    docs = record.get("docs_sorted") or record.get("docs_orig") or []
    docs_text = "\n\n".join(str(doc) for doc in docs)
    reference = _first_text(record.get("answers") or record.get("answer"))
    prompt = (
        "Answer the question based on the given passages. "
        "Only give the answer and do not output unrelated words.\n\n"
        f"Passages:\n{docs_text}\n\nQuestion:\n{question}"
    )
    row = _base_row(dataset, index, seed, prompt, reference)
    row["metadata"] = {
        key: record[key]
        for key in (
            "source_dataset",
            "source_index",
            "reuse_stream_meta",
            "exact_prefix_reuse_rep",
            "exact_prefix_reuse_source_index",
        )
        if key in record
    }
    return row


NORMALIZERS: dict[str, Normalizer] = {
    "sharegpt": normalize_sharegpt_record,
    "gsm8k": normalize_gsm8k_record,
    "cnn_dailymail": normalize_cnn_dailymail_record,
    "json_tool": normalize_json_tool_record,
    "instructcoder": normalize_code_edit_record,
    "rag_qa": normalize_qa_with_docs_record,
    "shared_prefix": normalize_qa_with_docs_record,
}


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"expected list in {path}")
        return [row for row in data if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_parquet_records(path: str | Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"pandas is required to load parquet: {exc}") from exc
    frame = pd.read_parquet(path)
    return [dict(row) for row in frame.to_dict(orient="records")]


def load_records(path: str | Path, input_format: str = "auto") -> list[dict[str, Any]]:
    resolved_format = input_format
    if input_format == "auto":
        suffix = Path(path).suffix.lower()
        resolved_format = "parquet" if suffix == ".parquet" else "json"
    if resolved_format == "json":
        return load_json_records(path)
    if resolved_format == "parquet":
        return load_parquet_records(path)
    raise ValueError(f"unsupported input format: {input_format}")


def normalize_records(
    records: Iterable[dict[str, Any]],
    dataset: str,
    seed: int,
    normalizer: Normalizer,
    limit: int | None = None,
) -> list[NormalizedRow]:
    rows: list[NormalizedRow] = []
    for index, record in enumerate(records):
        if limit is not None and len(rows) >= limit:
            break
        row = normalizer(record, dataset, index, seed)
        if row.get("prompt"):
            rows.append(row)
    return rows


def write_normalized_dataset(
    output_path: str | Path,
    sample_ids_path: str | Path,
    rows: Iterable[NormalizedRow],
    dataset: str,
    seed: int,
    source: str,
) -> None:
    materialized = list(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    ids_path = Path(sample_ids_path)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_payload = {
        "dataset": dataset,
        "seed": seed,
        "source": source,
        "count": len(materialized),
        "sample_ids": [str(row["sample_id"]) for row in materialized],
    }
    ids_path.write_text(
        json.dumps(ids_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
