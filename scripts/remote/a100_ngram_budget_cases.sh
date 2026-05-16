#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/TriePilot}"
PORT="${PORT:-30002}"
RUN_ID="${RUN_ID:-20260512_step0_per_request_budget_cases_seed20260512}"
RUN_DIR="${RUN_DIR:-${WORKSPACE}/runs/${RUN_ID}}"
MODEL_PATH="${MODEL_PATH:-/root/sglang_flex_test/models/Qwen/Qwen3-8B}"

mkdir -p "${RUN_DIR}"
cd "${WORKSPACE}"

TRACE_PATH="${RUN_DIR}/raw_step_events.jsonl"
SERVER_LOG="${RUN_DIR}/server.log"
SUMMARY_PATH="${RUN_DIR}/case_summary.json"
NOTES_PATH="${RUN_DIR}/notes.md"
GPU_MEMORY_PATH="${RUN_DIR}/gpu_memory.txt"

cleanup() {
  if [ -f "${RUN_DIR}/server.pid" ]; then
    pid="$(cat "${RUN_DIR}/server.pid")"
    if [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill -- -"${pid}" >/dev/null 2>&1 || kill "${pid}" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

if ss -ltn | grep -q ":${PORT} "; then
  echo "Port ${PORT} is already in use." >&2
  exit 1
fi

rm -f \
  "${TRACE_PATH}" \
  "${SERVER_LOG}" \
  "${SUMMARY_PATH}" \
  "${NOTES_PATH}" \
  "${GPU_MEMORY_PATH}" \
  "${RUN_DIR}/server.pid"

{
  echo "before"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader
} > "${GPU_MEMORY_PATH}" || true

export PORT
export MODEL_PATH
export SPECULATION=ngram
export DRAFT_TOKENS="${DRAFT_TOKENS:-16}"
export MATCH_WINDOW="${MATCH_WINDOW:-12}"
export BFS_BREADTH="${BFS_BREADTH:-4}"
export BRANCH_LENGTH="${BRANCH_LENGTH:-18}"
export WORKSPACE_SGLANG_SRC="${WORKSPACE}/third_party/sglang_flex/python"
export TRIEPILOT_TRACE_PATH="${TRACE_PATH}"
export TRIEPILOT_RUN_ID="${RUN_ID}"
export FLASHINFER_WORKSPACE_BASE="${RUN_DIR}/flashinfer_cache"

setsid ./scripts/run_server.sh >"${SERVER_LOG}" 2>&1 &
server_pid="$!"
echo "${server_pid}" > "${RUN_DIR}/server.pid"

ready=0
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
    echo "SGLang server exited before becoming healthy." >&2
    tail -200 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done

if [ "${ready}" != "1" ]; then
  echo "SGLang server did not become healthy on port ${PORT}." >&2
  tail -200 "${SERVER_LOG}" >&2 || true
  exit 1
fi

TRACE_PATH="${TRACE_PATH}" \
SUMMARY_PATH="${SUMMARY_PATH}" \
NOTES_PATH="${NOTES_PATH}" \
PORT="${PORT}" \
.venv/bin/python - <<'PY'
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

import os

port = os.environ["PORT"]
trace_path = Path(os.environ["TRACE_PATH"])
summary_path = Path(os.environ["SUMMARY_PATH"])
notes_path = Path(os.environ["NOTES_PATH"])
url = f"http://127.0.0.1:{port}/generate"

cases = {
    "caseA_all16": [16, 16, 16, 16, 16, 16, 16, 16],
    "caseB_half16_half0": [16, 16, 16, 16, 0, 0, 0, 0],
    "caseC_heterogeneous": [16, 8, 4, 2, 0, 16, 8, 0],
}


def post_json(payload: dict) -> object:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            if resp.status != 200:
                raise RuntimeError(f"status={resp.status}, body={body[:1000]}")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"status={exc.code}, body={body[:2000]}") from exc


def build_input_ids(case_index: int, request_index: int) -> list[int]:
    base = 100 + case_index * 1000 + request_index * 37
    return [base + (j % 23) for j in range(32)]


responses = {}
for case_index, (case_name, budgets) in enumerate(cases.items()):
    payload = {
        "rid": [f"{case_name}_{i}" for i in range(len(budgets))],
        "input_ids": [
            build_input_ids(case_index, i) for i in range(len(budgets))
        ],
        "sampling_params": [
            {
                "temperature": 0.0,
                "top_k": 1,
                "max_new_tokens": 16,
                "ignore_eos": True,
                "custom_params": {"triepilot_draft_budget": budget},
            }
            for budget in budgets
        ],
        "stream": False,
    }
    responses[case_name] = post_json(payload)
    time.sleep(1)

if not trace_path.exists() or trace_path.stat().st_size == 0:
    raise SystemExit(f"missing or empty trace: {trace_path}")

events = [
    json.loads(line)
    for line in trace_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

summary = {}
for case_name, budgets in cases.items():
    case_events = [
        event
        for event in events
        if any(str(rid).startswith(case_name) for rid in event.get("request_ids", []))
    ]
    if not case_events:
        raise SystemExit(f"no telemetry events found for {case_name}")

    first = case_events[0]
    expected_actual = sum(budgets)
    expected_verify_tokens = sum(max(budget, 1) for budget in budgets)
    if first["allocated_budgets"] != budgets:
        raise SystemExit(
            f"{case_name} budget mismatch: {first['allocated_budgets']} != {budgets}"
        )
    if first["actual_draft_nodes"] != expected_actual:
        raise SystemExit(
            f"{case_name} actual_draft_nodes={first['actual_draft_nodes']} "
            f"expected {expected_actual}"
        )
    if first["verify_input_tokens"] != expected_verify_tokens:
        raise SystemExit(
            f"{case_name} verify_input_tokens={first['verify_input_tokens']} "
            f"expected {expected_verify_tokens}"
        )

    summary[case_name] = {
        "budgets": budgets,
        "event_count": len(case_events),
        "first_actual_draft_nodes": first["actual_draft_nodes"],
        "first_verify_input_tokens": first["verify_input_tokens"],
        "first_can_run_cuda_graph": first["can_run_cuda_graph"],
        "verify_time_us_mean": statistics.fmean(
            event["verify_time_us"] for event in case_events
        ),
        "step_latency_us_mean": statistics.fmean(
            event["step_latency_us"] for event in case_events
        ),
        "accepted_tokens_sum": sum(event["accepted_tokens"] for event in case_events),
    }

summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

notes = [
    "# Step 0 Per-Request Budget Cases",
    "",
    f"Trace: `{trace_path}`",
    f"Summary: `{summary_path}`",
    "",
    "| case | budgets | first actual draft nodes | first verify input tokens | cuda graph | mean verify us | mean step us |",
    "| --- | --- | ---: | ---: | --- | ---: | ---: |",
]
for case_name, item in summary.items():
    notes.append(
        "| {case} | `{budgets}` | {actual} | {tokens} | {graph} | {verify:.2f} | {step:.2f} |".format(
            case=case_name,
            budgets=item["budgets"],
            actual=item["first_actual_draft_nodes"],
            tokens=item["first_verify_input_tokens"],
            graph=item["first_can_run_cuda_graph"],
            verify=item["verify_time_us_mean"],
            step=item["step_latency_us_mean"],
        )
    )
notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY

{
  echo "after"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader
} >> "${GPU_MEMORY_PATH}" || true

echo "A100 per-request budget case run directory: ${RUN_DIR}"
