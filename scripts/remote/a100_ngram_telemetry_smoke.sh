#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/TriePilot}"
PORT="${PORT:-30001}"
RUN_ID="${RUN_ID:-20260512_telemetry_ngram_smoke_seed20260512}"
RUN_DIR="${RUN_DIR:-${WORKSPACE}/runs/${RUN_ID}}"
MODEL_PATH="${MODEL_PATH:-/root/sglang_flex_test/models/Qwen/Qwen3-8B}"

mkdir -p "${RUN_DIR}"
cd "${WORKSPACE}"

TRACE_PATH="${RUN_DIR}/raw_step_events.jsonl"
SERVER_LOG="${RUN_DIR}/server.log"
BENCH_OUTPUT="${RUN_DIR}/raw_events_len32x16.jsonl"

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

rm -f "${TRACE_PATH}" "${BENCH_OUTPUT}" "${SERVER_LOG}" "${RUN_DIR}/server.pid"

export PORT
export MODEL_PATH
export SPECULATION=ngram
export DRAFT_TOKENS="${DRAFT_TOKENS:-8}"
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

.venv/bin/python scripts/run_bench.py \
  --conda-bin /root/anaconda3/bin/conda \
  --conda-env sglang \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --dataset-name random-ids \
  --num-prompts 8 \
  --max-concurrency 4 \
  --request-rate 4 \
  --random-input-len 32 \
  --random-output-len 16 \
  --random-range-ratio 0.0 \
  --output-file "${BENCH_OUTPUT}"

TRACE_PATH="${TRACE_PATH}" BENCH_OUTPUT="${BENCH_OUTPUT}" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

trace_path = Path(os.environ["TRACE_PATH"])
bench_output = Path(os.environ["BENCH_OUTPUT"])
required = {
    "event",
    "run_id",
    "step_id",
    "batch_size",
    "allocated_budget",
    "actual_draft_nodes",
    "verified_nodes",
    "accepted_tokens",
    "wasted_nodes",
    "accept_lens",
    "ngram_query_time_us",
    "verify_time_us",
    "step_latency_us",
    "can_run_cuda_graph",
}

if not bench_output.exists() or bench_output.stat().st_size == 0:
    raise SystemExit(f"missing or empty bench output: {bench_output}")
if not trace_path.exists() or trace_path.stat().st_size == 0:
    raise SystemExit(f"missing or empty telemetry trace: {trace_path}")

events = [
    json.loads(line)
    for line in trace_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not events:
    raise SystemExit(f"no telemetry events in {trace_path}")

missing = sorted(required - set(events[0]))
if missing:
    raise SystemExit(f"first telemetry event missing keys: {missing}")
if any(event["actual_draft_nodes"] != event["verified_nodes"] for event in events):
    raise SystemExit("actual_draft_nodes and verified_nodes diverged in static NGRAM smoke")

print(
    json.dumps(
        {
            "trace_path": str(trace_path),
            "bench_output": str(bench_output),
            "event_count": len(events),
            "first_event": events[0],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
)
PY

echo "A100 telemetry smoke run directory: ${RUN_DIR}"
