#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/TriePilot}"
DATASET_NAME="${DATASET_NAME:-random-ids}"
RUN_ID="${RUN_ID:-20260512_step1_static_ngram_tiers_${DATASET_NAME}_seed20260512}"
RUN_DIR="${RUN_DIR:-${WORKSPACE}/runs/${RUN_ID}}"
MODEL_PATH="${MODEL_PATH:-/root/sglang_flex_test/models/Qwen/Qwen3-8B}"
PORT_BASE="${PORT_BASE:-30100}"
BUDGETS="0 2 4 8 16 24 32"
BUDGETS="${TRIEPILOT_STEP1_BUDGETS:-${BUDGETS}}"
BENCH_DATASET_NAME="${BENCH_DATASET_NAME:-}"
DATASET_PATH="${DATASET_PATH:-}"
NORMALIZED_DATASET_PATH="${NORMALIZED_DATASET_PATH:-${WORKSPACE}/data/normalized/${DATASET_NAME}.jsonl}"
MAX_CONVERTED_ROWS="${MAX_CONVERTED_ROWS:-1000}"
MATCH_WINDOW="${MATCH_WINDOW:-12}"
BFS_BREADTH="${BFS_BREADTH:-4}"
BRANCH_LENGTH="${BRANCH_LENGTH:-18}"
NGRAM_MATCH_TYPE="${NGRAM_MATCH_TYPE:-BFS}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
REQUEST_RATE="${REQUEST_RATE:-8}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-32}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-16}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.0}"
SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN:-16}"
SHAREGPT_CONTEXT_LEN="${SHAREGPT_CONTEXT_LEN:-4096}"
SEED="${SEED:-20260512}"

mkdir -p "${RUN_DIR}"
cd "${WORKSPACE}"

if [ -z "${BENCH_DATASET_NAME}" ]; then
  if [ "${DATASET_NAME}" = "random-ids" ] || [ "${DATASET_NAME}" = "random" ]; then
    BENCH_DATASET_NAME="${DATASET_NAME}"
  else
    BENCH_DATASET_NAME="sharegpt"
  fi
fi

if [ -z "${DATASET_PATH}" ] && [ "${BENCH_DATASET_NAME}" = "sharegpt" ]; then
  DATASET_PATH="${NORMALIZED_DATASET_PATH}"
fi

BENCH_DATASET_PATH="${DATASET_PATH}"
if [ "${BENCH_DATASET_NAME}" = "sharegpt" ] && [[ "${DATASET_PATH}" == *.jsonl ]]; then
  converted_dir="${RUN_DIR}/bench_datasets"
  converted_path="${converted_dir}/${DATASET_NAME}_sharegpt.json"
  mkdir -p "${converted_dir}"
  DATASET_PATH="${DATASET_PATH}" CONVERTED_PATH="${converted_path}" MAX_CONVERTED_ROWS="${MAX_CONVERTED_ROWS}" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path


def _message_content(messages, role):
    for message in messages or []:
        if message.get("role") == role or message.get("from") == role:
            content = message.get("content", message.get("value", ""))
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _prompt_from_row(row):
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    content = _message_content(row.get("messages"), "user")
    if content:
        return content
    conversations = row.get("conversations") or row.get("conversation") or []
    if conversations:
        first = conversations[0]
        content = first.get("content", first.get("value", ""))
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _reference_from_row(row):
    for key in ("reference", "completion", "answer", "output"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    content = _message_content(row.get("messages"), "assistant")
    if content:
        return content
    conversations = row.get("conversations") or row.get("conversation") or []
    if len(conversations) >= 2:
        second = conversations[1]
        content = second.get("content", second.get("value", ""))
        if isinstance(content, str) and content.strip():
            return content
    return "OK"


source = Path(os.environ["DATASET_PATH"])
target = Path(os.environ["CONVERTED_PATH"])
limit = int(os.environ["MAX_CONVERTED_ROWS"])
rows = []
with source.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = _prompt_from_row(row)
        if not prompt:
            continue
        rows.append(
            {
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": _reference_from_row(row)},
                ]
            }
        )
        if len(rows) >= limit:
            break

if not rows:
    raise SystemExit(f"no usable prompts found in {source}")
target.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
print(f"converted {len(rows)} rows from {source} to {target}")
PY
  BENCH_DATASET_PATH="${converted_path}"
fi

SUMMARY_PATH="${RUN_DIR}/tier_summary.csv"
NOTES_PATH="${RUN_DIR}/notes.md"
ENV_PATH="${RUN_DIR}/env.json"
CONFIG_PATH="${RUN_DIR}/config.yaml"
GIT_COMMIT_PATH="${RUN_DIR}/git_commit.txt"
GPU_MEMORY_PATH="${RUN_DIR}/gpu_memory.txt"

current_pid=""
cleanup_server() {
  if [ -n "${current_pid}" ] && kill -0 "${current_pid}" >/dev/null 2>&1; then
    kill -- -"${current_pid}" >/dev/null 2>&1 || kill "${current_pid}" >/dev/null 2>&1 || true
    wait "${current_pid}" >/dev/null 2>&1 || true
  fi
  current_pid=""
}
trap cleanup_server EXIT

rm -f "${SUMMARY_PATH}" "${NOTES_PATH}" "${ENV_PATH}" "${CONFIG_PATH}" "${GIT_COMMIT_PATH}" "${GPU_MEMORY_PATH}"
git rev-parse HEAD > "${GIT_COMMIT_PATH}" || true

{
  echo "run_id: ${RUN_ID}"
  echo "model_path: ${MODEL_PATH}"
  echo "dataset_name: ${DATASET_NAME}"
  echo "bench_dataset_name: ${BENCH_DATASET_NAME}"
  echo "dataset_path: ${DATASET_PATH}"
  echo "bench_dataset_path: ${BENCH_DATASET_PATH}"
  echo "budgets: [${BUDGETS// /, }]"
  echo "match_window: ${MATCH_WINDOW}"
  echo "bfs_breadth: ${BFS_BREADTH}"
  echo "branch_length: ${BRANCH_LENGTH}"
  echo "ngram_match_type: ${NGRAM_MATCH_TYPE}"
  echo "num_prompts: ${NUM_PROMPTS}"
  echo "max_concurrency: ${MAX_CONCURRENCY}"
  echo "request_rate: ${REQUEST_RATE}"
  echo "random_input_len: ${RANDOM_INPUT_LEN}"
  echo "random_output_len: ${RANDOM_OUTPUT_LEN}"
  echo "random_range_ratio: ${RANDOM_RANGE_RATIO}"
  echo "sharegpt_output_len: ${SHAREGPT_OUTPUT_LEN}"
  echo "sharegpt_context_len: ${SHAREGPT_CONTEXT_LEN}"
  echo "seed: ${SEED}"
} > "${CONFIG_PATH}"

ENV_PATH="${ENV_PATH}" .venv/bin/python - <<'PY'
import json
import os
import platform
import subprocess
from pathlib import Path

env = {
    "python": platform.python_version(),
    "platform": platform.platform(),
}
for command, key in [
    (["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], "nvidia_smi"),
    (["/root/anaconda3/envs/sglang/bin/python", "-c", "import sglang; print(getattr(sglang, '__version__', 'unknown'))"], "sglang_version"),
]:
    try:
        env[key] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        env[key] = f"unavailable: {exc}"
Path(os.environ["ENV_PATH"]).write_text(json.dumps(env, indent=2), encoding="utf-8")
PY

{
  echo "before"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader
} > "${GPU_MEMORY_PATH}" || true

index=0
for budget in ${BUDGETS}; do
  tier_dir="${RUN_DIR}/budget_${budget}"
  mkdir -p "${tier_dir}"

  port="$((PORT_BASE + index))"
  index="$((index + 1))"
  if ss -ltn | grep -q ":${port} "; then
    echo "Port ${port} is already in use." >&2
    exit 1
  fi

  trace_path="${tier_dir}/raw_step_events.jsonl"
  server_log="${tier_dir}/server.log"
  if [[ "${BENCH_DATASET_NAME}" == random* ]]; then
    bench_output="${tier_dir}/raw_events_len${RANDOM_INPUT_LEN}x${RANDOM_OUTPUT_LEN}.jsonl"
  else
    bench_output="${tier_dir}/raw_events_${DATASET_NAME}_out${SHAREGPT_OUTPUT_LEN}.jsonl"
  fi
  rm -f "${trace_path}" "${server_log}" "${bench_output}" "${tier_dir}/server.pid"
  : > "${trace_path}"

  export PORT="${port}"
  export MODEL_PATH
  export MATCH_WINDOW
  export BFS_BREADTH
  export BRANCH_LENGTH
  export NGRAM_MATCH_TYPE
  export WORKSPACE_SGLANG_SRC="${WORKSPACE}/third_party/sglang_flex/python"
  export TRIEPILOT_TRACE_PATH="${trace_path}"
  export TRIEPILOT_RUN_ID="${RUN_ID}_budget_${budget}"
  export FLASHINFER_WORKSPACE_BASE="${tier_dir}/flashinfer_cache"

  if [ "${budget}" = "0" ]; then
    export SPECULATION="none"
    unset DRAFT_TOKENS || true
  else
    export SPECULATION="ngram"
    export DRAFT_TOKENS="${budget}"
  fi

  setsid ./scripts/run_server.sh >"${server_log}" 2>&1 &
  current_pid="$!"
  echo "${current_pid}" > "${tier_dir}/server.pid"

  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${current_pid}" >/dev/null 2>&1; then
      echo "SGLang server exited before becoming healthy for budget ${budget}." >&2
      tail -200 "${server_log}" >&2 || true
      exit 1
    fi
    sleep 5
  done

  if [ "${ready}" != "1" ]; then
    echo "SGLang server did not become healthy on port ${port} for budget ${budget}." >&2
    tail -200 "${server_log}" >&2 || true
    exit 1
  fi

  bench_args=(
    scripts/run_bench.py
    --conda-bin /root/anaconda3/bin/conda \
    --conda-env sglang \
    --host 127.0.0.1 \
    --port "${port}" \
    --model "${MODEL_PATH}" \
    --dataset-name "${BENCH_DATASET_NAME}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --request-rate "${REQUEST_RATE}" \
    --random-input-len "${RANDOM_INPUT_LEN}" \
    --random-output-len "${RANDOM_OUTPUT_LEN}" \
    --random-range-ratio "${RANDOM_RANGE_RATIO}" \
    --sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}" \
    --sharegpt-context-len "${SHAREGPT_CONTEXT_LEN}" \
    --seed "${SEED}" \
    --output-file "${bench_output}"
  )
  if [ -n "${BENCH_DATASET_PATH}" ]; then
    bench_args+=(--dataset-path "${BENCH_DATASET_PATH}")
  fi
  .venv/bin/python "${bench_args[@]}"

  cleanup_server

  if [ ! -s "${bench_output}" ]; then
    echo "Missing or empty bench output for budget ${budget}: ${bench_output}" >&2
    exit 1
  fi
  if [ "${budget}" != "0" ] && [ ! -s "${trace_path}" ]; then
    echo "Missing or empty NGRAM step trace for budget ${budget}: ${trace_path}" >&2
    exit 1
  fi
done

SUMMARY_PATH="${SUMMARY_PATH}" NOTES_PATH="${NOTES_PATH}" RUN_DIR="${RUN_DIR}" BUDGETS="${BUDGETS}" DATASET_NAME="${DATASET_NAME}" BENCH_DATASET_NAME="${BENCH_DATASET_NAME}" MATCH_WINDOW="${MATCH_WINDOW}" BFS_BREADTH="${BFS_BREADTH}" BRANCH_LENGTH="${BRANCH_LENGTH}" NGRAM_MATCH_TYPE="${NGRAM_MATCH_TYPE}" .venv/bin/python - <<'PY'
import csv
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


run_dir = Path(os.environ["RUN_DIR"])
budgets = [int(item) for item in os.environ["BUDGETS"].split()]
summary_path = Path(os.environ["SUMMARY_PATH"])
notes_path = Path(os.environ["NOTES_PATH"])
rows = []
dataset_name = os.environ["DATASET_NAME"]
bench_dataset_name = os.environ["BENCH_DATASET_NAME"]
match_window = int(os.environ["MATCH_WINDOW"])
bfs_breadth = int(os.environ["BFS_BREADTH"])
branch_length = int(os.environ["BRANCH_LENGTH"])
ngram_match_type = os.environ["NGRAM_MATCH_TYPE"]

for budget in budgets:
    tier_dir = run_dir / f"budget_{budget}"
    trace_path = tier_dir / "raw_step_events.jsonl"
    bench_paths = sorted(tier_dir.glob("raw_events_*.jsonl"))
    if not bench_paths:
        raise SystemExit(f"missing bench output under {tier_dir}")
    bench_output = bench_paths[0]
    bench_rows = read_jsonl(bench_output)
    bench_summary = bench_rows[0] if bench_rows else {}
    events = read_jsonl(trace_path)
    verified = sum(float(event.get("verified_nodes", 0)) for event in events)
    accepted = sum(float(event.get("accepted_tokens", 0)) for event in events)
    wasted = sum(float(event.get("wasted_nodes", 0)) for event in events)
    row = {
        "dataset_name": dataset_name,
        "bench_dataset_name": bench_dataset_name,
        "match_window": match_window,
        "bfs_breadth": bfs_breadth,
        "branch_length": branch_length,
        "ngram_match_type": ngram_match_type,
        "budget": budget,
        "speculation": "none" if budget == 0 else "ngram",
        "trace_events": len(events),
        "bench_rows": len(bench_rows),
        "request_throughput": float(bench_summary.get("request_throughput", 0)),
        "output_throughput": float(bench_summary.get("output_throughput", 0)),
        "mean_tpot_ms": float(bench_summary.get("mean_tpot_ms", 0)),
        "p99_tpot_ms": float(bench_summary.get("p99_tpot_ms", 0)),
        "mean_e2e_latency_ms": float(bench_summary.get("mean_e2e_latency_ms", 0)),
        "p99_e2e_latency_ms": float(bench_summary.get("p99_e2e_latency_ms", 0)),
        "accepted_tokens": accepted,
        "verified_nodes": verified,
        "accepted_per_verified_node": accepted / verified if verified else 0.0,
        "wasted_node_ratio": wasted / verified if verified else 0.0,
        "mean_verify_time_us": mean([float(event.get("verify_time_us", 0)) for event in events]),
        "mean_step_latency_us": mean([float(event.get("step_latency_us", 0)) for event in events]),
        "cuda_graph_ratio": mean([1.0 if event.get("can_run_cuda_graph") else 0.0 for event in events]),
        "mean_match_depth": mean([float(event.get("match_depth", 0)) for event in events]),
        "mean_candidate_count": mean([float(event.get("candidate_count", 0)) for event in events]),
        "mean_branch_entropy": mean([float(event.get("branch_entropy", 0)) for event in events]),
        "mean_top_branch_ratio": mean([float(event.get("top_branch_ratio", 0)) for event in events]),
        "trace_path": str(trace_path),
        "bench_output": str(bench_output),
    }
    if budget != 0 and events:
        required = {"match_depths", "candidate_counts", "branch_entropies", "top_branch_ratios", "filled_nodes"}
        missing = sorted(required - set(events[0]))
        if missing:
            raise SystemExit(f"budget {budget} first event missing feature fields: {missing}")
    rows.append(row)

with summary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

notes = [
    "# Step 1 Static NGRAM Budget Tiers",
    "",
    f"Run directory: `{run_dir}`",
    f"Summary: `{summary_path}`",
    f"Dataset: `{dataset_name}` (`{bench_dataset_name}`)",
    f"Shape: match_type={ngram_match_type}, match_window={match_window}, bfs_breadth={bfs_breadth}, branch_length={branch_length}",
    "",
    "| budget | speculation | out tok/s | mean TPOT ms | p99 TPOT ms | trace events | accepted/verified | wasted ratio | mean verify us | mean match depth | mean candidates |",
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    notes.append(
        "| {budget} | {speculation} | {out:.2f} | {tpot:.2f} | {p99_tpot:.2f} | {trace_events} | {apv:.4f} | {wasted:.4f} | {verify:.2f} | {depth:.2f} | {candidates:.2f} |".format(
            budget=row["budget"],
            speculation=row["speculation"],
            out=row["output_throughput"],
            tpot=row["mean_tpot_ms"],
            p99_tpot=row["p99_tpot_ms"],
            trace_events=row["trace_events"],
            apv=row["accepted_per_verified_node"],
            wasted=row["wasted_node_ratio"],
            verify=row["mean_verify_time_us"],
            depth=row["mean_match_depth"],
            candidates=row["mean_candidate_count"],
        )
    )
notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
PY

{
  echo "after"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader
} >> "${GPU_MEMORY_PATH}" || true

echo "A100 Step 1 static tiers run directory: ${RUN_DIR}"
