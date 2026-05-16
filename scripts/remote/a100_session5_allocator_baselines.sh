#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/TriePilot}"
RUN_ID="${RUN_ID:-20260513_session5_allocator_sweep_seed20260512}"
RUN_DIR="${RUN_DIR:-${WORKSPACE}/runs/${RUN_ID}}"
MODEL_PATH="${MODEL_PATH:-/root/sglang_flex_test/models/Qwen/Qwen3-8B}"
MIXED_PAIRS="${MIXED_PAIRS:-instructcoder:gsm8k instructcoder:sharegpt json_tool:sharegpt cnn_dailymail:random}"
LEFT_RATIOS="${LEFT_RATIOS:-0.5 0.2 0.8}"
B_BATCH_VALUES="${B_BATCH_VALUES:-16 32 64 100 160}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
REQUEST_RATE="${REQUEST_RATE:-8}"
SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN:-64}"
SHAREGPT_CONTEXT_LEN="${SHAREGPT_CONTEXT_LEN:-4096}"
MATCH_WINDOW="${MATCH_WINDOW:-12}"
BFS_BREADTH="${BFS_BREADTH:-4}"
BRANCH_LENGTH="${BRANCH_LENGTH:-18}"
MAX_DRAFT_TOKENS="${MAX_DRAFT_TOKENS:-16}"
BATCH_GLOBAL_DRAFT_TOKENS="${BATCH_GLOBAL_DRAFT_TOKENS:-2}"
TRIEPILOT_SHAPE_BUCKETS="${TRIEPILOT_SHAPE_BUCKETS:-}"
TRIEPILOT_SHAPE_BUCKET_MODE="${TRIEPILOT_SHAPE_BUCKET_MODE:-off}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MATERIALIZE_FILL_FILTERED="${MATERIALIZE_FILL_FILTERED:-1}"
MATERIALIZE_TOKENIZER_MODEL="${MATERIALIZE_TOKENIZER_MODEL:-${MODEL_PATH}}"
SEED="${SEED:-20260512}"
PORT_BASE="${PORT_BASE:-30300}"
METHODS="${TRIEPILOT_SESSION5_METHODS:-batch_global_budget2 equal_budget_allocation random_budget_allocation match_depth_greedy accept_ema_greedy triepilot_allocation}"

mkdir -p "${RUN_DIR}"
cd "${WORKSPACE}"

SWEEP_SUMMARY_PATH="${RUN_DIR}/session5_sweep_summary.csv"
ENV_PATH="${RUN_DIR}/env.json"
CONFIG_PATH="${RUN_DIR}/config.yaml"
GIT_COMMIT_PATH="${RUN_DIR}/git_commit.txt"
GPU_MEMORY_PATH="${RUN_DIR}/gpu_memory.txt"
NOTES_PATH="${RUN_DIR}/notes.md"

git rev-parse HEAD > "${GIT_COMMIT_PATH}" || true

{
  echo "run_id: ${RUN_ID}"
  echo "model_path: ${MODEL_PATH}"
  echo "mixed_pairs: [${MIXED_PAIRS// /, }]"
  echo "left_ratios: [${LEFT_RATIOS// /, }]"
  echo "b_batch_values: [${B_BATCH_VALUES// /, }]"
  echo "num_prompts: ${NUM_PROMPTS}"
  echo "max_concurrency: ${MAX_CONCURRENCY}"
  echo "request_rate: ${REQUEST_RATE}"
  echo "sharegpt_output_len: ${SHAREGPT_OUTPUT_LEN}"
  echo "sharegpt_context_len: ${SHAREGPT_CONTEXT_LEN}"
  echo "match_window: ${MATCH_WINDOW}"
  echo "bfs_breadth: ${BFS_BREADTH}"
  echo "branch_length: ${BRANCH_LENGTH}"
  echo "max_draft_tokens: ${MAX_DRAFT_TOKENS}"
  echo "batch_global_draft_tokens: ${BATCH_GLOBAL_DRAFT_TOKENS}"
  echo "triepilot_shape_buckets: ${TRIEPILOT_SHAPE_BUCKETS}"
  echo "triepilot_shape_bucket_mode: ${TRIEPILOT_SHAPE_BUCKET_MODE}"
  echo "cuda_graph_max_bs: ${CUDA_GRAPH_MAX_BS}"
  echo "max_running_requests: ${MAX_RUNNING_REQUESTS}"
  echo "materialize_fill_filtered: ${MATERIALIZE_FILL_FILTERED}"
  echo "materialize_tokenizer_model: ${MATERIALIZE_TOKENIZER_MODEL}"
  echo "methods: [${METHODS// /, }]"
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

current_pid=""
cleanup_server() {
  if [ -n "${current_pid}" ] && kill -0 "${current_pid}" >/dev/null 2>&1; then
    kill -- -"${current_pid}" >/dev/null 2>&1 || kill "${current_pid}" >/dev/null 2>&1 || true
    wait "${current_pid}" >/dev/null 2>&1 || true
  fi
  current_pid=""
}
trap cleanup_server EXIT

port_offset=0
rm -f "${SWEEP_SUMMARY_PATH}"

run_method() {
  local combo_dir="$1"
  local left_dataset="$2"
  local right_dataset="$3"
  local left_ratio="$4"
  local b_batch="$5"
  local bench_dataset_path="$6"
  local method="$7"

  local method_dir="${combo_dir}/${method}"
  mkdir -p "${method_dir}"
  local port="$((PORT_BASE + port_offset))"
  port_offset="$((port_offset + 1))"
  if ss -ltn | grep -q ":${port} "; then
    echo "Port ${port} is already in use." >&2
    exit 1
  fi

  local allocation_policy
  local draft_tokens
  local batch_budget
  local ablation_mode=""
  if [[ "${method}" =~ ^batch_global_budget([0-9]+)$ ]]; then
    allocation_policy="custom"
    draft_tokens="${BASH_REMATCH[1]}"
    batch_budget=""
  elif [ "${method}" = "triepilot_wo_trie_features" ]; then
    allocation_policy="triepilot_allocation"
    draft_tokens="${MAX_DRAFT_TOKENS}"
    batch_budget="${b_batch}"
    ablation_mode="no_trie_features"
  elif [ "${method}" = "triepilot_wo_history" ]; then
    allocation_policy="triepilot_allocation"
    draft_tokens="${MAX_DRAFT_TOKENS}"
    batch_budget="${b_batch}"
    ablation_mode="no_history"
  elif [ "${method}" = "triepilot_wo_serving_pressure" ]; then
    allocation_policy="triepilot_allocation"
    draft_tokens="${MAX_DRAFT_TOKENS}"
    batch_budget="${b_batch}"
    ablation_mode="no_serving_pressure"
  elif [ "${method}" = "triepilot_wo_strategy_bank" ]; then
    allocation_policy="triepilot_allocation"
    draft_tokens="${MAX_DRAFT_TOKENS}"
    batch_budget="${b_batch}"
    ablation_mode="no_strategy_bank"
  else
    allocation_policy="${method}"
    draft_tokens="${MAX_DRAFT_TOKENS}"
    batch_budget="${b_batch}"
  fi

  local trace_path="${method_dir}/raw_step_events.jsonl"
  local bench_output="${method_dir}/raw_events_${left_dataset}_${right_dataset}_r${left_ratio}_B${b_batch}_out${SHAREGPT_OUTPUT_LEN}.jsonl"
  local server_log="${method_dir}/server.log"
  rm -f "${trace_path}" "${bench_output}" "${server_log}" "${method_dir}/server.pid"
  : > "${trace_path}"

  export PORT="${port}"
  export MODEL_PATH
  export MATCH_WINDOW
  export BFS_BREADTH
  export BRANCH_LENGTH
  export WORKSPACE_SGLANG_SRC="${WORKSPACE}/third_party/sglang_flex/python"
  export TRIEPILOT_TRACE_PATH="${trace_path}"
  export TRIEPILOT_RUN_ID="${RUN_ID}_${left_dataset}_${right_dataset}_r${left_ratio}_B${b_batch}_${method}"
  export TRIEPILOT_ALLOCATION_POLICY="${allocation_policy}"
  export TRIEPILOT_RANDOM_SEED="${SEED}"
  export TRIEPILOT_SHAPE_BUCKETS
  export TRIEPILOT_SHAPE_BUCKET_MODE
  if [ -n "${ablation_mode}" ]; then
    export TRIEPILOT_ABLATION_MODE="${ablation_mode}"
  else
    unset TRIEPILOT_ABLATION_MODE || true
  fi
  export CUDA_GRAPH_MAX_BS
  export MAX_RUNNING_REQUESTS
  export FLASHINFER_WORKSPACE_BASE="${method_dir}/flashinfer_cache"
  export SPECULATION="ngram"
  export DRAFT_TOKENS="${draft_tokens}"
  if [ -n "${batch_budget}" ]; then
    export TRIEPILOT_BATCH_BUDGET="${batch_budget}"
  else
    unset TRIEPILOT_BATCH_BUDGET || true
  fi

  setsid ./scripts/run_server.sh >"${server_log}" 2>&1 &
  current_pid="$!"
  echo "${current_pid}" > "${method_dir}/server.pid"

  local ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${current_pid}" >/dev/null 2>&1; then
      echo "SGLang server exited before becoming healthy for ${method}." >&2
      tail -200 "${server_log}" >&2 || true
      exit 1
    fi
    sleep 5
  done

  if [ "${ready}" != "1" ]; then
    echo "SGLang server did not become healthy on port ${port} for ${method}." >&2
    tail -200 "${server_log}" >&2 || true
    exit 1
  fi

  .venv/bin/python scripts/run_bench.py \
    --conda-bin /root/anaconda3/bin/conda \
    --conda-env sglang \
    --host 127.0.0.1 \
    --port "${port}" \
    --model "${MODEL_PATH}" \
    --dataset-name sharegpt \
    --dataset-path "${bench_dataset_path}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --request-rate "${REQUEST_RATE}" \
    --sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}" \
    --sharegpt-context-len "${SHAREGPT_CONTEXT_LEN}" \
    --seed "${SEED}" \
    --output-file "${bench_output}"

  cleanup_server

  if [ ! -s "${bench_output}" ]; then
    echo "Missing or empty bench output for ${method}: ${bench_output}" >&2
    exit 1
  fi
  if [ ! -s "${trace_path}" ]; then
    echo "Missing or empty NGRAM step trace for ${method}: ${trace_path}" >&2
    exit 1
  fi
}

summarize_combo() {
  local combo_dir="$1"
  local left_dataset="$2"
  local right_dataset="$3"
  local left_ratio="$4"
  local b_batch="$5"
  local summary_path="${combo_dir}/allocator_summary.csv"
  local combo_notes_path="${combo_dir}/notes.md"

  SUMMARY_PATH="${summary_path}" \
  SWEEP_SUMMARY_PATH="${SWEEP_SUMMARY_PATH}" \
  NOTES_PATH="${combo_notes_path}" \
  RUN_DIR="${RUN_DIR}" \
  COMBO_DIR="${combo_dir}" \
  METHODS="${METHODS}" \
  LEFT_DATASET="${left_dataset}" \
  RIGHT_DATASET="${right_dataset}" \
  LEFT_RATIO="${left_ratio}" \
  B_BATCH="${b_batch}" \
  CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS}" \
  MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
  .venv/bin/python - <<'PY'
import csv
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def cuda_graph_token_shape_ok(event: dict) -> bool:
    if "cuda_graph_token_shape_ok" in event:
        return bool(event.get("cuda_graph_token_shape_ok"))
    draft_token_num = event.get("draft_token_num")
    if draft_token_num is None:
        return False
    return int(event.get("verify_input_tokens", 0)) == (
        int(event.get("batch_size", 0)) * int(draft_token_num)
    )


combo_dir = Path(os.environ["COMBO_DIR"])
methods = os.environ["METHODS"].split()
b_batch = int(os.environ["B_BATCH"])
rows = []

for method in methods:
    method_dir = combo_dir / method
    trace_path = method_dir / "raw_step_events.jsonl"
    bench_output = sorted(method_dir.glob("raw_events_*.jsonl"))[0]
    events = read_jsonl(trace_path)
    bench_rows = read_jsonl(bench_output)
    bench_summary = bench_rows[0] if bench_rows else {}
    verified = sum(float(event.get("verified_nodes", 0)) for event in events)
    accepted = sum(float(event.get("accepted_tokens", 0)) for event in events)
    wasted = sum(float(event.get("wasted_nodes", 0)) for event in events)
    allocated = [float(event.get("allocated_budget", 0)) for event in events]
    actual_draft_nodes = [float(event.get("actual_draft_nodes", 0)) for event in events]
    verify_input_tokens = [float(event.get("verify_input_tokens", 0)) for event in events]
    bucket_padding = [float(event.get("bucket_padding_nodes_total", 0)) for event in events]
    shape_padding = [float(event.get("shape_padding_tokens_total", 0)) for event in events]
    heterogeneous_events = 0
    strategy_hit_rates = []
    controller_times = []
    recovery_probe_counts = []
    request_local_probe_counts = []
    request_local_probe_events = []
    positive_observation_means = []
    max_observed_gains = []
    for event in events:
        budgets = event.get("allocated_budgets", [])
        if len(set(budgets)) > 1:
            heterogeneous_events += 1
        if not method.startswith("batch_global_budget") and sum(int(x) for x in budgets) > b_batch:
            raise SystemExit(f"{method} exceeded B_batch in {trace_path}: {budgets}")
        strategy_hit_rates.append(float(event.get("strategy_bank_hit_rate", 0.0)))
        controller_times.append(float(event.get("controller_time_us", 0.0)))
        recovery_probe_counts.append(float(event.get("recovery_probe_count", 0.0)))
        request_local_probe_count = float(event.get("request_local_probe_count", 0.0))
        request_local_probe_counts.append(request_local_probe_count)
        request_local_probe_events.append(1.0 if request_local_probe_count > 0 else 0.0)
        observations = event.get("positive_observations", []) or []
        if observations:
            positive_observation_means.append(
                sum(float(value) for value in observations) / len(observations)
            )
        observed_gains = event.get("max_observed_gain_per_node", []) or []
        if observed_gains:
            max_observed_gains.append(max(float(value) for value in observed_gains))
    first_event = events[0] if events else {}
    row = {
        "left_dataset": os.environ["LEFT_DATASET"],
        "right_dataset": os.environ["RIGHT_DATASET"],
        "left_ratio": os.environ["LEFT_RATIO"],
        "b_batch": b_batch,
        "cuda_graph_max_bs": int(os.environ["CUDA_GRAPH_MAX_BS"]),
        "max_running_requests": int(os.environ["MAX_RUNNING_REQUESTS"]),
        "method": method,
        "allocation_policy": first_event.get("allocation_policy", ""),
        "ablation_modes": ",".join(first_event.get("ablation_modes", [])),
        "batch_budget": first_event.get("batch_budget", ""),
        "server_draft_tokens": max(first_event.get("allocated_budgets", [0]) or [0]),
        "trace_events": len(events),
        "request_throughput": float(bench_summary.get("request_throughput", 0)),
        "output_throughput": float(bench_summary.get("output_throughput", 0)),
        "mean_tpot_ms": float(bench_summary.get("mean_tpot_ms", 0)),
        "p99_tpot_ms": float(bench_summary.get("p99_tpot_ms", 0)),
        "accepted_tokens": accepted,
        "verified_nodes": verified,
        "accepted_per_verified_node": accepted / verified if verified else 0.0,
        "wasted_node_ratio": wasted / verified if verified else 0.0,
        "mean_allocated_budget": mean(allocated),
        "max_allocated_budget": max(allocated) if allocated else 0.0,
        "mean_actual_draft_nodes": mean(actual_draft_nodes),
        "mean_verify_input_tokens": mean(verify_input_tokens),
        "mean_bucket_padding_nodes": mean(bucket_padding),
        "mean_shape_padding_tokens": mean(shape_padding),
        "heterogeneous_budget_event_ratio": heterogeneous_events / len(events) if events else 0.0,
        "mean_verify_time_us": mean([float(event.get("verify_time_us", 0)) for event in events]),
        "mean_target_forward_time_us": mean([float(event.get("target_forward_time_us", 0)) for event in events]),
        "mean_ngram_query_time_us": mean([float(event.get("ngram_query_time_us", 0)) for event in events]),
        "mean_step_latency_us": mean([float(event.get("step_latency_us", 0)) for event in events]),
        "cuda_graph_ratio": mean([1.0 if event.get("can_run_cuda_graph") else 0.0 for event in events]),
        "cuda_graph_token_shape_ok_ratio": mean([1.0 if cuda_graph_token_shape_ok(event) else 0.0 for event in events]),
        "mean_match_depth": mean([float(event.get("match_depth", 0)) for event in events]),
        "mean_accept_len_ema": mean([float(event.get("accept_len_ema", 0)) for event in events]),
        "strategy_bank_hit_rate": mean(strategy_hit_rates),
        "recovery_probe_count": sum(recovery_probe_counts),
        "request_local_probe_count": sum(request_local_probe_counts),
        "request_local_probe_event_ratio": mean(request_local_probe_events),
        "mean_positive_observations": mean(positive_observation_means),
        "max_observed_gain_per_node": max(max_observed_gains) if max_observed_gains else 0.0,
        "controller_overhead_p50_us": sorted(controller_times)[len(controller_times) // 2] if controller_times else 0.0,
        "controller_overhead_p99_us": sorted(controller_times)[min(len(controller_times) - 1, int(len(controller_times) * 0.99))] if controller_times else 0.0,
        "trace_path": str(trace_path),
        "bench_output": str(bench_output),
    }
    rows.append(row)

fieldnames = list(rows[0])
summary_path = Path(os.environ["SUMMARY_PATH"])
with summary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

sweep_path = Path(os.environ["SWEEP_SUMMARY_PATH"])
write_header = not sweep_path.exists()
with sweep_path.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerows(rows)

notes = [
    "# Session 5 Allocator Combo",
    "",
    f"Combo directory: `{combo_dir}`",
    f"Summary: `{summary_path}`",
    "",
    "| method | out tok/s | mean TPOT ms | p99 TPOT ms | accepted/verified | wasted ratio | mean B | mean input tokens | shape pad | target us | graph-shape ok | hit rate | local probes |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    notes.append(
        "| {method} | {out:.2f} | {tpot:.2f} | {p99:.2f} | {apv:.4f} | {wasted:.4f} | {mean_b:.2f} | {mean_input:.2f} | {shape_pad:.2f} | {target:.2f} | {shape:.2f} | {hit:.2f} | {local_probes:.0f} |".format(
            method=row["method"],
            out=row["output_throughput"],
            tpot=row["mean_tpot_ms"],
            p99=row["p99_tpot_ms"],
            apv=row["accepted_per_verified_node"],
            wasted=row["wasted_node_ratio"],
            mean_b=row["mean_allocated_budget"],
            mean_input=row["mean_verify_input_tokens"],
            shape_pad=row["mean_shape_padding_tokens"],
            target=row["mean_target_forward_time_us"],
            shape=row["cuda_graph_token_shape_ok_ratio"],
            hit=row["strategy_bank_hit_rate"],
            local_probes=row["request_local_probe_count"],
        )
    )
Path(os.environ["NOTES_PATH"]).write_text("\n".join(notes) + "\n", encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
PY
}

for pair in ${MIXED_PAIRS}; do
  left_dataset="${pair%%:*}"
  right_dataset="${pair##*:}"
  for left_ratio in ${LEFT_RATIOS}; do
    ratio_id="${left_ratio/./p}"
    workload_path="${RUN_DIR}/workload_${left_dataset}_${right_dataset}_r${ratio_id}.jsonl"
    bench_dataset_path="${RUN_DIR}/bench_${left_dataset}_${right_dataset}_r${ratio_id}.json"
    .venv/bin/python scripts/run_mixed_workload.py binary \
      --left-dataset "${left_dataset}" \
      --right-dataset "${right_dataset}" \
      --left-ratio "${left_ratio}" \
      --count "${NUM_PROMPTS}" \
      --seed "${SEED}" \
      --output "${workload_path}"

    materialize_cmd=(/root/anaconda3/envs/sglang/bin/python scripts/materialize_workload.py \
      --workload "${workload_path}" \
      --normalized-dir "${WORKSPACE}/data/normalized" \
      --output "${bench_dataset_path}" \
      --max-rows "${NUM_PROMPTS}")
    if [ "${MATERIALIZE_FILL_FILTERED}" = "1" ]; then
      materialize_cmd+=( \
        --fill-filtered \
        --tokenizer-model "${MATERIALIZE_TOKENIZER_MODEL}" \
        --context-len "${SHAREGPT_CONTEXT_LEN}" \
        --fixed-output-len "${SHAREGPT_OUTPUT_LEN}" \
      )
    fi
    "${materialize_cmd[@]}"

    for b_batch in ${B_BATCH_VALUES}; do
      combo_dir="${RUN_DIR}/${left_dataset}_${right_dataset}_r${ratio_id}_B${b_batch}"
      mkdir -p "${combo_dir}"
      for method in ${METHODS}; do
        run_method "${combo_dir}" "${left_dataset}" "${right_dataset}" "${left_ratio}" "${b_batch}" "${bench_dataset_path}" "${method}"
      done
      summarize_combo "${combo_dir}" "${left_dataset}" "${right_dataset}" "${left_ratio}" "${b_batch}"
    done
  done
done

{
  echo "after"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader
} >> "${GPU_MEMORY_PATH}" || true

{
  echo "# Session 5 Allocator Sweep"
  echo
  echo "Run directory: \`${RUN_DIR}\`"
  echo "Sweep summary: \`${SWEEP_SUMMARY_PATH}\`"
  echo
  echo "Pairs: ${MIXED_PAIRS}"
  echo "Ratios: ${LEFT_RATIOS}"
  echo "B_batch values: ${B_BATCH_VALUES}"
  echo "CUDA graph max batch size: ${CUDA_GRAPH_MAX_BS}"
  echo "Max running requests: ${MAX_RUNNING_REQUESTS}"
  echo "Materialize fill filtered: ${MATERIALIZE_FILL_FILTERED}"
  echo "Methods: ${METHODS}"
} > "${NOTES_PATH}"

echo "A100 Session 5 allocator sweep run directory: ${RUN_DIR}"
