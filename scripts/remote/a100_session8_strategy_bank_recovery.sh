#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/TriePilot}"
RUN_ID="${RUN_ID:-20260516_session8_selective_recovery_seed20260516}"
RUN_DIR="${RUN_DIR:-${WORKSPACE}/runs/${RUN_ID}}"
MODEL_PATH="${MODEL_PATH:-/root/sglang_flex_test/models/Qwen/Qwen3-8B}"
PHASES="${PHASES:-cold_code:instructcoder shift_math:gsm8k shift_json:json_tool warm_code_reuse:instructcoder}"
POLICIES="${POLICIES:-triepilot_no_recovery_B100 triepilot_selective_recovery_B100 equal_floor_B16 batch_global_budget2}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
REQUEST_RATE="${REQUEST_RATE:-8}"
SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN:-64}"
SHAREGPT_CONTEXT_LEN="${SHAREGPT_CONTEXT_LEN:-4096}"
MATCH_WINDOW="${MATCH_WINDOW:-12}"
BFS_BREADTH="${BFS_BREADTH:-4}"
BRANCH_LENGTH="${BRANCH_LENGTH:-18}"
MAX_DRAFT_TOKENS="${MAX_DRAFT_TOKENS:-16}"
B_BATCH="${B_BATCH:-100}"
RECOVERY_BUDGET="${RECOVERY_BUDGET:-2}"
RECOVERY_COOLDOWN_STEPS="${RECOVERY_COOLDOWN_STEPS:-16}"
RECOVERY_MIN_GAIN="${RECOVERY_MIN_GAIN:-0.20}"
REQUEST_LOCAL_RECOVERY_PROBE_BUDGET="${REQUEST_LOCAL_RECOVERY_PROBE_BUDGET:-2}"
REQUEST_LOCAL_RECOVERY_MAX_PROBES="${REQUEST_LOCAL_RECOVERY_MAX_PROBES:-2}"
REQUEST_LOCAL_RECOVERY_ACCEPT_THRESHOLD="${REQUEST_LOCAL_RECOVERY_ACCEPT_THRESHOLD:-0.75}"
REQUEST_LOCAL_RECOVERY_SUSTAIN_BUDGET="${REQUEST_LOCAL_RECOVERY_SUSTAIN_BUDGET:-4}"
TRIEPILOT_SHAPE_BUCKETS="${TRIEPILOT_SHAPE_BUCKETS:-0/1,2,4,8,16}"
TRIEPILOT_SHAPE_BUCKET_MODE="${TRIEPILOT_SHAPE_BUCKET_MODE:-batch_max}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
MATERIALIZE_TOKENIZER_MODEL="${MATERIALIZE_TOKENIZER_MODEL:-${MODEL_PATH}}"
SEED="${SEED:-20260516}"
PORT_BASE="${PORT_BASE:-31300}"

mkdir -p "${RUN_DIR}"
cd "${WORKSPACE}"

SUMMARY_PATH="${RUN_DIR}/phase_summary.csv"
COMPARISON_PATH="${RUN_DIR}/recovery_comparison.csv"
INTEGRITY_PATH="${RUN_DIR}/integrity_session8_selective_recovery.json"
ANALYSIS_PATH="${RUN_DIR}/analysis_session8_selective_recovery.md"
CONFIG_PATH="${RUN_DIR}/config.yaml"
ENV_PATH="${RUN_DIR}/env.json"
GIT_COMMIT_PATH="${RUN_DIR}/git_commit.txt"
GPU_MEMORY_PATH="${RUN_DIR}/gpu_memory.txt"

git rev-parse HEAD > "${GIT_COMMIT_PATH}" || true

{
  echo "run_id: ${RUN_ID}"
  echo "model_path: ${MODEL_PATH}"
  echo "phases: [${PHASES// /, }]"
  echo "policies: [${POLICIES// /, }]"
  echo "num_prompts: ${NUM_PROMPTS}"
  echo "max_concurrency: ${MAX_CONCURRENCY}"
  echo "request_rate: ${REQUEST_RATE}"
  echo "sharegpt_output_len: ${SHAREGPT_OUTPUT_LEN}"
  echo "sharegpt_context_len: ${SHAREGPT_CONTEXT_LEN}"
  echo "max_draft_tokens: ${MAX_DRAFT_TOKENS}"
  echo "b_batch: ${B_BATCH}"
  echo "recovery_budget: ${RECOVERY_BUDGET}"
  echo "recovery_cooldown_steps: ${RECOVERY_COOLDOWN_STEPS}"
  echo "recovery_min_gain: ${RECOVERY_MIN_GAIN}"
  echo "request_local_recovery_probe_budget: ${REQUEST_LOCAL_RECOVERY_PROBE_BUDGET}"
  echo "request_local_recovery_max_probes: ${REQUEST_LOCAL_RECOVERY_MAX_PROBES}"
  echo "request_local_recovery_accept_threshold: ${REQUEST_LOCAL_RECOVERY_ACCEPT_THRESHOLD}"
  echo "request_local_recovery_sustain_budget: ${REQUEST_LOCAL_RECOVERY_SUSTAIN_BUDGET}"
  echo "shape_buckets: ${TRIEPILOT_SHAPE_BUCKETS}"
  echo "shape_bucket_mode: ${TRIEPILOT_SHAPE_BUCKET_MODE}"
  echo "cuda_graph_max_bs: ${CUDA_GRAPH_MAX_BS}"
  echo "max_running_requests: ${MAX_RUNNING_REQUESTS}"
  echo "seed: ${SEED}"
} > "${CONFIG_PATH}"

ENV_PATH="${ENV_PATH}" .venv/bin/python - <<'PY'
import json
import os
import platform
import subprocess
from pathlib import Path

env = {"python": platform.python_version(), "platform": platform.platform()}
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
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader
} > "${GPU_MEMORY_PATH}" || true

rm -f "${SUMMARY_PATH}" "${COMPARISON_PATH}" "${INTEGRITY_PATH}" "${ANALYSIS_PATH}"

for phase_spec in ${PHASES}; do
  phase_name="${phase_spec%%:*}"
  dataset="${phase_spec##*:}"
  workload_path="${RUN_DIR}/workload_${phase_name}_${dataset}.jsonl"
  bench_dataset_path="${RUN_DIR}/bench_${phase_name}_${dataset}.json"
  .venv/bin/python scripts/run_mixed_workload.py homogeneous \
    --dataset "${dataset}" \
    --count "${NUM_PROMPTS}" \
    --seed "${SEED}" \
    --output "${workload_path}"
  /root/anaconda3/envs/sglang/bin/python scripts/materialize_workload.py \
    --workload "${workload_path}" \
    --normalized-dir "${WORKSPACE}/data/normalized" \
    --output "${bench_dataset_path}" \
    --max-rows "${NUM_PROMPTS}" \
    --fill-filtered \
    --tokenizer-model "${MATERIALIZE_TOKENIZER_MODEL}" \
    --context-len "${SHAREGPT_CONTEXT_LEN}" \
    --fixed-output-len "${SHAREGPT_OUTPUT_LEN}"
done

current_pid=""
cleanup_server() {
  if [ -n "${current_pid}" ] && kill -0 "${current_pid}" >/dev/null 2>&1; then
    kill -- -"${current_pid}" >/dev/null 2>&1 || kill "${current_pid}" >/dev/null 2>&1 || true
    wait "${current_pid}" >/dev/null 2>&1 || true
  fi
  current_pid=""
}
trap cleanup_server EXIT

line_count() {
  local path="$1"
  if [ -s "${path}" ]; then
    wc -l < "${path}"
  else
    echo 0
  fi
}

port_offset=0
for policy in ${POLICIES}; do
  policy_dir="${RUN_DIR}/${policy}"
  mkdir -p "${policy_dir}"
  trace_path="${policy_dir}/raw_step_events.jsonl"
  server_log="${policy_dir}/server.log"
  rm -f "${trace_path}" "${server_log}" "${policy_dir}/server.pid"
  : > "${trace_path}"

  allocation_policy="triepilot_allocation"
  draft_tokens="${MAX_DRAFT_TOKENS}"
  batch_budget="${B_BATCH}"
  recovery_enabled="0"
  request_local_recovery_enabled="0"
  case "${policy}" in
    triepilot_no_recovery_B100)
      allocation_policy="triepilot_allocation"
      recovery_enabled="0"
      ;;
    triepilot_selective_recovery_B100)
      allocation_policy="triepilot_allocation"
      recovery_enabled="1"
      ;;
    triepilot_request_local_recovery_B100)
      allocation_policy="triepilot_allocation"
      recovery_enabled="0"
      request_local_recovery_enabled="1"
      ;;
    equal_floor_B16)
      allocation_policy="equal_budget_allocation"
      batch_budget="16"
      recovery_enabled="0"
      ;;
    batch_global_budget2)
      allocation_policy="custom"
      draft_tokens="2"
      batch_budget=""
      recovery_enabled="0"
      ;;
    *)
      echo "Unknown Session 8 policy: ${policy}" >&2
      exit 1
      ;;
  esac

  port="$((PORT_BASE + port_offset))"
  port_offset="$((port_offset + 1))"
  if ss -ltn | grep -q ":${port} "; then
    echo "Port ${port} is already in use." >&2
    exit 1
  fi

  export PORT="${port}"
  export MODEL_PATH
  export MATCH_WINDOW
  export BFS_BREADTH
  export BRANCH_LENGTH
  export WORKSPACE_SGLANG_SRC="${WORKSPACE}/third_party/sglang_flex/python"
  export TRIEPILOT_TRACE_PATH="${trace_path}"
  export TRIEPILOT_RUN_ID="${RUN_ID}_${policy}"
  export TRIEPILOT_ALLOCATION_POLICY="${allocation_policy}"
  export TRIEPILOT_RANDOM_SEED="${SEED}"
  export TRIEPILOT_SHAPE_BUCKETS
  export TRIEPILOT_SHAPE_BUCKET_MODE
  export CUDA_GRAPH_MAX_BS
  export MAX_RUNNING_REQUESTS
  export FLASHINFER_WORKSPACE_BASE="${policy_dir}/flashinfer_cache"
  export SPECULATION="ngram"
  export DRAFT_TOKENS="${draft_tokens}"
  export TRIEPILOT_SELECTIVE_RECOVERY="${recovery_enabled}"
  export TRIEPILOT_SELECTIVE_RECOVERY_BUDGET="${RECOVERY_BUDGET}"
  export TRIEPILOT_SELECTIVE_RECOVERY_COOLDOWN_STEPS="${RECOVERY_COOLDOWN_STEPS}"
  export TRIEPILOT_SELECTIVE_RECOVERY_MIN_GAIN="${RECOVERY_MIN_GAIN}"
  export TRIEPILOT_REQUEST_LOCAL_RECOVERY="${request_local_recovery_enabled}"
  export TRIEPILOT_REQUEST_LOCAL_RECOVERY_PROBE_BUDGET="${REQUEST_LOCAL_RECOVERY_PROBE_BUDGET}"
  export TRIEPILOT_REQUEST_LOCAL_RECOVERY_MAX_PROBES="${REQUEST_LOCAL_RECOVERY_MAX_PROBES}"
  export TRIEPILOT_REQUEST_LOCAL_RECOVERY_ACCEPT_THRESHOLD="${REQUEST_LOCAL_RECOVERY_ACCEPT_THRESHOLD}"
  export TRIEPILOT_REQUEST_LOCAL_RECOVERY_SUSTAIN_BUDGET="${REQUEST_LOCAL_RECOVERY_SUSTAIN_BUDGET}"
  if [ -n "${batch_budget}" ]; then
    export TRIEPILOT_BATCH_BUDGET="${batch_budget}"
  else
    unset TRIEPILOT_BATCH_BUDGET || true
  fi

  setsid ./scripts/run_server.sh >"${server_log}" 2>&1 &
  current_pid="$!"
  echo "${current_pid}" > "${policy_dir}/server.pid"

  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${current_pid}" >/dev/null 2>&1; then
      echo "SGLang server exited before becoming healthy for ${policy}." >&2
      tail -200 "${server_log}" >&2 || true
      exit 1
    fi
    sleep 5
  done
  if [ "${ready}" != "1" ]; then
    echo "SGLang server did not become healthy on port ${port} for ${policy}." >&2
    tail -200 "${server_log}" >&2 || true
    exit 1
  fi

  for phase_spec in ${PHASES}; do
    phase_name="${phase_spec%%:*}"
    dataset="${phase_spec##*:}"
    phase_dir="${policy_dir}/${phase_name}"
    mkdir -p "${phase_dir}"
    bench_dataset_path="${RUN_DIR}/bench_${phase_name}_${dataset}.json"
    bench_output="${phase_dir}/raw_events_${phase_name}_${dataset}.jsonl"
    phase_trace="${phase_dir}/raw_step_events_${phase_name}.jsonl"
    before_lines="$(line_count "${trace_path}")"
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
    after_lines="$(line_count "${trace_path}")"
    if [ "${after_lines}" -le "${before_lines}" ]; then
      echo "No new step events for ${policy}/${phase_name}." >&2
      exit 1
    fi
    sed -n "$((before_lines + 1)),${after_lines}p" "${trace_path}" > "${phase_trace}"
  done

  cleanup_server
done

SUMMARY_PATH="${SUMMARY_PATH}" \
COMPARISON_PATH="${COMPARISON_PATH}" \
INTEGRITY_PATH="${INTEGRITY_PATH}" \
ANALYSIS_PATH="${ANALYSIS_PATH}" \
RUN_DIR="${RUN_DIR}" \
PHASES="${PHASES}" \
POLICIES="${POLICIES}" \
NUM_PROMPTS="${NUM_PROMPTS}" \
B_BATCH="${B_BATCH}" \
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


def quantile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, int(len(values) * q))
    return values[index]


def graph_shape_ok(event: dict) -> bool:
    if "cuda_graph_token_shape_ok" in event:
        return bool(event["cuda_graph_token_shape_ok"])
    expected = int(event.get("batch_size", 0)) * int(event.get("draft_token_num", 0))
    return int(event.get("verify_input_tokens", 0)) == expected


run_dir = Path(os.environ["RUN_DIR"])
phase_specs = [item.split(":", 1) for item in os.environ["PHASES"].split()]
policies = os.environ["POLICIES"].split()
rows = []
budget_violations = 0
server_log_errors = []
trace_events_total = 0
all_bench_completed = True
all_traces_nonempty = True

for policy in policies:
    server_log = (run_dir / policy / "server.log").read_text(
        encoding="utf-8", errors="ignore"
    )
    if any(token in server_log for token in ["Traceback", "CUDA error", "illegal memory access", "RuntimeError"]):
        server_log_errors.append(policy)
    for phase_name, dataset in phase_specs:
        phase_dir = run_dir / policy / phase_name
        trace_path = phase_dir / f"raw_step_events_{phase_name}.jsonl"
        bench_path = phase_dir / f"raw_events_{phase_name}_{dataset}.jsonl"
        events = read_jsonl(trace_path)
        bench_rows = read_jsonl(bench_path)
        bench = bench_rows[0] if bench_rows else {}
        trace_events_total += len(events)
        all_traces_nonempty = all_traces_nonempty and bool(events)
        completed = int(bench.get("completed", bench.get("successful_requests", 0)) or 0)
        all_bench_completed = all_bench_completed and completed == int(os.environ["NUM_PROMPTS"])
        verified = sum(float(event.get("verified_nodes", 0)) for event in events)
        accepted = sum(float(event.get("accepted_tokens", 0)) for event in events)
        wasted = sum(float(event.get("wasted_nodes", 0)) for event in events)
        recovery_probe_count = sum(int(event.get("recovery_probe_count", 0)) for event in events)
        request_local_probe_count = sum(int(event.get("request_local_probe_count", 0)) for event in events)
        nonzero_budget = []
        allocated_sums = []
        recovery_event_flags = []
        request_local_event_flags = []
        for event in events:
            budgets = [int(value) for value in event.get("allocated_budgets", [])]
            if (
                not policy.startswith("batch_global_budget")
                and sum(budgets) > int(os.environ["B_BATCH"])
            ):
                budget_violations += 1
            if budgets:
                nonzero_budget.extend([1.0 if budget > 0 else 0.0 for budget in budgets])
                allocated_sums.append(sum(budgets))
            recovery_event_flags.append(1.0 if int(event.get("recovery_probe_count", 0)) > 0 else 0.0)
            request_local_event_flags.append(1.0 if int(event.get("request_local_probe_count", 0)) > 0 else 0.0)
        rows.append(
            {
                "policy": policy,
                "phase": phase_name,
                "dataset": dataset,
                "trace_events": len(events),
                "completed": completed,
                "verified_nodes": verified,
                "accepted_tokens": accepted,
                "accepted_per_verified_node": accepted / verified if verified else 0.0,
                "wasted_node_ratio": wasted / verified if verified else 0.0,
                "mean_allocated_budget": mean(allocated_sums),
                "nonzero_request_budget_share": mean(nonzero_budget),
                "recovery_probe_count": recovery_probe_count,
                "recovery_probe_event_share": mean(recovery_event_flags),
                "request_local_probe_count": request_local_probe_count,
                "request_local_probe_event_share": mean(request_local_event_flags),
                "strategy_bank_hit_rate": mean([float(event.get("strategy_bank_hit_rate", 0.0)) for event in events]),
                "mean_tpot_ms": float(bench.get("mean_tpot_ms", 0.0)),
                "p99_tpot_ms": float(bench.get("p99_tpot_ms", 0.0)),
                "output_throughput": float(bench.get("output_throughput", 0.0)),
                "mean_controller_us": mean([float(event.get("controller_time_us", 0.0)) for event in events]),
                "p99_controller_us": quantile([float(event.get("controller_time_us", 0.0)) for event in events], 0.99),
                "cuda_graph_token_shape_ok_ratio": mean([1.0 if graph_shape_ok(event) else 0.0 for event in events]),
                "trace_path": str(trace_path),
                "bench_output": str(bench_path),
            }
        )

with Path(os.environ["SUMMARY_PATH"]).open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

by_key = {(row["policy"], row["phase"]): row for row in rows}
comparisons = []
for phase_name, _ in phase_specs:
    base = by_key.get(("triepilot_no_recovery_B100", phase_name))
    recovery_policies = [
        "triepilot_selective_recovery_B100",
        "triepilot_request_local_recovery_B100",
    ]
    if not base:
        continue
    for recovery_policy in recovery_policies:
        recovery = by_key.get((recovery_policy, phase_name))
        if not recovery:
            continue
        comparisons.append(
            {
                "comparison": f"{recovery_policy}_minus_no_recovery_{phase_name}",
                "phase": phase_name,
                "verified_delta": recovery["verified_nodes"] - base["verified_nodes"],
                "accepted_delta": recovery["accepted_tokens"] - base["accepted_tokens"],
                "apv_delta": recovery["accepted_per_verified_node"] - base["accepted_per_verified_node"],
                "mean_tpot_delta_ms": recovery["mean_tpot_ms"] - base["mean_tpot_ms"],
                "p99_tpot_delta_ms": recovery["p99_tpot_ms"] - base["p99_tpot_ms"],
                "recovery_probe_count": recovery["recovery_probe_count"],
                "request_local_probe_count": recovery["request_local_probe_count"],
            }
        )

with Path(os.environ["COMPARISON_PATH"]).open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)

integrity = {
    "summary_rows": len(rows),
    "trace_events_total": trace_events_total,
    "all_traces_nonempty": all_traces_nonempty,
    "all_bench_completed": all_bench_completed,
    "budget_violation_count": budget_violations,
    "min_cuda_graph_token_shape_ok_ratio": min(row["cuda_graph_token_shape_ok_ratio"] for row in rows),
    "server_log_error_policies": server_log_errors,
}
Path(os.environ["INTEGRITY_PATH"]).write_text(
    json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8"
)

notes = [
    "# Session 8 Selective Recovery",
    "",
    f"Run directory: `{run_dir}`",
    "",
    "| policy | phase | verified | accepted | APV | recovery probes | request-local probes | nonzero req budget | mean TPOT | p99 TPOT | graph shape ok |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    notes.append(
        "| {policy} | {phase} | {verified:.0f} | {accepted:.0f} | {apv:.4f} | {probes:.0f} | {local_probes:.0f} | {nonzero:.3f} | {mean_tpot:.2f} | {p99:.2f} | {graph:.2f} |".format(
            policy=row["policy"],
            phase=row["phase"],
            verified=row["verified_nodes"],
            accepted=row["accepted_tokens"],
            apv=row["accepted_per_verified_node"],
            probes=row["recovery_probe_count"],
            local_probes=row["request_local_probe_count"],
            nonzero=row["nonzero_request_budget_share"],
            mean_tpot=row["mean_tpot_ms"],
            p99=row["p99_tpot_ms"],
            graph=row["cuda_graph_token_shape_ok_ratio"],
        )
    )
notes.extend(["", "## Selective Recovery vs No Recovery", ""])
for row in comparisons:
    notes.append(
        "- {comparison}: verified_delta={verified_delta:.0f}, accepted_delta={accepted_delta:.0f}, APV_delta={apv_delta:.4f}, mean_TPOT_delta={mean_tpot_delta_ms:.2f}ms, p99_TPOT_delta={p99_tpot_delta_ms:.2f}ms, recovery_probes={recovery_probe_count:.0f}, request_local_probes={request_local_probe_count:.0f}".format(**row)
    )
notes.extend(["", "## Integrity", ""])
for key, value in integrity.items():
    notes.append(f"- {key}: {value}")
Path(os.environ["ANALYSIS_PATH"]).write_text("\n".join(notes) + "\n", encoding="utf-8")
PY

{
  echo "after"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader
} >> "${GPU_MEMORY_PATH}" || true

echo "A100 Session 8 selective recovery run directory: ${RUN_DIR}"
