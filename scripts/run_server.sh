#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/root/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-sglang}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/root/anaconda3/envs/${CONDA_ENV}}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/sglang_flex_test/models/Qwen/Qwen3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
SPECULATION="${SPECULATION:-none}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set CONDA_ENV_PATH or PYTHON_BIN explicitly." >&2
  exit 1
fi

export PATH="${CONDA_ENV_PATH}/bin:$(dirname "${CONDA_BIN}"):${PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${PWD}}"
WORKSPACE_SGLANG_SRC="${WORKSPACE_SGLANG_SRC:-${PWD}/third_party/sglang_flex/python}"
LEGACY_SGLANG_SRC="${LEGACY_SGLANG_SRC:-/root/sglang_flex_test/sglang_flex/python}"
if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${WORKSPACE_ROOT}:${WORKSPACE_SGLANG_SRC}:${PYTHONPATH}"
elif [ -d "${WORKSPACE_SGLANG_SRC}/sglang" ]; then
  export PYTHONPATH="${WORKSPACE_ROOT}:${WORKSPACE_SGLANG_SRC}"
else
  export PYTHONPATH="${WORKSPACE_ROOT}:${LEGACY_SGLANG_SRC}"
fi
if [ -n "${TRIEPILOT_DEBUG_PYTHONPATH:-}" ]; then
  echo "PYTHONPATH=${PYTHONPATH}"
fi
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${PWD}/runs/flashinfer_cache}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"

cmd=(
  "${PYTHON_BIN}" -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
)

if [ -n "${TRIEPILOT_TRACE_PATH:-}" ]; then
  cmd+=(--triepilot-trace-path "${TRIEPILOT_TRACE_PATH}")
fi

if [ -n "${TRIEPILOT_RUN_ID:-}" ]; then
  cmd+=(--triepilot-run-id "${TRIEPILOT_RUN_ID}")
fi

if [ -n "${TRIEPILOT_ALLOCATION_POLICY:-}" ]; then
  cmd+=(--triepilot-allocation-policy "${TRIEPILOT_ALLOCATION_POLICY}")
fi

if [ -n "${TRIEPILOT_BATCH_BUDGET:-}" ]; then
  cmd+=(--triepilot-batch-budget "${TRIEPILOT_BATCH_BUDGET}")
fi

if [ -n "${TRIEPILOT_RANDOM_SEED:-}" ]; then
  cmd+=(--triepilot-random-seed "${TRIEPILOT_RANDOM_SEED}")
fi

if [ -n "${TRIEPILOT_ACCEPT_EMA_ALPHA:-}" ]; then
  cmd+=(--triepilot-accept-ema-alpha "${TRIEPILOT_ACCEPT_EMA_ALPHA}")
fi

if [ "${SPECULATION}" = "ngram" ]; then
  DRAFT_TOKENS="${DRAFT_TOKENS:-8}"
  MATCH_WINDOW="${MATCH_WINDOW:-12}"
  BFS_BREADTH="${BFS_BREADTH:-4}"
  BRANCH_LENGTH="${BRANCH_LENGTH:-18}"
  if [ "${MATCH_WINDOW}" -ge "${BRANCH_LENGTH}" ]; then
    echo "MATCH_WINDOW (${MATCH_WINDOW}) must be less than BRANCH_LENGTH (${BRANCH_LENGTH}) for SGLang NGRAM." >&2
    exit 1
  fi
  cmd+=(
    --speculative-algorithm NGRAM
    --speculative-num-draft-tokens "${DRAFT_TOKENS}"
    --speculative-ngram-max-match-window-size "${MATCH_WINDOW}"
    --speculative-ngram-max-bfs-breadth "${BFS_BREADTH}"
    --speculative-ngram-branch-length "${BRANCH_LENGTH}"
  )
elif [ "${SPECULATION}" != "none" ]; then
  echo "Unsupported SPECULATION=${SPECULATION}; expected none or ngram" >&2
  exit 1
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
