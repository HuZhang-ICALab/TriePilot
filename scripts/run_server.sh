#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/root/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-sglang}"
MODEL_PATH="${MODEL_PATH:-/root/sglang_flex_test/models/Qwen/Qwen3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
SPECULATION="${SPECULATION:-none}"

cmd=(
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
)

if [ "${SPECULATION}" = "ngram" ]; then
  cmd+=(
    --speculative-algorithm NGRAM
    --speculative-num-draft-tokens "${DRAFT_TOKENS:-8}"
    --speculative-ngram-max-match-window-size "${MATCH_WINDOW:-12}"
    --speculative-ngram-max-bfs-breadth "${BFS_BREADTH:-4}"
  )
elif [ "${SPECULATION}" != "none" ]; then
  echo "Unsupported SPECULATION=${SPECULATION}; expected none or ngram" >&2
  exit 1
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
