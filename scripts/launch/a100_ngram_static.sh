#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-30000}"
DRAFT_TOKENS="${DRAFT_TOKENS:-8}"
MATCH_WINDOW="${MATCH_WINDOW:-12}"
BFS_BREADTH="${BFS_BREADTH:-4}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${PYTHONPATH:-/root/sglang_flex_test/sglang_flex/python}"

python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --mem-fraction-static 0.75 \
  --cuda-graph-max-bs 32 \
  --max-running-requests 32 \
  --speculative-algorithm NGRAM \
  --speculative-num-draft-tokens "${DRAFT_TOKENS}" \
  --speculative-ngram-max-match-window-size "${MATCH_WINDOW}" \
  --speculative-ngram-max-bfs-breadth "${BFS_BREADTH}"
