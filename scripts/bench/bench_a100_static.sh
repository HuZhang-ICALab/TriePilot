#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-30000}"
OUT="${OUT:-experiments/results/a100_static_random_c32.jsonl}"
mkdir -p "$(dirname "${OUT}")"

python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --model "${MODEL}" \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 256 \
  --random-range-ratio 0.5 \
  --num-prompts 320 \
  --max-concurrency 32 \
  --request-rate 32 \
  --output-file "${OUT}"

