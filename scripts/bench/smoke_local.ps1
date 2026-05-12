param(
  [string]$Model = "Qwen/Qwen2.5-0.5B-Instruct",
  [int]$Port = 30000
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "experiments/results" | Out-Null
python -m sglang.bench_serving `
  --backend sglang `
  --host 127.0.0.1 `
  --port $Port `
  --model $Model `
  --dataset-name random `
  --random-input-len 64 `
  --random-output-len 32 `
  --num-prompts 8 `
  --max-concurrency 1 `
  --output-file experiments/results/local_3050_smoke.jsonl

