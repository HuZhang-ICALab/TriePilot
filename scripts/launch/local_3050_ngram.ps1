param(
  [string]$Model = "Qwen/Qwen2.5-0.5B-Instruct",
  [int]$Port = 30000
)

$ErrorActionPreference = "Stop"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PYTHONPATH = "$PWD\third_party\sglang_flex\python;$env:PYTHONPATH"
python -m sglang.launch_server `
  --model-path $Model `
  --host 0.0.0.0 `
  --port $Port `
  --mem-fraction-static 0.30 `
  --cuda-graph-max-bs 1 `
  --max-running-requests 1 `
  --speculative-algorithm NGRAM `
  --speculative-num-draft-tokens 2 `
  --speculative-ngram-max-match-window-size 4 `
  --speculative-ngram-max-bfs-breadth 1
