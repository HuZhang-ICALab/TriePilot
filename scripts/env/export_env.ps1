param(
  [string]$Python = ".\.miniconda\envs\triepilot\python.exe"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "experiments/env" | Out-Null
$out = "experiments/env/local_3050_env.txt"
"# TriePilot local environment" | Out-File -Encoding utf8 $out
Get-Date | Out-File -Append -Encoding utf8 $out
if (Test-Path ".git") {
  try {
    $gitHead = git rev-parse --verify HEAD 2>$null
  } catch {
    $gitHead = $null
  }
  if ($gitHead) {
    $gitHead | Out-File -Append -Encoding utf8 $out
  } else {
    "git: repository has no commits yet" | Out-File -Append -Encoding utf8 $out
  }
} else {
  "git: not a repository" | Out-File -Append -Encoding utf8 $out
}
& $Python -V | Out-File -Append -Encoding utf8 $out
nvidia-smi | Out-File -Append -Encoding utf8 $out
& $Python -m pip freeze | Out-File -Append -Encoding utf8 $out
Write-Host "Wrote $out"
