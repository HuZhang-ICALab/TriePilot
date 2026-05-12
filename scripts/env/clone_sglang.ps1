param(
  [string]$Repo = "https://github.com/sgl-project/sglang.git",
  [string]$Commit = "",
  [string]$Target = "third_party/sglang"
)

$ErrorActionPreference = "Stop"
if (Test-Path $Target) {
  Write-Host "$Target already exists."
} else {
  git clone $Repo $Target
}
if ($Commit -ne "") {
  git -C $Target checkout $Commit
}
git -C $Target rev-parse HEAD | Set-Content -Encoding utf8 "experiments/env/sglang_commit.txt"

