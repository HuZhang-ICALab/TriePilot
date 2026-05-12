param(
  [string]$Remote = "root@10.106.17.98",
  [int]$Port = 10022,
  [string]$IdentityFile = "E:\flexkv\.codex_ssh\flexkv_4gpu_ed25519",
  [string]$RemotePath = "/root/sglang_flex_test/sglang_flex",
  [string]$Target = "third_party\sglang_flex"
)

$ErrorActionPreference = "Stop"
if (Test-Path $Target) {
  Write-Host "$Target already exists. Remove or rename it before resyncing."
  exit 0
}

scp -r -P $Port -i $IdentityFile -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new "${Remote}:${RemotePath}" "third_party\"
Write-Host "Synced FlexKV SGLang source to $Target"

