param(
  [string]$InstallDir = "$PWD\.miniconda"
)

$ErrorActionPreference = "Stop"
$url = "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$installer = Join-Path $env:TEMP "Miniconda3-latest-Windows-x86_64.exe"

Write-Host "Downloading Miniconda from Tsinghua mirror..."
curl.exe -L $url -o $installer

Write-Host "Installing Miniconda to $InstallDir ..."
Start-Process -FilePath $installer -ArgumentList @("/S", "/D=$InstallDir") -Wait

$conda = Join-Path $InstallDir "Scripts\conda.exe"
& $conda config --file (Join-Path $InstallDir ".condarc") --set show_channel_urls yes
Copy-Item -Force -Path "configs/env/condarc" -Destination (Join-Path $InstallDir ".condarc")

Write-Host "Creating triepilot Python 3.10.19 environment aligned with FlexKV..."
& $conda create -y -n triepilot python=3.10.19
Write-Host "Done. Activate with:"
Write-Host "$InstallDir\Scripts\activate triepilot"
