$ErrorActionPreference = "Stop"
$pipDir = Join-Path $env:APPDATA "pip"
New-Item -ItemType Directory -Force -Path $pipDir | Out-Null
Copy-Item -Force -Path "configs/env/pip.ini" -Destination (Join-Path $pipDir "pip.ini")
Write-Host "Configured pip to use Tsinghua mirror at $pipDir\pip.ini"

