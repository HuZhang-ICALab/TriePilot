param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
& .\scripts\env\setup_pip_sources.ps1
$version = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
if (-not $version.StartsWith("3.10.")) {
  throw "FlexKV alignment requires Python 3.10.x, target 3.10.19. Got $version from $Python"
}
& $Python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Write-Host "Created .venv with domestic pip source."
