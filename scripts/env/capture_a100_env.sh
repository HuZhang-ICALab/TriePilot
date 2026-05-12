#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/TriePilot}"
OUT_DIR="${OUT_DIR:-${WORKSPACE}/runs/$(date +%Y%m%d_%H%M%S)_env_capture}"
CONDA_BIN="${CONDA_BIN:-/root/anaconda3/bin/conda}"
PYTHON_BIN=("${CONDA_BIN}" run -n sglang python)

mkdir -p "${OUT_DIR}"

git_commit="unavailable"
if command -v git >/dev/null 2>&1 && [ -d "${WORKSPACE}/.git" ]; then
  git_commit="$(git -C "${WORKSPACE}" rev-parse --verify HEAD 2>/dev/null || echo no_commit)"
fi
printf '%s\n' "${git_commit}" > "${OUT_DIR}/git_commit.txt"

CAPTURE_SCRIPT="${OUT_DIR}/capture_env.py"
cat > "${CAPTURE_SCRIPT}" <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])
conda = sys.argv[2]

def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

def env_probe(env):
    code = (
        "import importlib, json, platform; "
        "mods=['torch','triton','sglang','vllm','datasets','transformers','yaml']; "
        "data={'python': platform.python_version(), 'packages': {}}; "
        "\nfor m in mods:\n"
        "    try:\n"
        "        mod=importlib.import_module(m); data['packages'][m]=getattr(mod,'__version__','installed')\n"
        "    except Exception as exc:\n"
        "        data['packages'][m]=f'unavailable: {type(exc).__name__}'\n"
        "print(json.dumps(data, sort_keys=True))"
    )
    raw = run([conda, "run", "-n", env, "python", "-c", code])
    try:
        return json.loads(raw.splitlines()[-1])
    except Exception:
        return {"probe_error": raw}

payload = {
    "host": run(["hostname"]),
    "platform": platform.platform(),
    "nvidia_smi": run(["nvidia-smi"]),
    "cuda_visible_devices": run(["bash", "-lc", "printf '%s' \"${CUDA_VISIBLE_DEVICES:-unset}\""]),
    "conda_envs": {
        "sglang": env_probe("sglang"),
        "vllm": env_probe("vllm"),
    },
    "asset_roots": {
        "models": "/root/sglang_flex_test/models",
        "datasets": "/root/sglang_flex_test/datasets",
    },
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
"${PYTHON_BIN[@]}" "${CAPTURE_SCRIPT}" "${OUT_DIR}/env.json" "${CONDA_BIN}"
rm -f "${CAPTURE_SCRIPT}"

"${CONDA_BIN}" run -n sglang python -m pip freeze > "${OUT_DIR}/sglang_pip_freeze.txt" || true
"${CONDA_BIN}" run -n vllm python -m pip freeze > "${OUT_DIR}/vllm_pip_freeze.txt" || true
find /root/sglang_flex_test/models -maxdepth 2 -mindepth 1 -print | sort > "${OUT_DIR}/models_manifest.txt" || true
find /root/sglang_flex_test/datasets -maxdepth 2 -mindepth 1 -print | sort > "${OUT_DIR}/datasets_manifest.txt" || true

echo "Wrote ${OUT_DIR}"
