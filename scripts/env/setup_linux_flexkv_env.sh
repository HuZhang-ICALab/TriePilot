#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-triepilot}"
SGLANG_SRC="${SGLANG_SRC:-third_party/sglang_flex/python}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found" >&2
  exit 1
fi

conda config --set show_channel_urls yes
conda config --remove-key default_channels >/dev/null 2>&1 || true
conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --set custom_channels.conda-forge https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
conda config --set custom_channels.pytorch https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
conda config --set custom_channels.nvidia https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10.19
fi

conda activate "$ENV_NAME"
python -m pip config set global.index-url "$PIP_INDEX_URL"
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

if [ -d "$SGLANG_SRC" ]; then
  python -m pip install -e "$SGLANG_SRC"
else
  echo "SGLang source not found at $SGLANG_SRC; skip editable install" >&2
fi

python scripts/env/check_flexkv_alignment.py

