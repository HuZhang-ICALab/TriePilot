#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-${MODEL:-/root/sglang_flex_test/models/Qwen/Qwen3-8B}}"
export PORT="${PORT:-30000}"
export SPECULATION=ngram
export DRAFT_TOKENS="${DRAFT_TOKENS:-8}"
export MATCH_WINDOW="${MATCH_WINDOW:-12}"
export BFS_BREADTH="${BFS_BREADTH:-4}"
export BRANCH_LENGTH="${BRANCH_LENGTH:-18}"

exec "${REPO_ROOT}/scripts/run_server.sh"
