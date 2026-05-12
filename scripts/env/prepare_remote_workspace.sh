#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/TriePilot}"

case "${WORKSPACE}" in
  /root/TriePilot|/root/TriePilot/*) ;;
  *)
    echo "Refusing to prepare unexpected workspace: ${WORKSPACE}" >&2
    exit 1
    ;;
esac

mkdir -p \
  "${WORKSPACE}/data/normalized" \
  "${WORKSPACE}/data/sample_ids" \
  "${WORKSPACE}/logs" \
  "${WORKSPACE}/matrix" \
  "${WORKSPACE}/runs" \
  "${WORKSPACE}/workloads/homogeneous" \
  "${WORKSPACE}/workloads/mixed" \
  "${WORKSPACE}/workloads/shift"

cat > "${WORKSPACE}/README.remote.md" <<'EOF'
# TriePilot A100 Workspace

This directory is additive experiment workspace state for TriePilot.
It must not replace or mutate the existing conda environments, model weights,
or source datasets under:

- /root/anaconda3/envs/sglang
- /root/anaconda3/envs/vllm
- /root/sglang_flex_test/models
- /root/sglang_flex_test/datasets
EOF

echo "Prepared ${WORKSPACE}"
