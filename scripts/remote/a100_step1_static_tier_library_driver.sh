#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/TriePilot}"
DRIVER_RUN_ID="${DRIVER_RUN_ID:-20260512_step1_static_tier_library_driver_seed20260512}"
DRIVER_DIR="${WORKSPACE}/runs/${DRIVER_RUN_ID}"
DATASETS="${TRIEPILOT_STEP1_DATASETS:-instructcoder json_tool sharegpt gsm8k cnn_dailymail shared_prefix}"
SHAPES="${TRIEPILOT_STEP1_SHAPES:-mw12_b4:12:4:18 mw24_b8:24:8:34}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
REQUEST_RATE="${REQUEST_RATE:-8}"
SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN:-16}"
SHAREGPT_CONTEXT_LEN="${SHAREGPT_CONTEXT_LEN:-4096}"
SEED="${SEED:-20260512}"
PORT_BASE="${PORT_BASE:-30100}"

cd "${WORKSPACE}"
mkdir -p "${DRIVER_DIR}"
: > "${DRIVER_DIR}/completed_runs.txt"
: > "${DRIVER_DIR}/driver.progress.log"

for shape_spec in ${SHAPES}; do
  IFS=: read -r shape_name match_window bfs_breadth branch_length <<<"${shape_spec}"
  for dataset in ${DATASETS}; do
    run_id="20260512_step1_static_ngram_tiers_${dataset}_${shape_name}_seed${SEED}"
    echo "[$(date -Is)] starting ${run_id}" | tee -a "${DRIVER_DIR}/driver.progress.log"
    DATASET_NAME="${dataset}" \
      RUN_ID="${run_id}" \
      MATCH_WINDOW="${match_window}" \
      BFS_BREADTH="${bfs_breadth}" \
      BRANCH_LENGTH="${branch_length}" \
      NUM_PROMPTS="${NUM_PROMPTS}" \
      MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
      REQUEST_RATE="${REQUEST_RATE}" \
      SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN}" \
      SHAREGPT_CONTEXT_LEN="${SHAREGPT_CONTEXT_LEN}" \
      SEED="${SEED}" \
      PORT_BASE="${PORT_BASE}" \
      ./scripts/remote/a100_step1_static_tiers.sh
    echo "${run_id}" >> "${DRIVER_DIR}/completed_runs.txt"
    echo "[$(date -Is)] completed ${run_id}" | tee -a "${DRIVER_DIR}/driver.progress.log"
  done
done

DRIVER_DIR="${DRIVER_DIR}" WORKSPACE="${WORKSPACE}" .venv/bin/python - <<'PY'
import csv
import os
from pathlib import Path

workspace = Path(os.environ["WORKSPACE"])
run_root = workspace / "runs"
driver_dir = Path(os.environ["DRIVER_DIR"])
run_ids = [
    line.strip()
    for line in (driver_dir / "completed_runs.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
rows = []
for run_id in run_ids:
    path = run_root / run_id / "tier_summary.csv"
    if not path.exists():
        raise SystemExit(f"missing summary: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
if not rows:
    raise SystemExit("no summary rows collected")

combined = driver_dir / "combined_tier_summary.csv"
with combined.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

keys = sorted({(r["dataset_name"], r["match_window"], r["bfs_breadth"]) for r in rows})
notes = [
    "# Step 1 Static Tier Library Driver",
    "",
    f"Combined summary: `{combined}`",
    "",
    "| dataset | shape | best APV budget | best mean TPOT budget | best p99 TPOT budget |",
    "| --- | --- | ---: | ---: | ---: |",
]
for key in keys:
    subset = [r for r in rows if (r["dataset_name"], r["match_window"], r["bfs_breadth"]) == key]

    def number(row, col):
        try:
            return float(row[col])
        except Exception:
            return 0.0

    best_apv = max(subset, key=lambda r: number(r, "accepted_per_verified_node"))
    best_tpot = min(
        subset,
        key=lambda r: number(r, "mean_tpot_ms") if number(r, "mean_tpot_ms") > 0 else float("inf"),
    )
    best_p99 = min(
        subset,
        key=lambda r: number(r, "p99_tpot_ms") if number(r, "p99_tpot_ms") > 0 else float("inf"),
    )
    dataset, mw, bfs = key
    notes.append(
        f"| {dataset} | mw{mw}/b{bfs} | {best_apv['budget']} | {best_tpot['budget']} | {best_p99['budget']} |"
    )
(driver_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
print(combined)
PY
