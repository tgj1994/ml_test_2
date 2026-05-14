#!/usr/bin/env bash
# Run all 5 cells in parallel for one (--training-start, --cache-tag) variant.
# Usage:
#   ./run_extended_v2.sh 2018-12-01 train1y
#   ./run_extended_v2.sh 2017-12-01 train2y
set -euo pipefail
cd "$(dirname "$0")/.."

START="${1:?training-start (e.g. 2018-12-01)}"
TAG="${2:?cache-tag (e.g. train1y)}"

LOG_DIR="logs/extended_retrain_${TAG}"
mkdir -p "${LOG_DIR}"

CELLS=(
  utc2130_sm
  utc2130_sm_v3
  utc2130_sm_v3_complete
  utc2130_sm_v5
  utc2130_sm_v6
)

PIDS=()
for c in "${CELLS[@]}"; do
  echo "[launch] cell=${c}  start=${START}  tag=${TAG}"
  OMP_NUM_THREADS=2 uv run python retrain/retrain_extended_v2.py \
    --cell "${c}" \
    --training-start "${START}" \
    --cache-tag "${TAG}" \
    > "${LOG_DIR}/${c}.log" 2>&1 &
  PIDS+=($!)
done

echo "[wait] PIDs: ${PIDS[*]}"
fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
    echo "[error] pid=${pid} failed"
  fi
done

echo "[done] tag=${TAG}  fail=${fail}"
exit $fail
