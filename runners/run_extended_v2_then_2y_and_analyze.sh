#!/usr/bin/env bash
# Wait until the 1y retrain finishes (110 cache files = 22 labels × 5 cells),
# then launch 2y, then run all analyses (extended_windows / pbo / dsr) for both
# tags and the 'base' tag.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_DIR="data"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

CACHE_1Y="${DATA_DIR}/preds_cache_extended_train1y"
CACHE_2Y="${DATA_DIR}/preds_cache_extended_train2y"
EXPECTED=110   # 22 labels * 5 cells

echo "[wait] for 1y retrain to finish (target ${EXPECTED} parquet files in ${CACHE_1Y}) ..."
while true; do
  if pgrep -f 'retrain_extended_v2.py.*train1y' >/dev/null; then
    n=$(ls "${CACHE_1Y}" 2>/dev/null | wc -l | tr -d ' ')
    sleep 60
    continue
  fi
  n=$(ls "${CACHE_1Y}" 2>/dev/null | wc -l | tr -d ' ')
  echo "[wait] 1y processes ended. cache count=${n}/${EXPECTED}"
  break
done

echo "[1y-done] launching 2y retrain ..."
"$SCRIPT_DIR/run_extended_v2.sh" 2017-12-01 train2y > "${LOG_DIR}/_extended_v2_train2y.log" 2>&1
echo "[2y-done] retrain finished. running analyses for all 3 cache tags ..."

for tag in base train1y train2y; do
  echo "================================================================"
  echo "[analyze] tag=${tag}"
  echo "================================================================"

  uv run python analysis/analyze_extended_windows_v2.py --cache-tag "${tag}" \
    > "${LOG_DIR}/_analyze_extwin_${tag}.log" 2>&1
  echo "  [extwin] done"

  uv run python analysis/analyze_pbo_extended_v2.py --cache-tag "${tag}" \
    > "${LOG_DIR}/_analyze_pbo_${tag}.log" 2>&1
  echo "  [pbo]    done"

  uv run python analysis/analyze_dsr_extended.py --cache-tag "${tag}" \
    > "${LOG_DIR}/_analyze_dsr_${tag}.log" 2>&1
  echo "  [dsr]    done"
done

echo "[all-done] retrain + analysis pipeline finished"
