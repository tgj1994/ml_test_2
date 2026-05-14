#!/usr/bin/env bash
# Run v5 (top-60 feature pruning) and v6 (tighter regularization) across
# SM and M cadences, all 3 NaN-handling variants.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "ALLSTART_V5V6 $(stamp)"
for VARIANT in \
    utc2130_sm_v5 utc2130_sm_v5_close utc2130_sm_v5_complete \
    utc2130_v5 utc2130_v5_close utc2130_v5_complete \
    utc2130_sm_v6 utc2130_sm_v6_close utc2130_sm_v6_complete \
    utc2130_v6 utc2130_v6_close utc2130_v6_complete \
; do
  LOG="logs/${VARIANT}.log"
  echo "VARIANT_BEGIN ${VARIANT} $(stamp)  log=${LOG}"
  ver="v0"; [[ "$VARIANT" =~ _v([0-9]+) ]] && ver="v${BASH_REMATCH[1]}"
  uv run python "main_th_sweep/${ver}/main_th_sweep_${VARIANT}.py" --retrain >"${LOG}" 2>&1
  RC=$?
  echo "VARIANT_END   ${VARIANT} $(stamp)  rc=${RC}"
  if [ "${RC}" -ne 0 ]; then
    echo "VARIANT_FAILED ${VARIANT} — see ${LOG}"
    exit "${RC}"
  fi
done
echo "ALLDONE_V5V6 $(stamp)"

echo "REPORT_BEGIN $(stamp)"
uv run python analysis/compare_v0_v3.py >logs/compare_v0_v3.log 2>&1
echo "REPORT_END   $(stamp)"
