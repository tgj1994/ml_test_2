#!/usr/bin/env bash
# Run v4 (= v3 + Global M2 + restored macro z_60d) across SM and M cadences,
# all 3 NaN-handling variants. Then auto-generate the comparison report.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "ALLSTART_V4 $(stamp)"
for VARIANT in \
    utc2130_sm_v4 utc2130_sm_v4_close utc2130_sm_v4_complete \
    utc2130_v4 utc2130_v4_close utc2130_v4_complete \
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
echo "ALLDONE_V4 $(stamp)"

# Generate comparison report.
echo "REPORT_BEGIN $(stamp)"
uv run python analysis/compare_v0_v3.py >logs/compare_v0_v3.log 2>&1
echo "REPORT_END   $(stamp)"
