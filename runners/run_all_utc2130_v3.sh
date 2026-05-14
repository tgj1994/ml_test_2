#!/usr/bin/env bash
# Sequentially run v3 SM variants.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "ALLSTART_V3 $(stamp)"
for VARIANT in utc2130_sm_v3 utc2130_sm_v3_close utc2130_sm_v3_complete; do
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
echo "ALLDONE_V3 $(stamp)"
