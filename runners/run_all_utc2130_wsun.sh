#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "ALLSTART_WSUN $(stamp)"
for VARIANT in utc2130_wsun utc2130_wsun_close utc2130_wsun_complete; do
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
echo "ALLDONE_WSUN $(stamp)"
