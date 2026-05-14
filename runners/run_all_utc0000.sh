#!/usr/bin/env bash
# Mirror every utc2130 variant at UTC midnight (00:00 UTC) decision time.
# Reports go under reports/utc0000/. Resume-safe: skips variants already done.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

# 45 utc2130 variants = 3 cadences × 5 feature-sets × 3 NaN-policies.
# Order: cheapest M-cadence first, then SM, then WSUN.
VARIANTS=(
  # M cadence × 5 feature-sets × 3 NaN-policies = 15
  utc2130                 utc2130_close              utc2130_complete
  utc2130_v3              utc2130_v3_close           utc2130_v3_complete
  utc2130_v4              utc2130_v4_close           utc2130_v4_complete
  utc2130_v5              utc2130_v5_close           utc2130_v5_complete
  utc2130_v6              utc2130_v6_close           utc2130_v6_complete
  # SM cadence × 5 × 3 = 15
  utc2130_sm              utc2130_sm_close           utc2130_sm_complete
  utc2130_sm_v3           utc2130_sm_v3_close        utc2130_sm_v3_complete
  utc2130_sm_v4           utc2130_sm_v4_close        utc2130_sm_v4_complete
  utc2130_sm_v5           utc2130_sm_v5_close        utc2130_sm_v5_complete
  utc2130_sm_v6           utc2130_sm_v6_close        utc2130_sm_v6_complete
  # WSUN cadence × 5 × 3 = 15
  utc2130_wsun            utc2130_wsun_close         utc2130_wsun_complete
  utc2130_wsun_v3         utc2130_wsun_v3_close      utc2130_wsun_v3_complete
  utc2130_wsun_v4         utc2130_wsun_v4_close      utc2130_wsun_v4_complete
  utc2130_wsun_v5         utc2130_wsun_v5_close      utc2130_wsun_v5_complete
  utc2130_wsun_v6         utc2130_wsun_v6_close      utc2130_wsun_v6_complete
)

# Optional tier filter: 'msm' = only M+SM, 'wsun' = only WSUN, 'all' = both (default).
TIER="${1:-all}"

echo "ALLSTART_UTC0000 $(stamp)  count=${#VARIANTS[@]} tier=${TIER}"
for SRC in "${VARIANTS[@]}"; do
  TGT="${SRC/utc2130/utc0000}"
  # Tier filter
  IS_WSUN=false; [[ "$SRC" == *wsun* ]] && IS_WSUN=true
  if [ "$TIER" = "msm"  ] && [ "$IS_WSUN" = "true"  ]; then continue; fi
  if [ "$TIER" = "wsun" ] && [ "$IS_WSUN" = "false" ]; then continue; fi
  # Resume-safe: if utc0000 report dir already exists, skip.
  if [ -d "reports/utc0000/${TGT}_730d" ]; then
    echo "VARIANT_SKIP  ${TGT} $(stamp)  (already done)"
    continue
  fi
  LOG="logs/${TGT}.log"
  echo "VARIANT_BEGIN ${TGT} $(stamp)  log=${LOG}"
  uv run python main_th_sweep_utc0000.py "${SRC}" --retrain >"${LOG}" 2>&1
  RC=$?
  echo "VARIANT_END   ${TGT} $(stamp)  rc=${RC}"
  if [ "${RC}" -ne 0 ]; then
    echo "VARIANT_FAILED ${TGT} — see ${LOG}"
    exit "${RC}"
  fi
done
echo "ALLDONE_UTC0000 $(stamp)  tier=${TIER}"
