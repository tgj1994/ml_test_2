#!/usr/bin/env bash
#
# Full sweep entry point for ml_test_2.
#
# DIFFERENCE FROM ml_test: data/ in this repo is the 2019-12-01 UTC cutoff
# version (≈ 22% fewer rows than the full-history ml_test). The training
# pipeline, calibration, model code, and phase order are identical — the
# only experimental knob is the training start date.
#
# Hardware target (server_spec): AMD EPYC 75F3, 64 vCPU, 566 GiB RAM, no GPU,
# Ubuntu 22.04. Load target ~62.5% (throttle=20 × threads=2 = 40 logical
# cores). Both EBM and XGB use isotonic calibration (cv=3); XGB device=cpu.
#
# Phase order (matches the local Windows watchdog chain used during
# development — v5/v5_2/v5_3 are pruned variants whose keep lists are
# regenerated dynamically by v5_keep_selector_ebm.py after the base EBM
# sweeps produce per-cell feature_importance CSVs):
#
#   1. EBM base sweeps         : v0 → v3 → v4 → v6
#   2. v5_selector             : cum 0.80 → src/v5_keep.py
#   3. v5 EBM
#   4. v5_2_selector           : cum 0.92 → src/v5_2_keep.py
#   5. v5_2 EBM
#   6. v7 EBM                  : broadest input (FRED Tier-1 + GDELT)
#   7. v5_3_selector           : cum 0.93 → src/v5_3_keep.py
#   8. v5_3 EBM
#   9. XGB chain               : same 8 versions, calib=isotonic
#
# Usage (on the server, after `git clone`):
#   chmod +x start.sh
#   nohup bash start.sh > sweep.log 2>&1 &
#   echo $! > sweep.pid
#   tail -f sweep.log

set -euo pipefail
trap 'echo "[start.sh] FAILED at line $LINENO (rc=$?)" >&2' ERR

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

THROTTLE="${THROTTLE:-20}"     # parallel sweep children (each uses THREADS cores)
THREADS="${THREADS:-2}"         # EBM_N_JOBS / XGB n_jobs per child
NOMINAL_CORES=$(( THROTTLE * THREADS ))
TOTAL_CORES="$(nproc --all 2>/dev/null || echo 64)"

echo "[start.sh] ============================================================"
echo "[start.sh] ml_test_2 full sweep (2019-12-01 UTC cutoff)"
echo "[start.sh] root      : $ROOT"
echo "[start.sh] host      : $(hostname)"
echo "[start.sh] cores     : ${NOMINAL_CORES} of ${TOTAL_CORES} ( ~$(( 100 * NOMINAL_CORES / TOTAL_CORES ))% )"
echo "[start.sh] throttle  : $THROTTLE   threads/child: $THREADS"
echo "[start.sh] calib     : isotonic cv=3 (EBM via EBM_CAL_METHOD, XGB default)"
echo "[start.sh] started   : $(date --iso-8601=seconds)"
echo "[start.sh] ============================================================"

# ---- 1. uv (install if missing) -----------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[start.sh] uv not found — installing to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[start.sh] uv version: $(uv --version)"

# ---- 2. Python env ------------------------------------------------------
echo "[start.sh] uv sync ..."
uv sync

# ---- 3. system hygiene --------------------------------------------------
mkdir -p logs reports
ulimit -n 65536 2>/dev/null || true
export PYTHONUNBUFFERED=1

# ---- 4. data integrity sanity check -------------------------------------
echo "[start.sh] verifying 2019-12-01 cutoff in data/ ..."
uv run python - <<'PYCHECK'
import pandas as pd
from pathlib import Path
cutoff = pd.Timestamp("2019-12-01", tz="UTC")
files = ["data/btc_1d.parquet", "data/eth_1d.parquet", "data/macro_fred.parquet",
         "data/coinmetrics.parquet", "data/m2.parquet", "data/gdelt.parquet"]
for f in files:
    df = pd.read_parquet(f)
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        first = df.index.min()
    elif "close_time" in df.columns:
        first = pd.to_datetime(df["close_time"], utc=True).min()
    else:
        first = "(no datetime index/column)"
    print(f"  {f:36s} first={first}  rows={len(df)}")
PYCHECK

# ---- 5. helpers ---------------------------------------------------------
LOGDIR="$ROOT/logs"

run_ebm() {
    local v="$1"
    local logf="${LOGDIR}/run_${v}_sweep.log"
    echo "[start.sh] $(date +%H:%M:%S) EBM ${v}  -> $(basename "$logf")"
    EBM_CAL_METHOD=isotonic \
    uv run python runners/run_all_M_sm.py \
        --only-version "$v" \
        --retrain \
        --throttle "$THROTTLE" \
        --threads "$THREADS" \
        --model ebm \
        --skip-done \
        > "$logf" 2>&1
    echo "[start.sh] $(date +%H:%M:%S) EBM ${v}  done"
}

run_xgb() {
    local v="$1"
    local logf="${LOGDIR}/run_${v}_sweep_xgb.log"
    echo "[start.sh] $(date +%H:%M:%S) XGB ${v}  -> $(basename "$logf")"
    XGB_DEVICE=cpu \
    uv run python runners/run_all_M_sm.py \
        --only-version "$v" \
        --retrain \
        --throttle "$THROTTLE" \
        --threads "$THREADS" \
        --model xgb \
        --skip-done \
        > "$logf" 2>&1
    echo "[start.sh] $(date +%H:%M:%S) XGB ${v}  done"
}

run_selector() {
    local cum="$1"
    local out="$2"
    local var="$3"
    local logf="${LOGDIR}/$(basename "$out" .py)_selector.log"
    echo "[start.sh] $(date +%H:%M:%S) selector cum=${cum} -> ${out}"
    uv run python analysis/v5_keep_selector_ebm.py \
        --cumulative "$cum" \
        --output-file "$out" \
        --var-name "$var" \
        > "$logf" 2>&1
    if [[ ! -f "$out" ]]; then
        echo "[start.sh] ERROR: selector did not write $out" >&2
        exit 1
    fi
    echo "[start.sh] $(date +%H:%M:%S) selector done -> $(wc -l < "$out") lines in $out"
}

# ---- 6. EBM chain -------------------------------------------------------
for v in v0 v3 v4 v6; do
    run_ebm "$v"
done

run_selector 0.80 src/v5_keep.py   V5_KEEP_COLS
run_ebm v5

run_selector 0.92 src/v5_2_keep.py V5_2_KEEP_COLS
run_ebm v5_2

run_ebm v7

run_selector 0.93 src/v5_3_keep.py V5_3_KEEP_COLS
run_ebm v5_3

# ---- 7. XGB chain -------------------------------------------------------
for v in v0 v3 v4 v5 v5_2 v6 v7 v5_3; do
    run_xgb "$v"
done

echo "[start.sh] ============================================================"
echo "[start.sh] ALL PHASES DONE at $(date --iso-8601=seconds)"
echo "[start.sh] reports under reports/ , logs under logs/"
echo "[start.sh] ============================================================"
