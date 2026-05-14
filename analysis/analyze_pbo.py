"""PBO via Bailey-López de Prado's CSCV — VECTORISED implementation.

Per-cell precompute (M strategies × S blocks):
  block_sum, block_sumsq, block_n  shape (M, S)

For all C(S, S/2) combinations stacked into combo_indicator (S, K):
  IS_sum   = block_sum   @ combo_indicator   →  (M, K)
  IS_sumsq = block_sumsq @ combo_indicator
  IS_n     = block_n     @ combo_indicator
  OOS_*    = total_* − IS_*

Sharpe from running stats:
  mean = sum / n
  var  = sumsq / n − mean²    (population)
  SR   = mean / sqrt(var)     (per-period)

Then per combo column: argmax over SR_IS gives the IS-best strategy s*.
Its OOS rank ω = rank(SR_OOS[s*]) / M. PBO = P(ω > 0.5).

This runs in seconds per cell instead of hours.
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm"
OUT_DIR = ROOT / "reports" / "dsr_fwer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 730
S_BLOCKS = 16
PROB_THRESHOLDS = [round(0.50 + i * 0.01, 2) for i in range(21)]
LABELS = [f"th{i}" for i in range(10, 21)] + [f"dynk{i}" for i in range(10, 21)]

CELLS = [
    "utc2130_sm_v5",
    "utc2130_sm_v3_complete",
    "utc2130_sm",
    "utc2130_sm_v3",
    "utc2130_sm_v6",
]


def _block_stats(returns: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split returns into S contiguous blocks → per-block (sum, sumsq, n)."""
    n = len(returns)
    edges = np.linspace(0, n, S + 1, dtype=int)
    sums = np.zeros(S); sqs = np.zeros(S); ns = np.zeros(S)
    for i in range(S):
        seg = returns[edges[i]:edges[i + 1]]
        ns[i] = len(seg)
        if len(seg) > 0:
            sums[i] = seg.sum()
            sqs[i] = (seg * seg).sum()
    return sums, sqs, ns


def precompute_cell(cell: str, close: pd.Series) -> tuple[np.ndarray, ...]:
    """Returns (block_sum, block_sumsq, block_n) shape (M, S) for one cell."""
    sums_all = []; sqs_all = []; ns_all = []
    for label in LABELS:
        cache_path = CACHE_DIR / f"preds_ebm_{label}_{cell}.parquet"
        if not cache_path.exists():
            continue
        preds = pd.read_parquet(cache_path)
        cutoff = preds.index.max() - pd.Timedelta(days=WINDOW_DAYS)
        p = preds.dropna(subset=["prob_up"]).loc[lambda d: d.index >= cutoff]
        if len(p) < 30:
            continue
        cl = close.reindex(p.index)
        for prob in PROB_THRESHOLDS:
            res = run_backtest(p, cl, up_threshold=prob, down_threshold=0.50)
            if len(res.equity_curve) < S_BLOCKS * 5:
                sums_all.append(np.zeros(S_BLOCKS))
                sqs_all.append(np.zeros(S_BLOCKS))
                ns_all.append(np.zeros(S_BLOCKS))
                continue
            r = res.equity_curve.pct_change().dropna().to_numpy()
            s_, q_, n_ = _block_stats(r, S_BLOCKS)
            sums_all.append(s_); sqs_all.append(q_); ns_all.append(n_)
    return np.array(sums_all), np.array(sqs_all), np.array(ns_all)


def cscv_pbo_vectorised(block_sum, block_sumsq, block_n, S: int = 16) -> dict:
    M = block_sum.shape[0]
    half = S // 2
    combos = list(combinations(range(S), half))
    K = len(combos)
    # Build combo_indicator: (S, K) where col k has 1 in IS positions
    indicator = np.zeros((S, K), dtype=np.float64)
    for k, c in enumerate(combos):
        for s in c:
            indicator[s, k] = 1.0
    # OOS indicator = 1 − IS
    oos_ind = 1.0 - indicator

    # Aggregate IS / OOS sums via matrix mult
    IS_sum   = block_sum   @ indicator    # (M, K)
    IS_sumsq = block_sumsq @ indicator
    IS_n     = block_n     @ indicator
    OOS_sum   = block_sum   @ oos_ind
    OOS_sumsq = block_sumsq @ oos_ind
    OOS_n     = block_n     @ oos_ind

    def sharpe_matrix(sumv, sqv, nv):
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sumv / nv
            var = sqv / nv - mean * mean    # population variance
            sr = mean / np.sqrt(var)
            sr[~np.isfinite(sr)] = -np.inf
        return sr

    SR_IS  = sharpe_matrix(IS_sum,  IS_sumsq,  IS_n)
    SR_OOS = sharpe_matrix(OOS_sum, OOS_sumsq, OOS_n)

    # For each combo column k, find IS argmax; look up OOS rank
    best_idx = np.argmax(SR_IS, axis=0)         # (K,)
    # OOS rank: descending order, 1 = top, M = bottom
    # rank = M - argpos in sorted ascending; use argsort then invert
    order = np.argsort(-SR_OOS, axis=0)         # descending; (M, K)
    ranks = np.empty_like(order)
    rng = np.arange(M)[:, None]                 # (M, 1)
    ranks[order, np.arange(K)[None, :]] = rng + 1
    rank_best = ranks[best_idx, np.arange(K)]   # (K,)
    omega = (rank_best - 0.5) / M
    # lambda = log(omega/(1-omega)); guard against 0/1
    omega_clip = np.clip(omega, 1e-9, 1 - 1e-9)
    lam = np.log(omega_clip / (1 - omega_clip))

    pbo = float((lam > 0).mean())
    is_sr_best = SR_IS[best_idx, np.arange(K)]
    oos_sr_best = SR_OOS[best_idx, np.arange(K)]
    return dict(
        pbo=pbo,
        median_omega=float(np.median(omega)),
        mean_lambda=float(np.mean(lam)),
        median_lambda=float(np.median(lam)),
        mean_is_sr=float(np.nanmean(is_sr_best)),
        mean_oos_sr=float(np.nanmean(oos_sr_best)),
        loss_in_translation=float(np.nanmean(is_sr_best) - np.nanmean(oos_sr_best)),
        n_combos=K, n_strategies=M,
    )


def main():
    print("[1/3] loading close (UTC2130) ...", flush=True)
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close = bar.set_index("close_time")["close"]

    print(f"[2/3] PBO on {len(CELLS)} cells × C(16,8)=12,870 combos ...",
          flush=True)
    rows = []
    for cell in CELLS:
        print(f"  cell={cell}: precomputing ...", flush=True)
        bs, bq, bn = precompute_cell(cell, close)
        print(f"    {bs.shape[0]} strategies × {bs.shape[1]} blocks", flush=True)
        print(f"  cell={cell}: CSCV ...", flush=True)
        res = cscv_pbo_vectorised(bs, bq, bn, S=S_BLOCKS)
        res["cell"] = cell
        rows.append(res)
        print(f"    PBO={res['pbo']:.4f}  median ω={res['median_omega']:.4f}  "
              f"IS SR mean={res['mean_is_sr']:.4f}  OOS SR mean={res['mean_oos_sr']:.4f}",
              flush=True)

    df = pd.DataFrame(rows)[["cell", "pbo", "median_omega",
                             "median_lambda", "mean_lambda",
                             "mean_is_sr", "mean_oos_sr",
                             "loss_in_translation",
                             "n_strategies", "n_combos"]]
    df.to_csv(OUT_DIR / "pbo_summary.csv", index=False)
    print("\n=== PBO summary (intra-cell selection overfit) ===")
    with pd.option_context("display.max_columns", None,
                           "display.width", 220,
                           "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))
    print(f"\nsaved → {OUT_DIR/'pbo_summary.csv'}")
    print("\nReading guide:")
    print("  PBO = P(IS-best falls below OOS median rank)")
    print("  PBO < 0.5  → generalises (good)")
    print("  PBO < 0.3  → strong evidence of OOS robustness")
    print("  loss_in_translation = IS-SR − OOS-SR; large positive = overfit")


if __name__ == "__main__":
    main()
