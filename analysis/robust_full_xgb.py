"""XGB twin of robust_full_ebm.py.

Identical battery of tests (MaxDD, Sharpe, PSR, DSR, PBO, Quarterly, Regime),
but reads the XGB sweep outputs instead of EBM:

  inputs:
    reports/xgb/utc2130/<variant>_<window>d/
        label_threshold_sweep_summary_{static,dynamic}_xgb_<variant>_<window>d.csv
    data/preds_cache_xgb/preds_xgb_<label>_<variant>.parquet

  outputs (under reports/xgb/utc2130/):
    robustness_all_xgb.csv             headline (full cell grid)
    robustness_quarterly_all_xgb.csv
    robustness_regime_all_xgb.csv
    robustness_pbo_all_xgb.csv

All cells with n_trades >= 4 are kept (top-N filtering removed by user
directive 2026-05-13).
"""
from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the stat helpers from the EBM module — they're model-agnostic.
from analysis.robust_full_ebm import (  # noqa: E402
    max_drawdown,
    daily_returns_from_equity,
    sharpe,
    psr,
    dsr,
    quarterly_returns,
    regime_breakdown,
)
from src.backtest import run_backtest  # noqa: E402
from src.utc2130 import build_utc2130_daily  # noqa: E402


REPORTS = ROOT / "reports" / "xgb" / "utc2130"
CACHE_DIR = ROOT / "data" / "preds_cache_xgb"
MODEL = "xgb"
WINDOW_730 = 730
WINDOW_365 = 365
S_BLOCKS = 16


def load_all_cells() -> pd.DataFrame:
    rows = []
    for d in sorted(REPORTS.iterdir()):
        if not d.is_dir() or not d.name.endswith(f"_{WINDOW_730}d"):
            continue
        variant = d.name[: -len(str(WINDOW_730)) - 2]
        for mode in ("static", "dynamic"):
            f = (d / f"label_threshold_sweep_summary_{mode}_{MODEL}_"
                     f"{variant}_{WINDOW_730}d.csv")
            if not f.exists():
                continue
            df = pd.read_csv(f)
            for _, r in df.iterrows():
                rows.append({
                    "variant": variant,
                    "mode": mode,
                    "label_or_k": r["slug"].split("_")[0],
                    "value": float(r["value"]),
                    "best_prob_th": float(r["best_prob_th"]),
                    "best_return": float(r["best_return"]),
                    "best_trades": int(r["best_trades"]),
                })
    df_all = pd.DataFrame(rows)
    df_pool = df_all[df_all["best_trades"] >= 4].copy()
    return (df_pool.sort_values("best_return", ascending=False)
                   .reset_index(drop=True))


def cache_path_for(variant: str, label_or_k: str) -> Path:
    return CACHE_DIR / f"preds_{MODEL}_{label_or_k}_{variant}.parquet"


def backtest_window(preds: pd.DataFrame, daily_close: pd.Series,
                    prob_th: float, window_days: int):
    end = preds.index.max()
    start = end - pd.Timedelta(days=window_days - 1)
    sub = preds.loc[preds.index >= start]
    if len(sub) < 30:
        return None
    return run_backtest(sub, daily_close, up_threshold=prob_th,
                        down_threshold=0.5)


def pbo_for_cell(variant: str, daily_close: pd.Series,
                 prob_th_grid: list[float], window_days: int = WINDOW_730,
                 s_blocks: int = S_BLOCKS) -> dict:
    """CSCV-style PBO over the full label × prob_TH grid of one variant."""
    parquets = sorted(CACHE_DIR.glob(f"preds_{MODEL}_*_{variant}.parquet"))
    rets_list = []
    for p in parquets:
        label = p.stem.replace(f"preds_{MODEL}_", "").replace(f"_{variant}", "")
        try:
            preds = pd.read_parquet(p)
        except Exception:
            continue
        for th in prob_th_grid:
            bt = backtest_window(preds, daily_close, th, window_days)
            if bt is None or bt.equity_curve.empty:
                continue
            rets = bt.equity_curve.pct_change().fillna(0.0)
            if len(rets) < s_blocks * 4:
                continue
            rets_list.append(rets.values)
    if not rets_list:
        return {"variant": variant, "PBO": float("nan"),
                "n_strategies": 0, "n_blocks": s_blocks}

    min_len = min(len(r) for r in rets_list)
    R = np.stack([r[:min_len] for r in rets_list])
    M, T = R.shape
    block_size = T // s_blocks
    R = R[:, : block_size * s_blocks]
    Rb = R.reshape(M, s_blocks, block_size)
    block_sum = Rb.sum(axis=2)
    block_sumsq = (Rb ** 2).sum(axis=2)
    block_n = np.full_like(block_sum, block_size, dtype=float)

    half = s_blocks // 2
    combos = list(combinations(range(s_blocks), half))
    K = len(combos)
    indicator = np.zeros((s_blocks, K), dtype=float)
    for k, idx in enumerate(combos):
        indicator[list(idx), k] = 1.0

    IS_sum = block_sum @ indicator
    IS_sumsq = block_sumsq @ indicator
    IS_n = block_n @ indicator
    OOS_sum = block_sum.sum(axis=1, keepdims=True) - IS_sum
    OOS_sumsq = block_sumsq.sum(axis=1, keepdims=True) - IS_sumsq
    OOS_n = block_n.sum(axis=1, keepdims=True) - IS_n

    def _sharpe(s, sq, n):
        mu = s / n
        var = sq / n - mu * mu
        sd = np.sqrt(np.where(var > 0, var, np.nan))
        return mu / sd

    SR_IS = _sharpe(IS_sum, IS_sumsq, IS_n)
    SR_OOS = _sharpe(OOS_sum, OOS_sumsq, OOS_n)
    is_best = np.argmax(SR_IS, axis=0)
    rank_oos = (np.argsort(np.argsort(SR_OOS, axis=0), axis=0) + 1) / float(M)
    omega = rank_oos[is_best, np.arange(K)]
    return {"variant": variant, "PBO": float((omega < 0.5).mean()),
            "n_strategies": M, "n_blocks": s_blocks,
            "min_period_T": int(block_size * s_blocks)}


def main() -> int:
    pool = load_all_cells()
    print(f"loaded {len(pool)} XGB cells with n_trades>=4")

    daily = build_utc2130_daily(ROOT / "data" / "btc_15m.parquet")
    daily.index = pd.to_datetime(daily["close_time"], utc=True)
    daily_close = daily["close"].copy()
    daily_close.index = daily.index

    headline_rows = []
    quarterly_rows = []
    regime_rows = []
    n_trials_dsr = 22 * 21

    total = len(pool)
    for i, (_, row) in enumerate(pool.iterrows(), start=1):
        variant = row["variant"]
        label = row["label_or_k"]
        prob_th = float(row["best_prob_th"])
        if i % 200 == 0 or i == total:
            print(f"  [{i}/{total}] processed")
        p = cache_path_for(variant, label)
        if not p.exists():
            continue
        preds = pd.read_parquet(p)
        bt = backtest_window(preds, daily_close, prob_th, WINDOW_730)
        if bt is None or bt.equity_curve.empty:
            continue
        eq = bt.equity_curve
        bh = bt.buy_and_hold_curve
        rets = daily_returns_from_equity(eq)
        headline_rows.append({
            "variant": variant,
            "mode": row["mode"],
            "label_or_k": label,
            "best_prob_th": prob_th,
            "best_return_730d": row["best_return"],
            "max_drawdown": max_drawdown(eq),
            "sharpe_annual": sharpe(rets),
            "PSR": psr(rets, sr_star=0.0),
            "DSR": dsr(rets, n_trials=n_trials_dsr),
            "n_trades": bt.n_trades,
        })
        q = quarterly_returns(eq, bh)
        if not q.empty:
            q["variant"] = variant
            q["label_or_k"] = label
            quarterly_rows.append(q)
        rg = regime_breakdown(eq, daily_close)
        if not rg.empty:
            rg["variant"] = variant
            rg["label_or_k"] = label
            regime_rows.append(rg)

    headline = pd.DataFrame(headline_rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    headline.to_csv(REPORTS / "robustness_all_xgb.csv", index=False)
    if quarterly_rows:
        pd.concat(quarterly_rows, ignore_index=True).to_csv(
            REPORTS / "robustness_quarterly_all_xgb.csv", index=False)
    if regime_rows:
        pd.concat(regime_rows, ignore_index=True).to_csv(
            REPORTS / "robustness_regime_all_xgb.csv", index=False)

    # PBO across ALL unique variants
    pbo_rows = []
    prob_th_grid = [round(0.50 + i * 0.01, 2) for i in range(21)]
    all_variants = pool["variant"].drop_duplicates().tolist()
    print(f"\nPBO over {len(all_variants)} unique variants...")
    for i, cell in enumerate(all_variants, start=1):
        print(f"  [{i}/{len(all_variants)}] PBO {cell}")
        pbo_rows.append(pbo_for_cell(cell, daily_close, prob_th_grid))
    pd.DataFrame(pbo_rows).to_csv(
        REPORTS / "robustness_pbo_all_xgb.csv", index=False)

    # Console summary (top 20 only — CSV holds the full table)
    print()
    print("=" * 130)
    print(f"ROBUSTNESS HEADLINE — full {len(headline)} XGB cells (top 20 by 730d return)")
    print("=" * 130)
    out = headline.head(20).copy()
    out["best_return_730d"] = (out["best_return_730d"] * 100).round(2).astype(str) + "%"
    out["max_drawdown"] = (out["max_drawdown"] * 100).round(2).astype(str) + "%"
    for c in ("sharpe_annual", "PSR", "DSR"):
        out[c] = out[c].round(3)
    print(out.to_string(index=False))
    print(f"\n(see robustness_all_xgb.csv for all {len(headline)} cells)")

    print()
    print("=" * 130)
    print(f"PBO — all {len(pbo_rows)} variants (S={S_BLOCKS}, N_trials={n_trials_dsr})")
    print("=" * 130)
    pbo_df = pd.DataFrame(pbo_rows).sort_values("PBO")
    for _, r in pbo_df.head(20).iterrows():
        print(f"  {r.get('variant','-'):30s}  PBO={r.get('PBO',float('nan')):.3f}  "
              f"M={int(r.get('n_strategies', 0)):3d}  S={r.get('n_blocks',S_BLOCKS)}")
    if len(pbo_df) > 20:
        print(f"  ... (full {len(pbo_df)} rows in robustness_pbo_all_xgb.csv)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
