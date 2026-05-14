"""Add Top1 (sm_v5 alone) and Top2 (sm_v5 + sm_v3_complete) to the
comparison battery. Runs:
  - Trade quality + extended risk metrics
  - Regime-conditional analysis
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.utc2130 import build_utc2130_daily

# Reuse helpers from the previous scripts via direct import
from analyze_trade_quality import (collect_metrics, _backtest_one,
                                    _split_capital)
from analyze_regime import (_classify_regime, _per_regime_stats,
                             _split_capital_eq_and_trades)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "reports" / "dsr_fwer"

WINDOWS = (730, 365)

# Cells, ordered by 730d PSR
CELL_1 = ("utc2130_sm_v5",          "dynk12", 0.63)   # PSR 0.973
CELL_2 = ("utc2130_sm_v3_complete", "th13",   0.68)   # PSR 0.953
CELL_3 = ("utc2130_sm",             "dynk11", 0.64)   # PSR 0.952
CELL_4 = ("utc2130_sm_v3",          "th14",   0.66)   # PSR 0.852
CELL_5 = ("utc2130_sm_v6",          "dynk14", 0.66)   # PSR 0.842

# Compositions
CONFIGS = {
    "Top1_sm_v5":    [CELL_1],
    "Top2_PSR":      [CELL_1, CELL_2],
    "Manual_v5_sm":  [CELL_1, CELL_3],   # baseline reference
    "Top3_PSR":      [CELL_1, CELL_2, CELL_3],
    "Top5_PSR":      [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5],
}


def run_quality():
    print("[trade quality]", flush=True)
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close = bar.set_index("close_time")["close"]
    rows = []
    for name, cells in CONFIGS.items():
        for window in WINDOWS:
            if len(cells) == 1:
                cell, label, prob = cells[0]
                res = _backtest_one(cell, label, prob, close, window)
                eq, trades = res.equity_curve, res.trades
            else:
                eq, trades = _split_capital(cells, close, window)
            rows.append(collect_metrics(name, eq, trades, window))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "top12_quality_summary.csv", index=False)

    cols = ["strategy", "window_days", "total_return_pct",
            "n_trades", "win_rate", "profit_factor", "RR_ratio",
            "expectancy_pct", "max_consec_losses",
            "max_dd_pct", "calmar", "ulcer_index", "time_underwater_pct",
            "tail_ratio", "sortino_annual"]
    with pd.option_context("display.max_columns", None,
                           "display.width", 240,
                           "display.float_format", "{:.3f}".format):
        for w in WINDOWS:
            print(f"\n--- window={w}d ---")
            print(df.loc[df["window_days"] == w, cols].to_string(index=False))

    return df


def run_regime():
    print("\n[regime]", flush=True)
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close = bar.set_index("close_time")["close"]
    regime_full = _classify_regime(close)
    bh_daily = close.pct_change().fillna(0.0)

    rows = []
    for window in WINDOWS:
        cutoff = close.index.max() - pd.Timedelta(days=window)
        regime = regime_full.loc[regime_full.index >= cutoff]
        bh_w = bh_daily.loc[regime.index]
        bh_by_regime = {}
        for r in ("bull", "bear", "sideways"):
            mask = regime == r
            if mask.sum() == 0:
                bh_by_regime[r] = {"total_return_pct": 0.0}
            else:
                bh_by_regime[r] = {
                    "total_return_pct":
                        float(np.prod(1 + bh_w[mask].values) - 1) * 100}

        for name, cells in CONFIGS.items():
            if len(cells) == 1:
                cell, label, prob = cells[0]
                res = _backtest_one(cell, label, prob, close, window)
                eq, trades = res.equity_curve, res.trades
            else:
                eq, trades = _split_capital_eq_and_trades(cells, close, window)
            for r in _per_regime_stats(eq, regime, trades):
                bh_match = bh_by_regime[r["regime"]]
                rows.append({
                    "strategy": name, "window_days": window,
                    **r,
                    "buy_hold_pct": bh_match["total_return_pct"],
                    "outperform_pp": (r["total_return_pct"]
                                       - bh_match["total_return_pct"]),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "top12_regime_summary.csv", index=False)

    cols = ["strategy", "window_days", "regime", "n_days",
            "total_return_pct", "buy_hold_pct", "outperform_pp",
            "n_trades_closed", "win_rate", "sharpe_per_period"]
    with pd.option_context("display.max_columns", None,
                           "display.width", 240,
                           "display.float_format", "{:.3f}".format):
        for w in WINDOWS:
            print(f"\n--- window={w}d ---")
            print(df.loc[df["window_days"] == w, cols].to_string(index=False))
    return df


def main():
    run_quality()
    run_regime()


if __name__ == "__main__":
    main()
