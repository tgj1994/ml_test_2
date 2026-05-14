"""Regime-conditional analysis for production candidates.

Regime definition (BTC daily close, UTC 21:30):
  - bull     : 30d rolling return > +5%
  - bear     : 30d rolling return < -5%
  - sideways : in between

For each strategy {Single_A, Single_B, Top3_SPLIT, Top5_SPLIT}:
  Per regime:
    - n_days
    - days_in_market
    - total_return_pct  (compounded over regime days)
    - win_rate          (over trades that closed in this regime)
    - n_trades_closed
    - mean_daily_return_pct
    - sharpe_per_period (regime-only daily returns)
    - outperform_vs_BH_pp

Outputs (under reports/dsr_fwer/):
  - regime_summary.csv   : per (strategy, window, regime) row
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


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm"
OUT_DIR = ROOT / "reports" / "dsr_fwer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = (730, 365)
TRADING_DAYS = 365.0
REGIME_LOOKBACK_DAYS = 30
BULL_TH = 0.05
BEAR_TH = -0.05

CELLS_TOP5 = [
    ("utc2130_sm_v5",          "dynk12", 0.63),
    ("utc2130_sm_v3_complete", "th13",   0.68),
    ("utc2130_sm",             "dynk11", 0.64),
    ("utc2130_sm_v3",          "th14",   0.66),
    ("utc2130_sm_v6",          "dynk14", 0.66),
]
CELLS_TOP3 = CELLS_TOP5[:3]


def _classify_regime(close: pd.Series) -> pd.Series:
    """For each day, regime is based on past 30d return ending that day."""
    ret_30d = close.pct_change(REGIME_LOOKBACK_DAYS)
    regime = pd.Series(index=close.index, dtype=object)
    regime[ret_30d > BULL_TH] = "bull"
    regime[ret_30d < BEAR_TH] = "bear"
    regime[(ret_30d >= BEAR_TH) & (ret_30d <= BULL_TH)] = "sideways"
    regime = regime.fillna("sideways")
    return regime


def _backtest_one(cell, label, prob, close, window, stake=100.0):
    cache = CACHE_DIR / f"preds_ebm_{label}_{cell}.parquet"
    preds = pd.read_parquet(cache)
    cutoff = preds.index.max() - pd.Timedelta(days=window)
    p = preds.dropna(subset=["prob_up"]).loc[lambda d: d.index >= cutoff]
    cl = close.reindex(p.index)
    return run_backtest(p, cl, stake_usd=stake, up_threshold=prob,
                        down_threshold=0.50)


def _split_capital_eq_and_trades(cells, close, window):
    sub_eq = []; trades = []
    for cell, label, prob in cells:
        res = _backtest_one(cell, label, prob, close, window,
                            stake=100.0 / len(cells))
        sub_eq.append(res.equity_curve)
        trades.extend(res.trades)
    eq = sub_eq[0]
    for s in sub_eq[1:]:
        eq, s2 = eq.align(s, join="inner")
        eq = eq + s2
    return eq, trades


def _per_regime_stats(eq: pd.Series, regime: pd.Series, trades) -> list[dict]:
    daily_ret = eq.pct_change().fillna(0.0)
    regime = regime.reindex(eq.index, method="ffill")
    bh = eq.iloc[0] * (1 + (eq.index.to_series().map(
        lambda d: 0)))  # placeholder; we'll compute B&H separately
    rows = []
    # Total compounded return per regime (chained daily)
    for r in ("bull", "bear", "sideways"):
        mask = regime == r
        n_days = int(mask.sum())
        if n_days == 0:
            rows.append(dict(regime=r, n_days=0, total_return_pct=0.0,
                             mean_daily_return_pct=float("nan"),
                             sharpe_per_period=float("nan"),
                             n_trades_closed=0, win_rate=float("nan")))
            continue
        regime_ret = daily_ret[mask]
        compound_ret = float(np.prod(1 + regime_ret.values) - 1) * 100
        mean_ret = float(regime_ret.mean() * 100)
        sd = float(regime_ret.std(ddof=1)) if len(regime_ret) > 1 else 0.0
        sr = (regime_ret.mean() / sd * math.sqrt(TRADING_DAYS)) \
            if sd > 0 else float("nan")
        # Trades closed in this regime
        in_regime_trades = []
        for t in trades:
            d = pd.Timestamp(t.exit_date)
            if d.tz is None:
                d = d.tz_localize("UTC")
            if d in regime.index:
                if regime.loc[d] == r:
                    in_regime_trades.append(t)
            else:
                # nearest within regime series
                pos = regime.index.get_indexer([d], method="nearest")[0]
                if regime.iloc[pos] == r:
                    in_regime_trades.append(t)
        n_t = len(in_regime_trades)
        n_w = sum(1 for t in in_regime_trades if t.pnl_usd > 0)
        rows.append(dict(
            regime=r, n_days=n_days,
            total_return_pct=compound_ret,
            mean_daily_return_pct=mean_ret,
            sharpe_per_period=float(sr),
            n_trades_closed=n_t,
            win_rate=(n_w / n_t * 100) if n_t else float("nan"),
        ))
    return rows


def main():
    print("[1/3] loading close + classifying regimes ...", flush=True)
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close = bar.set_index("close_time")["close"]
    regime_full = _classify_regime(close)
    print(f"  regime distribution (all): "
          f"{regime_full.value_counts().to_dict()}")

    # B&H reference per regime
    bh_daily = close.pct_change().fillna(0.0)

    rows = []
    for window in WINDOWS:
        cutoff = close.index.max() - pd.Timedelta(days=window)
        regime = regime_full.loc[regime_full.index >= cutoff]
        cl_w = close.loc[regime.index]
        bh_w = bh_daily.loc[regime.index]

        # B&H per regime (compound)
        bh_rows = []
        for r in ("bull", "bear", "sideways"):
            mask = regime == r
            n_days = int(mask.sum())
            if n_days == 0:
                bh_rows.append(dict(regime=r, total_return_pct=0.0,
                                    mean_daily_return_pct=float("nan")))
                continue
            cmp_ret = float(np.prod(1 + bh_w[mask].values) - 1) * 100
            bh_rows.append(dict(
                regime=r,
                total_return_pct=cmp_ret,
                mean_daily_return_pct=float(bh_w[mask].mean() * 100),
            ))
        bh_by_regime = {r["regime"]: r for r in bh_rows}

        # Run strategies
        strategies = []
        res_a = _backtest_one("utc2130_sm_v5", "dynk12", 0.63, close, window)
        strategies.append(("Single_A_sm_v5", res_a.equity_curve, res_a.trades))
        res_b = _backtest_one("utc2130_sm", "dynk11", 0.64, close, window)
        strategies.append(("Single_B_sm", res_b.equity_curve, res_b.trades))
        eq3, trades3 = _split_capital_eq_and_trades(CELLS_TOP3, close, window)
        strategies.append(("Top3_SPLIT", eq3, trades3))
        eq5, trades5 = _split_capital_eq_and_trades(CELLS_TOP5, close, window)
        strategies.append(("Top5_SPLIT", eq5, trades5))

        for name, eq, trades in strategies:
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
    df.to_csv(OUT_DIR / "regime_summary.csv", index=False)

    print("\n=== Regime-conditional performance ===")
    cols = ["strategy", "window_days", "regime", "n_days",
            "total_return_pct", "buy_hold_pct", "outperform_pp",
            "n_trades_closed", "win_rate", "sharpe_per_period",
            "mean_daily_return_pct"]
    with pd.option_context("display.max_columns", None,
                           "display.width", 220,
                           "display.float_format", "{:.3f}".format):
        for w in WINDOWS:
            print(f"\n--- window = {w}d ---")
            sub = df.loc[df["window_days"] == w, cols]
            print(sub.to_string(index=False))

    print(f"\nsaved → {OUT_DIR/'regime_summary.csv'}")


if __name__ == "__main__":
    main()
