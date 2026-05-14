"""Trade-level quality + extended risk metrics for the production candidates:
  - Single A    = utc2130_sm_v5 / dynk12 / prob=0.63
  - Single B    = utc2130_sm    / dynk11 / prob=0.64
  - Top-3 SPLIT = sm_v5 + sm_v3_complete + sm
  - Top-5 SPLIT = + sm_v3 + sm_v6

For each strategy × {730d, 365d}:
  - Profit factor
  - Expectancy ($/trade)
  - Avg win / Avg loss ratio
  - Largest win / Largest loss
  - Max consecutive losses (and consecutive wins)
  - Tail ratio (P95 / |P5|) of daily returns
  - Sortino ratio (annualised)
  - Calmar ratio
  - Time underwater %
  - Ulcer Index
  - Top-5 drawdown depths
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest, BacktestResult
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm"
OUT_DIR = ROOT / "reports" / "dsr_fwer"
TRADING_DAYS = 365.0

WINDOWS = (730, 365)

CELLS_TOP5 = [
    ("utc2130_sm_v5",          "dynk12", 0.63),
    ("utc2130_sm_v3_complete", "th13",   0.68),
    ("utc2130_sm",             "dynk11", 0.64),
    ("utc2130_sm_v3",          "th14",   0.66),
    ("utc2130_sm_v6",          "dynk14", 0.66),
]
CELLS_TOP3 = CELLS_TOP5[:3]


# ---------- helpers ----------

def _profit_factor(trades) -> float:
    gain = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
    loss = sum(-t.pnl_usd for t in trades if t.pnl_usd <= 0)
    return float("inf") if loss == 0 else gain / loss


def _consecutive(seq: list[bool]) -> int:
    """Longest run of True in seq."""
    cur = 0; mx = 0
    for x in seq:
        if x:
            cur += 1; mx = max(mx, cur)
        else:
            cur = 0
    return mx


def _drawdowns(eq: pd.Series, top_n: int = 5) -> tuple[list[float], float]:
    """Identify all complete drawdown peaks→troughs. Returns top-N depths and Ulcer."""
    cummax = eq.cummax()
    dd = eq / cummax - 1
    # Underwater segments (dd < 0)
    underwater = dd < -1e-9
    # Find drawdown episodes
    episodes = []
    in_dd = False; episode_min = 0.0
    for v in dd.values:
        if v < -1e-9:
            if not in_dd:
                in_dd = True; episode_min = v
            else:
                episode_min = min(episode_min, v)
        else:
            if in_dd:
                episodes.append(episode_min)
                in_dd = False
    if in_dd:
        episodes.append(episode_min)
    episodes.sort()  # ascending = most negative first
    top = episodes[:top_n]
    # Ulcer index
    ui = float(np.sqrt((dd.values ** 2).mean()) * 100)
    return [d * 100 for d in top], ui


def _time_underwater_pct(eq: pd.Series) -> float:
    cummax = eq.cummax()
    dd = eq / cummax - 1
    return float((dd < -1e-9).mean() * 100)


def _calmar(total_ret_pct: float, window_days: int, max_dd_pct: float) -> float:
    if max_dd_pct == 0:
        return float("inf")
    annual_ret = ((1 + total_ret_pct / 100) ** (TRADING_DAYS / window_days) - 1) * 100
    return annual_ret / abs(max_dd_pct)


def _sortino(daily_rets: np.ndarray) -> float:
    if len(daily_rets) < 5:
        return float("nan")
    mu = daily_rets.mean()
    downside = daily_rets[daily_rets < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return float("inf")
    return float(mu / downside.std(ddof=1) * math.sqrt(TRADING_DAYS))


def _tail_ratio(daily_rets: np.ndarray) -> float:
    if len(daily_rets) < 20:
        return float("nan")
    p95 = np.percentile(daily_rets, 95)
    p05 = np.percentile(daily_rets, 5)
    return float(p95 / abs(p05)) if p05 != 0 else float("inf")


def _maxdd(eq: pd.Series) -> float:
    cummax = eq.cummax()
    return float((eq / cummax - 1).min()) * 100


def _backtest_one(cell, label, prob, close_series, window, stake=100.0):
    cache = CACHE_DIR / f"preds_ebm_{label}_{cell}.parquet"
    preds = pd.read_parquet(cache)
    cutoff = preds.index.max() - pd.Timedelta(days=window)
    p = preds.dropna(subset=["prob_up"]).loc[lambda d: d.index >= cutoff]
    cl = close_series.reindex(p.index)
    return run_backtest(p, cl, stake_usd=stake, up_threshold=prob,
                        down_threshold=0.50)


def _split_capital(cells, close, window):
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


# ---------- per-strategy metric ----------

def collect_metrics(name, eq, trades, window):
    final_eq = float(eq.iloc[-1])
    initial_eq = 100.0
    total_ret = (final_eq / initial_eq - 1) * 100
    daily_rets = eq.pct_change().dropna().to_numpy()

    # Trade-level
    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t.pnl_usd > 0)
    n_losses = n_trades - n_wins
    win_rate = n_wins / n_trades * 100 if n_trades else 0.0
    pf = _profit_factor(trades) if trades else float("nan")
    avg_win = (sum(t.pnl_usd for t in trades if t.pnl_usd > 0) / n_wins
               if n_wins else 0.0)
    avg_loss = (sum(-t.pnl_usd for t in trades if t.pnl_usd <= 0) / n_losses
                if n_losses else 0.0)
    rr = (avg_win / avg_loss) if avg_loss > 0 else float("inf")
    largest_win = max((t.pnl_usd for t in trades), default=0.0)
    largest_loss = min((t.pnl_usd for t in trades), default=0.0)
    expectancy = sum(t.pnl_usd for t in trades) / n_trades if n_trades else 0.0
    expectancy_pct = sum(t.return_pct for t in trades) / n_trades * 100 \
        if n_trades else 0.0
    is_loss = [t.pnl_usd <= 0 for t in trades]
    is_win = [t.pnl_usd > 0 for t in trades]
    max_consec_loss = _consecutive(is_loss)
    max_consec_win = _consecutive(is_win)

    # Distribution
    tail_ratio = _tail_ratio(daily_rets)
    skew = float(pd.Series(daily_rets).skew())
    kurt = float(pd.Series(daily_rets).kurt())

    # Drawdown
    top5_dd, ulcer = _drawdowns(eq)
    max_dd = _maxdd(eq)
    time_uw = _time_underwater_pct(eq)
    calmar = _calmar(total_ret, window, max_dd)

    # Risk-adjusted
    sortino = _sortino(daily_rets)

    return {
        "strategy": name,
        "window_days": window,
        "total_return_pct": total_ret,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "RR_ratio": rr,
        "largest_win_usd": largest_win,
        "largest_loss_usd": largest_loss,
        "expectancy_usd": expectancy,
        "expectancy_pct": expectancy_pct,
        "max_consec_losses": max_consec_loss,
        "max_consec_wins": max_consec_win,
        "max_dd_pct": max_dd,
        "calmar": calmar,
        "ulcer_index": ulcer,
        "time_underwater_pct": time_uw,
        "top1_dd_pct": top5_dd[0] if top5_dd else 0.0,
        "top2_dd_pct": top5_dd[1] if len(top5_dd) > 1 else 0.0,
        "top3_dd_pct": top5_dd[2] if len(top5_dd) > 2 else 0.0,
        "top4_dd_pct": top5_dd[3] if len(top5_dd) > 3 else 0.0,
        "top5_dd_pct": top5_dd[4] if len(top5_dd) > 4 else 0.0,
        "tail_ratio": tail_ratio,
        "skew_daily": skew,
        "kurt_daily": kurt,
        "sortino_annual": sortino,
    }


def main():
    print("[1/2] loading close ...", flush=True)
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close = bar.set_index("close_time")["close"]

    print("[2/2] computing metrics for 4 strategies × 2 windows ...", flush=True)
    rows = []
    for window in WINDOWS:
        # Single A
        res_a = _backtest_one("utc2130_sm_v5", "dynk12", 0.63, close, window)
        rows.append(collect_metrics("Single_A_sm_v5",
                                    res_a.equity_curve, res_a.trades, window))
        # Single B
        res_b = _backtest_one("utc2130_sm", "dynk11", 0.64, close, window)
        rows.append(collect_metrics("Single_B_sm",
                                    res_b.equity_curve, res_b.trades, window))
        # Top-3
        eq3, trades3 = _split_capital(CELLS_TOP3, close, window)
        rows.append(collect_metrics("Top3_SPLIT", eq3, trades3, window))
        # Top-5
        eq5, trades5 = _split_capital(CELLS_TOP5, close, window)
        rows.append(collect_metrics("Top5_SPLIT", eq5, trades5, window))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "trade_quality_summary.csv", index=False)

    # Print key cols
    cols = ["strategy", "window_days", "total_return_pct",
            "n_trades", "win_rate", "profit_factor", "RR_ratio",
            "expectancy_pct", "max_consec_losses",
            "max_dd_pct", "calmar", "ulcer_index", "time_underwater_pct",
            "tail_ratio", "sortino_annual"]
    with pd.option_context("display.max_columns", None,
                           "display.width", 220,
                           "display.float_format", "{:.3f}".format):
        for w in WINDOWS:
            print(f"\n=== window={w}d ===")
            print(df.loc[df["window_days"] == w, cols].to_string(index=False))

        print("\n=== top-5 drawdown depths (%) — 730d ===")
        cols2 = ["strategy", "top1_dd_pct", "top2_dd_pct", "top3_dd_pct",
                 "top4_dd_pct", "top5_dd_pct"]
        print(df.loc[df["window_days"] == 730, cols2].to_string(index=False))

    print(f"\nsaved → {OUT_DIR/'trade_quality_summary.csv'}")


if __name__ == "__main__":
    main()
