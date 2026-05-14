"""TWAP / VWAP execution simulation — addresses reviewer critique #2.

The original backtest assumes instant execution at the 21:30 UTC daily bar
close. Reviewer noted that mechanical execution at a fixed timestamp is
exposed to HFT front-running and market impact. We test the strategies'
robustness to non-instant execution by pricing entries / exits as the
TWAP or VWAP across the next N 15-minute bars after the signal bar.

Modes evaluated:
  instant       : fill at daily close (the original assumption)
  TWAP_1h       : average close of next 4 × 15m bars after signal close
  TWAP_2h       : average close of next 8 × 15m bars
  VWAP_1h       : volume-weighted close of next 4 × 15m bars
  VWAP_2h       : volume-weighted close of next 8 × 15m bars

Both BUY and SELL transitions use the same execution mode. Fee = 10 bps
per side (unchanged).

Strategies: Top1 / Top2 / Top3 / Top5 SPLIT-CAPITAL ensembles.
Windows: 730d, 365d.

Output (under reports/execution_sim/):
  per_strategy_execution.csv     long format (strategy × window × mode)
  matrix_return.csv              wide pivot of total_return_pct
  matrix_maxdd.csv               wide pivot of max_dd_pct
  summary.txt                    paper-ready table
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm"
OUT_DIR = ROOT / "reports" / "execution_sim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = (730, 365)
FEE_BPS = 10.0
TRADING_DAYS = 365.0
STAKE_USD = 100.0

ENSEMBLES = {
    "Top1": [("utc2130_sm_v5", "dynk12", 0.63)],
    "Top2": [("utc2130_sm_v5", "dynk12", 0.63),
             ("utc2130_sm_v3_complete", "th13", 0.68)],
    "Top3": [("utc2130_sm_v5", "dynk12", 0.63),
             ("utc2130_sm_v3_complete", "th13", 0.68),
             ("utc2130_sm", "dynk11", 0.64)],
    "Top5": [("utc2130_sm_v5", "dynk12", 0.63),
             ("utc2130_sm_v3_complete", "th13", 0.68),
             ("utc2130_sm", "dynk11", 0.64),
             ("utc2130_sm_v3", "th14", 0.66),
             ("utc2130_sm_v6", "dynk14", 0.66)],
}

EXECUTION_MODES = (
    ("instant",  "instant",  0, "instant fill at daily bar close"),
    ("TWAP_1h",  "TWAP",     4, "TWAP across next 4 × 15m bars (1 hour)"),
    ("TWAP_2h",  "TWAP",     8, "TWAP across next 8 × 15m bars (2 hours)"),
    ("VWAP_1h",  "VWAP",     4, "VWAP across next 4 × 15m bars (1 hour)"),
    ("VWAP_2h",  "VWAP",     8, "VWAP across next 8 × 15m bars (2 hours)"),
)


def build_execution_price_lookup(m15: pd.DataFrame, daily_close_times: pd.Index,
                                  n_bars: int, mode: str) -> pd.Series:
    """For each daily close timestamp, compute the TWAP or VWAP of the
    NEXT n_bars 15m bars (open_time strictly > close_time of daily bar).

    For instant mode, return the daily close itself (handled separately by
    caller; here mode != instant)."""
    m15 = m15.sort_values("open_time").reset_index(drop=True)
    open_times = m15["open_time"].values
    closes = m15["close"].values
    volumes = m15["volume"].values

    # daily close_time = 21:30 UTC. The first 15m bar with open_time >= 21:30
    # is itself open_time = 21:30 (which has close_time = 21:44:59.999).
    # We want the bars whose open_time > daily close_time (= bars STARTING
    # AFTER the signal close). The first such bar opens at 21:45 UTC.
    out = pd.Series(index=daily_close_times, dtype=float)
    for ts in daily_close_times:
        # binary-search the next 15m bar with open_time > ts
        # daily close_time is e.g. 21:30:00 UTC of day t; next open_time =
        # 21:30 (= the same bar) or 21:45. We want execution AFTER signal,
        # so skip equal/earlier — find first open_time > ts.
        # ts may be ns precision; m15 is ns too.
        pos = np.searchsorted(open_times, np.datetime64(ts.to_datetime64()),
                              side="right")
        # If we want bars STARTING AT or AFTER the signal close (21:30),
        # use side="left". Reviewer concern: execution at signal time =>
        # the bar that opens at 21:30 IS the signal bar. We skip it
        # (already used to compute the signal) and take the bars AFTER.
        # So side="right" — first index with open_time > ts.
        if pos + n_bars > len(m15):
            out.loc[ts] = float("nan")
            continue
        seg_close = closes[pos:pos + n_bars]
        seg_vol = volumes[pos:pos + n_bars]
        if mode == "TWAP":
            out.loc[ts] = float(seg_close.mean())
        elif mode == "VWAP":
            tot_vol = seg_vol.sum()
            if tot_vol > 0:
                out.loc[ts] = float((seg_close * seg_vol).sum() / tot_vol)
            else:
                out.loc[ts] = float(seg_close.mean())
    return out


def run_backtest_with_exec_prices(
        preds: pd.DataFrame, close_daily: pd.Series, exec_price: pd.Series,
        up_th: float, down_th: float, fee_bps: float, stake: float):
    """Backtest with separate signal price (close_daily) for decision and
    exec_price for entry/exit fills. Mirrors src.backtest.run_backtest but
    uses exec_price for cash → qty conversions."""
    df = preds.dropna(subset=["prob_up"]).copy()
    df["close"] = close_daily.reindex(df.index)
    df["exec"] = exec_price.reindex(df.index)
    df = df.dropna(subset=["close", "exec"])
    fee = fee_bps / 1e4
    cash = stake; qty = 0.0; in_pos = False
    entry_idx = None; entry_cash = 0.0; entry_price = float("nan")
    trades = []
    eq_arr = []
    bh_arr = []
    bh_anchor = float(df["close"].iloc[0])
    days_in_market = 0
    for k, (idx, row) in enumerate(df.iterrows()):
        sig_price = float(row["close"])      # used for mark-to-market only
        exec_p = float(row["exec"])           # actual fill price
        prob = float(row["prob_up"])
        if prob >= up_th: signal = "BUY"
        elif prob < down_th: signal = "SELL"
        else: signal = "HOLD"
        if in_pos and signal == "SELL":
            cash = qty * exec_p * (1 - fee)
            pnl = cash - entry_cash
            trades.append({"entry_date": df.index[entry_idx],
                            "entry_price": entry_price,
                            "exit_date": idx,
                            "exit_price": exec_p,
                            "bars_held": k - entry_idx,
                            "pnl_usd": pnl,
                            "return_pct": pnl / entry_cash})
            qty = 0.0; in_pos = False; entry_idx = None
        if not in_pos and signal == "BUY":
            entry_cash = cash
            qty = (cash * (1 - fee)) / exec_p
            cash = 0.0
            entry_idx = k
            entry_price = exec_p
            in_pos = True
        if in_pos:
            days_in_market += 1
            eq_arr.append(qty * sig_price)
        else:
            eq_arr.append(cash)
        bh_arr.append(stake * sig_price / bh_anchor)
    if in_pos and entry_idx is not None:
        last = df.iloc[-1]
        cash = qty * float(last["exec"]) * (1 - fee)
        pnl = cash - entry_cash
        trades.append({"entry_date": df.index[entry_idx],
                        "entry_price": entry_price,
                        "exit_date": df.index[-1],
                        "exit_price": float(last["exec"]),
                        "bars_held": len(df) - 1 - entry_idx,
                        "pnl_usd": pnl,
                        "return_pct": pnl / entry_cash})
    eq = pd.Series(eq_arr, index=df.index, name="equity")
    return eq, trades


def metrics_from_eq(eq: pd.Series, trades, stake=STAKE_USD) -> dict:
    if len(eq) < 5:
        return {}
    final = float(eq.iloc[-1])
    daily = eq.pct_change().dropna().to_numpy()
    cummax = eq.cummax()
    max_dd = float((eq / cummax - 1.0).min())
    n_t = len(trades)
    n_w = sum(1 for t in trades if t["pnl_usd"] > 0)
    if len(daily) < 5 or daily.std(ddof=1) == 0:
        sr = float("nan")
    else:
        sr = (daily.mean() / daily.std(ddof=1)) * math.sqrt(TRADING_DAYS)
    return {
        "total_return_pct": (final / stake - 1) * 100,
        "n_trades": n_t,
        "n_wins": n_w,
        "win_rate_pct": (n_w / n_t * 100) if n_t else 0.0,
        "max_dd_pct": max_dd * 100,
        "sharpe": sr,
        "final_equity": final,
    }


def main() -> None:
    print("[1/4] loading data ...", flush=True)
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close_daily = bar.set_index("close_time")["close"]
    m15 = pd.read_parquet(DATA_DIR / "btc_15m.parquet")
    m15["open_time"] = pd.to_datetime(m15["open_time"])
    if m15["open_time"].dt.tz is None:
        m15["open_time"] = m15["open_time"].dt.tz_localize("UTC")

    print("[2/4] precomputing execution price lookups (TWAP_1h/2h, VWAP_1h/2h) ...",
          flush=True)
    daily_idx = bar["close_time"]
    exec_lookups: dict[str, pd.Series] = {"instant": close_daily}
    for tag, mode, n_bars, _desc in EXECUTION_MODES[1:]:
        s = build_execution_price_lookup(m15, daily_idx, n_bars, mode)
        s.index = pd.DatetimeIndex(s.index)
        exec_lookups[tag] = s
        print(f"  {tag}: {s.notna().sum()} timestamps priced "
              f"({s.isna().sum()} NaN at series tail)")

    print("[3/4] running backtests ...", flush=True)
    rows = []
    for ens_name, cells in ENSEMBLES.items():
        for window in WINDOWS:
            for tag, mode, n_bars, desc in EXECUTION_MODES:
                # SPLIT-CAPITAL: each cell trades stake/N independently
                sub_eq = []; trades_all = []
                stake_each = STAKE_USD / len(cells)
                for cell, label, prob in cells:
                    cache = CACHE_DIR / f"preds_ebm_{label}_{cell}.parquet"
                    preds = pd.read_parquet(cache)
                    cutoff = preds.index.max() - pd.Timedelta(days=window)
                    p = preds.dropna(subset=["prob_up"]).loc[lambda d: d.index >= cutoff]
                    eq, trades = run_backtest_with_exec_prices(
                        p, close_daily, exec_lookups[tag],
                        up_th=prob, down_th=0.50,
                        fee_bps=FEE_BPS, stake=stake_each)
                    sub_eq.append(eq)
                    trades_all.extend(trades)
                eq = sub_eq[0]
                for s in sub_eq[1:]:
                    eq, s2 = eq.align(s, join="inner")
                    eq = eq + s2
                m = metrics_from_eq(eq, trades_all)
                rows.append({"ensemble": ens_name, "window_days": window,
                             "exec_mode": tag, "exec_desc": desc,
                             **m})
            # Per-ensemble print
            sub = [r for r in rows if r["ensemble"] == ens_name
                   and r["window_days"] == window]
            instant = next(r for r in sub if r["exec_mode"] == "instant")
            for r in sub:
                if r["exec_mode"] == "instant":
                    continue
                d_ret = r["total_return_pct"] - instant["total_return_pct"]
                print(f"  {ens_name:<5s} {window:>3}d  {r['exec_mode']:<8s}  "
                      f"ret={r['total_return_pct']:+.2f}%  "
                      f"Δ vs instant = {d_ret:+.2f}pp  "
                      f"maxDD={r['max_dd_pct']:.2f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_strategy_execution.csv", index=False)

    # Wide matrices
    pivot_ret = df.pivot_table(index=["ensemble", "window_days"],
                                columns="exec_mode",
                                values="total_return_pct").reset_index()
    pivot_dd = df.pivot_table(index=["ensemble", "window_days"],
                                columns="exec_mode",
                                values="max_dd_pct").reset_index()
    pivot_ret.to_csv(OUT_DIR / "matrix_return.csv", index=False)
    pivot_dd.to_csv(OUT_DIR / "matrix_maxdd.csv", index=False)

    print("\n[4/4] summary ...", flush=True)
    lines = []
    lines.append("# Execution-mode sensitivity — TWAP / VWAP simulation")
    lines.append("")
    lines.append("Decision time: 21:30 UTC daily bar close.")
    lines.append("Execution price: averaged across the next N × 15m bars after the signal close.")
    lines.append("Fee = 10 bps per side (unchanged across modes).")
    lines.append("")
    for w in WINDOWS:
        sub = df.loc[df["window_days"] == w]
        lines.append(f"## window = {w}d — total return %")
        lines.append("")
        lines.append("| ensemble | instant | TWAP_1h | TWAP_2h | VWAP_1h | VWAP_2h | Δ instant→TWAP_1h |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for ens in ENSEMBLES.keys():
            r = sub.loc[sub["ensemble"] == ens]
            ret = {row["exec_mode"]: row["total_return_pct"]
                    for _, row in r.iterrows()}
            d = ret.get("TWAP_1h", float("nan")) - ret.get("instant", 0)
            lines.append(f"| {ens} | {ret.get('instant', float('nan')):+.2f} "
                         f"| {ret.get('TWAP_1h', float('nan')):+.2f} "
                         f"| {ret.get('TWAP_2h', float('nan')):+.2f} "
                         f"| {ret.get('VWAP_1h', float('nan')):+.2f} "
                         f"| {ret.get('VWAP_2h', float('nan')):+.2f} "
                         f"| {d:+.2f} pp |")
        lines.append("")
        lines.append(f"## window = {w}d — max drawdown %")
        lines.append("")
        lines.append("| ensemble | instant | TWAP_1h | TWAP_2h | VWAP_1h | VWAP_2h |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for ens in ENSEMBLES.keys():
            r = sub.loc[sub["ensemble"] == ens]
            dd = {row["exec_mode"]: row["max_dd_pct"]
                   for _, row in r.iterrows()}
            lines.append(f"| {ens} | {dd.get('instant', float('nan')):.2f} "
                         f"| {dd.get('TWAP_1h', float('nan')):.2f} "
                         f"| {dd.get('TWAP_2h', float('nan')):.2f} "
                         f"| {dd.get('VWAP_1h', float('nan')):.2f} "
                         f"| {dd.get('VWAP_2h', float('nan')):.2f} |")
        lines.append("")
    (OUT_DIR / "summary.txt").write_text("\n".join(lines))
    print(f"saved → {OUT_DIR/'per_strategy_execution.csv'}")
    print(f"saved → {OUT_DIR/'matrix_return.csv'}")
    print(f"saved → {OUT_DIR/'matrix_maxdd.csv'}")
    print(f"saved → {OUT_DIR/'summary.txt'}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
