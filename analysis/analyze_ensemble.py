"""Ensemble backtest — compare 4 ensemble methods vs single-cell baselines.

Components:
  A = utc2130_sm_v5,  label dynk12, prob_TH 0.63   (PSR=0.973)
  B = utc2130_sm,     label dynk11, prob_TH 0.64   (PSR=0.952)

Ensembles:
  E1. SPLIT  — capital split 50/50, each cell trades independently
  E2. VOTE   — vote-based position 0/50/100% based on # of agreeing BUY signals;
               SELL when both cells signal SELL or below entry threshold
  E3. AND    — BUY only when BOTH signal BUY; SELL when EITHER signals SELL
  E4. OR     — BUY when EITHER signals BUY; SELL only when BOTH signal SELL

Each compared to single A and single B over 730d and 365d holdouts.

Outputs (under reports/dsr_fwer/):
  - ensemble_summary.csv          : per-strategy metrics row
  - ensemble_quarterly.csv        : per-strategy quarterly outperform_pp
  - ensemble_equity_curves.png    : 730d equity curves of all strategies
  - ensemble_equity_365d.png      : same for 365d
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import run_backtest, Trade, BacktestResult
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm"
OUT_DIR = ROOT / "reports" / "dsr_fwer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = (730, 365)
STAKE_USD = 100.0
FEE_BPS = 10.0
TRADING_DAYS = 365.0
EULER_GAMMA = 0.5772156649015329

CELL_A_NAME = "utc2130_sm_v5"; LABEL_A = "dynk12"; PROB_A = 0.63
CELL_B_NAME = "utc2130_sm";    LABEL_B = "dynk11"; PROB_B = 0.64


# =========================================================================
# Ensemble backtest engines
# =========================================================================

def _ensemble_backtest(prob_a: pd.Series, prob_b: pd.Series,
                       close: pd.Series,
                       mode: str,
                       th_a: float, th_b: float,
                       fee_bps: float = FEE_BPS,
                       stake: float = STAKE_USD) -> BacktestResult:
    """Combined-signal backtests (VOTE / AND / OR) with single equity curve.

    Position fraction f ∈ {0.0, 0.5, 1.0}.  Cash earns 0%; in-position equity
    tracks price. Transitioning between fractions costs `fee_bps` on the
    *changed* notional only.
    """
    df = pd.concat({"prob_a": prob_a, "prob_b": prob_b, "close": close},
                   axis=1).dropna()
    fee = fee_bps / 1e4

    cash = stake
    qty = 0.0
    cur_frac = 0.0  # 0, 0.5, 1.0
    entry_price_for_trade = None
    entry_date_for_trade = None
    entry_frac = 0.0
    entry_cash_basis = stake
    trades: list[Trade] = []

    equity_arr = []
    bh_arr = []
    bh_anchor = float(df["close"].iloc[0])

    days_buy = days_sell = days_hold = 0
    days_in_market = 0

    for date, row in df.iterrows():
        pa = float(row["prob_a"]); pb = float(row["prob_b"])
        price = float(row["close"])
        sig_a = "BUY" if pa >= th_a else ("SELL" if pa < 0.50 else "HOLD")
        sig_b = "BUY" if pb >= th_b else ("SELL" if pb < 0.50 else "HOLD")

        # Determine target fraction based on mode
        if mode == "VOTE":
            n_buy = int(sig_a == "BUY") + int(sig_b == "BUY")
            n_sell = int(sig_a == "SELL") + int(sig_b == "SELL")
            if n_sell == 2:
                target = 0.0
            elif n_buy == 2:
                target = 1.0
            elif n_buy == 1 and n_sell == 0:
                target = 0.5
            elif n_buy == 1 and n_sell == 1:
                target = 0.5
            else:  # both HOLD or 0 buy 0 sell
                target = cur_frac  # hold prior position
        elif mode == "AND":
            both_buy = (sig_a == "BUY") and (sig_b == "BUY")
            either_sell = (sig_a == "SELL") or (sig_b == "SELL")
            if both_buy:
                target = 1.0
            elif either_sell:
                target = 0.0
            else:
                target = cur_frac
        elif mode == "OR":
            either_buy = (sig_a == "BUY") or (sig_b == "BUY")
            both_sell = (sig_a == "SELL") and (sig_b == "SELL")
            if either_buy:
                target = 1.0
            elif both_sell:
                target = 0.0
            else:
                target = cur_frac
        else:
            raise ValueError(mode)

        # accounting bucket for printout
        if target > cur_frac:
            days_buy += 1
        elif target < cur_frac:
            days_sell += 1
        else:
            days_hold += 1

        # Transition
        if target != cur_frac:
            # Compute current equity (mark-to-market)
            equity_now = cash + qty * price
            # Adjust to target fraction with fee on the changed notional
            target_notional = equity_now * target
            current_notional = qty * price
            delta = target_notional - current_notional  # positive=buy more, neg=sell some
            fee_paid = abs(delta) * fee
            equity_after = equity_now - fee_paid
            target_notional_post = equity_after * target
            new_qty = target_notional_post / price
            new_cash = equity_after - new_qty * price
            # Trade bookkeeping: closing the old fraction → open trade record
            if cur_frac > 0.0 and target == 0.0:
                # full exit
                pnl = equity_after - entry_cash_basis
                ret = pnl / entry_cash_basis
                trades.append(Trade(
                    entry_date=entry_date_for_trade,
                    entry_price=entry_price_for_trade,
                    entry_prob=float("nan"),
                    exit_date=date,
                    exit_price=price,
                    exit_prob=float("nan"),
                    bars_held=0,
                    pnl_usd=pnl,
                    return_pct=ret,
                ))
                entry_price_for_trade = None
                entry_date_for_trade = None
            elif cur_frac == 0.0 and target > 0.0:
                # fresh entry
                entry_price_for_trade = price
                entry_date_for_trade = date
                entry_cash_basis = equity_now  # treat full account as basis
            # partial reshape (e.g., 0.5 → 1.0 or 1.0 → 0.5) — keep entry; pnl
            # accrues continuously through equity curve
            qty = new_qty
            cash = new_cash
            cur_frac = target

        if cur_frac > 0.0:
            days_in_market += 1
        equity_arr.append(cash + qty * price)
        bh_arr.append(stake * price / bh_anchor)

    # force-close
    if cur_frac > 0.0:
        last_price = float(df["close"].iloc[-1])
        equity_now = cash + qty * last_price
        fee_paid = qty * last_price * fee
        equity_after = equity_now - fee_paid
        if entry_cash_basis is not None:
            pnl = equity_after - entry_cash_basis
            ret = pnl / entry_cash_basis
            trades.append(Trade(
                entry_date=entry_date_for_trade,
                entry_price=entry_price_for_trade,
                entry_prob=float("nan"),
                exit_date=df.index[-1],
                exit_price=last_price,
                exit_prob=float("nan"),
                bars_held=0,
                pnl_usd=pnl,
                return_pct=ret,
            ))

    eq = pd.Series(equity_arr, index=df.index, name="equity")
    bh = pd.Series(bh_arr, index=df.index, name="bh")
    final_eq = float(eq.iloc[-1])
    wins = sum(1 for t in trades if t.pnl_usd > 0)
    return BacktestResult(
        trades=trades,
        equity_curve=eq,
        buy_and_hold_curve=bh,
        total_pnl_usd=final_eq - stake,
        total_return_pct=(final_eq - stake) / stake,
        win_rate=(wins / len(trades)) if trades else 0.0,
        n_trades=len(trades),
        days_in_market=days_in_market,
        days_buy=days_buy,
        days_sell=days_sell,
        days_sideways=days_hold,
        buy_and_hold_return_pct=(float(df["close"].iloc[-1]) / bh_anchor) - 1.0,
        final_equity=final_eq,
    )


def _split_capital_backtest(prob_a, prob_b, close,
                            th_a, th_b,
                            fee_bps: float = FEE_BPS,
                            stake: float = STAKE_USD) -> BacktestResult:
    """Two independent half-stake sub-strategies; sum equity curves."""
    res_a = run_backtest(
        pd.DataFrame({"prob_up": prob_a}).dropna(),
        close.reindex(prob_a.dropna().index),
        stake_usd=stake / 2, fee_bps=fee_bps,
        up_threshold=th_a, down_threshold=0.50)
    res_b = run_backtest(
        pd.DataFrame({"prob_up": prob_b}).dropna(),
        close.reindex(prob_b.dropna().index),
        stake_usd=stake / 2, fee_bps=fee_bps,
        up_threshold=th_b, down_threshold=0.50)
    eq_a, eq_b = res_a.equity_curve, res_b.equity_curve
    eq_a, eq_b = eq_a.align(eq_b, join="inner")
    eq = eq_a + eq_b
    bh = res_a.buy_and_hold_curve.reindex(eq.index)
    final_eq = float(eq.iloc[-1])
    # combine trades
    all_trades = list(res_a.trades) + list(res_b.trades)
    wins = sum(1 for t in all_trades if t.pnl_usd > 0)
    return BacktestResult(
        trades=all_trades,
        equity_curve=eq,
        buy_and_hold_curve=bh,
        total_pnl_usd=final_eq - stake,
        total_return_pct=(final_eq - stake) / stake,
        win_rate=(wins / len(all_trades)) if all_trades else 0.0,
        n_trades=len(all_trades),
        days_in_market=int(((eq_a > stake / 2) | (eq_b > stake / 2)).sum()),
        days_buy=res_a.days_buy + res_b.days_buy,
        days_sell=res_a.days_sell + res_b.days_sell,
        days_sideways=res_a.days_sideways + res_b.days_sideways,
        buy_and_hold_return_pct=res_a.buy_and_hold_return_pct,
        final_equity=final_eq,
    )


# =========================================================================
# Metrics
# =========================================================================

def _sharpe_psr(rets: np.ndarray, sr_benchmark_per_period: float = 0.0) -> dict:
    n = len(rets)
    if n < 5 or rets.std(ddof=1) == 0:
        return dict(sharpe=float("nan"), psr=float("nan"),
                    skew=float("nan"), kurt=float("nan"), n=n)
    mu = rets.mean(); sd = rets.std(ddof=1)
    sr_pp = mu / sd
    skew = float(stats.skew(rets, bias=False))
    kurt = float(stats.kurtosis(rets, fisher=False, bias=False))
    sr_var = (1 - skew * sr_pp + (kurt - 1) / 4 * sr_pp ** 2) / (n - 1)
    psr = float(stats.norm.cdf(
        (sr_pp - sr_benchmark_per_period) / math.sqrt(sr_var))) \
        if sr_var > 0 else float("nan")
    return dict(sharpe=sr_pp * math.sqrt(TRADING_DAYS),
                psr=psr, skew=skew, kurt=kurt, n=n,
                sr_var=sr_var, sr_pp=sr_pp)


def _drawdown(eq: pd.Series) -> tuple[float, int, pd.Timestamp]:
    """(max_drawdown_pct, max_dd_duration_days, max_dd_trough_date)."""
    cummax = eq.cummax()
    dd = eq / cummax - 1
    trough = dd.idxmin()
    max_dd = float(dd.min())
    # duration: from peak before trough to recovery (or end)
    peak_date = eq.loc[:trough].idxmax()
    after = eq.loc[trough:]
    peak_value = eq.loc[peak_date]
    recovered = after[after >= peak_value]
    end_date = recovered.index[0] if len(recovered) else after.index[-1]
    return max_dd, (end_date - peak_date).days, trough


def _longest_inactive(eq: pd.Series) -> int:
    flat = (eq.diff().abs() < 1e-9).astype(int)
    runs = []; cur = 0
    for f in flat.values:
        if f: cur += 1
        else:
            if cur: runs.append(cur)
            cur = 0
    if cur: runs.append(cur)
    return max(runs) if runs else 0


def _quarterly(eq: pd.Series, bh: pd.Series) -> pd.DataFrame:
    eq_q = eq.resample("QE").last()
    bh_q = bh.resample("QE").last()
    s_ret = eq_q.pct_change().dropna() * 100
    b_ret = bh_q.pct_change().dropna() * 100
    s_ret, b_ret = s_ret.align(b_ret, join="inner")
    df = pd.DataFrame({
        "strategy_pct": s_ret.values,
        "buy_hold_pct": b_ret.values,
        "outperform_pp": (s_ret - b_ret).values,
    }, index=s_ret.index.tz_localize(None).to_period("Q"))
    return df


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    print("[1/3] loading data ...", flush=True)
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close = bar.set_index("close_time")["close"]

    preds_a = pd.read_parquet(CACHE_DIR / f"preds_ebm_{LABEL_A}_{CELL_A_NAME}.parquet")
    preds_b = pd.read_parquet(CACHE_DIR / f"preds_ebm_{LABEL_B}_{CELL_B_NAME}.parquet")

    summary_rows: list[dict] = []
    quarterly_rows: list[dict] = []

    eq_curves: dict[tuple[int, str], pd.Series] = {}

    print("[2/3] running 6 strategies × 2 windows ...", flush=True)
    for window in WINDOWS:
        cutoff = min(preds_a.index.max(), preds_b.index.max()) \
                 - pd.Timedelta(days=window)
        pa = preds_a["prob_up"].loc[preds_a.index >= cutoff].dropna()
        pb = preds_b["prob_up"].loc[preds_b.index >= cutoff].dropna()
        # align
        common = pa.index.intersection(pb.index)
        pa, pb = pa.loc[common], pb.loc[common]
        cl = close.reindex(common)

        strategies: dict[str, BacktestResult] = {}

        # Single A
        res_a = run_backtest(
            pd.DataFrame({"prob_up": pa}), cl,
            up_threshold=PROB_A, down_threshold=0.50)
        strategies["Single_A_sm_v5"] = res_a

        # Single B
        res_b = run_backtest(
            pd.DataFrame({"prob_up": pb}), cl,
            up_threshold=PROB_B, down_threshold=0.50)
        strategies["Single_B_sm"] = res_b

        # Ensembles
        strategies["E1_SPLIT_50_50"] = _split_capital_backtest(
            pa, pb, cl, PROB_A, PROB_B)
        strategies["E2_VOTE"] = _ensemble_backtest(
            pa, pb, cl, "VOTE", PROB_A, PROB_B)
        strategies["E3_AND"]  = _ensemble_backtest(
            pa, pb, cl, "AND",  PROB_A, PROB_B)
        strategies["E4_OR"]   = _ensemble_backtest(
            pa, pb, cl, "OR",   PROB_A, PROB_B)

        for name, res in strategies.items():
            eq = res.equity_curve
            if len(eq) < 5:
                continue
            drets = eq.pct_change().dropna().to_numpy()
            sp = _sharpe_psr(drets, sr_benchmark_per_period=0.0)
            max_dd, dd_dur, dd_trough = _drawdown(eq)
            longest_flat = _longest_inactive(eq)
            qb = _quarterly(eq, res.buy_and_hold_curve)
            n_outperform = int((qb["outperform_pp"] > 0).sum())
            n_total = len(qb)

            summary_rows.append({
                "strategy": name,
                "window_days": window,
                "total_return_pct": res.total_return_pct * 100,
                "buy_hold_pct": res.buy_and_hold_return_pct * 100,
                "outperform_pp": (res.total_return_pct
                                  - res.buy_and_hold_return_pct) * 100,
                "n_trades": res.n_trades,
                "win_rate": res.win_rate * 100,
                "max_drawdown_pct": max_dd * 100,
                "max_dd_duration_days": dd_dur,
                "longest_inactive_days": longest_flat,
                "sharpe_annual": sp["sharpe"],
                "psr": sp["psr"],
                "n_quarters": n_total,
                "n_quarters_outperform": n_outperform,
                "pct_quarters_outperform":
                    (n_outperform / n_total * 100) if n_total else 0.0,
                "median_q_outperform_pp": float(qb["outperform_pp"].median()),
                "worst_q_outperform_pp": float(qb["outperform_pp"].min()),
            })
            for q, qrow in qb.iterrows():
                quarterly_rows.append({
                    "strategy": name, "window_days": window,
                    "quarter": str(q),
                    "strategy_pct": qrow["strategy_pct"],
                    "buy_hold_pct": qrow["buy_hold_pct"],
                    "outperform_pp": qrow["outperform_pp"],
                })
            eq_curves[(window, name)] = eq

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "ensemble_summary.csv", index=False)
    pd.DataFrame(quarterly_rows).to_csv(
        OUT_DIR / "ensemble_quarterly.csv", index=False)

    # Plots
    print("[3/3] plotting ...", flush=True)
    try:
        import matplotlib.pyplot as plt
        for window in WINDOWS:
            fig, ax = plt.subplots(figsize=(11, 6))
            for name in ["Single_A_sm_v5", "Single_B_sm",
                         "E1_SPLIT_50_50", "E2_VOTE", "E3_AND", "E4_OR"]:
                eq = eq_curves.get((window, name))
                if eq is None:
                    continue
                ax.plot(eq.index, eq.values, label=name, linewidth=1.4)
            # B&H from one of them
            any_eq_key = next(k for k in eq_curves if k[0] == window)
            # we need B&H curve; reconstruct from price
            cl_w = close.reindex(eq_curves[any_eq_key].index)
            bh = STAKE_USD * cl_w / float(cl_w.iloc[0])
            ax.plot(bh.index, bh.values, label="Buy & Hold",
                    linestyle="--", color="black", alpha=0.7)
            ax.set_title(f"Ensemble vs Single — {window}d holdout (UTC 21:30)")
            ax.set_ylabel("USD (stake = $100)")
            ax.legend(fontsize=9, loc="best")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"ensemble_equity_{window}d.png", dpi=130)
            plt.close(fig)
    except Exception as e:
        print(f"  plot failed: {e}")

    # Print summary
    with pd.option_context("display.max_columns", None,
                           "display.width", 250,
                           "display.float_format", "{:.3f}".format):
        print("\n=== 730d holdout ===")
        cols = ["strategy", "total_return_pct", "buy_hold_pct", "outperform_pp",
                "n_trades", "win_rate", "max_drawdown_pct",
                "max_dd_duration_days", "longest_inactive_days",
                "sharpe_annual", "psr", "n_quarters_outperform",
                "n_quarters", "median_q_outperform_pp", "worst_q_outperform_pp"]
        print(summary.loc[summary["window_days"] == 730, cols].to_string(index=False))
        print("\n=== 365d holdout ===")
        print(summary.loc[summary["window_days"] == 365, cols].to_string(index=False))


if __name__ == "__main__":
    main()
