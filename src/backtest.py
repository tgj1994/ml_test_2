"""Backtest a 3-state probability strategy on the most recent ~365 days.

Signal mapping (per daily prediction `prob_up`):
  - prob_up >= UP_THRESHOLD (0.7)            -> BUY (deploy full cash)
  - prob_up <  DOWN_THRESHOLD (0.5)          -> SELL (exit to cash)
  - 0.5 <= prob_up < 0.7                     -> SIDEWAYS (do nothing)

Sideways means: if flat, stay flat; if in a position, hold it. Only a clear
DOWN signal forces an exit. Compounding mode: full cash balance is deployed
on each entry, so realised P&L feeds the next stake.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


STAKE_USD = 100.0
FEE_BPS = 10.0  # 0.10 % per side, conservative for spot

UP_THRESHOLD = 0.7
DOWN_THRESHOLD = 0.5


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    entry_prob: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_prob: float
    bars_held: int
    pnl_usd: float
    return_pct: float


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    buy_and_hold_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    total_pnl_usd: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    days_in_market: int = 0
    days_buy: int = 0
    days_sell: int = 0
    days_sideways: int = 0
    buy_and_hold_return_pct: float = 0.0
    final_equity: float = 0.0


def _classify(prob: float) -> str:
    if prob >= UP_THRESHOLD:
        return "BUY"
    if prob < DOWN_THRESHOLD:
        return "SELL"
    return "HOLD"


def run_backtest(preds: pd.DataFrame, close: pd.Series,
                 stake_usd: float = STAKE_USD,
                 fee_bps: float = FEE_BPS,
                 up_threshold: float = UP_THRESHOLD,
                 down_threshold: float = DOWN_THRESHOLD) -> BacktestResult:
    """3-state compounding backtest.

    preds: index=date, columns include `prob_up`.
    close: daily close prices on the same index.
    """
    df = preds.dropna(subset=["prob_up"]).copy()
    df["close"] = close.reindex(df.index)
    df = df.dropna(subset=["close"])
    fee = fee_bps / 1e4

    cash = stake_usd
    qty = 0.0
    in_pos = False
    entry_idx: int | None = None
    entry_price = np.nan
    entry_prob = np.nan
    entry_cash = np.nan
    trades: list[Trade] = []

    equity: list[float] = []
    bh: list[float] = []
    bh_anchor = float(df["close"].iloc[0])

    days_in_market = 0
    days_buy = days_sell = days_hold = 0

    rows = list(df.itertuples(index=True))
    for k, row in enumerate(rows):
        date = row.Index
        price = float(row.close)
        prob = float(row.prob_up)

        if prob >= up_threshold:
            signal = "BUY"
            days_buy += 1
        elif prob < down_threshold:
            signal = "SELL"
            days_sell += 1
        else:
            signal = "HOLD"
            days_hold += 1

        # exit on SELL signal only
        if in_pos and signal == "SELL":
            cash = qty * price * (1 - fee)
            pnl = cash - entry_cash
            ret = pnl / entry_cash
            trades.append(Trade(
                entry_date=df.index[entry_idx],
                entry_price=entry_price,
                entry_prob=entry_prob,
                exit_date=date,
                exit_price=price,
                exit_prob=prob,
                bars_held=k - entry_idx,
                pnl_usd=pnl,
                return_pct=ret,
            ))
            qty = 0.0
            in_pos = False
            entry_idx = None

        # entry on BUY signal only
        if not in_pos and signal == "BUY":
            entry_cash = cash
            qty = (cash * (1 - fee)) / price
            cash = 0.0
            entry_idx = k
            entry_price = price
            entry_prob = prob
            in_pos = True

        if in_pos:
            days_in_market += 1
            equity.append(qty * price)
        else:
            equity.append(cash)
        bh.append(stake_usd * price / bh_anchor)

    # force-close any remaining position
    if in_pos and entry_idx is not None:
        date = df.index[-1]
        price = float(df["close"].iloc[-1])
        prob = float(df["prob_up"].iloc[-1])
        cash = qty * price * (1 - fee)
        pnl = cash - entry_cash
        ret = pnl / entry_cash
        trades.append(Trade(
            entry_date=df.index[entry_idx],
            entry_price=entry_price,
            entry_prob=entry_prob,
            exit_date=date,
            exit_price=price,
            exit_prob=prob,
            bars_held=len(df) - 1 - entry_idx,
            pnl_usd=pnl,
            return_pct=ret,
        ))
        qty = 0.0
        in_pos = False

    eq = pd.Series(equity, index=df.index, name="equity")
    bh_curve = pd.Series(bh, index=df.index, name="buy_and_hold")
    wins = sum(1 for t in trades if t.pnl_usd > 0)
    final_equity = float(eq.iloc[-1])
    total_pnl = final_equity - stake_usd

    return BacktestResult(
        trades=trades,
        equity_curve=eq,
        buy_and_hold_curve=bh_curve,
        total_pnl_usd=total_pnl,
        total_return_pct=total_pnl / stake_usd,
        win_rate=(wins / len(trades)) if trades else 0.0,
        n_trades=len(trades),
        days_in_market=days_in_market,
        days_buy=days_buy,
        days_sell=days_sell,
        days_sideways=days_hold,
        buy_and_hold_return_pct=(float(df["close"].iloc[-1]) / bh_anchor) - 1.0,
        final_equity=final_equity,
    )


def write_report(results: dict[str, "BacktestResult"],
                 importances: dict[str, pd.DataFrame],
                 out_dir: Path) -> None:
    """Write equity-curve plot, trade logs, and feature-importance CSVs for
    each model in `results`. Equity curves of all models share one chart."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    bh_drawn = False
    for name, res in results.items():
        ax.plot(res.equity_curve.index, res.equity_curve.values, label=f"{name} strategy")
        if not bh_drawn:
            ax.plot(res.buy_and_hold_curve.index, res.buy_and_hold_curve.values,
                    label="Buy & Hold", linestyle="--", color="black", alpha=0.7)
            bh_drawn = True
    ax.set_title("BTC 3-state strategy — model comparison")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curve.png", dpi=120)
    plt.close(fig)

    for name, res in results.items():
        trades_df = pd.DataFrame([t.__dict__ for t in res.trades])
        if not trades_df.empty:
            trades_df.to_csv(out_dir / f"trades_{name}.csv", index=False)
    for name, fi in importances.items():
        fi.to_csv(out_dir / f"feature_importance_{name}.csv", index=False)
