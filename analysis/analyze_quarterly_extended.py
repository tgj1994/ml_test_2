"""Quarterly returns for the 4 ensembles (Top1/2/3/5) at 1095d / 1460d windows.

For each (model, window):
  - Build split-capital ensemble equity curve from cached extended preds.
  - Resample equity to quarter-end → quarterly returns (% per quarter).
  - Same for B&H.
  - Outperform = strategy_q − bh_q (pp).

Outputs:
  reports/utc2130_extended/quarterly_returns_extended_long.csv   # long form
  reports/utc2130_extended/quarterly_returns_extended_wide.csv   # pivot
  reports/utc2130_extended/quarterly_returns_extended_summary.csv  # per (model,window)
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.backtest import run_backtest
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm_ebm_extended"
OUT_DIR = ROOT / "reports" / "utc2130_extended"

WINDOWS = (1095, 1460)
STAKE_USD = 100.0


@dataclass
class CellPick:
    cell: str
    label: str
    prob_TH: float


SUB_MODEL_PICKS = {
    "sm_v5_dynk12":         CellPick("utc2130_sm_v5",          "dynk12", 0.63),
    "sm_v3_complete_th13":  CellPick("utc2130_sm_v3_complete", "th13",   0.68),
    "sm_dynk11":            CellPick("utc2130_sm",             "dynk11", 0.64),
    "sm_v3_th14":           CellPick("utc2130_sm_v3",          "th14",   0.66),
    "sm_v6_dynk14":         CellPick("utc2130_sm_v6",          "dynk14", 0.66),
}

ENSEMBLES = {
    "top1": ["sm_v5_dynk12"],
    "top2": ["sm_v5_dynk12", "sm_v3_complete_th13"],
    "top3": ["sm_v5_dynk12", "sm_v3_complete_th13", "sm_dynk11"],
    "top5": ["sm_v5_dynk12", "sm_v3_complete_th13", "sm_dynk11",
             "sm_v3_th14", "sm_v6_dynk14"],
}


def _ensemble_curves(picks: list[CellPick], window_days: int,
                     close: pd.Series) -> tuple[pd.Series, pd.Series]:
    N = len(picks)
    sub_eq: list[pd.Series] = []
    bh_curve = None
    for pk in picks:
        cache = CACHE_DIR / f"preds_ebm_{pk.label}_{pk.cell}.parquet"
        preds = pd.read_parquet(cache)
        nn = preds.dropna(subset=["prob_up"])
        cutoff = nn.index.max() - pd.Timedelta(days=window_days)
        p = nn.loc[nn.index >= cutoff]
        cl = close.reindex(p.index)
        res = run_backtest(p, cl, stake_usd=STAKE_USD / N,
                           up_threshold=pk.prob_TH, down_threshold=0.50)
        sub_eq.append(res.equity_curve)
        if bh_curve is None:
            bh_curve = res.buy_and_hold_curve
    aligned = sub_eq[0]
    for s in sub_eq[1:]:
        a, s2 = aligned.align(s, join="inner")
        aligned = a + s2
    eq = aligned
    bh = bh_curve.reindex(eq.index)
    return eq, bh


def _quarterly_pct(curve: pd.Series) -> pd.Series:
    return curve.resample("QE").last().pct_change().dropna() * 100


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close = bar.set_index("close_time")["close"]

    long_rows = []
    summary_rows = []
    for model_name, sub_keys in ENSEMBLES.items():
        picks = [SUB_MODEL_PICKS[k] for k in sub_keys]
        for w in WINDOWS:
            eq, bh = _ensemble_curves(picks, w, close)
            sq = _quarterly_pct(eq)
            bq = _quarterly_pct(bh)
            sq, bq = sq.align(bq, join="inner")
            diff = sq - bq
            for q_ts in sq.index:
                long_rows.append({
                    "model": model_name,
                    "window_days": w,
                    "quarter_end": q_ts.strftime("%Y-Q%q") if hasattr(q_ts, "strftime") else str(q_ts),
                    "quarter": f"{q_ts.year}-Q{((q_ts.month-1)//3)+1}",
                    "strategy_pct": round(float(sq.loc[q_ts]), 2),
                    "bh_pct": round(float(bq.loc[q_ts]), 2),
                    "outperform_pp": round(float(diff.loc[q_ts]), 2),
                })
            n = len(diff)
            n_out = int((diff > 0).sum())
            summary_rows.append({
                "model": model_name,
                "window_days": w,
                "n_quarters": n,
                "n_outperform": n_out,
                "pct_outperform": round(n_out / n * 100, 2) if n else 0.0,
                "median_strat_pct": round(float(sq.median()), 2),
                "median_bh_pct": round(float(bq.median()), 2),
                "median_outperform_pp": round(float(diff.median()), 2),
                "best_outperform_pp": round(float(diff.max()), 2),
                "worst_outperform_pp": round(float(diff.min()), 2),
                "best_quarter": f"{diff.idxmax().year}-Q{((diff.idxmax().month-1)//3)+1}",
                "worst_quarter": f"{diff.idxmin().year}-Q{((diff.idxmin().month-1)//3)+1}",
            })

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT_DIR / "quarterly_returns_extended_long.csv", index=False)

    # wide pivot: rows=quarter, cols=(model, window) → outperform_pp
    pivot_strat = long_df.pivot_table(
        index="quarter", columns=["model", "window_days"],
        values="strategy_pct", aggfunc="first").sort_index()
    pivot_bh = long_df.pivot_table(
        index="quarter", columns=["window_days"],
        values="bh_pct", aggfunc="first").sort_index()
    pivot_op = long_df.pivot_table(
        index="quarter", columns=["model", "window_days"],
        values="outperform_pp", aggfunc="first").sort_index()
    with pd.ExcelWriter(OUT_DIR / "quarterly_returns_extended.xlsx") if False else open(
            OUT_DIR / "quarterly_returns_strategy_wide.csv", "w") as f:
        pivot_strat.to_csv(f)
    pivot_bh.to_csv(OUT_DIR / "quarterly_returns_bh_wide.csv")
    pivot_op.to_csv(OUT_DIR / "quarterly_returns_outperform_wide.csv")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "quarterly_returns_extended_summary.csv", index=False)

    print("=" * 78)
    print("Quarterly returns — strategy %% per quarter")
    print("=" * 78)
    print(pivot_strat.round(2).to_string())
    print("\n" + "=" * 78)
    print("Quarterly outperform vs B&H (pp)")
    print("=" * 78)
    print(pivot_op.round(2).to_string())
    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    with pd.option_context("display.max_columns", None,
                           "display.width", 220):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
