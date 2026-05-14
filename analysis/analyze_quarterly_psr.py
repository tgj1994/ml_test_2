"""Compute PSR + quarterly B&H-outperform breakdown for every (cell, window)
using the cell's BEST (label, prob_TH).

PSR is read from reports/dsr_fwer/fwer_summary.csv (already computed in
analyze_dsr_fwer.py). Quarterly outperform_pp is recomputed from cached
predictions.

Outputs (under reports/dsr_fwer/):
  - quarterly_psr.csv          : long-format (cell, window, quarter, strategy_pct,
                                  buy_hold_pct, outperform_pp)
  - quarterly_psr_wide.csv     : wide matrix (cell, window, label, prob_TH, PSR,
                                  Q1..Qn outperform_pp, n_quarters_outperform,
                                  total_strategy_pct, total_bh_pct)
  - quarterly_consistency.csv  : sorted by % of quarters outperforming B&H
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.utc0000 import build_utc0000_daily
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm"
OUT_DIR = ROOT / "reports" / "dsr_fwer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = (730, 365)


def _kind(cell: str) -> str:
    return "utc2130" if "utc2130" in cell else "utc0000"


def quarterly_breakdown(eq: pd.Series, bh: pd.Series) -> pd.DataFrame:
    """Quarter-end resample → simple returns; index = quarter end (Period)."""
    eq_q = eq.resample("QE").last()
    bh_q = bh.resample("QE").last()
    s_ret = eq_q.pct_change().dropna() * 100
    b_ret = bh_q.pct_change().dropna() * 100
    s_ret, b_ret = s_ret.align(b_ret, join="inner")
    df = pd.DataFrame({
        "strategy_pct": s_ret.values,
        "buy_hold_pct": b_ret.values,
        "outperform_pp": (s_ret - b_ret).values,
    }, index=s_ret.index)
    df.index = df.index.to_period("Q")
    return df


def main() -> None:
    print("[1/4] reading PSR table ...", flush=True)
    fwer = pd.read_csv(OUT_DIR / "fwer_summary.csv")

    print("[2/4] building close series ...", flush=True)
    bar0 = build_utc0000_daily(DATA_DIR / "btc_15m.parquet")
    bar21 = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close_utc0000 = bar0.set_index("close_time")["close"]
    close_utc2130 = bar21.set_index("close_time")["close"]

    print(f"[3/4] computing quarterly breakdown for {len(fwer)} (cell,window) pairs ...",
          flush=True)
    long_rows: list[dict] = []
    wide_rows: list[dict] = []

    for i, row in fwer.iterrows():
        cell = row["cell"]
        window = int(row["window_days"])
        label = row["best_label"]
        prob = float(row["best_prob_TH"])
        psr = float(row["psr"])
        kind = _kind(cell)
        cs = close_utc2130 if kind == "utc2130" else close_utc0000

        cache_path = CACHE_DIR / f"preds_ebm_{label}_{cell}.parquet"
        if not cache_path.exists():
            continue
        preds = pd.read_parquet(cache_path)
        cutoff = preds.index.max() - pd.Timedelta(days=window)
        p = preds.dropna(subset=["prob_up"])
        p = p.loc[p.index >= cutoff]
        if len(p) < 30:
            continue
        cl = cs.reindex(p.index)
        res = run_backtest(p, cl, up_threshold=prob, down_threshold=0.5)
        eq = res.equity_curve
        bh = res.buy_and_hold_curve
        if len(eq) < 5:
            continue

        qb = quarterly_breakdown(eq, bh)
        # long
        for q, qrow in qb.iterrows():
            long_rows.append({
                "cell": cell, "kind": kind, "window_days": window,
                "label": label, "prob_TH": prob, "psr": psr,
                "quarter": str(q),
                "strategy_pct": qrow["strategy_pct"],
                "buy_hold_pct": qrow["buy_hold_pct"],
                "outperform_pp": qrow["outperform_pp"],
            })
        # wide row
        wr = {
            "cell": cell, "kind": kind, "window_days": window,
            "label": label, "prob_TH": prob, "psr": psr,
            "total_strategy_pct": float(res.total_return_pct * 100),
            "total_bh_pct": float(res.buy_and_hold_return_pct * 100),
            "total_outperform_pp": float((res.total_return_pct
                                          - res.buy_and_hold_return_pct) * 100),
            "n_quarters": len(qb),
            "n_quarters_outperform": int((qb["outperform_pp"] > 0).sum()),
            "pct_quarters_outperform":
                float((qb["outperform_pp"] > 0).mean() * 100),
            "median_quarter_outperform_pp": float(qb["outperform_pp"].median()),
            "worst_quarter_outperform_pp": float(qb["outperform_pp"].min()),
            "best_quarter_outperform_pp": float(qb["outperform_pp"].max()),
        }
        for q, qrow in qb.iterrows():
            wr[f"Q_{q}_outperform_pp"] = float(qrow["outperform_pp"])
        wide_rows.append(wr)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)

    long_df.to_csv(OUT_DIR / "quarterly_psr.csv", index=False)
    wide_df.to_csv(OUT_DIR / "quarterly_psr_wide.csv", index=False)
    print(f"  long  → {OUT_DIR/'quarterly_psr.csv'} ({len(long_df)} rows)",
          flush=True)
    print(f"  wide  → {OUT_DIR/'quarterly_psr_wide.csv'} ({len(wide_df)} rows)",
          flush=True)

    print("[4/4] consistency ranking ...", flush=True)
    cons = wide_df.copy()
    cons["sig_psr_095"] = cons["psr"] >= 0.95
    # Rank by (PSR, then pct_quarters_outperform, then total_outperform_pp)
    cons = cons.sort_values(
        ["window_days", "psr", "pct_quarters_outperform", "total_outperform_pp"],
        ascending=[True, False, False, False])
    cons.to_csv(OUT_DIR / "quarterly_consistency.csv", index=False)

    # Headline summary
    print("\n=== TOP 15 (window=730d) by PSR — with quarter-consistency ===")
    cols = ["cell", "label", "prob_TH", "psr",
            "total_strategy_pct", "total_bh_pct", "total_outperform_pp",
            "pct_quarters_outperform", "n_quarters_outperform", "n_quarters",
            "median_quarter_outperform_pp",
            "worst_quarter_outperform_pp", "best_quarter_outperform_pp"]
    s730 = cons.loc[cons["window_days"] == 730, cols].head(15)
    with pd.option_context("display.max_columns", None,
                           "display.width", 220,
                           "display.float_format", "{:.3f}".format):
        print(s730.to_string(index=False))

    print("\n=== TOP 15 (window=365d) by PSR ===")
    s365 = cons.loc[cons["window_days"] == 365, cols].head(15)
    with pd.option_context("display.max_columns", None,
                           "display.width", 220,
                           "display.float_format", "{:.3f}".format):
        print(s365.to_string(index=False))

    # Group summary by UTC kind
    print("\n=== BY UTC kind × window — average consistency ===")
    grp = wide_df.groupby(["kind", "window_days"]).agg(
        n_cells=("cell", "count"),
        mean_psr=("psr", "mean"),
        median_psr=("psr", "median"),
        mean_pct_outperform=("pct_quarters_outperform", "mean"),
        n_psr_095=("psr", lambda s: int((s >= 0.95).sum())),
        n_outperform_total=("total_outperform_pp",
                            lambda s: int((s > 0).sum())),
    ).reset_index()
    print(grp.to_string(index=False))


if __name__ == "__main__":
    main()
