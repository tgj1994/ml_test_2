"""One-shot ranking of EBM 365d sweep results.

Outputs three top-10 rankings:
  1. by absolute return
  2. by t-stat (statistical robustness, n>=8 trades required)
  3. by composite (avg percentile rank of return + tstat + alpha-vs-buyhold)

Reads label_threshold_sweep_summary_{static|dynamic}_{model_kind}_{variant}_{window}.csv.
The model suffix in the filename is required since the EBM/XGB sweeps now
write separately-suffixed summary CSVs into the same report directories.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


REPORTS = Path(__file__).resolve().parent.parent / "reports" / "utc2130"
MODEL_KIND = os.environ.get("MODEL_KIND", "ebm").lower()
AGG = REPORTS / f"aggregate_{MODEL_KIND}_365d.csv"


def main() -> None:
    m = pd.read_csv(AGG)
    m["alpha_vs_bh"] = m["best_return"] - m["buy_hold"]

    HAS_STATS = m["n"] >= 8
    POSITIVE = m["best_return"] > 0
    NOT_OVERFIT = m["best_trades"] >= 4

    print(f"total candidates: {len(m)}")
    print(f"  has trades stats (n>=8): {HAS_STATS.sum()}")
    print(f"  positive return         : {POSITIVE.sum()}")
    print(f"  not overfit (>=4 trades): {NOT_OVERFIT.sum()}")
    print()

    # 1. Top 10 by return
    ret_pool = m[POSITIVE & NOT_OVERFIT].copy()
    top_ret = ret_pool.sort_values("best_return", ascending=False).head(10).copy()

    # 2. Top 10 by t-stat robustness
    rob_pool = m[HAS_STATS & POSITIVE].copy()
    top_rob = rob_pool.sort_values("tstat", ascending=False).head(10).copy()

    # 3. Composite: avg percentile of return + tstat + alpha_vs_bh
    comp_pool = m[HAS_STATS & POSITIVE].copy()
    comp_pool["pr_ret"] = comp_pool["best_return"].rank(pct=True)
    comp_pool["pr_tst"] = comp_pool["tstat"].rank(pct=True)
    comp_pool["pr_alpha"] = comp_pool["alpha_vs_bh"].rank(pct=True)
    comp_pool["composite"] = (
        comp_pool["pr_ret"] + comp_pool["pr_tst"] + comp_pool["pr_alpha"]
    ) / 3
    top_comp = comp_pool.sort_values("composite", ascending=False).head(10).copy()

    def fmt(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        d = df[cols].copy()
        round_cols = (
            "value",
            "best_prob_th",
            "best_return",
            "alpha_vs_bh",
            "sharpe",
            "tstat",
            "best_winrate",
            "composite",
        )
        for c in round_cols:
            if c in d.columns:
                d[c] = d[c].round(4)
        return d

    cols_ret = [
        "variant", "mode", "label_or_k", "value", "best_prob_th",
        "best_return", "alpha_vs_bh", "best_trades", "sharpe", "tstat", "best_winrate",
    ]
    cols_comp = [
        "variant", "mode", "label_or_k", "value", "best_prob_th",
        "best_return", "alpha_vs_bh", "best_trades", "sharpe", "tstat", "composite",
    ]

    sep = "=" * 110
    print(sep)
    print("1) Top 10 by RETURN  (positive return + >=4 trades)")
    print(sep)
    print(fmt(top_ret, cols_ret).to_string(index=False))

    print()
    print(sep)
    print("2) Top 10 by STATISTICAL ROBUSTNESS  (t-stat desc, n>=8 trades, positive ret)")
    print(sep)
    print(fmt(top_rob, cols_ret).to_string(index=False))

    print()
    print(sep)
    print("3) Top 10 by COMPOSITE  (avg percentile of return + tstat + alpha_vs_bh)")
    print(sep)
    print(fmt(top_comp, cols_comp).to_string(index=False))

    print()
    print(f"reference: buy_and_hold over last 365d = {m['buy_hold'].iloc[0]*100:.2f}%")


if __name__ == "__main__":
    main()
