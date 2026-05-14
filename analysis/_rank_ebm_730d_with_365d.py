"""Rank EBM models by 730d return, with the 365d return for the same
(variant, mode, label/k) tuple shown alongside for sanity check.

Reads the new model-suffixed summary CSVs that today's run produces:
  reports/utc2130/{variant}_{window}/
    label_threshold_sweep_summary_static_ebm_{variant}_{window}.csv
    label_threshold_sweep_summary_dynamic_ebm_{variant}_{window}.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPORTS = Path(__file__).resolve().parent.parent / "reports" / "utc2130"


def collect(window: str) -> pd.DataFrame:
    rows = []
    for d in sorted(REPORTS.iterdir()):
        if not d.is_dir() or not d.name.endswith(f"_{window}"):
            continue
        variant = d.name[: -len(window) - 1]
        for mode in ("static", "dynamic"):
            f = d / f"label_threshold_sweep_summary_{mode}_ebm_{variant}_{window}.csv"
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
                    "best_winrate": float(r["best_winrate"]),
                    "buy_hold": float(r["buy_hold"]),
                })
    return pd.DataFrame(rows)


def main() -> int:
    df730 = collect("730d")
    df365 = collect("365d")
    print(f"730d rows: {len(df730)} from {df730.variant.nunique()} variants")
    print(f"365d rows: {len(df365)} from {df365.variant.nunique()} variants")
    print(f"730d buy_hold (median): {df730.buy_hold.median()*100:+.2f}%")
    print(f"365d buy_hold (median): {df365.buy_hold.median()*100:+.2f}%")
    print()

    # Join: same (variant, mode, label_or_k) across the two windows.
    keys = ["variant", "mode", "label_or_k"]
    join = df730[keys + [
        "value", "best_prob_th", "best_return", "best_trades",
        "best_winrate"]].rename(columns={
            "best_prob_th": "th_730d",
            "best_return": "ret_730d",
            "best_trades": "trades_730d",
            "best_winrate": "winrate_730d",
        })
    j365 = df365[keys + [
        "best_prob_th", "best_return", "best_trades",
        "best_winrate"]].rename(columns={
            "best_prob_th": "th_365d",
            "best_return": "ret_365d",
            "best_trades": "trades_365d",
            "best_winrate": "winrate_365d",
        })
    merged = join.merge(j365, on=keys, how="left")

    # Filter to candidates with at least a few trades to avoid degenerate "best"
    pool = merged[merged["trades_730d"] >= 4].copy()
    top = pool.sort_values("ret_730d", ascending=False).head(10).copy()

    # Format for printing
    cols_print = [
        "variant", "mode", "label_or_k", "value",
        "th_730d", "ret_730d", "trades_730d", "winrate_730d",
        "th_365d", "ret_365d", "trades_365d", "winrate_365d",
    ]
    out = top[cols_print].copy()
    for c in ("ret_730d", "ret_365d"):
        out[c] = (out[c] * 100).round(2).astype(str) + "%"
    for c in ("winrate_730d", "winrate_365d"):
        out[c] = (out[c] * 100).round(1).astype(str) + "%"
    for c in ("value", "th_730d", "th_365d"):
        out[c] = out[c].round(4)

    print("=" * 130)
    print("Top 10 EBM models sorted by 730d return")
    print("=" * 130)
    print(out.to_string(index=False))

    print()
    print(f"reference:")
    print(f"  730d buy & hold = {df730.buy_hold.iloc[0]*100:+.2f}%")
    print(f"  365d buy & hold = {df365.buy_hold.iloc[0]*100:+.2f}%")

    out_csv = REPORTS / "aggregate_ebm_730d_with_365d_top10.csv"
    top.to_csv(out_csv, index=False)
    print(f"\nfull top-10 saved -> {out_csv}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
