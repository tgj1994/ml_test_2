"""SHAP-style feature-category attribution for Top1/Top2/Top3/Top5.

Approach (paper-grade attribution from existing artefacts):
  - For each ensemble's component cells, load the per-cell feature_importance
    CSV (gain values from the trained calibrated XGBoost). Each label slug
    has its own dump; we average gains across the 22 (label) slugs to get a
    cell-level importance vector that is robust to a single label's noise.
  - Map every feature name to one of 11 categories using
    feature_descriptions.json.  Add a 12th meta-category "real_onchain" for
    the v3+ Coin Metrics MVRV/NVT/hashrate variables (mvrv2_*) which the
    JSON does not yet enumerate.
  - For each ensemble: sum gains per category across all component cells
    (each cell weighted equally, since SPLIT-CAPITAL ensembles allocate
    1/N capital each).
  - Report normalised category share (gain / total gain) and the top-5
    individual features per ensemble.
  - Specifically test: does mvrv2_hashrate_z (VIF 1.07 in §7.16) appear in
    the top features of any ensemble? Same for mvrv2_z_ema900, mvrv2_nvt_z.

Output:
  reports/shap_attribution/category_share.csv
  reports/shap_attribution/top_features_per_ensemble.csv
  reports/shap_attribution/orthogonal_onchain_presence.csv
  reports/shap_attribution/summary.txt
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "reports" / "shap_attribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENSEMBLES = {
    "Top1": ["utc2130_sm_v5"],
    "Top2": ["utc2130_sm_v5", "utc2130_sm_v3_complete"],
    "Top3": ["utc2130_sm_v5", "utc2130_sm_v3_complete", "utc2130_sm"],
    "Top5": ["utc2130_sm_v5", "utc2130_sm_v3_complete", "utc2130_sm",
             "utc2130_sm_v3", "utc2130_sm_v6"],
}

# v3+ on-chain MVRV / NVT / hashrate (Coin Metrics) — meta-category
REAL_ONCHAIN = {"mvrv2_z_200w", "mvrv2_z_ema900", "mvrv2_z_multi",
                 "mvrv2_mayer", "mvrv2_nvt_z", "mvrv2_hashrate_z"}

# Orthogonality test targets (from §7.16 VIF analysis)
ORTHOGONAL_TEST = ["mvrv2_hashrate_z", "mvrv2_z_ema900", "mvrv2_nvt_z"]

# Korean display labels for paper
CATEGORY_DISPLAY = {
    "daily_ta":     "TA-A: Daily TA",
    "weekly_ta":    "TA-A: Weekly TA",
    "monthly_ta":   "TA-A: Monthly TA",
    "window":       "TA-B: Window summaries",
    "intraday":     "TA-C: 15m intraday",
    "sideways":     "TA-D: Sideways/regime",
    "mvrv_proxy":   "Block H: Price-based MVRV proxy (v0)",
    "macro":        "Block E: Cross-asset macro",
    "futures":      "Block F: Funding/basis",
    "sentiment":    "Block F: Fear & Greed",
    "calendar":     "Block G: Calendar",
    "real_onchain": "Block H: Real on-chain (Coin Metrics, v3+)",
    "other":        "Other / unclassified",
}


def load_categories() -> dict[str, str]:
    """feature → category mapping, with mvrv2_* mapped to real_onchain."""
    fd = json.loads((ROOT / "feature_descriptions.json").read_text())
    out: dict[str, str] = {}
    for f in fd["features"]:
        out[f["name"]] = f["category"]
    for f in REAL_ONCHAIN:
        out[f] = "real_onchain"
    return out


def cell_avg_importance(cell: str) -> pd.DataFrame:
    """Average per-feature gain across all label fits for one cell (730d
    sweep). Returns DataFrame [feature, mean_gain, n_fits, mean_rank,
    mean_n_features]."""
    rep_dir = ROOT / "reports" / "utc2130" / f"{cell}_730d"
    fi_files = sorted(glob.glob(str(rep_dir / "feature_importance_ebm_*.csv")))
    rows = []
    for f in fi_files:
        df = pd.read_csv(f)
        df["rank"] = df["gain"].rank(ascending=False, method="min").astype(int)
        df["n_features_in_fit"] = len(df)
        df["fit_id"] = os.path.basename(f)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    cat = pd.concat(rows, ignore_index=True)
    agg = (cat.groupby("feature")
              .agg(mean_gain=("gain", "mean"),
                   n_fits=("gain", "size"),
                   mean_rank=("rank", "mean"),
                   mean_n_features=("n_features_in_fit", "mean"),
                   max_gain=("gain", "max"))
              .reset_index())
    return agg.sort_values("mean_gain", ascending=False)


def main() -> None:
    print("[1/4] loading category map ...", flush=True)
    cat_map = load_categories()
    print(f"  features mapped: {len(cat_map)}")

    print("[2/4] aggregating per-cell importance ...", flush=True)
    cell_imps: dict[str, pd.DataFrame] = {}
    all_cells = sorted(set().union(*ENSEMBLES.values()))
    for cell in all_cells:
        df = cell_avg_importance(cell)
        if df.empty:
            print(f"  ! {cell}: no feature_importance dumps")
            continue
        df["category"] = df["feature"].map(lambda f: cat_map.get(f, "other"))
        cell_imps[cell] = df
        print(f"  {cell}: {len(df)} unique features, "
              f"top gain={df['mean_gain'].iloc[0]:.3f} "
              f"({df['feature'].iloc[0]})")

    print("[3/4] computing per-ensemble category shares ...", flush=True)
    cat_rows = []
    feat_rows = []
    for name, cells in ENSEMBLES.items():
        agg_by_feat: dict[str, float] = {}
        for cell in cells:
            df = cell_imps.get(cell)
            if df is None:
                continue
            for _, r in df.iterrows():
                agg_by_feat[r["feature"]] = (
                    agg_by_feat.get(r["feature"], 0.0)
                    + r["mean_gain"] / len(cells))  # equal-weight ensemble
        total = sum(agg_by_feat.values())
        # Category share
        cat_total: dict[str, float] = {}
        for f, g in agg_by_feat.items():
            cat = cat_map.get(f, "other")
            cat_total[cat] = cat_total.get(cat, 0.0) + g
        for cat, g in cat_total.items():
            cat_rows.append({
                "ensemble": name, "category": cat,
                "category_label": CATEGORY_DISPLAY.get(cat, cat),
                "gain_sum": g,
                "share_pct": (g / total * 100) if total > 0 else 0.0,
            })
        # Top features
        top10 = sorted(agg_by_feat.items(), key=lambda x: -x[1])[:10]
        for rank, (feat, g) in enumerate(top10, start=1):
            feat_rows.append({
                "ensemble": name, "rank": rank, "feature": feat,
                "category": cat_map.get(feat, "other"),
                "gain": g,
                "share_pct": (g / total * 100) if total > 0 else 0.0,
            })

    cat_df = pd.DataFrame(cat_rows).sort_values(
        ["ensemble", "share_pct"], ascending=[True, False])
    feat_df = pd.DataFrame(feat_rows)
    cat_df.to_csv(OUT_DIR / "category_share.csv", index=False)
    feat_df.to_csv(OUT_DIR / "top_features_per_ensemble.csv", index=False)

    # Orthogonality test
    print("[4/4] orthogonality presence test ...", flush=True)
    orth_rows = []
    for name, cells in ENSEMBLES.items():
        for tgt in ORTHOGONAL_TEST:
            present_in: list[str] = []
            ranks: list[float] = []
            gains: list[float] = []
            for cell in cells:
                df = cell_imps.get(cell)
                if df is None:
                    continue
                sub = df.loc[df["feature"] == tgt]
                if len(sub):
                    r = sub.iloc[0]
                    present_in.append(cell)
                    ranks.append(r["mean_rank"])
                    gains.append(r["mean_gain"])
            orth_rows.append({
                "ensemble": name, "feature": tgt,
                "present_in_n_cells": len(present_in),
                "n_total_cells": len(cells),
                "cells": ",".join(present_in),
                "mean_rank": np.mean(ranks) if ranks else float("nan"),
                "mean_gain": np.mean(gains) if gains else float("nan"),
            })
    orth_df = pd.DataFrame(orth_rows)
    orth_df.to_csv(OUT_DIR / "orthogonal_onchain_presence.csv", index=False)

    # Summary
    lines = []
    lines.append("# Feature-category attribution — Top1 / Top2 / Top3 / Top5")
    lines.append("")
    lines.append("## Category share (% of total ensemble-level gain)")
    lines.append("")
    pivot = (cat_df.pivot_table(index="category_label", columns="ensemble",
                                  values="share_pct", fill_value=0.0)
                  [list(ENSEMBLES.keys())])
    pivot = pivot.sort_values("Top1", ascending=False)
    lines.append("| category | Top1 | Top2 | Top3 | Top5 |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, row in pivot.iterrows():
        lines.append(f"| {label} | {row['Top1']:.1f}% | {row['Top2']:.1f}% "
                     f"| {row['Top3']:.1f}% | {row['Top5']:.1f}% |")
    lines.append("")
    lines.append("## Top-5 features per ensemble")
    for name in ENSEMBLES.keys():
        sub = feat_df.loc[feat_df["ensemble"] == name].head(5)
        lines.append("")
        lines.append(f"### {name}")
        lines.append("| rank | feature | category | gain | share% |")
        lines.append("|---:|---|---|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(f"| {int(r['rank'])} | `{r['feature']}` | "
                         f"{r['category']} | {r['gain']:.3f} | "
                         f"{r['share_pct']:.1f}% |")
    lines.append("")
    lines.append("## Orthogonal on-chain presence test (§7.16 follow-up)")
    lines.append("Does the model actually USE the v3+ Coin Metrics features "
                 "shown in §7.16 to be orthogonal to TA?")
    lines.append("")
    lines.append("| ensemble | feature | present in | mean rank | mean gain |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in orth_df.iterrows():
        present = (f"{int(r['present_in_n_cells'])}/{int(r['n_total_cells'])}"
                   if r["present_in_n_cells"] > 0 else "0")
        rank = (f"{r['mean_rank']:.1f}"
                if not pd.isna(r["mean_rank"]) else "—")
        gain = (f"{r['mean_gain']:.3f}"
                if not pd.isna(r["mean_gain"]) else "—")
        lines.append(f"| {r['ensemble']} | `{r['feature']}` | {present} | "
                     f"{rank} | {gain} |")
    (OUT_DIR / "summary.txt").write_text("\n".join(lines))
    print(f"\nsaved → {OUT_DIR/'category_share.csv'}")
    print(f"saved → {OUT_DIR/'top_features_per_ensemble.csv'}")
    print(f"saved → {OUT_DIR/'orthogonal_onchain_presence.csv'}")
    print(f"saved → {OUT_DIR/'summary.txt'}")
    print()
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
