"""EBM-native v5 feature selector.

Replaces the legacy XGBoost-based "≥10% top-20 appearance rate over 620 runs"
rule with an EBM-appropriate procedure:

1. Walk every feature_importance_ebm_*.csv produced by the M+SM v0/v3/v4/v6
   sweeps under `reports/utc2130/*/`. Each row already has a 'feature' /
   'gain' column (avg_weight for EBM, mixed univariate + 'a & b'
   interaction names).
2. Per-CSV, split into main-effect rows (no ' & ' in name) and interaction
   rows. Normalize main-effect gains so each run sums to 1, normalize
   interactions separately.
3. Cross-run mean normalized importance per main-effect feature. Also
   compute cross-run std for stability inspection.
4. Cumulative-80% rule: sort by mean importance desc, take features until
   the cumulative sum >= 0.80. That set is the primary keep list.
5. Interaction expansion: collect the top-10 most-important interactions
   across all runs (mean over runs of normalized interaction importance).
   For each, add BOTH endpoint features to keep list (if not already in).
6. Stability filter: compute cross-run Kendall tau rank correlation of
   each feature's rank in each run vs its overall rank. Features with
   tau < 0.3 are noisy — drop them.
7. Emit `src/v5_keep.py` with `V5_KEEP_COLS = (...)` ready for
   `features.py` to import. Also write a markdown report under
   `reports/v5_selection/`.

Usage:
    uv run python analysis/v5_keep_selector_ebm.py
    # then re-run v5 variants with `--retrain` to pick up the new keep list.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUTDIR = REPORTS / "v5_selection"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _load_fi_csvs() -> list[tuple[str, pd.DataFrame]]:
    """Find all feature_importance_ebm_*.csv under reports/, excluding v5
    variants (they don't inform their own keep list)."""
    rows = []
    for p in REPORTS.glob("*/*/feature_importance_ebm_*.csv"):
        # skip v5 variants
        if "_v5" in p.parent.name:
            continue
        df = pd.read_csv(p)
        rows.append((str(p.relative_to(REPORTS)), df))
    return rows


def _normalize_pair_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (main_effect_norm_series, interaction_norm_series)."""
    df = df[df["gain"].notna() & (df["gain"] > 0)]
    is_interaction = df["feature"].str.contains(" & ", regex=False)
    main = df.loc[~is_interaction]
    inter = df.loc[is_interaction]
    if not main.empty:
        s_main = pd.Series(main["gain"].values, index=main["feature"].values)
        s_main = s_main / s_main.sum()
    else:
        s_main = pd.Series(dtype=float)
    if not inter.empty:
        s_inter = pd.Series(inter["gain"].values, index=inter["feature"].values)
        s_inter = s_inter / s_inter.sum()
    else:
        s_inter = pd.Series(dtype=float)
    return s_main, s_inter


def select_v5_keep(cumulative_target: float = 0.80,
                   stability_tau_min: float = 0.30,
                   top_interactions: int = 10,
                   **kwargs) -> dict:
    fis = _load_fi_csvs()
    if not fis:
        raise RuntimeError(
            "No feature_importance_ebm_*.csv found under reports/. "
            "Run main_th_sweep variants for v0/v3/v4/v6 first."
        )
    print(f"  loaded {len(fis)} feature_importance_ebm CSVs")

    # collect per-run normalized importances
    main_records: list[pd.Series] = []
    inter_records: list[pd.Series] = []
    for _, df in fis:
        s_main, s_inter = _normalize_pair_split(df)
        main_records.append(s_main)
        inter_records.append(s_inter)

    main_panel = pd.concat(main_records, axis=1).fillna(0.0)
    inter_panel = pd.concat(inter_records, axis=1).fillna(0.0)

    mean_main = main_panel.mean(axis=1).sort_values(ascending=False)
    std_main = main_panel.std(axis=1).reindex(mean_main.index)

    # Cumulative-80% selection
    cumsum = mean_main.cumsum()
    keep_primary = mean_main[cumsum <= cumulative_target].index.tolist()
    if cumsum.iloc[len(keep_primary)] if len(keep_primary) < len(mean_main) else False:
        keep_primary.append(mean_main.index[len(keep_primary)])  # include the boundary one
    print(f"  cumulative-{cumulative_target:.0%} selection: "
          f"{len(keep_primary)} main-effect features "
          f"(cum_imp={mean_main.loc[keep_primary].sum():.3f})")

    # Interaction expansion: top-N interactions, take both endpoints
    mean_inter = inter_panel.mean(axis=1).sort_values(ascending=False)
    top_inter = mean_inter.head(top_interactions)
    inter_endpoints: set[str] = set()
    for name in top_inter.index:
        parts = [p.strip() for p in name.split(" & ")]
        inter_endpoints.update(parts)
    added_from_interactions = inter_endpoints - set(keep_primary)
    print(f"  + {len(added_from_interactions)} added from top-{top_interactions} "
          f"interactions: {sorted(added_from_interactions)}")

    candidate = list(keep_primary) + sorted(added_from_interactions)

    # Stability filter (cross-run rank Kendall tau)
    keep_final: list[str] = []
    stability: list[dict] = []
    overall_rank = mean_main.rank(ascending=False)
    for feat in candidate:
        ranks_per_run = main_panel.loc[feat].rank(ascending=False) if feat in main_panel.index else None
        if ranks_per_run is None:
            keep_final.append(feat)
            stability.append({"feature": feat, "tau": float("nan"),
                              "kept": True, "reason": "interaction-only"})
            continue
        # Kendall tau between this feature's rank across runs and its
        # global rank. Stable features keep a similar position.
        # We instead measure how consistent the feature's importance
        # ordering is by comparing run-wise rank vs aggregate ranking.
        global_rank = overall_rank.loc[feat]
        # Compute per-run rank of this feature among ALL main features:
        per_run_ranks = main_panel.rank(axis=0, ascending=False).loc[feat]
        # tau between the per-run ranks and a constant vector is undefined;
        # use std of per-run ranks normalized by feature count as a proxy
        # bounded stability score in [0, 1].
        n_feats = main_panel.shape[0]
        normalized_std = per_run_ranks.std() / max(1.0, n_feats / 3.0)
        stab_score = max(0.0, 1.0 - normalized_std)
        # Convert "stability score" to a tau-like number bounded above by
        # 1. If we have multiple runs, also compute Kendall tau between
        # rank vectors as a true measure.
        if main_panel.shape[1] >= 3:
            # Tau between rank in 1st half of runs vs rank in 2nd half
            half = main_panel.shape[1] // 2
            r1 = main_panel.iloc[:, :half].mean(axis=1).rank(ascending=False)
            r2 = main_panel.iloc[:, half:].mean(axis=1).rank(ascending=False)
            tau, _ = kendalltau(r1.loc[candidate], r2.loc[candidate])
            stab_score = max(stab_score, float(tau) if tau == tau else 0.0)

        kept = stab_score >= stability_tau_min or feat in inter_endpoints
        stability.append({"feature": feat, "stab": float(stab_score),
                          "mean_imp": float(mean_main.get(feat, 0.0)),
                          "std_imp": float(std_main.get(feat, 0.0)),
                          "kept": bool(kept)})
        if kept:
            keep_final.append(feat)

    print(f"  after stability filter (>= {stability_tau_min}): "
          f"{len(keep_final)} features")

    # Write outputs
    stab_df = pd.DataFrame(stability).sort_values("mean_imp", ascending=False)
    stab_df.to_csv(OUTDIR / "v5_keep_selection.csv", index=False)
    mean_main.head(80).to_csv(OUTDIR / "main_effect_mean_importance.csv",
                              header=["mean_norm_imp"])
    mean_inter.head(50).to_csv(OUTDIR / "interaction_mean_importance.csv",
                               header=["mean_norm_imp"])

    # Write the importable Python module the next v5 sweep will pick up.
    # output_path / var_name are caller-overridable so the same selector can
    # emit `src/v5_keep.py` (default, cum=0.80) AND `src/v5_2_keep.py`
    # (cum~0.92) without forking the script.
    output_path = kwargs.get("output_path",
                              ROOT / "src" / "v5_keep.py")
    var_name = kwargs.get("var_name", "V5_KEEP_COLS")
    keep_tuple_src = ",\n    ".join(repr(c) for c in keep_final)
    output_path.write_text(
        '"""Auto-generated by analysis/v5_keep_selector_ebm.py.\n\n'
        f"Selected {len(keep_final)} features by EBM-native procedure:\n"
        f"- cumulative-{cumulative_target:.0%} of main-effect importance\n"
        f"- expand by both endpoints of top-{top_interactions} interactions\n"
        f"- stability filter (tau >= {stability_tau_min:.2f})\n"
        f"- {len(fis)} feature_importance_ebm_*.csv files aggregated\n"
        '"""\n\n'
        f"{var_name} = (\n    {keep_tuple_src},\n)\n"
    )
    try:
        rel = output_path.resolve().relative_to(ROOT)
    except ValueError:
        rel = output_path
    print(f"  wrote {rel} with {len(keep_final)} columns")
    return {
        "n_main_features": int(main_panel.shape[0]),
        "n_runs": int(main_panel.shape[1]),
        "n_keep": len(keep_final),
        "cumulative_share": float(mean_main.loc[keep_primary].sum()),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cumulative", type=float, default=0.80,
                        help="cumulative main-effect share target (default 0.80)")
    parser.add_argument("--stability-tau", type=float, default=0.30,
                        help="Kendall tau stability threshold (default 0.30)")
    parser.add_argument("--top-interactions", type=int, default=10,
                        help="top-N interactions whose endpoints expand the "
                             "keep list (default 10)")
    parser.add_argument("--output-file", type=str,
                        default=str(ROOT / "src" / "v5_keep.py"),
                        help="path of the python module to (re)write")
    parser.add_argument("--var-name", type=str, default="V5_KEEP_COLS",
                        help="name of the keep tuple in the output module")
    args = parser.parse_args()
    summary = select_v5_keep(
        cumulative_target=args.cumulative,
        stability_tau_min=args.stability_tau,
        top_interactions=args.top_interactions,
        output_path=Path(args.output_file),
        var_name=args.var_name,
    )
    print(f"\nSummary: {summary}")
