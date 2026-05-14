"""Full SHAP analysis for paper §7.22.

For each of the 5 production sub-cells (live_models_utc2130/*.pkl):
  1. Load the calibrated XGBoost. Extract the 3 base XGBClassifier folds and
     average their SHAP TreeExplainer outputs.
  2. Compute SHAP values across the 730-day test period.
  3. Aggregate global mean |SHAP| and compare to the gain ranking from §7.19.
  4. Save:
       - shap_global_mean_abs.csv  (per-cell mean |SHAP|)
       - summary_beeswarm_<cell>.png
       - dependence_<cell>_<feat>.png  for top-3 features per cell
       - local_shap_<cell>_<date>.png  waterfall for representative days
  5. Surrogate depth-3 decision tree fit on the model's prob_up output —
     extract IF-THEN rules.
  6. Single-feature SHAP × value scatter for the most-important feature.

Outputs (under reports/shap_full/):
  global_mean_abs.csv
  surrogate_rules.txt
  plots/<png files>
  summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.tree import DecisionTreeRegressor, export_text


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "live_models_utc2130"
OUT_DIR = ROOT / "reports" / "shap_full"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 730

# 5 production sub-cells (from coin_service_utc2130 SUB_MODELS)
SUB_CELLS = {
    "sm_v5_dynk12":          {"feature_set": "v5", "policy": "raw"},
    "sm_v3_complete_th13":   {"feature_set": "v3", "policy": "complete"},
    "sm_dynk11":             {"feature_set": "v0", "policy": "raw"},
    "sm_v3_th14":            {"feature_set": "v3", "policy": "raw"},
    "sm_v6_dynk14":          {"feature_set": "v6", "policy": "raw"},
}


def build_X_for_cell(sub_cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    from src.utc2130 import build_utc2130_daily
    from src.features import build_features

    daily_df = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    feature_set = sub_cfg["feature_set"]
    policy = sub_cfg["policy"]
    kwargs = dict(daily_df=daily_df, external_lag_days=0)
    if feature_set == "v3":
        kwargs["use_features_v3"] = True
    elif feature_set == "v5":
        kwargs["use_features_v5"] = True
    elif feature_set == "v6":
        kwargs["use_features_v4"] = True
    X, _, close = build_features(DATA_DIR, label_threshold=0.015, **kwargs)
    if policy == "complete":
        X = X.sort_index().ffill().bfill()
    return X, close


def extract_average_shap(calibrated, X_aligned: pd.DataFrame) -> np.ndarray:
    """Average SHAP across the 3 calibration folds. Each fold has its own
    XGBClassifier; we run TreeExplainer on each, then average.

    Returns array of shape (n_rows, n_features).
    """
    fold_shaps = []
    for cc in calibrated.calibrated_classifiers_:
        xgb = cc.estimator
        explainer = shap.TreeExplainer(xgb)
        sv = explainer.shap_values(X_aligned)
        # XGBoost binary: shap_values returns array (n, n_features)
        # (for newer SHAP versions it may return Explanation; handle both)
        if hasattr(sv, "values"):
            sv = sv.values
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]
        fold_shaps.append(np.asarray(sv))
    return np.mean(fold_shaps, axis=0)


def run_one_cell(sub_key: str, sub_cfg: dict) -> dict:
    print(f"\n[{sub_key}] loading model + X ...", flush=True)
    pkl_path = MODELS_DIR / f"model_{sub_key}.pkl"
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    calibrated = bundle["model"]
    cols = bundle["feature_columns"]

    X_full, close = build_X_for_cell(sub_cfg)
    # Align X to the model's feature_columns
    X = X_full.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[cols]
    # Cap at training cutoff (= the bundle's last training date) to ensure
    # we explain the model on data it has actually seen for SHAP integrity.
    cutoff = pd.Timestamp(bundle["trained_at"]).tz_convert("UTC") \
        if "trained_at" in bundle else X.index.max()
    X_test = X.loc[X.index >= (X.index.max() - pd.Timedelta(days=WINDOW_DAYS))]
    X_test = X_test.loc[X_test.index <= cutoff]
    X_test = X_test.dropna(how="all")
    print(f"  X_test shape: {X_test.shape}")

    print(f"[{sub_key}] computing SHAP (3-fold average) ...", flush=True)
    shap_values = extract_average_shap(calibrated, X_test)
    print(f"  shap shape: {shap_values.shape}")

    # Global aggregate
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    feat_df = pd.DataFrame({
        "feature": cols,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    feat_df["rank"] = feat_df.index + 1

    # Save beeswarm plot
    print(f"[{sub_key}] beeswarm plot ...", flush=True)
    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, X_test, feature_names=cols,
                       max_display=15, show=False, plot_type="dot")
    fig.suptitle(f"SHAP summary — {sub_key} (730d test)", y=1.0, fontsize=12)
    fig.tight_layout()
    plt.savefig(PLOTS_DIR / f"summary_beeswarm_{sub_key}.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # Dependence plots for top-3 features
    print(f"[{sub_key}] dependence plots ...", flush=True)
    for tgt in feat_df["feature"].head(3):
        try:
            fig = plt.figure(figsize=(7, 5))
            shap.dependence_plot(tgt, shap_values, X_test,
                                  feature_names=cols, show=False,
                                  interaction_index="auto")
            fig.suptitle(f"SHAP dependence — {sub_key} / {tgt}",
                         y=1.0, fontsize=11)
            fig.tight_layout()
            plt.savefig(PLOTS_DIR / f"dependence_{sub_key}_{tgt}.png",
                        dpi=130, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # noqa
            print(f"  ! dependence plot for {tgt} failed: {e}")
            plt.close("all")

    # Local SHAP waterfalls — pick 3 representative days
    print(f"[{sub_key}] local waterfalls ...", flush=True)
    # Get raw probabilities from the calibrated model
    probs = calibrated.predict_proba(X_test)[:, 1]
    prob_series = pd.Series(probs, index=X_test.index)

    # Pick top 1 prob, bottom 1 prob, and one near-threshold day
    high_idx = prob_series.idxmax()
    low_idx = prob_series.idxmin()
    th = bundle.get("prob_TH", 0.65)
    near_idx = (prob_series - th).abs().idxmin()
    sample_idx = [("HIGH", high_idx), ("LOW", low_idx), ("NEAR_TH", near_idx)]
    base_value_per_fold = []
    for cc in calibrated.calibrated_classifiers_:
        try:
            base_value_per_fold.append(
                shap.TreeExplainer(cc.estimator).expected_value)
        except Exception:
            base_value_per_fold.append(0.0)
    base_val = float(np.mean([
        v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0
        else float(v) for v in base_value_per_fold]))

    for tag, idx in sample_idx:
        try:
            i = X_test.index.get_loc(idx)
            sv_row = shap_values[i]
            x_row = X_test.iloc[i]
            # Build SHAP Explanation manually
            exp = shap.Explanation(values=sv_row,
                                    base_values=base_val,
                                    data=x_row.values,
                                    feature_names=cols)
            fig = plt.figure(figsize=(8, 6))
            shap.plots.waterfall(exp, max_display=12, show=False)
            fig.suptitle(f"SHAP waterfall — {sub_key} / "
                         f"{idx.strftime('%Y-%m-%d')} ({tag}, "
                         f"prob={prob_series.loc[idx]:.3f})",
                         fontsize=11)
            fig.tight_layout()
            plt.savefig(PLOTS_DIR / f"waterfall_{sub_key}_{tag}.png",
                        dpi=130, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # noqa
            print(f"  ! local waterfall {tag} failed: {e}")
            plt.close("all")

    # Surrogate decision tree → IF-THEN rules
    # Train depth-3 regressor on the model's prob_up output
    print(f"[{sub_key}] surrogate tree ...", flush=True)
    # Drop rows with all-NaN to satisfy sklearn
    X_tree = X_test.fillna(X_test.median(numeric_only=True)).fillna(0)
    surrogate = DecisionTreeRegressor(max_depth=3, random_state=42)
    surrogate.fit(X_tree, probs)
    rules = export_text(surrogate, feature_names=list(X_tree.columns),
                          max_depth=3, decimals=3)

    return {
        "sub_key": sub_key,
        "feat_df": feat_df,
        "rules": rules,
        "high_idx": high_idx,
        "low_idx": low_idx,
        "near_idx": near_idx,
        "high_prob": float(prob_series.loc[high_idx]),
        "low_prob": float(prob_series.loc[low_idx]),
        "near_prob": float(prob_series.loc[near_idx]),
        "prob_TH": th,
    }


def main() -> None:
    all_global_rows = []
    rules_text = []
    summary_lines = []
    summary_lines.append("# Full SHAP analysis — Top1/Top2/Top3/Top5 sub-cells")
    summary_lines.append("")

    for sub_key, sub_cfg in SUB_CELLS.items():
        try:
            res = run_one_cell(sub_key, sub_cfg)
        except Exception as e:  # noqa
            print(f"!! {sub_key} failed: {e}")
            continue

        feat_df = res["feat_df"]
        feat_df["sub_key"] = sub_key
        all_global_rows.append(feat_df)

        # Top-10 feature table for this cell
        summary_lines.append(f"## {sub_key} — top-10 features by mean |SHAP|")
        summary_lines.append("")
        summary_lines.append("| rank | feature | mean |SHAP| | mean signed SHAP |")
        summary_lines.append("|---:|---|---:|---:|")
        for _, r in feat_df.head(10).iterrows():
            summary_lines.append(
                f"| {int(r['rank'])} | `{r['feature']}` | "
                f"{r['mean_abs_shap']:.4f} | "
                f"{r['mean_signed_shap']:+.4f} |")
        summary_lines.append("")
        summary_lines.append(
            f"Representative days: HIGH = {res['high_idx'].date()} "
            f"(prob {res['high_prob']:.3f}), "
            f"NEAR_TH = {res['near_idx'].date()} "
            f"(prob {res['near_prob']:.3f}, threshold {res['prob_TH']:.2f}), "
            f"LOW = {res['low_idx'].date()} "
            f"(prob {res['low_prob']:.3f}).")
        summary_lines.append("")

        rules_text.append(f"=== {sub_key} (depth-3 surrogate tree) ===\n")
        rules_text.append(res["rules"])
        rules_text.append("\n")

    if all_global_rows:
        global_df = pd.concat(all_global_rows, ignore_index=True)
        global_df.to_csv(OUT_DIR / "global_mean_abs.csv", index=False)
        print(f"\nsaved → {OUT_DIR/'global_mean_abs.csv'}")

    (OUT_DIR / "surrogate_rules.txt").write_text("".join(rules_text))
    (OUT_DIR / "summary.txt").write_text("\n".join(summary_lines))
    print(f"saved → {OUT_DIR/'surrogate_rules.txt'}")
    print(f"saved → {OUT_DIR/'summary.txt'}")
    print(f"plots → {PLOTS_DIR}")


if __name__ == "__main__":
    main()
