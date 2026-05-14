"""On-chain proxy / real-MVRV feature analysis.

Addresses paper reviewer critique #3:
  > "유료 API 한계로 365일 종가 이동평균을 활용한 proxy 지표를 사용했음.
     이는 본질적으로 가격 데이터의 파생 변수이므로 진정한 의미의 on-chain
     수급 데이터가 주는 직교성(orthogonality)을 잃을 수 있음."

Key clarification we need to surface in the paper:
  - **v0 feature set**  uses the price-based proxy: d_mvrv_z, d_mvrv_z_chg7
  - **v3 / v4 / v5 / v6 feature sets** REPLACE these with REAL on-chain
    MVRV variants from Coin Metrics: mvrv2_z_200w, mvrv2_z_ema900,
    mvrv2_z_multi, mvrv2_mayer, mvrv2_nvt_z, mvrv2_hashrate_z

So the production winner (Top1 = utc2130_sm_v5, uses v5=v4 features) actually
relies on REAL on-chain MVRV — not the proxy. The proxy critique applies
specifically to v0-only cells (utc2130_sm = our Top3 component).

Outputs (under reports/proxy_analysis/):
  proxy_importance_summary.csv  : feature importance ranks for proxy + real
  proxy_correlation.csv         : Pearson |corr| between proxy/real and TA
  proxy_summary.txt             : narrative summary for the paper
  proxy_vif.json                : VIF stats
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "reports" / "proxy_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# v0 (legacy) — price-based proxy
PROXY_FEATURES = ["d_mvrv_z", "d_mvrv_z_chg7"]

# v3+ — real on-chain MVRV / NVT / hashrate from Coin Metrics
REAL_ONCHAIN_FEATURES = ["mvrv2_z_200w", "mvrv2_z_ema900", "mvrv2_z_multi",
                          "mvrv2_mayer", "mvrv2_nvt_z", "mvrv2_hashrate_z"]

ALL_ONCHAIN = PROXY_FEATURES + REAL_ONCHAIN_FEATURES

TOP_CELLS = ["utc2130_sm_v5", "utc2130_sm_v3_complete", "utc2130_sm",
             "utc2130_sm_v3", "utc2130_sm_v6"]


def cell_feature_set(cell: str) -> str:
    """Map cell suffix to its feature_set tag."""
    if "_v5" in cell:
        return "v5"
    if "_v6" in cell:
        return "v6_eq_v4"  # v6 = v4 features + tighter reg
    if "_v4" in cell:
        return "v4"
    if "_v3" in cell:
        return "v3"
    return "v0"


def load_feature_importance() -> pd.DataFrame:
    rows = []
    for cell in TOP_CELLS:
        fs = cell_feature_set(cell)
        rep_dir = ROOT / "reports" / "utc2130" / f"{cell}_730d"
        fi_files = sorted(glob.glob(str(rep_dir / "feature_importance_ebm_*.csv")))
        for f in fi_files:
            df = pd.read_csv(f)
            n = len(df)
            df["rank"] = df["gain"].rank(ascending=False, method="min").astype(int)
            label = os.path.basename(f).split("_")[3]
            for feat in ALL_ONCHAIN:
                sub = df.loc[df["feature"] == feat]
                if len(sub):
                    r = sub.iloc[0]
                    rows.append({"cell": cell, "feature_set": fs, "label": label,
                                 "feature": feat,
                                 "kind": "proxy" if feat in PROXY_FEATURES else "real_onchain",
                                 "gain": float(r["gain"]),
                                 "rank": int(r["rank"]), "n_features": n,
                                 "rank_pct": float(r["rank"]) / n * 100})
    return pd.DataFrame(rows)


def correlation_with_TA() -> pd.DataFrame:
    """Compute |corr| of each proxy/real feature against every TA-block feature."""
    from src.utc2130 import build_utc2130_daily
    from src.features import build_features

    bar = build_utc2130_daily(ROOT / "data" / "btc_15m.parquet")

    rows = []
    for feature_set, kwargs in [("v0", {}),
                                  ("v3", {"use_features_v3": True})]:
        X, _, _ = build_features(ROOT / "data", label_threshold=0.015,
                                  daily_df=bar, external_lag_days=0, **kwargs)
        targets = [c for c in ALL_ONCHAIN if c in X.columns]
        if not targets:
            continue
        ta_cols = [c for c in X.columns
                   if c not in ALL_ONCHAIN
                   and c.startswith(("d_", "w_", "M_", "sw_"))]
        for tgt in targets:
            for col in ta_cols:
                if col == tgt:
                    continue
                r = X[[tgt, col]].dropna()
                if len(r) < 50:
                    continue
                corr = float(r[tgt].corr(r[col]))
                if not np.isfinite(corr):
                    continue
                rows.append({"feature_set": feature_set, "subject": tgt,
                             "kind": "proxy" if tgt in PROXY_FEATURES else "real_onchain",
                             "ta_feature": col,
                             "corr": corr, "abs_corr": abs(corr),
                             "n_obs": len(r)})
    return pd.DataFrame(rows).sort_values(
        ["feature_set", "subject", "abs_corr"], ascending=[True, True, False])


def vif_analysis() -> dict:
    from src.utc2130 import build_utc2130_daily
    from src.features import build_features
    from sklearn.linear_model import LinearRegression

    bar = build_utc2130_daily(ROOT / "data" / "btc_15m.parquet")
    out = {"v0": {}, "v3": {}}
    for feature_set, kwargs in [("v0", {}),
                                  ("v3", {"use_features_v3": True})]:
        X, _, _ = build_features(ROOT / "data", label_threshold=0.015,
                                  daily_df=bar, external_lag_days=0, **kwargs)
        ta_cols = [c for c in X.columns
                   if c not in ALL_ONCHAIN
                   and c.startswith(("d_", "w_", "M_", "sw_"))]
        for feat in ALL_ONCHAIN:
            if feat not in X.columns:
                continue
            sub = X[[feat] + ta_cols].dropna()
            if len(sub) < 100:
                out[feature_set][feat] = {"r2": None, "vif": None,
                                            "n_obs": len(sub),
                                            "predictors": len(ta_cols)}
                continue
            y = sub[feat].values
            Xm = sub[ta_cols].values
            reg = LinearRegression().fit(Xm, y)
            r2 = reg.score(Xm, y)
            vif = 1.0 / (1.0 - r2) if r2 < 0.999 else float("inf")
            out[feature_set][feat] = {"r2": float(r2), "vif": float(vif),
                                        "predictors": len(ta_cols),
                                        "n_obs": int(len(sub))}
    return out


def main() -> None:
    print("[1/3] feature importance summary ...", flush=True)
    fi = load_feature_importance()
    fi.to_csv(OUT_DIR / "proxy_importance_summary.csv", index=False)
    print(f"  rows: {len(fi)}")
    if len(fi):
        for kind in ("proxy", "real_onchain"):
            sub = fi.loc[fi["kind"] == kind]
            if len(sub):
                print(f"  {kind}: {len(sub)} (cell, label, feat) entries; "
                      f"mean rank {sub['rank'].mean():.1f}/"
                      f"{sub['n_features'].mean():.0f}")

    print("\n[2/3] correlations with TA blocks ...", flush=True)
    corr = correlation_with_TA()
    corr.to_csv(OUT_DIR / "proxy_correlation.csv", index=False)
    print(f"  rows: {len(corr)}")
    for fset in ("v0", "v3"):
        for kind in ("proxy", "real_onchain"):
            sub = corr.loc[(corr["feature_set"] == fset) & (corr["kind"] == kind)]
            if len(sub) == 0:
                continue
            print(f"\n  {fset} / {kind}: top |corr|")
            top = sub.groupby("subject", group_keys=False).head(5)
            for _, r in top.iterrows():
                print(f"    {r['subject']:<22s} ↔ {r['ta_feature']:<24s} "
                      f"{r['corr']:+.3f}")

    print("\n[3/3] VIF ...", flush=True)
    vif = vif_analysis()
    (OUT_DIR / "proxy_vif.json").write_text(json.dumps(vif, indent=2))
    for fset, d in vif.items():
        if not d:
            continue
        print(f"\n  feature_set = {fset}:")
        for feat, stats in d.items():
            if stats.get("r2") is None:
                print(f"    {feat:<22s} n/a")
                continue
            severity = ("severe" if stats["vif"] > 10
                        else "problematic" if stats["vif"] > 5
                        else "acceptable")
            print(f"    {feat:<22s} R²={stats['r2']:.4f}  "
                  f"VIF={stats['vif']:.2f}  ({severity})")

    # Build narrative
    lines = []
    lines.append("# On-chain proxy vs real on-chain feature analysis")
    lines.append("")
    lines.append("## Critique addressed")
    lines.append("Reviewer noted that v0 uses a price-based MVRV-Z proxy "
                 "(365-day moving average of close as 'realised price' approx)")
    lines.append("and asked: does this proxy provide information beyond TA blocks?")
    lines.append("")
    lines.append("## Resolution at the design level")
    lines.append("**v3+ feature sets already replace the price-based proxy with "
                 "REAL on-chain Coin Metrics MVRV-Z variants** "
                 "(mvrv2_z_200w, mvrv2_z_ema900, mvrv2_z_multi, mvrv2_mayer, "
                 "mvrv2_nvt_z, mvrv2_hashrate_z). The 4 production ensembles "
                 "(Top1/2/3/5) primarily use v5=v4=v3+ feature sets with REAL "
                 "on-chain data. Only the v0 sub-cell (utc2130_sm in Top3/Top5) "
                 "still relies on the proxy.")
    lines.append("")
    lines.append("## Quantitative findings")
    lines.append("")
    lines.append("### Feature importance (top-N per cell × label fits)")
    if len(fi):
        for kind in ("proxy", "real_onchain"):
            sub = fi.loc[fi["kind"] == kind]
            if len(sub):
                lines.append(f"  {kind}: appears in {len(sub)} top-N dumps "
                             f"(mean rank {sub['rank'].mean():.1f}/"
                             f"{sub['n_features'].mean():.0f})")
                feat_counts = sub.groupby("feature").size().sort_values(ascending=False)
                for f, n in feat_counts.items():
                    sub_f = sub.loc[sub["feature"] == f]
                    lines.append(f"    {f:<22s}  in {n} fits, "
                                 f"mean rank {sub_f['rank'].mean():.1f}")
    lines.append("")
    lines.append("### VIF (predict each on-chain feature from TA blocks A/B)")
    for fset in ("v0", "v3"):
        if not vif[fset]:
            continue
        lines.append(f"  feature_set = {fset}:")
        for feat, stats in vif[fset].items():
            if stats.get("r2") is None:
                continue
            severity = ("severe" if stats["vif"] > 10
                        else "problematic" if stats["vif"] > 5
                        else "acceptable")
            lines.append(f"    {feat:<22s}  R²={stats['r2']:.4f}, "
                         f"VIF={stats['vif']:.2f} ({severity})")
    lines.append("")
    lines.append("### Top-3 |corr| with TA features")
    for fset in ("v0", "v3"):
        for kind in ("proxy", "real_onchain"):
            sub = corr.loc[(corr["feature_set"] == fset) & (corr["kind"] == kind)]
            if len(sub) == 0:
                continue
            lines.append(f"  {fset} / {kind}:")
            top = sub.groupby("subject", group_keys=False).head(3)
            for _, r in top.iterrows():
                lines.append(f"    {r['subject']:<22s} ↔ "
                             f"{r['ta_feature']:<24s} {r['corr']:+.3f}")
    lines.append("")
    (OUT_DIR / "proxy_summary.txt").write_text("\n".join(lines))
    print(f"\nsaved → {OUT_DIR/'proxy_summary.txt'}")


if __name__ == "__main__":
    main()
