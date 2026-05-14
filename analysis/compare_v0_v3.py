"""Compare v0 (legacy synthetic MVRV proxy) vs v3 (improved + add-ons)
across the three retrain cadences (M, SM, WSUN), three NaN-handling
variants (raw, close, complete), and two backtest windows (730d, 365d).

Goes BEYOND top-1 return by computing consistency / robustness metrics:

  best_ret           top-1 prob_TH return per (label, window) pair
  mean_ret           mean across the 11 label thresholds (avg performance)
  median_ret         median across labels (robust to one outlier label)
  std_ret            std-dev across labels (lower = more consistent)
  min_ret            worst-label return (worst-case)
  pct_positive_cells % of (label × prob_TH) cells with positive return
  top3_mean_ret      avg return at top-3 prob_TH per label (robust to TH miss)
  cross_window_corr  correlation between 730d & 365d label×prob return matrices
                     (high = same label/prob choice generalises across periods)

Output:
  reports/compare_v0_v3.md   — markdown report
  reports/compare_v0_v3.csv  — tabular metrics (one row per variant×window×mode)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORT_ROOT = ROOT / "reports" / "utc2130"
OUT_DIR = ROOT / "reports"

CADENCES = ["SM", "M", "WSUN"]
NAN_VARIANTS = ["raw", "close", "complete"]
FEATURE_SETS = ["v0", "v3", "v4", "v5", "v6"]
WINDOWS = [730, 365]
MODES = ["static", "dynamic"]

# Map (cadence, nan_variant, feature_set) -> directory suffix used in reports/utc2130/
def _suffix(cadence: str, nan_variant: str, feature_set: str) -> str:
    """Reconstruct the suffix used by the runner.

    cadence: 'M' | 'SM' | 'WSUN'
    nan_variant: 'raw' | 'close' | 'complete'
    feature_set: 'v0' | 'v3' | 'v4'

    Examples:
      ('M',    'close',    'v0') -> 'utc2130_close'
      ('SM',   'raw',      'v0') -> 'utc2130_sm'
      ('WSUN', 'complete', 'v0') -> 'utc2130_wsun_complete'
      ('M',    'raw',      'v3') -> 'utc2130_v3'
      ('SM',   'close',    'v3') -> 'utc2130_sm_v3_close'
      ('M',    'raw',      'v4') -> 'utc2130_v4'
      ('SM',   'close',    'v4') -> 'utc2130_sm_v4_close'
    """
    parts = ["utc2130"]
    if cadence == "SM":
        parts.append("sm")
    elif cadence == "WSUN":
        parts.append("wsun")
    if feature_set in ("v3", "v4", "v5", "v6"):
        parts.append(feature_set)
    if nan_variant != "raw":
        parts.append(nan_variant)
    return "_".join(parts)


def _load_window_matrix(suffix: str, window: int, mode: str) -> pd.DataFrame:
    """Return label × prob_TH return matrix for one variant×window×mode."""
    suffix_w = f"{suffix}_{window}d"
    p = REPORT_ROOT / suffix_w / f"label_x_prob_{mode}_heatmap_{suffix_w}.csv"
    if not p.exists():
        return pd.DataFrame()
    m = pd.read_csv(p, index_col=0)
    m.columns = [float(c) for c in m.columns]
    m.index = [float(i) for i in m.index]
    return m


def _consistency_metrics(matrix: pd.DataFrame) -> dict:
    """Compute summary stats over a label × prob_TH return matrix."""
    if matrix.empty:
        return {k: float("nan") for k in (
            "best_ret", "mean_ret", "median_ret", "std_ret",
            "min_ret", "max_ret", "pct_positive_cells",
            "top3_mean_ret", "best_label", "best_prob")}

    arr = matrix.to_numpy(dtype=float)
    # best per-label: take max across prob axis
    best_per_label = np.nanmax(arr, axis=1)
    top3_per_label = np.sort(arr, axis=1)[:, -3:].mean(axis=1)

    flat = arr[~np.isnan(arr)]
    pct_pos = float((flat > 0).mean()) if flat.size else float("nan")

    overall_best_idx = np.unravel_index(np.nanargmax(arr), arr.shape)
    best_label = matrix.index[overall_best_idx[0]]
    best_prob = matrix.columns[overall_best_idx[1]]

    return {
        "best_ret": float(np.nanmax(arr)),
        "mean_ret": float(np.nanmean(best_per_label)),
        "median_ret": float(np.nanmedian(best_per_label)),
        "std_ret": float(np.nanstd(best_per_label)),
        "min_ret": float(np.nanmin(best_per_label)),
        "max_ret": float(np.nanmax(best_per_label)),
        "pct_positive_cells": pct_pos,
        "top3_mean_ret": float(np.nanmean(top3_per_label)),
        "best_label": float(best_label),
        "best_prob": float(best_prob),
    }


def _cross_window_corr(suffix: str, mode: str) -> float:
    m_730 = _load_window_matrix(suffix, 730, mode)
    m_365 = _load_window_matrix(suffix, 365, mode)
    if m_730.empty or m_365.empty:
        return float("nan")
    a = m_730.values.flatten()
    b = m_365.values.flatten()
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _gather_all() -> pd.DataFrame:
    rows = []
    for cadence in CADENCES:
        for nan_v in NAN_VARIANTS:
            for fs in FEATURE_SETS:
                suffix = _suffix(cadence, nan_v, fs)
                report_dir = REPORT_ROOT / f"{suffix}_730d"
                if not report_dir.exists():
                    continue
                for window in WINDOWS:
                    for mode in MODES:
                        matrix = _load_window_matrix(suffix, window, mode)
                        if matrix.empty:
                            continue
                        m = _consistency_metrics(matrix)
                        m.update({
                            "cadence": cadence, "nan_variant": nan_v,
                            "feature_set": fs, "window": window,
                            "label_mode": mode, "suffix": suffix,
                            "cross_window_corr": _cross_window_corr(suffix, mode)
                                if window == 730 else float("nan"),
                        })
                        rows.append(m)
    return pd.DataFrame(rows)


def _fmt_pct(x: float, sign: bool = True) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:+.2f}%" if sign else f"{x*100:.2f}%"


def _section_table(df: pd.DataFrame, window: int, mode: str) -> list[str]:
    sub = df[(df["window"] == window) & (df["label_mode"] == mode)].copy()
    if sub.empty:
        return [f"  (no data for window={window} mode={mode})", ""]
    # rank by mean_ret (consistency-first metric)
    sub = sub.sort_values("mean_ret", ascending=False).reset_index(drop=True)
    out = []
    out.append(f"### {mode.upper()} labels — {window}-day window  "
               f"(ranked by mean return across labels)")
    out.append("")
    out.append("| variant | best | mean | median | std | min | top3-mean | %pos |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in sub.iterrows():
        variant_id = f"{r['cadence']} · {r['nan_variant']} · {r['feature_set']}"
        out.append(
            f"| {variant_id} "
            f"| {_fmt_pct(r['best_ret'])} "
            f"| **{_fmt_pct(r['mean_ret'])}** "
            f"| {_fmt_pct(r['median_ret'])} "
            f"| {_fmt_pct(r['std_ret'], sign=False)} "
            f"| {_fmt_pct(r['min_ret'])} "
            f"| {_fmt_pct(r['top3_mean_ret'])} "
            f"| {_fmt_pct(r['pct_positive_cells'], sign=False)} |"
        )
    out.append("")
    return out


def _v0_v3_delta(df: pd.DataFrame, window: int, mode: str) -> list[str]:
    sub = df[(df["window"] == window) & (df["label_mode"] == mode)]
    if sub.empty:
        return []
    pivot_metrics = ["best_ret", "mean_ret", "min_ret", "pct_positive_cells", "top3_mean_ret"]
    out = []
    out.append(f"### v3 − v0 delta — {mode.upper()} labels — {window}-day window")
    out.append("")
    out.append("| (cadence · nan) | Δ best | Δ mean | Δ min | Δ %pos | Δ top3-mean |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for cadence in CADENCES:
        for nan_v in NAN_VARIANTS:
            v0 = sub[(sub["cadence"] == cadence) & (sub["nan_variant"] == nan_v)
                     & (sub["feature_set"] == "v0")]
            v3 = sub[(sub["cadence"] == cadence) & (sub["nan_variant"] == nan_v)
                     & (sub["feature_set"] == "v3")]
            if v0.empty or v3.empty:
                continue
            r0 = v0.iloc[0]; r3 = v3.iloc[0]
            row = f"| {cadence} · {nan_v} "
            for m in pivot_metrics:
                d = r3[m] - r0[m]
                if m == "pct_positive_cells":
                    row += f"| {d*100:+.1f}p "
                else:
                    row += f"| {d*100:+.2f}p "
            row += "|"
            out.append(row)
    out.append("")
    return out


def _cross_window_section(df: pd.DataFrame, mode: str) -> list[str]:
    """Cross-window consistency = correlation between 730d and 365d return matrices."""
    sub = df[(df["window"] == 730) & (df["label_mode"] == mode)
             & df["cross_window_corr"].notna()].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("cross_window_corr", ascending=False)
    out = []
    out.append(f"### Cross-window robustness ({mode}) — corr(730d, 365d) "
               f"of label×prob return matrices")
    out.append("")
    out.append("Higher correlation = strategy choice that wins in 730d also "
               "wins in 365d (regime-stable).")
    out.append("")
    out.append("| variant | corr(730d, 365d) |")
    out.append("|---|---:|")
    for _, r in sub.iterrows():
        variant_id = f"{r['cadence']} · {r['nan_variant']} · {r['feature_set']}"
        out.append(f"| {variant_id} | {r['cross_window_corr']:+.3f} |")
    out.append("")
    return out


def _generate_md(df: pd.DataFrame, out_path: Path) -> None:
    lines = []
    lines.append("# UTC2130 — v0 vs v3 Comparison Report")
    lines.append("")
    lines.append(f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- **Decision time**: 21:30 UTC daily (= 06:30 KST next day, post macro release)")
    lines.append("- **External data lag**: 0 days (macro/funding/fng all published by 21:30 UTC)")
    lines.append("- **Coin Metrics**: lag = 1 day (community publish delay)")
    lines.append("- **Training start**: 2019-12-01")
    lines.append("- **Backtest windows**: 730d (2024-05+) & 365d (2025-05+)")
    lines.append("")
    lines.append("## Variant grid")
    lines.append("")
    lines.append("- **Cadence**: M (1st of month) / SM (1st & 16th) / WSUN (every Sunday)")
    lines.append("- **NaN handling**: raw / close (offday-flag) / complete (ffill+bfill)")
    lines.append("- **Feature set**:")
    lines.append("  - **v0**: legacy 365d-MA MVRV proxy (`d_mvrv_z`, `d_mvrv_z_chg7`)")
    lines.append("  - **v3** (= v2 + add-ons): 6 improved MVRV-Z features (`mvrv2_*`) "
                 "+ 12 add-ons (cross-asset corr, funding skew/velocity, hashrate "
                 "ribbon, address momentum, vol regime) − 8 redundant cols dropped")
    lines.append("- **Labels**: 11 static thresholds (1.0–2.0%) + 11 dynamic k values "
                 "(τ = k · σ_30d, k=1.0–2.0)")
    lines.append("- **Probability sweep**: 21 prob_TH values 0.50–0.70")
    lines.append("")
    lines.append("## Consistency metrics")
    lines.append("")
    lines.append("- **best**: top-1 return across all (label, prob_TH) pairs")
    lines.append("- **mean**: mean of best-prob_TH return across the 11 labels — "
                 "average performance, primary consistency metric")
    lines.append("- **median**: median across labels (robust to 1 outlier label)")
    lines.append("- **std**: std-dev across labels (lower = more uniform)")
    lines.append("- **min**: worst-label return (defensive sanity)")
    lines.append("- **top3-mean**: avg of top-3 prob_TH per label "
                 "(robust to slight TH choice)")
    lines.append("- **%pos**: % of (label × prob_TH) cells with positive return")
    lines.append("- **cross-window corr**: corr between 730d and 365d return matrices "
                 "(high = regime-stable choice)")
    lines.append("")

    for window in WINDOWS:
        lines.append(f"## {window}-day window")
        lines.append("")
        for mode in MODES:
            lines.extend(_section_table(df, window, mode))
        for mode in MODES:
            lines.extend(_v0_v3_delta(df, window, mode))

    lines.append("## Cross-window robustness")
    lines.append("")
    for mode in MODES:
        lines.extend(_cross_window_section(df, mode))

    # Champion sections — best consistency-first picks
    lines.append("## Champions")
    lines.append("")
    for window in WINDOWS:
        lines.append(f"### {window}-day window")
        for mode in MODES:
            sub = df[(df["window"] == window) & (df["label_mode"] == mode)]
            if sub.empty:
                continue
            best = sub.sort_values("mean_ret", ascending=False).iloc[0]
            lines.append(f"  - **{mode.upper()}** consistency-first ({sub.iloc[0]['label_mode']}): "
                         f"{best['cadence']} · {best['nan_variant']} · {best['feature_set']} "
                         f"→ mean {_fmt_pct(best['mean_ret'])}, "
                         f"best {_fmt_pct(best['best_ret'])}, "
                         f"min {_fmt_pct(best['min_ret'])}")
        lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> None:
    df = _gather_all()
    if df.empty:
        print("No variant directories found under reports/utc2130 — nothing to report.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "compare_v0_v3.csv"
    df.to_csv(csv_path, index=False)
    md_path = OUT_DIR / "compare_v0_v3.md"
    _generate_md(df, md_path)

    print(f"Loaded {len(df)} variant rows.")
    print(f"  CSV: {csv_path}")
    print(f"  MD : {md_path}")


if __name__ == "__main__":
    main()
