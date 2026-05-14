"""Regenerate the utc2130 markdown reports from already-saved threshold_sweep
CSVs. Use after updating utc2130_runner._generate_markdown to add per-label
detailed tables — this lets us upgrade the markdown for variants whose
walk-forward already finished without rerunning the model.

Pass --include-running to also regenerate variants that are still being
written (only safe after the variant has finished its walk-forward and the
threshold_sweep CSVs have been written).
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import re
from pathlib import Path

import pandas as pd

from src.utc2130_runner import (
    REPORT_ROOT, PROB_THRESHOLDS, _generate_markdown,
)


# Map suffix → refit_calendar so the markdown header lists the right cadence.
SUFFIX_TO_CALENDAR = {
    "utc2130": "M",
    "utc2130_close": "M",
    "utc2130_complete": "M",
    "utc2130_sm": "SM",
    "utc2130_sm_close": "SM",
    "utc2130_sm_complete": "SM",
    "utc2130_wsun": "W-SUN",
    "utc2130_wsun_close": "W-SUN",
    "utc2130_wsun_complete": "W-SUN",
}

SUFFIX_TO_TITLE = {
    "utc2130": "",
    "utc2130_close": "offday-flag",
    "utc2130_complete": "ffill+bfill",
    "utc2130_sm": "SM 1st&16th",
    "utc2130_sm_close": "SM 1st&16th, offday-flag",
    "utc2130_sm_complete": "SM 1st&16th, ffill+bfill",
    "utc2130_wsun": "W-SUN weekly",
    "utc2130_wsun_close": "W-SUN weekly, offday-flag",
    "utc2130_wsun_complete": "W-SUN weekly, ffill+bfill",
}


_RX_STATIC = re.compile(r"^threshold_sweep_th(\d+)_(.+?)_(\d+)d\.csv$")
_RX_DYNAMIC = re.compile(r"^threshold_sweep_dynk(\d+)_(.+?)_(\d+)d\.csv$")


def _build_runs_from_dir(report_dir: Path, suffix: str, window_days: int):
    """Reconstruct the static_runs and dynamic_runs lists from CSVs."""
    suffix_w = f"{suffix}_{window_days}d"
    static_runs = []
    dynamic_runs = []

    for csv_path in sorted(report_dir.glob(f"threshold_sweep_th*_{suffix_w}.csv")):
        m = _RX_STATIC.match(csv_path.name)
        if not m:
            continue
        n_str, csv_suffix, w_str = m.group(1), m.group(2), m.group(3)
        if csv_suffix != suffix or int(w_str) != window_days:
            continue
        value = int(n_str) / 1000  # th15 → 0.015
        df = pd.read_csv(csv_path).sort_values("threshold").reset_index(drop=True)
        best_idx = int(df["total_return_pct"].idxmax())
        best = df.iloc[best_idx]
        # Try to read summary CSV for n_up / n_dn / ignore_pct / acc / prob_mean.
        summary_path = report_dir / f"label_threshold_sweep_summary_static_{suffix_w}.csv"
        meta = _meta_for_value(summary_path, value)
        slug = csv_path.stem.removeprefix("threshold_sweep_")
        static_runs.append({
            "value": value,
            "slug": slug,
            "best_prob_th": float(best["threshold"]),
            "best_return": float(best["total_return_pct"]),
            "best_equity": float(best["final_equity_usd"]),
            "best_trades": int(best["trades"]),
            "best_winrate": float(best["win_rate"]),
            "buy_hold": float(best["buy_and_hold_pct"]),
            "acc_at_05": meta.get("acc_at_05", float("nan")),
            "prob_mean": meta.get("prob_mean", float("nan")),
            "n_train_labels_up": int(meta.get("n_train_labels_up", -1)),
            "n_train_labels_dn": int(meta.get("n_train_labels_dn", -1)),
            "ignore_pct": float(meta.get("ignore_pct", float("nan"))),
        })

    for csv_path in sorted(report_dir.glob(f"threshold_sweep_dynk*_{suffix_w}.csv")):
        m = _RX_DYNAMIC.match(csv_path.name)
        if not m:
            continue
        n_str, csv_suffix, w_str = m.group(1), m.group(2), m.group(3)
        if csv_suffix != suffix or int(w_str) != window_days:
            continue
        value = int(n_str) / 10  # dynk15 → 1.5
        df = pd.read_csv(csv_path).sort_values("threshold").reset_index(drop=True)
        best_idx = int(df["total_return_pct"].idxmax())
        best = df.iloc[best_idx]
        summary_path = report_dir / f"label_threshold_sweep_summary_dynamic_{suffix_w}.csv"
        meta = _meta_for_value(summary_path, value)
        slug = csv_path.stem.removeprefix("threshold_sweep_")
        dynamic_runs.append({
            "value": value,
            "slug": slug,
            "best_prob_th": float(best["threshold"]),
            "best_return": float(best["total_return_pct"]),
            "best_equity": float(best["final_equity_usd"]),
            "best_trades": int(best["trades"]),
            "best_winrate": float(best["win_rate"]),
            "buy_hold": float(best["buy_and_hold_pct"]),
            "acc_at_05": meta.get("acc_at_05", float("nan")),
            "prob_mean": meta.get("prob_mean", float("nan")),
            "n_train_labels_up": int(meta.get("n_train_labels_up", -1)),
            "n_train_labels_dn": int(meta.get("n_train_labels_dn", -1)),
            "ignore_pct": float(meta.get("ignore_pct", float("nan"))),
        })

    return static_runs, dynamic_runs


def _meta_for_value(summary_path: Path, value: float) -> dict:
    if not summary_path.exists():
        return {}
    df = pd.read_csv(summary_path)
    # Try to find row by value; column name is 'value'.
    if "value" not in df.columns:
        return {}
    row = df[df["value"].round(6) == round(value, 6)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _build_matrix(runs: list[dict]) -> pd.DataFrame:
    """Re-derive label × prob_TH return matrix from the per-label CSVs."""
    if not runs:
        return pd.DataFrame()
    rows = {}
    # We need per-(label, prob) returns. Read each CSV again.
    base_dir = REPORT_ROOT  # noqa: F841 (handled by per-run csv path)
    return pd.DataFrame()  # placeholder — we use saved heatmap CSV instead


def _load_matrix(report_dir: Path, mode: str, suffix_w: str) -> pd.DataFrame:
    path = report_dir / f"label_x_prob_{mode}_heatmap_{suffix_w}.csv"
    if not path.exists():
        return pd.DataFrame()
    m = pd.read_csv(path, index_col=0)
    m.columns = [float(c) for c in m.columns]
    m.index = [float(i) for i in m.index]
    return m


def regen_one(report_dir: Path, suffix: str, window_days: int) -> bool:
    suffix_w = f"{suffix}_{window_days}d"
    static_runs, dynamic_runs = _build_runs_from_dir(report_dir, suffix, window_days)
    if not static_runs and not dynamic_runs:
        return False
    static_matrix = _load_matrix(report_dir, "static", suffix_w)
    dynamic_matrix = _load_matrix(report_dir, "dynamic", suffix_w)

    title_extra = f" — {SUFFIX_TO_TITLE.get(suffix, suffix)}" \
                   if SUFFIX_TO_TITLE.get(suffix) else ""
    refit_calendar = SUFFIX_TO_CALENDAR.get(suffix, "M")
    md_path = report_dir / f"label_prob_threshold_tables_{suffix_w}.md"

    _generate_markdown(static_runs, dynamic_runs,
                       static_matrix, dynamic_matrix,
                       md_path, window_days,
                       title_extra=title_extra,
                       report_dir=report_dir,
                       refit_calendar=refit_calendar)
    return True


def main():
    if not REPORT_ROOT.exists():
        print(f"no report root at {REPORT_ROOT}")
        return
    n_done = 0
    for suffix, _title in SUFFIX_TO_TITLE.items():
        for window_days in (730, 365):
            suffix_w = f"{suffix}_{window_days}d"
            report_dir = REPORT_ROOT / suffix_w
            if not report_dir.exists():
                continue
            ok = regen_one(report_dir, suffix, window_days)
            if ok:
                print(f"  regen {suffix_w} -> "
                      f"{report_dir / f'label_prob_threshold_tables_{suffix_w}.md'}")
                n_done += 1
            else:
                print(f"  skip  {suffix_w} (no threshold_sweep CSVs)")
    print(f"\nregenerated {n_done} markdown(s)")


if __name__ == "__main__":
    main()
