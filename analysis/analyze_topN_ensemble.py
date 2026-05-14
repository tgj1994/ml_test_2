"""Top-N SPLIT-capital ensembles within UTC groups + combined.

For each (group, N) configuration:
  1. From fwer_summary.csv, pick the top-N cells by 730d PSR (within that
     group's 730d sub-table).
  2. Each cell trades its own (label, prob_TH) on 1/N capital.
  3. Compute ensemble metrics on BOTH 730d and 365d holdouts (the same N
     cells, no re-selection per window).

Groups:
  utc2130_only        : 33 cells
  utc0000_only        : 30 cells
  combined            : 63 cells
  utc2130_static_only : cells whose best label is 'th*' on 730d
  utc2130_dynamic_only: cells whose best label is 'dynk*' on 730d
  utc0000_static_only / utc0000_dynamic_only

Selection is on 730d PSR; evaluation uses BOTH windows. (730d is in-sample
to selection — that's known. The 365d evaluation is the more honest "does
the ensemble hold up on a different window with the same component picks".)

Outputs (under reports/dsr_fwer/):
  - topN_ensemble_summary.csv
  - topN_ensemble_components.csv     # which cells went into each config
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import run_backtest
from src.utc0000 import build_utc0000_daily
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm"
OUT_DIR = ROOT / "reports" / "dsr_fwer"

STAKE_USD = 100.0
TRADING_DAYS = 365.0
N_VALUES = (2, 3, 5, 8, 10)


def _kind(cell: str) -> str:
    return "utc2130" if "utc2130" in cell else "utc0000"


def _label_kind(label: str) -> str:
    return "static" if label.startswith("th") else "dynamic"


def _drawdown(eq: pd.Series):
    cummax = eq.cummax()
    dd = eq / cummax - 1
    trough = dd.idxmin()
    max_dd = float(dd.min())
    peak_date = eq.loc[:trough].idxmax()
    after = eq.loc[trough:]
    peak_value = eq.loc[peak_date]
    recovered = after[after >= peak_value]
    end_date = recovered.index[0] if len(recovered) else after.index[-1]
    return max_dd, (end_date - peak_date).days


def _longest_inactive(eq: pd.Series) -> int:
    flat = (eq.diff().abs() < 1e-9).astype(int)
    runs = []; cur = 0
    for f in flat.values:
        if f: cur += 1
        else:
            if cur: runs.append(cur); cur = 0
    if cur: runs.append(cur)
    return max(runs) if runs else 0


def _sharpe_psr(rets: np.ndarray) -> dict:
    n = len(rets)
    if n < 5 or rets.std(ddof=1) == 0:
        return dict(sharpe=float("nan"), psr=float("nan"))
    mu = rets.mean(); sd = rets.std(ddof=1)
    sr_pp = mu / sd
    skew = float(stats.skew(rets, bias=False))
    kurt = float(stats.kurtosis(rets, fisher=False, bias=False))
    sr_var = (1 - skew * sr_pp + (kurt - 1) / 4 * sr_pp ** 2) / (n - 1)
    psr = float(stats.norm.cdf(sr_pp / math.sqrt(sr_var))) \
        if sr_var > 0 else float("nan")
    return dict(sharpe=sr_pp * math.sqrt(TRADING_DAYS), psr=psr)


def _quarterly(eq: pd.Series, bh: pd.Series) -> dict:
    eq_q = eq.resample("QE").last()
    bh_q = bh.resample("QE").last()
    s_ret = eq_q.pct_change().dropna() * 100
    b_ret = bh_q.pct_change().dropna() * 100
    s_ret, b_ret = s_ret.align(b_ret, join="inner")
    out = (s_ret - b_ret)
    return dict(
        n_quarters=len(out),
        n_outperform=int((out > 0).sum()),
        pct_outperform=float((out > 0).mean() * 100) if len(out) else 0.0,
        median_pp=float(out.median()) if len(out) else float("nan"),
        worst_pp=float(out.min()) if len(out) else float("nan"),
        best_pp=float(out.max()) if len(out) else float("nan"),
    )


def split_capital_ensemble(picks: list[dict], window_days: int,
                           closes: dict[str, pd.Series]) -> dict:
    """SPLIT-capital ensemble. picks = list of dicts with cell, label,
    prob_TH. Each runs on stake/N capital."""
    N = len(picks)
    sub_eq: list[pd.Series] = []
    sub_trades = 0
    sub_wins = 0
    bh_curve = None
    for pk in picks:
        cell = pk["cell"]; label = pk["label"]; prob = pk["prob_TH"]
        kind = _kind(cell)
        cs = closes[kind]
        cache_path = CACHE_DIR / f"preds_ebm_{label}_{cell}.parquet"
        if not cache_path.exists():
            return {}
        preds = pd.read_parquet(cache_path)
        cutoff = preds.index.max() - pd.Timedelta(days=window_days)
        p = preds.dropna(subset=["prob_up"]).loc[lambda d: d.index >= cutoff]
        if len(p) < 30:
            return {}
        cl = cs.reindex(p.index)
        res = run_backtest(p, cl, stake_usd=STAKE_USD / N,
                           up_threshold=prob, down_threshold=0.50)
        sub_eq.append(res.equity_curve)
        sub_trades += res.n_trades
        sub_wins += sum(1 for t in res.trades if t.pnl_usd > 0)
        if bh_curve is None:
            bh_curve = res.buy_and_hold_curve

    # align & sum
    aligned = sub_eq[0]
    for s in sub_eq[1:]:
        aligned, s2 = aligned.align(s, join="inner")
        aligned = aligned + s2
    eq = aligned
    bh = bh_curve.reindex(eq.index)
    if eq.empty:
        return {}
    final_eq = float(eq.iloc[-1])
    drets = eq.pct_change().dropna().to_numpy()
    sp = _sharpe_psr(drets)
    max_dd, dd_dur = _drawdown(eq)
    lng_flat = _longest_inactive(eq)
    bh_ret = float(bh.iloc[-1] / bh.iloc[0]) - 1
    qb = _quarterly(eq, bh)
    return dict(
        N=N,
        total_return_pct=(final_eq / STAKE_USD - 1) * 100,
        bh_pct=bh_ret * 100,
        outperform_pp=((final_eq / STAKE_USD - 1) - bh_ret) * 100,
        n_trades=sub_trades, win_rate=(sub_wins / sub_trades * 100) if sub_trades else 0.0,
        max_dd_pct=max_dd * 100, max_dd_dur_days=dd_dur,
        longest_inactive_days=lng_flat,
        sharpe=sp["sharpe"], psr=sp["psr"],
        **{f"q_{k}": v for k, v in qb.items()},
    )


def main():
    print("[1/4] loading PSR table + closes ...", flush=True)
    fwer = pd.read_csv(OUT_DIR / "fwer_summary.csv")
    bar0 = build_utc0000_daily(DATA_DIR / "btc_15m.parquet")
    bar21 = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    closes = {
        "utc0000": bar0.set_index("close_time")["close"],
        "utc2130": bar21.set_index("close_time")["close"],
    }
    fwer["kind"] = fwer["cell"].apply(_kind)
    fwer["label_kind"] = fwer["best_label"].apply(_label_kind)

    f730 = fwer.loc[fwer["window_days"] == 730].copy()
    f730 = f730.sort_values("psr", ascending=False)

    # Build ensemble configurations
    configs: list[tuple[str, pd.DataFrame]] = []
    for n in N_VALUES:
        configs.append((f"utc2130_top{n}",
                        f730.loc[f730["kind"] == "utc2130"].head(n)))
        configs.append((f"utc0000_top{n}",
                        f730.loc[f730["kind"] == "utc0000"].head(n)))
        configs.append((f"combined_top{n}",
                        f730.head(n)))
        # static-only
        configs.append((f"utc2130_static_top{n}",
                        f730.loc[(f730["kind"] == "utc2130")
                                 & (f730["label_kind"] == "static")].head(n)))
        configs.append((f"utc0000_static_top{n}",
                        f730.loc[(f730["kind"] == "utc0000")
                                 & (f730["label_kind"] == "static")].head(n)))
        # dynamic-only
        configs.append((f"utc2130_dynamic_top{n}",
                        f730.loc[(f730["kind"] == "utc2130")
                                 & (f730["label_kind"] == "dynamic")].head(n)))
        configs.append((f"utc0000_dynamic_top{n}",
                        f730.loc[(f730["kind"] == "utc0000")
                                 & (f730["label_kind"] == "dynamic")].head(n)))

    # Add the manual 2-cell baseline (sm_v5 + sm) for reference
    manual = pd.DataFrame([
        {"cell": "utc2130_sm_v5", "best_label": "dynk12", "best_prob_TH": 0.63,
         "psr": 0.973},
        {"cell": "utc2130_sm",    "best_label": "dynk11", "best_prob_TH": 0.64,
         "psr": 0.952},
    ])
    configs.append(("utc2130_manual_v5_sm", manual))

    print(f"[2/4] running {len(configs)} ensemble configs × 2 windows ...", flush=True)
    summary_rows = []
    component_rows = []
    for name, df in configs:
        if len(df) == 0:
            continue
        picks = [{"cell": r["cell"], "label": r["best_label"],
                  "prob_TH": r["best_prob_TH"]} for _, r in df.iterrows()]
        for window in (730, 365):
            metrics = split_capital_ensemble(picks, window, closes)
            if not metrics:
                continue
            summary_rows.append({"config": name, "window_days": window,
                                 **metrics})
        for _, r in df.iterrows():
            component_rows.append({"config": name, "cell": r["cell"],
                                   "label": r["best_label"],
                                   "prob_TH": r["best_prob_TH"],
                                   "cell_psr_730d": r["psr"]})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "topN_ensemble_summary.csv", index=False)
    pd.DataFrame(component_rows).to_csv(
        OUT_DIR / "topN_ensemble_components.csv", index=False)
    print(f"  summary → {OUT_DIR/'topN_ensemble_summary.csv'} ({len(summary)} rows)")

    # Print clean tables
    print("[3/4] summary tables ...\n")
    cols = ["config", "N", "total_return_pct", "bh_pct", "outperform_pp",
            "n_trades", "win_rate", "max_dd_pct", "max_dd_dur_days",
            "longest_inactive_days", "sharpe", "psr",
            "q_pct_outperform", "q_worst_pp", "q_median_pp"]

    # Group results by 730d window first
    for w in (730, 365):
        s = summary.loc[summary["window_days"] == w].copy()
        # Highlight the manual + groups of interest
        with pd.option_context("display.max_columns", None,
                               "display.width", 240,
                               "display.float_format", "{:.2f}".format):
            print(f"=== window = {w}d ===")
            for prefix in ["utc2130_top", "utc0000_top", "combined_top",
                           "utc2130_static_top", "utc0000_static_top",
                           "utc2130_dynamic_top", "utc0000_dynamic_top",
                           "utc2130_manual"]:
                sub = s.loc[s["config"].str.startswith(prefix)]
                if len(sub):
                    print(sub[cols].to_string(index=False))
                    print()


if __name__ == "__main__":
    main()
