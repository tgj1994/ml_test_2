"""DSR + PSR + MaxDD analysis on extended-window caches, parameterized by tag.

Computes for each cache (--cache-tag), each window (1095, 1460), each cell:
  * the BEST (label, prob_TH) trajectory by total_return_pct
  * its annualised Sharpe, PSR (Bailey-Lopez de Prado), and DSR with
    intra-cell N = #labels × #prob_TH = 462 trials
  * MaxDD %, drawdown duration

Plus FWER (Bonferroni / Holm) and BH-FDR across the per-cell selections.
Plus PSR/DSR/MaxDD for the 4 ensembles (top1/top2/top3/top5) at the
service-config (label_slug, prob_TH) split-capital combination.

Output dir mirrors analyze_pbo_extended_v2.py:
  - 'base' → reports/utc2130_extended/
  - else   → reports/utc2130_extended_{tag}/

Run:
    uv run python analyze_dsr_extended.py --cache-tag train1y
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import run_backtest
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

WINDOWS = (365, 730, 1095, 1460)  # 1y, 2y, 3y, 4y
PROB_THRESHOLDS = [round(0.50 + i * 0.01, 2) for i in range(21)]
TRADING_DAYS = 365.0
EULER_GAMMA = 0.5772156649015329
GAMMA_E = math.e
STAKE_USD = 100.0


CELLS = [
    "utc2130_sm_v5",
    "utc2130_sm_v3_complete",
    "utc2130_sm",
    "utc2130_sm_v3",
    "utc2130_sm_v6",
]


@dataclass
class CellSpec:
    cell_name: str
    label_slug: str
    prob_TH: float


SERVICE_CELLS: list[CellSpec] = [
    CellSpec("utc2130_sm_v5",          "dynk12", 0.63),
    CellSpec("utc2130_sm_v3_complete", "th13",   0.68),
    CellSpec("utc2130_sm",             "dynk11", 0.64),
    CellSpec("utc2130_sm_v3",          "th14",   0.66),
    CellSpec("utc2130_sm_v6",          "dynk14", 0.66),
]


ENSEMBLES = {
    "top1": ["utc2130_sm_v5/dynk12"],
    "top2": ["utc2130_sm_v5/dynk12", "utc2130_sm_v3_complete/th13"],
    "top3": ["utc2130_sm_v5/dynk12", "utc2130_sm_v3_complete/th13",
             "utc2130_sm/dynk11"],
    "top5": ["utc2130_sm_v5/dynk12", "utc2130_sm_v3_complete/th13",
             "utc2130_sm/dynk11", "utc2130_sm_v3/th14",
             "utc2130_sm_v6/dynk14"],
}


def _cell_key(c: CellSpec) -> str:
    return f"{c.cell_name}/{c.label_slug}"


def parse_cache(filename: str, cell: str) -> str | None:
    m = re.match(rf"preds_ebm_(th\d+|dynk\d+)_{re.escape(cell)}\.parquet$",
                 filename)
    if not m:
        return None
    return m.group(1)


def daily_strategy_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe_components(rets: np.ndarray) -> dict:
    n = len(rets)
    if n < 5 or rets.std(ddof=1) == 0:
        return dict(sharpe=float("nan"), sharpe_per_period=float("nan"),
                    skew=float("nan"), kurt=float("nan"), n=n,
                    sr_var=float("nan"), t_stat=float("nan"),
                    p_value=float("nan"))
    mu = rets.mean(); sd = rets.std(ddof=1)
    sr_pp = mu / sd
    sr_annual = sr_pp * math.sqrt(TRADING_DAYS)
    skew = float(stats.skew(rets, bias=False))
    kurt = float(stats.kurtosis(rets, fisher=False, bias=False))
    sr_var = (1 - skew * sr_pp + (kurt - 1) / 4 * sr_pp ** 2) / (n - 1)
    if sr_var <= 0 or not math.isfinite(sr_var):
        sr_var = float("nan")
    t_stat = sr_pp / math.sqrt(sr_var) if (sr_var and sr_var > 0) else float("nan")
    p_value = 1.0 - stats.norm.cdf(t_stat) if math.isfinite(t_stat) else float("nan")
    return dict(sharpe=sr_annual, sharpe_per_period=sr_pp, skew=skew, kurt=kurt,
                n=n, sr_var=sr_var, t_stat=t_stat, p_value=p_value)


def expected_max_sr(N: int, sr_var: float) -> float:
    if N <= 1 or sr_var <= 0 or not math.isfinite(sr_var):
        return 0.0
    a = stats.norm.ppf(1.0 - 1.0 / N)
    b = stats.norm.ppf(1.0 - 1.0 / (N * GAMMA_E))
    return math.sqrt(sr_var) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(sr_hat: float, sr_var: float, sr0: float) -> tuple[float, float]:
    if not math.isfinite(sr_hat) or not (sr_var and sr_var > 0):
        return float("nan"), float("nan")
    z = (sr_hat - sr0) / math.sqrt(sr_var)
    return z, float(stats.norm.cdf(z))


def max_drawdown(eq: pd.Series) -> tuple[float, int]:
    cummax = eq.cummax()
    dd = eq / cummax - 1
    if dd.isna().all():
        return float("nan"), 0
    trough = dd.idxmin()
    max_dd = float(dd.min())
    peak_date = eq.loc[:trough].idxmax()
    after = eq.loc[trough:]
    peak_value = eq.loc[peak_date]
    recovered = after[after >= peak_value]
    end_date = recovered.index[0] if len(recovered) else after.index[-1]
    return max_dd, (end_date - peak_date).days


def build_close() -> pd.Series:
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    return bar.set_index("close_time")["close"]


def all_trajectories(cache_dir: Path, close: pd.Series) -> pd.DataFrame:
    rows: list[dict] = []
    for cell in CELLS:
        files = sorted(f for f in os.listdir(cache_dir)
                       if parse_cache(f, cell) is not None)
        for f in files:
            label = parse_cache(f, cell)
            preds_full = pd.read_parquet(cache_dir / f)
            for window in WINDOWS:
                cutoff = preds_full.index.max() - pd.Timedelta(days=window)
                preds = preds_full.dropna(subset=["prob_up"])
                preds = preds.loc[preds.index >= cutoff]
                if len(preds) < 30:
                    continue
                cl = close.reindex(preds.index)
                if cl.isna().all():
                    continue
                for th in PROB_THRESHOLDS:
                    res = run_backtest(preds, cl, stake_usd=STAKE_USD,
                                       up_threshold=th, down_threshold=0.5)
                    eq = res.equity_curve
                    if len(eq) < 5:
                        continue
                    drets = daily_strategy_returns(eq).to_numpy()
                    comps = sharpe_components(drets)
                    max_dd, dd_dur = max_drawdown(eq)
                    rows.append({
                        "cell": cell,
                        "window_days": window,
                        "label": label,
                        "prob_TH": th,
                        "n_trades": res.n_trades,
                        "win_rate": res.win_rate,
                        "total_return_pct": res.total_return_pct,
                        "buy_hold_pct": res.buy_and_hold_return_pct,
                        "final_equity": res.final_equity,
                        "max_dd_pct": round(max_dd * 100, 2)
                                       if not pd.isna(max_dd) else None,
                        "max_dd_dur_days": dd_dur,
                        **comps,
                    })
        print(f"  scanned cell={cell} ({len(files)} labels)", flush=True)
    return pd.DataFrame(rows)


def per_cell_dsr(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, window), grp in df.groupby(["cell", "window_days"]):
        N = len(grp)
        if N == 0:
            continue
        best = grp.loc[grp["total_return_pct"].idxmax()]
        sr_var_cross_annual = float(grp["sharpe"].dropna().var(ddof=1))
        if not math.isfinite(sr_var_cross_annual) or sr_var_cross_annual <= 0:
            sr_var_cross_annual = float("nan")
        sr_var_cross_pp = (sr_var_cross_annual / TRADING_DAYS
                           if math.isfinite(sr_var_cross_annual)
                           else float("nan"))
        sr_exp_pp = (expected_max_sr(N, sr_var_cross_pp)
                     if math.isfinite(sr_var_cross_pp) else float("nan"))
        sr_exp_annual = (sr_exp_pp * math.sqrt(TRADING_DAYS)
                         if math.isfinite(sr_exp_pp) else float("nan"))
        dsr_z, psr = deflated_sharpe(best["sharpe_per_period"], best["sr_var"],
                                     sr_exp_pp)
        rows.append({
            "cell": cell,
            "window_days": window,
            "n_trials": N,
            "best_label": best["label"],
            "best_prob_TH": best["prob_TH"],
            "best_total_return_pct": best["total_return_pct"],
            "best_buy_hold_pct": best["buy_hold_pct"],
            "best_n_trades": best["n_trades"],
            "best_win_rate": best["win_rate"],
            "best_sharpe": best["sharpe"],
            "best_max_dd_pct": best["max_dd_pct"],
            "best_max_dd_dur_days": best["max_dd_dur_days"],
            "best_p_value": best["p_value"],
            "sr_cross_var_annual": sr_var_cross_annual,
            "sr_expected_max_annual": sr_exp_annual,
            "dsr_z": dsr_z,
            "psr": psr,
        })
    return pd.DataFrame(rows).sort_values(["window_days", "best_total_return_pct"],
                                          ascending=[True, False])


def fwer(per_cell: pd.DataFrame) -> pd.DataFrame:
    p = per_cell.copy()
    pv = p["best_p_value"].fillna(1.0).to_numpy()
    M = len(pv)
    order = np.argsort(pv)
    sorted_p = pv[order]
    bonf = np.minimum(pv * M, 1.0)
    holm_sorted = np.minimum.accumulate(
        np.maximum(sorted_p * (M - np.arange(M)), 0)[::-1])[::-1]
    holm = np.empty(M); holm[order] = np.minimum(holm_sorted, 1.0)
    bh_sorted = np.minimum.accumulate(
        (sorted_p * M / (np.arange(M) + 1))[::-1])[::-1]
    bh = np.empty(M); bh[order] = np.minimum(bh_sorted, 1.0)
    p["bonferroni_p"] = bonf
    p["holm_p"] = holm
    p["bh_fdr"] = bh
    return p


def ensemble_metrics(model_name: str, picks: list[CellSpec], window_days: int,
                     cache_dir: Path, close: pd.Series, n_trials: int) -> dict:
    N = len(picks)
    sub_eq: list[pd.Series] = []
    for c in picks:
        path = cache_dir / f"preds_ebm_{c.label_slug}_{c.cell_name}.parquet"
        if not path.exists():
            return {}
        preds = pd.read_parquet(path)
        nn = preds.dropna(subset=["prob_up"])
        cutoff = nn.index.max() - pd.Timedelta(days=window_days)
        p = nn.loc[nn.index >= cutoff]
        if len(p) < 30:
            return {}
        cl = close.reindex(p.index)
        res = run_backtest(p, cl, stake_usd=STAKE_USD / N,
                           up_threshold=c.prob_TH, down_threshold=0.5)
        sub_eq.append(res.equity_curve)
    aligned = sub_eq[0]
    for s in sub_eq[1:]:
        a, s2 = aligned.align(s, join="inner")
        aligned = a + s2
    eq = aligned
    if eq.empty:
        return {}
    drets = daily_strategy_returns(eq).to_numpy()
    comps = sharpe_components(drets)
    max_dd, dd_dur = max_drawdown(eq)
    # DSR with N=n_trials (use cross-trial var of the underlying single-cell
    # universe for the SAME window — we approximate it as 1.0 SR^2 / TRADING_DAYS
    # if not provided by caller; for ensemble this is a back-of-envelope DSR).
    return {
        "model": model_name,
        "window_days": window_days,
        "N_subs": N,
        "total_return_pct": round((float(eq.iloc[-1]) / STAKE_USD - 1) * 100, 2),
        "sharpe": round(comps["sharpe"], 3) if pd.notna(comps["sharpe"]) else None,
        "sharpe_per_period": comps["sharpe_per_period"],
        "sr_var": comps["sr_var"],
        "max_dd_pct": round(max_dd * 100, 2) if pd.notna(max_dd) else None,
        "max_dd_dur_days": dd_dur,
        "n_obs": comps["n"],
        "p_value": comps["p_value"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-tag", required=True)
    args = ap.parse_args()
    if args.cache_tag == "base":
        cache_dir = DATA_DIR / "preds_cache_ebm_ebm_extended"
        out_dir = ROOT / "reports" / "utc2130_extended"
    else:
        cache_dir = DATA_DIR / f"preds_cache_ebm_ebm_extended_{args.cache_tag}"
        out_dir = ROOT / "reports" / f"utc2130_extended_{args.cache_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"DSR/PSR/MaxDD analysis — cache_tag={args.cache_tag}")
    print(f"  cache : {cache_dir.relative_to(ROOT)}")
    print(f"  output: {out_dir.relative_to(ROOT)}")
    print("=" * 78, flush=True)

    print("[1/4] building close series ...", flush=True)
    close = build_close()

    print("[2/4] enumerating trajectories ...", flush=True)
    df = all_trajectories(cache_dir, close)
    df.to_csv(out_dir / "all_trajectories.csv", index=False)
    print(f"  saved {len(df)} → all_trajectories.csv", flush=True)

    print("[3/4] per-cell DSR ...", flush=True)
    pcb = per_cell_dsr(df)
    pcb.to_csv(out_dir / "per_cell_best.csv", index=False)
    print(f"  saved {len(pcb)} → per_cell_best.csv", flush=True)

    fw = fwer(pcb)
    fw.to_csv(out_dir / "fwer_summary.csv", index=False)
    print(f"  saved → fwer_summary.csv", flush=True)

    print("[4/4] ensemble metrics + DSR ...", flush=True)
    by_key = {_cell_key(c): c for c in SERVICE_CELLS}
    ens_rows = []
    # also include single sub-models (N=1) at their service settings
    for c in SERVICE_CELLS:
        for w in WINDOWS:
            r = ensemble_metrics(_cell_key(c), [c], w, cache_dir, close, n_trials=1)
            if r:
                ens_rows.append(r)
    for model_name, keys in ENSEMBLES.items():
        picks = [by_key[k] for k in keys]
        for w in WINDOWS:
            r = ensemble_metrics(model_name, picks, w, cache_dir, close, n_trials=len(keys))
            if r:
                ens_rows.append(r)
    ens = pd.DataFrame(ens_rows)
    if not ens.empty:
        # DSR on the ensemble: use the per-cell (cell, window) sr_var_cross from pcb
        # to deflate; for ensemble, we conservatively use the max across sub-cells.
        deflate_lookup = {(r["cell"], r["window_days"]): r["sr_cross_var_annual"]
                          for _, r in pcb.iterrows()}
        dsr_rows = []
        for _, r in ens.iterrows():
            # heuristic deflation: look up sr_var for the heaviest sub-model cell
            sr_var_cross_annual = float("nan")
            if "/" in r["model"]:
                cell = r["model"].split("/")[0]
                sr_var_cross_annual = deflate_lookup.get((cell, r["window_days"]),
                                                         float("nan"))
            else:
                # ensemble — average across the constituent sub-cells
                cells_in = [k.split("/")[0] for k in ENSEMBLES[r["model"]]]
                vals = [deflate_lookup.get((c, r["window_days"]), np.nan)
                        for c in cells_in]
                vals = [v for v in vals if math.isfinite(v)]
                if vals:
                    sr_var_cross_annual = float(np.mean(vals))
            sr_var_cross_pp = (sr_var_cross_annual / TRADING_DAYS
                               if math.isfinite(sr_var_cross_annual)
                               else float("nan"))
            sr_exp_pp = (expected_max_sr(462, sr_var_cross_pp)
                         if math.isfinite(sr_var_cross_pp) else float("nan"))
            sr_exp_annual = (sr_exp_pp * math.sqrt(TRADING_DAYS)
                             if math.isfinite(sr_exp_pp) else float("nan"))
            dsr_z, psr = deflated_sharpe(r["sharpe_per_period"], r["sr_var"],
                                         sr_exp_pp)
            dsr_rows.append({
                **r.to_dict(),
                "sr_expected_max_annual_at_N462": sr_exp_annual,
                "dsr_z_at_N462": dsr_z,
                "psr_at_N462": psr,
            })
        ens = pd.DataFrame(dsr_rows)
    cols_first = ["model", "window_days", "N_subs", "total_return_pct",
                  "sharpe", "max_dd_pct", "max_dd_dur_days",
                  "psr_at_N462", "dsr_z_at_N462",
                  "sr_expected_max_annual_at_N462",
                  "p_value", "n_obs"]
    if not ens.empty:
        ens = ens[[c for c in cols_first if c in ens.columns]
                  + [c for c in ens.columns if c not in cols_first]]
    ens.to_csv(out_dir / "ensemble_dsr.csv", index=False)
    print(f"  saved → ensemble_dsr.csv  ({len(ens)} rows)", flush=True)
    print("\n=== ensemble (sub-models + top1/2/3/5) ===")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        if not ens.empty:
            print(ens[[c for c in cols_first if c in ens.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
