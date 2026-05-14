"""DSR (Deflated Sharpe Ratio) + FWER analysis over the full UTC0000/UTC2130 cell × label × prob_TH grid.

Addresses reviewer critique #1 (multiple-hypothesis testing / data snooping).

Pipeline:
  1. Build two daily-close series (utc0000, utc2130).
  2. For every cached prediction parquet (cell × label_slug),
     for every prob_TH in [0.50..0.70 step 0.01], for windows {730d, 365d}:
       - run the same compounding backtest used in the paper,
       - extract daily strategy returns from the equity curve,
       - compute annualised SR, skew, kurtosis, SR variance, t-stat, p-value.
  3. For each (cell, window), pick the BEST trajectory by total return;
     compute Bailey & Lopez de Prado (2014) DSR with N = 22 × 21 = 462
     intra-cell trials.
  4. Across all (cell, window) selections, compute Bonferroni / Holm /
     Benjamini-Hochberg FDR adjustments.
  5. Also report a "global" DSR using N = total number of trajectories.

Outputs (under reports/dsr_fwer/):
  - all_trajectories.csv          : one row per (cell, window, label, prob_TH)
  - per_cell_best.csv             : one row per (cell, window) — best trajectory + DSR
  - fwer_summary.csv              : per_cell_best with Bonferroni/Holm/BH columns
  - global_summary.txt            : top-level numbers for the paper
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import os
import re
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROB_THRESHOLDS = [round(0.50 + i * 0.01, 2) for i in range(21)]
WINDOWS = (730, 365)
TRADING_DAYS = 365.0  # crypto trades 24/7
EULER_GAMMA = 0.5772156649015329
GAMMA_E = math.e

EXCLUDE_SUFFIX_TOKENS = ("smoke",)


def parse_cell(filename: str) -> tuple[str, str] | None:
    m = re.match(r"preds_ebm_(th\d+|dynk\d+)_(.+)\.parquet$", filename)
    if not m:
        return None
    return m.group(1), m.group(2)


def utc_kind(suffix: str) -> str | None:
    if "utc0000" in suffix:
        return "utc0000"
    if "utc2130" in suffix:
        return "utc2130"
    return None


def daily_strategy_returns(equity: pd.Series) -> pd.Series:
    """Daily simple returns from the equity curve (intra-day flat-cash days
    contribute 0%, since equity stays equal to cash). Drop the first row."""
    return equity.pct_change().dropna()


def sharpe_components(rets: np.ndarray) -> dict:
    """Annualised SR + the moments needed for Bailey-Lopez de Prado."""
    n = len(rets)
    if n < 5 or rets.std(ddof=1) == 0:
        return {"sharpe": float("nan"), "skew": float("nan"),
                "kurt": float("nan"), "n": n,
                "sr_var": float("nan"), "t_stat": float("nan"),
                "p_value": float("nan")}
    mu = rets.mean()
    sd = rets.std(ddof=1)
    sr_per_period = mu / sd
    sr_annual = sr_per_period * math.sqrt(TRADING_DAYS)
    skew = float(stats.skew(rets, bias=False))
    kurt = float(stats.kurtosis(rets, fisher=False, bias=False))  # non-excess
    # Bailey & Lopez de Prado (2014), Eq. (5): variance of SR estimator
    # sigma_SR^2 = (1 - skew*SR + (kurt-1)/4 * SR^2) / (n-1)
    sr_var = (1 - skew * sr_per_period
              + (kurt - 1) / 4.0 * sr_per_period ** 2) / (n - 1)
    if sr_var <= 0 or not math.isfinite(sr_var):
        sr_var = float("nan")
    t_stat = sr_per_period / math.sqrt(sr_var) if sr_var > 0 else float("nan")
    p_value = 1.0 - stats.norm.cdf(t_stat) if math.isfinite(t_stat) else float("nan")
    return {
        "sharpe": sr_annual,
        "sharpe_per_period": sr_per_period,
        "skew": skew,
        "kurt": kurt,
        "n": n,
        "sr_var": sr_var,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def expected_max_sr(N: int, sr_var: float) -> float:
    """Bailey & Lopez de Prado (2014), Eq. (6):
    E[max SR_n] ≈ sqrt(var_SR) · ((1−γ)·Φ^-1(1−1/N) + γ·Φ^-1(1−1/(N·e)))
    where γ ≈ 0.5772, var_SR is the cross-trial variance of SR estimates."""
    if N <= 1 or sr_var <= 0 or not math.isfinite(sr_var):
        return 0.0
    a = stats.norm.ppf(1.0 - 1.0 / N)
    b = stats.norm.ppf(1.0 - 1.0 / (N * GAMMA_E))
    return math.sqrt(sr_var) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe_ratio(sr_hat: float, sr_var_hat: float, sr0: float, n_obs: int) -> tuple[float, float]:
    """Bailey & Lopez de Prado (2014) PSR with a non-zero benchmark SR.
       Returns (DSR_z, PSR=Phi(z))."""
    if not math.isfinite(sr_hat) or sr_var_hat <= 0 or not math.isfinite(sr_var_hat):
        return float("nan"), float("nan")
    z = (sr_hat - sr0) / math.sqrt(sr_var_hat)
    psr = float(stats.norm.cdf(z))
    return z, psr


def run_one_cell(suffix: str, kind: str,
                 close_series: pd.Series) -> list[dict]:
    """Iterate every label_slug × prob_TH × window for one cell."""
    rows: list[dict] = []
    label_files = sorted([f for f in os.listdir(CACHE_DIR)
                          if f.startswith("preds_ebm_") and f.endswith(f"_{suffix}.parquet")])
    if not label_files:
        return rows
    for f in label_files:
        parsed = parse_cell(f)
        if parsed is None:
            continue
        label, suffix_chk = parsed
        if suffix_chk != suffix:
            continue
        try:
            preds_full = pd.read_parquet(CACHE_DIR / f)
        except Exception as e:
            print(f"  ! skip {f}: {e}")
            continue
        if preds_full.empty:
            continue
        for window_days in WINDOWS:
            cutoff = preds_full.index.max() - pd.Timedelta(days=window_days)
            preds = preds_full.dropna(subset=["prob_up"])
            preds = preds.loc[preds.index >= cutoff]
            if len(preds) < 30:
                continue
            close_w = close_series.reindex(preds.index)
            if close_w.isna().all():
                continue
            for th in PROB_THRESHOLDS:
                res = run_backtest(preds, close_w,
                                   up_threshold=th, down_threshold=0.5)
                if len(res.equity_curve) < 5:
                    continue
                drets = daily_strategy_returns(res.equity_curve).to_numpy()
                comps = sharpe_components(drets)
                rows.append({
                    "cell": suffix,
                    "kind": kind,
                    "window_days": window_days,
                    "label": label,
                    "prob_TH": th,
                    "n_trades": res.n_trades,
                    "win_rate": res.win_rate,
                    "total_return_pct": res.total_return_pct,
                    "buy_hold_pct": res.buy_and_hold_return_pct,
                    "final_equity": res.final_equity,
                    **comps,
                })
    return rows


def main() -> None:
    print("[1/4] building close series ...", flush=True)
    bar0 = build_utc0000_daily(DATA_DIR / "btc_15m.parquet")
    bar21 = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close_utc0000 = bar0.set_index("close_time")["close"]
    close_utc2130 = bar21.set_index("close_time")["close"]
    print(f"  utc0000 close: {len(close_utc0000)} rows")
    print(f"  utc2130 close: {len(close_utc2130)} rows")

    # discover cells
    files = sorted([f for f in os.listdir(CACHE_DIR)
                    if f.startswith("preds_ebm_")
                    and ("utc0000" in f or "utc2130" in f)])
    cells = sorted({parse_cell(f)[1] for f in files if parse_cell(f) is not None})
    cells = [c for c in cells if not any(tok in c for tok in EXCLUDE_SUFFIX_TOKENS)]
    print(f"[2/4] {len(cells)} cells found ...", flush=True)

    all_rows: list[dict] = []
    for i, cell in enumerate(cells, 1):
        kind = utc_kind(cell)
        if kind is None:
            continue
        cs = close_utc0000 if kind == "utc0000" else close_utc2130
        rows = run_one_cell(cell, kind, cs)
        all_rows.extend(rows)
        print(f"  [{i:>3}/{len(cells)}] {cell:<40s} → {len(rows)} trajectories",
              flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "all_trajectories.csv", index=False)
    print(f"  saved {len(df)} trajectories → {OUT_DIR/'all_trajectories.csv'}",
          flush=True)

    print("[3/4] per-cell best + DSR ...", flush=True)
    # For each (cell, window), pick BEST trajectory by total_return.
    # Compute DSR using intra-cell N = (#labels × #prob_TH) and
    # cross-trial var(SR) measured from THAT cell's trajectories.
    per_cell_rows = []
    for (cell, window), grp in df.groupby(["cell", "window_days"]):
        N = len(grp)
        if N == 0:
            continue
        best = grp.loc[grp["total_return_pct"].idxmax()]
        sr_cross_var = float(grp["sharpe"].dropna().var(ddof=1))  # cross-trial
        if not math.isfinite(sr_cross_var) or sr_cross_var <= 0:
            sr_cross_var = float("nan")
        # Convert annualised cross-var back to per-period for the formula:
        sr_cross_var_per_period = sr_cross_var / TRADING_DAYS \
            if math.isfinite(sr_cross_var) else float("nan")
        sr_expected = expected_max_sr(N, sr_cross_var_per_period) \
            if math.isfinite(sr_cross_var_per_period) else float("nan")
        sr_expected_annual = sr_expected * math.sqrt(TRADING_DAYS) \
            if math.isfinite(sr_expected) else float("nan")
        dsr_z, psr = deflated_sharpe_ratio(
            best["sharpe_per_period"], best["sr_var"],
            sr_expected, int(best["n"]))
        per_cell_rows.append({
            "cell": cell,
            "window_days": window,
            "n_trials": N,
            "kind": best["kind"],
            "best_label": best["label"],
            "best_prob_TH": best["prob_TH"],
            "best_total_return_pct": best["total_return_pct"],
            "best_buy_hold_pct": best["buy_hold_pct"],
            "best_n_trades": best["n_trades"],
            "best_win_rate": best["win_rate"],
            "best_sharpe": best["sharpe"],
            "best_p_value": best["p_value"],
            "sr_cross_var_annual": sr_cross_var,
            "sr_expected_max_annual": sr_expected_annual,
            "dsr_z": dsr_z,
            "psr": psr,
        })
    pcb = pd.DataFrame(per_cell_rows).sort_values(
        ["window_days", "best_total_return_pct"], ascending=[True, False])
    pcb.to_csv(OUT_DIR / "per_cell_best.csv", index=False)
    print(f"  per-cell best → {OUT_DIR/'per_cell_best.csv'} ({len(pcb)} rows)",
          flush=True)

    print("[4/4] FWER (Bonferroni / Holm) + BH-FDR ...", flush=True)
    # Apply across the per_cell_best p-values — i.e. one selection per cell
    # × window. This is the "family of selections" the paper makes.
    pvals = pcb["best_p_value"].fillna(1.0).to_numpy()
    M = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(M, dtype=int)
    ranks[order] = np.arange(1, M + 1)

    # Bonferroni
    bonf = np.minimum(pvals * M, 1.0)
    # Holm-Bonferroni
    holm = np.empty(M)
    sorted_p = pvals[order]
    cmax = 0.0
    for k, p in enumerate(sorted_p):
        adj = min(1.0, (M - k) * p)
        cmax = max(cmax, adj)
        holm[order[k]] = cmax
    # Benjamini-Hochberg FDR
    bh = np.empty(M)
    cmin = 1.0
    for k in range(M - 1, -1, -1):
        p = sorted_p[k]
        adj = min(1.0, p * M / (k + 1))
        cmin = min(cmin, adj)
        bh[order[k]] = cmin

    pcb["rank"] = ranks
    pcb["p_bonferroni"] = bonf
    pcb["p_holm"] = holm
    pcb["q_bh_fdr"] = bh
    pcb["sig_bonf_005"] = pcb["p_bonferroni"] < 0.05
    pcb["sig_holm_005"] = pcb["p_holm"] < 0.05
    pcb["sig_bh_fdr_005"] = pcb["q_bh_fdr"] < 0.05
    pcb["dsr_significant_psr_095"] = pcb["psr"] >= 0.95
    pcb.to_csv(OUT_DIR / "fwer_summary.csv", index=False)
    print(f"  FWER summary → {OUT_DIR/'fwer_summary.csv'}", flush=True)

    # Global DSR — N = total trajectories
    total_N = len(df)
    sr_cross_var_global = float(df["sharpe"].dropna().var(ddof=1))
    sr_cross_var_global_pp = sr_cross_var_global / TRADING_DAYS
    sr_expected_global = expected_max_sr(total_N, sr_cross_var_global_pp)
    global_best = df.loc[df["total_return_pct"].idxmax()]
    dsr_z_global, psr_global = deflated_sharpe_ratio(
        global_best["sharpe_per_period"], global_best["sr_var"],
        sr_expected_global, int(global_best["n"]))

    # Summary text
    lines = []
    lines.append("# DSR + FWER summary")
    lines.append("")
    lines.append(f"Total trajectories evaluated: {total_N}")
    lines.append(f"Distinct (cell × window) selections: {M}")
    lines.append("")
    lines.append("## Global best across all trajectories")
    lines.append(f"  cell={global_best['cell']} window={int(global_best['window_days'])}d "
                 f"label={global_best['label']} prob_TH={global_best['prob_TH']:.2f}")
    lines.append(f"  total_return = {global_best['total_return_pct']*100:+.2f}%  "
                 f"trades={int(global_best['n_trades'])}  win={global_best['win_rate']*100:.1f}%  "
                 f"SR(annual) = {global_best['sharpe']:.3f}")
    lines.append(f"  E[max SR | N={total_N}] (annual) = "
                 f"{sr_expected_global*math.sqrt(TRADING_DAYS):.3f}")
    lines.append(f"  DSR z = {dsr_z_global:.3f}    PSR = {psr_global:.4f}")
    lines.append(f"  PSR ≥ 0.95 ? {'YES' if psr_global >= 0.95 else 'NO'}")
    lines.append("")
    lines.append("## FWER results (per-cell-best, M selections)")
    lines.append(f"  significant @ 0.05 — Bonferroni : "
                 f"{int(pcb['sig_bonf_005'].sum())} / {M}")
    lines.append(f"  significant @ 0.05 — Holm       : "
                 f"{int(pcb['sig_holm_005'].sum())} / {M}")
    lines.append(f"  significant @ 0.05 — BH-FDR     : "
                 f"{int(pcb['sig_bh_fdr_005'].sum())} / {M}")
    lines.append(f"  DSR PSR ≥ 0.95 cells           : "
                 f"{int(pcb['dsr_significant_psr_095'].sum())} / {M}")
    lines.append("")
    lines.append("## Top-10 cells by DSR PSR (window=730d)")
    top10 = pcb.loc[pcb["window_days"] == 730].sort_values("psr", ascending=False).head(10)
    for _, r in top10.iterrows():
        lines.append(
            f"  {r['cell']:<40s} "
            f"label={r['best_label']:<7s} prob={r['best_prob_TH']:.2f}  "
            f"ret={r['best_total_return_pct']*100:+7.2f}%  "
            f"SR={r['best_sharpe']:.3f}  "
            f"DSR_z={r['dsr_z']:.2f}  PSR={r['psr']:.3f}  "
            f"p_bonf={r['p_bonferroni']:.3g}  q_bh={r['q_bh_fdr']:.3g}"
        )

    out_txt = OUT_DIR / "global_summary.txt"
    out_txt.write_text("\n".join(lines))
    print(f"  summary → {out_txt}", flush=True)
    print("\n".join(lines[:18]))


if __name__ == "__main__":
    main()
