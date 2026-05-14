"""Stationary Block Bootstrap confidence intervals for Top1/Top2/Top3/Top5.

Addresses reviewer critique #1: small trade count (Top1=14, Top2=22 over
730d) limits statistical confidence in headline metrics. We quantify the
uncertainty rather than dismiss it.

Method:
  - Stationary bootstrap (Politis & Romano 1994): pick random block lengths
    from a geometric distribution with mean L, sample with replacement.
    Mean block length L = 7 (days) by default — balances serial-correlation
    preservation with sample diversity.
  - 5000 bootstrap iterations per (ensemble, window).
  - For each iteration: rebuild the daily-returns series by concatenating
    blocks until length = original; recompute total_return, Sharpe (annual),
    Sortino, Calmar, maxDD on the bootstrap sample.
  - Report 95% CI as the [2.5, 97.5] empirical percentile.
  - Plus Wilson score CI on win rate from the original trade list (not
    bootstrap — it's a binomial proportion CI for n trades, k wins).

Output:
  reports/dsr_fwer/bootstrap_ci.csv             (one row per ensemble × window)
  reports/dsr_fwer/bootstrap_ci_summary.txt     (paper-ready table)
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
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm"
OUT_DIR = ROOT / "reports" / "dsr_fwer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = (730, 365)
N_BOOT = 5000
BLOCK_MEAN = 7.0
TRADING_DAYS = 365.0
RNG = np.random.default_rng(42)

ENSEMBLES = {
    "Top1": [("utc2130_sm_v5", "dynk12", 0.63)],
    "Top2": [("utc2130_sm_v5", "dynk12", 0.63),
             ("utc2130_sm_v3_complete", "th13", 0.68)],
    "Top3": [("utc2130_sm_v5", "dynk12", 0.63),
             ("utc2130_sm_v3_complete", "th13", 0.68),
             ("utc2130_sm", "dynk11", 0.64)],
    "Top5": [("utc2130_sm_v5", "dynk12", 0.63),
             ("utc2130_sm_v3_complete", "th13", 0.68),
             ("utc2130_sm", "dynk11", 0.64),
             ("utc2130_sm_v3", "th14", 0.66),
             ("utc2130_sm_v6", "dynk14", 0.66)],
}


def split_capital_eq_and_trades(cells, close, window: int) -> tuple[pd.Series, list]:
    sub_eq = []; trades: list = []
    for cell, label, prob in cells:
        cache = CACHE_DIR / f"preds_ebm_{label}_{cell}.parquet"
        preds = pd.read_parquet(cache)
        cutoff = preds.index.max() - pd.Timedelta(days=window)
        p = preds.dropna(subset=["prob_up"]).loc[lambda d: d.index >= cutoff]
        cl = close.reindex(p.index)
        res = run_backtest(p, cl, stake_usd=100.0 / len(cells),
                           up_threshold=prob, down_threshold=0.50)
        sub_eq.append(res.equity_curve)
        trades.extend(res.trades)
    eq = sub_eq[0]
    for s in sub_eq[1:]:
        eq, s2 = eq.align(s, join="inner")
        eq = eq + s2
    return eq, trades


def stationary_block_bootstrap_indices(n: int, L_mean: float,
                                        rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap: at each step, with
    probability 1/L_mean either start a new random block, or continue the
    current one. Returns int array of length n with bootstrapped indices
    into [0, n)."""
    out = np.empty(n, dtype=np.int64)
    p_new = 1.0 / L_mean
    out[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p_new:
            out[t] = rng.integers(0, n)
        else:
            out[t] = (out[t - 1] + 1) % n
    return out


def metrics_from_returns(rets: np.ndarray, stake: float = 100.0) -> dict:
    """Compute total_return%, Sharpe(annual), Sortino, maxDD%, Calmar from
    a daily returns series."""
    if len(rets) < 5 or rets.std(ddof=1) == 0:
        return dict(total_return_pct=0.0, sharpe=float("nan"),
                    sortino=float("nan"), max_dd_pct=0.0, calmar=float("nan"))
    eq = stake * np.cumprod(1.0 + rets)
    final = eq[-1]
    total_pct = (final / stake - 1.0) * 100.0
    mu = rets.mean(); sd = rets.std(ddof=1)
    sr = (mu / sd) * math.sqrt(TRADING_DAYS)
    downside = rets[rets < 0]
    sortino = ((mu / downside.std(ddof=1)) * math.sqrt(TRADING_DAYS)
                if len(downside) > 1 and downside.std(ddof=1) > 0
                else float("inf"))
    cummax = np.maximum.accumulate(eq)
    max_dd_pct = float((eq / cummax - 1.0).min()) * 100.0
    days = len(rets)
    annualised = ((final / stake) ** (TRADING_DAYS / days) - 1.0) * 100.0
    calmar = annualised / abs(max_dd_pct) if max_dd_pct != 0 else float("inf")
    return dict(total_return_pct=total_pct, sharpe=sr,
                sortino=sortino, max_dd_pct=max_dd_pct, calmar=calmar)


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    lo = (centre - half) / denom
    hi = (centre + half) / denom
    return lo, hi


def bootstrap_one(daily_rets: np.ndarray, n_boot: int = N_BOOT) -> dict:
    n = len(daily_rets)
    metrics_lists = {"total_return_pct": [], "sharpe": [], "sortino": [],
                     "max_dd_pct": [], "calmar": []}
    for _ in range(n_boot):
        idx = stationary_block_bootstrap_indices(n, BLOCK_MEAN, RNG)
        sample = daily_rets[idx]
        m = metrics_from_returns(sample)
        for k in metrics_lists:
            metrics_lists[k].append(m[k])
    out = {}
    for k, vals in metrics_lists.items():
        arr = np.array([v for v in vals if math.isfinite(v)])
        if len(arr) < 100:
            out[k] = (float("nan"), float("nan"), float("nan"))
            continue
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        med = float(np.percentile(arr, 50.0))
        out[k] = (lo, med, hi)
    return out


def main() -> None:
    print("[1/3] loading close ...", flush=True)
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close = bar.set_index("close_time")["close"]

    print(f"[2/3] running bootstrap ({N_BOOT} iters × 4 ensembles × 2 windows) ...",
          flush=True)
    rows = []
    for name, cells in ENSEMBLES.items():
        for window in WINDOWS:
            eq, trades = split_capital_eq_and_trades(cells, close, window)
            if eq.empty:
                continue
            daily = eq.pct_change().dropna().to_numpy()
            obs = metrics_from_returns(daily)
            ci = bootstrap_one(daily)
            n_trades = len(trades)
            n_wins = sum(1 for t in trades if t.pnl_usd > 0)
            wr_lo, wr_hi = wilson_ci(n_wins, n_trades)
            rows.append({
                "ensemble": name, "window_days": window,
                "n_days": len(daily),
                "n_trades": n_trades, "n_wins": n_wins,
                "win_rate_pct": (n_wins / n_trades * 100) if n_trades else 0.0,
                "win_rate_lo_pct": wr_lo * 100,
                "win_rate_hi_pct": wr_hi * 100,
                "obs_total_return_pct": obs["total_return_pct"],
                "ci_total_return_lo": ci["total_return_pct"][0],
                "ci_total_return_med": ci["total_return_pct"][1],
                "ci_total_return_hi": ci["total_return_pct"][2],
                "obs_sharpe": obs["sharpe"],
                "ci_sharpe_lo": ci["sharpe"][0],
                "ci_sharpe_med": ci["sharpe"][1],
                "ci_sharpe_hi": ci["sharpe"][2],
                "obs_sortino": obs["sortino"],
                "ci_sortino_lo": ci["sortino"][0],
                "ci_sortino_med": ci["sortino"][1],
                "ci_sortino_hi": ci["sortino"][2],
                "obs_max_dd_pct": obs["max_dd_pct"],
                "ci_max_dd_lo": ci["max_dd_pct"][0],
                "ci_max_dd_med": ci["max_dd_pct"][1],
                "ci_max_dd_hi": ci["max_dd_pct"][2],
                "obs_calmar": obs["calmar"],
                "ci_calmar_lo": ci["calmar"][0],
                "ci_calmar_med": ci["calmar"][1],
                "ci_calmar_hi": ci["calmar"][2],
            })
            print(f"  {name} {window}d  done  obs SR={obs['sharpe']:.2f}  "
                  f"CI [{ci['sharpe'][0]:.2f}, {ci['sharpe'][2]:.2f}]  "
                  f"obs ret={obs['total_return_pct']:+.1f}%  "
                  f"CI [{ci['total_return_pct'][0]:+.1f}%, "
                  f"{ci['total_return_pct'][2]:+.1f}%]  "
                  f"win {n_wins}/{n_trades} CI [{wr_lo*100:.1f}%, "
                  f"{wr_hi*100:.1f}%]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "bootstrap_ci.csv", index=False)

    print("[3/3] writing summary ...", flush=True)
    lines = []
    lines.append(f"# Stationary block bootstrap 95% CI ({N_BOOT} iters, "
                 f"block mean L={BLOCK_MEAN}d) + Wilson score CI on win rate")
    lines.append("")
    for w in WINDOWS:
        sub = df.loc[df["window_days"] == w]
        if sub.empty:
            continue
        lines.append(f"## window = {w}d")
        lines.append("")
        lines.append("| ensemble | obs ret% | 95% CI ret% | obs SR | "
                     "95% CI SR | obs maxDD% | 95% CI maxDD% | "
                     "obs Calmar | 95% CI Calmar | win | Wilson 95% CI |")
        lines.append("|---|---:|---|---:|---|---:|---|---:|---|---:|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['ensemble']} "
                f"| {r['obs_total_return_pct']:+.2f} "
                f"| [{r['ci_total_return_lo']:+.2f}, "
                f"{r['ci_total_return_hi']:+.2f}] "
                f"| {r['obs_sharpe']:.2f} "
                f"| [{r['ci_sharpe_lo']:.2f}, {r['ci_sharpe_hi']:.2f}] "
                f"| {r['obs_max_dd_pct']:.2f} "
                f"| [{r['ci_max_dd_lo']:.2f}, {r['ci_max_dd_hi']:.2f}] "
                f"| {r['obs_calmar']:.2f} "
                f"| [{r['ci_calmar_lo']:.2f}, "
                f"{r['ci_calmar_hi']:.2f}] "
                f"| {int(r['n_wins'])}/{int(r['n_trades'])} "
                f"| [{r['win_rate_lo_pct']:.1f}%, "
                f"{r['win_rate_hi_pct']:.1f}%] |")
        lines.append("")
    (OUT_DIR / "bootstrap_ci_summary.txt").write_text("\n".join(lines))
    print(f"saved → {OUT_DIR/'bootstrap_ci.csv'}")
    print(f"saved → {OUT_DIR/'bootstrap_ci_summary.txt'}")


if __name__ == "__main__":
    main()
