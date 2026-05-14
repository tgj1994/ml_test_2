"""Full statistical robustness suite for the new EBM sweep results.

Replicates the same battery of tests that coin_analysis ran historically:

  1. MaxDD         — worst peak-to-trough drawdown of the equity curve
  2. Sharpe        — annualised SR from daily strategy returns
  3. PSR           — Probabilistic Sharpe Ratio (Bailey & López de Prado 2012)
                     P(true SR > 0 | observed returns), accounting for skew/kurt
  4. DSR           — Deflated Sharpe Ratio (Bailey & López de Prado 2014)
                     PSR with the threshold inflated for multiple testing
  5. PBO           — Probability of Backtest Overfitting (Bailey et al. 2017)
                     CSCV: how often the IS-best strategy ranks below the OOS
                     median across all S=16-block combinatorial splits
  6. Quarterly     — equity resampled to quarter end → per-quarter returns
                     alongside the per-quarter B&H, with outperform-pp counts
  7. Regime        — daily strategy return broken down by bull / bear /
                     sideways regime (30d rolling return ±5% bands)

Runs against the BEST-per-cell (variant × label × prob_TH) chosen on the 730d
window — same convention coin_analysis used.

Outputs (under reports/utc2130/):
  robustness_top10_ebm.csv             headline metrics (Sharpe, PSR, DSR,
                                       MaxDD, total_return) per top-10 model
  robustness_quarterly_top10_ebm.csv   per-quarter equity vs B&H
  robustness_regime_top10_ebm.csv      per-regime daily aggregates
  robustness_pbo_top_cells_ebm.csv     PBO per cell (full label × prob_TH grid)
  robustness_dsr_top10_ebm.csv         DSR with N = total trial count
  robustness_summary_top10_ebm.md      one-page Korean summary the user can scan
"""
from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.utc2130 import build_utc2130_daily


REPORTS = ROOT / "reports" / "utc2130"
CACHE_DIR = ROOT / "data" / "preds_cache_ebm"
WINDOW_730 = 730
WINDOW_365 = 365
TRADING_DAYS = 365.0  # BTC trades 24/7
EULER_GAMMA = 0.5772156649015329
S_BLOCKS = 16


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

def load_all_cells() -> pd.DataFrame:
    """Pull EVERY (variant, mode, label_or_k) cell from the freshly written
    per-window summary CSVs. Trades < 4 are dropped (statistics not meaningful).

    Sorted by 730d best_return desc so the console summary head() shows the
    strongest cells first, but every row is preserved in the output CSV.
    """
    rows = []
    for d in sorted(REPORTS.iterdir()):
        if not d.is_dir() or not d.name.endswith(f"_{WINDOW_730}d"):
            continue
        variant = d.name[: -len(str(WINDOW_730)) - 2]
        for mode in ("static", "dynamic"):
            f = (d / f"label_threshold_sweep_summary_{mode}_ebm_"
                     f"{variant}_{WINDOW_730}d.csv")
            if not f.exists():
                continue
            df = pd.read_csv(f)
            for _, r in df.iterrows():
                rows.append({
                    "variant": variant,
                    "mode": mode,
                    "label_or_k": r["slug"].split("_")[0],
                    "value": float(r["value"]),
                    "best_prob_th": float(r["best_prob_th"]),
                    "best_return": float(r["best_return"]),
                    "best_trades": int(r["best_trades"]),
                })
    df_all = pd.DataFrame(rows)
    df_pool = df_all[df_all["best_trades"] >= 4].copy()
    return (df_pool.sort_values("best_return", ascending=False)
                   .reset_index(drop=True))


# Backwards-compat alias — older scripts (drill_three.py etc.) may import this.
def load_top10() -> pd.DataFrame:
    return load_all_cells().head(10).copy()


def cache_path_for(variant: str, label_or_k: str) -> Path:
    return CACHE_DIR / f"preds_ebm_{label_or_k}_{variant}.parquet"


# ----------------------------------------------------------------------------
# Single-strategy backtest + metric extraction
# ----------------------------------------------------------------------------

def backtest_window(preds: pd.DataFrame, daily_close: pd.Series,
                    prob_th: float, window_days: int):
    """Run the same compounding backtest at a fixed up_threshold."""
    end = preds.index.max()
    start = end - pd.Timedelta(days=window_days - 1)
    sub = preds.loc[preds.index >= start]
    if len(sub) < 30:
        return None
    return run_backtest(sub, daily_close, up_threshold=prob_th,
                        down_threshold=0.5)


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough drawdown (fraction, negative)."""
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def daily_returns_from_equity(equity: pd.Series) -> pd.Series:
    """Per-period (daily here) return from a cumulative equity curve."""
    return equity.pct_change().dropna()


def sharpe(returns: pd.Series) -> float:
    if returns.std(ddof=1) == 0 or returns.empty:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def psr(returns: pd.Series, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio per Bailey & López de Prado (2012).

    PSR(SR*) = Φ( (SR_hat - SR*) * sqrt(n-1) /
                  sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2) )

    SR_hat is the per-period SR (NOT annualised) for the formula.
    """
    r = returns.dropna()
    n = len(r)
    if n < 8:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    sr_hat = mu / sd
    # Convert SR* (annualised) to per-period
    sr_star_pp = sr_star / math.sqrt(TRADING_DAYS)
    sk = float(stats.skew(r, bias=False))
    ku = float(stats.kurtosis(r, fisher=False, bias=False))  # NOT excess
    denom = math.sqrt(max(1e-12,
        1.0 - sk * sr_hat + (ku - 1.0) / 4.0 * sr_hat * sr_hat))
    z = (sr_hat - sr_star_pp) * math.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def dsr_threshold(n_trials: int, sr_variance: float) -> float:
    """Bailey & López de Prado (2014) inflated SR threshold for N trials.

    SR*_deflated = sqrt(Var(SR)) *
                   ( (1 - γ) * Φ^-1(1 - 1/N) + γ * Φ^-1(1 - 1/(N*e)) )

    where γ is the Euler-Mascheroni constant.
    """
    if n_trials <= 1:
        return 0.0
    inv1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    inv2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    threshold = math.sqrt(max(0.0, sr_variance)) * (
        (1.0 - EULER_GAMMA) * inv1 + EULER_GAMMA * inv2)
    return float(threshold)


def dsr(returns: pd.Series, n_trials: int) -> float:
    """DSR = PSR using the deflated threshold."""
    r = returns.dropna()
    n = len(r)
    if n < 8:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    sr_hat = mu / sd
    sk = float(stats.skew(r, bias=False))
    ku = float(stats.kurtosis(r, fisher=False, bias=False))
    # SR variance per Mertens (2002) → also feeds DSR threshold
    sr_var = (1.0 - sk * sr_hat + (ku - 1.0) / 4.0 * sr_hat * sr_hat) / (n - 1)
    sr_star_pp = dsr_threshold(n_trials, sr_var)
    denom = math.sqrt(max(1e-12,
        1.0 - sk * sr_hat + (ku - 1.0) / 4.0 * sr_hat * sr_hat))
    z = (sr_hat - sr_star_pp) * math.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


# ----------------------------------------------------------------------------
# Quarterly + regime breakdowns
# ----------------------------------------------------------------------------

def quarterly_returns(equity: pd.Series, bh: pd.Series) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    eq_q = equity.resample("QE").last().pct_change().dropna()
    bh_q = bh.resample("QE").last().pct_change().dropna()
    out = pd.DataFrame({
        "strategy_pct": eq_q * 100.0,
        "buy_hold_pct": bh_q * 100.0,
    })
    out["outperform_pp"] = out["strategy_pct"] - out["buy_hold_pct"]
    return out.reset_index().rename(columns={"index": "quarter_end"})


def regime_breakdown(equity: pd.Series, daily_close: pd.Series) -> pd.DataFrame:
    """Bull/Bear/Sideways by 30d rolling return ±5% bands on daily_close."""
    daily_close = daily_close.reindex(equity.index).dropna()
    eq = equity.reindex(daily_close.index).dropna()
    if eq.empty:
        return pd.DataFrame()
    ret30 = daily_close.pct_change(30)
    regime = pd.Series("sideways", index=daily_close.index)
    regime[ret30 > 0.05] = "bull"
    regime[ret30 < -0.05] = "bear"
    daily_strategy = eq.pct_change().fillna(0.0)
    daily_bh = daily_close.pct_change().fillna(0.0)

    rows = []
    for r in ("bull", "bear", "sideways"):
        mask = regime == r
        n_days = int(mask.sum())
        if n_days == 0:
            continue
        strat = daily_strategy.loc[mask]
        bh = daily_bh.loc[mask]
        # Compound the daily strategy returns within the regime
        total_strat = float((1.0 + strat).prod() - 1.0)
        total_bh = float((1.0 + bh).prod() - 1.0)
        rows.append({
            "regime": r,
            "n_days": n_days,
            "total_return_pct": total_strat * 100.0,
            "bh_return_pct": total_bh * 100.0,
            "outperform_pp": (total_strat - total_bh) * 100.0,
            "mean_daily_return_pct": float(strat.mean() * 100.0),
            "sharpe_per_period": (float(strat.mean() / strat.std(ddof=1))
                                   if strat.std(ddof=1) > 0 else 0.0),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# CSCV / PBO over a single cell's full (label × prob_TH) grid
# ----------------------------------------------------------------------------

def pbo_for_cell(variant: str, daily_close: pd.Series,
                 prob_th_grid: list[float],
                 window_days: int = WINDOW_730,
                 s_blocks: int = S_BLOCKS) -> dict:
    """Build the strategy × time returns matrix for one cell and compute PBO.

    M = number of (label, prob_TH) trajectories with sufficient labelled days
    in the chosen window. S = even number of equal-sized contiguous blocks.
    PBO = P(rank_OOS(IS-best) < 0.5)  across all C(S, S/2) splits.
    """
    parquets = sorted(CACHE_DIR.glob(f"preds_ebm_*_{variant}.parquet"))
    rets_list = []
    labels = []
    for p in parquets:
        # extract label_or_k from filename: preds_ebm_<label>_<variant>.parquet
        label = p.stem.replace("preds_ebm_", "").replace(f"_{variant}", "")
        try:
            preds = pd.read_parquet(p)
        except Exception:
            continue
        for th in prob_th_grid:
            bt = backtest_window(preds, daily_close, th, window_days)
            if bt is None or bt.equity_curve.empty:
                continue
            rets = bt.equity_curve.pct_change().fillna(0.0)
            if len(rets) < s_blocks * 4:
                continue
            rets_list.append(rets.values)
            labels.append((label, th))
    if not rets_list:
        return {"variant": variant, "PBO": float("nan"),
                "n_strategies": 0, "n_blocks": s_blocks}

    # Align lengths (trim to min)
    min_len = min(len(r) for r in rets_list)
    R = np.stack([r[:min_len] for r in rets_list])  # (M, T)
    M, T = R.shape
    block_size = T // s_blocks
    R = R[:, : block_size * s_blocks]  # truncate to multiple of S
    Rb = R.reshape(M, s_blocks, block_size)
    block_sum = Rb.sum(axis=2)                   # (M, S)
    block_sumsq = (Rb ** 2).sum(axis=2)
    block_n = np.full_like(block_sum, block_size, dtype=float)

    half = s_blocks // 2
    combos = list(combinations(range(s_blocks), half))
    K = len(combos)
    indicator = np.zeros((s_blocks, K), dtype=float)
    for k, idx in enumerate(combos):
        indicator[list(idx), k] = 1.0

    IS_sum = block_sum @ indicator                # (M, K)
    IS_sumsq = block_sumsq @ indicator
    IS_n = block_n @ indicator
    OOS_sum = block_sum.sum(axis=1, keepdims=True) - IS_sum
    OOS_sumsq = block_sumsq.sum(axis=1, keepdims=True) - IS_sumsq
    OOS_n = block_n.sum(axis=1, keepdims=True) - IS_n

    def _sharpe(s, sq, n):
        mu = s / n
        var = sq / n - mu * mu
        sd = np.sqrt(np.where(var > 0, var, np.nan))
        return mu / sd

    SR_IS = _sharpe(IS_sum, IS_sumsq, IS_n)
    SR_OOS = _sharpe(OOS_sum, OOS_sumsq, OOS_n)

    is_best = np.argmax(SR_IS, axis=0)            # (K,)
    rank_oos = (np.argsort(np.argsort(SR_OOS, axis=0), axis=0)
                + 1) / float(M)                    # 1..M → rank in [1/M, 1]
    omega = rank_oos[is_best, np.arange(K)]
    pbo = float((omega < 0.5).mean())
    return {
        "variant": variant,
        "PBO": pbo,
        "n_strategies": M,
        "n_blocks": s_blocks,
        "min_period_T": int(block_size * s_blocks),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    pool = load_all_cells()
    print(f"loaded {len(pool)} cells (variant x mode x label) "
          f"with n_trades>=4 from 730d summaries")

    daily = build_utc2130_daily(ROOT / "data" / "btc_15m.parquet")
    daily.index = pd.to_datetime(daily["close_time"], utc=True)
    daily_close = daily["close"].copy()
    daily_close.index = daily.index

    headline_rows: list[dict] = []
    quarterly_rows: list[dict] = []
    regime_rows: list[dict] = []

    # Estimate N_trials for DSR: total label-prob_TH count across all variants
    # The sweep evaluates 22 labels × 21 prob_TH values × 48 variants × 2 windows
    # ≈ 44,352 trials per window; use 22*21=462 within a variant as Bailey-style
    # intra-cell N (matches coin_analysis convention).
    n_trials_dsr = 22 * 21

    total = len(pool)
    for i, (_, row) in enumerate(pool.iterrows(), start=1):
        variant = row["variant"]
        label = row["label_or_k"]
        prob_th = float(row["best_prob_th"])
        if i % 200 == 0 or i == total:
            print(f"  [{i}/{total}] processed")
        p = cache_path_for(variant, label)
        if not p.exists():
            continue
        preds = pd.read_parquet(p)
        bt = backtest_window(preds, daily_close, prob_th, WINDOW_730)
        if bt is None or bt.equity_curve.empty:
            continue
        eq = bt.equity_curve
        bh = bt.buy_and_hold_curve

        rets = daily_returns_from_equity(eq)
        mdd = max_drawdown(eq)
        s = sharpe(rets)
        p_psr = psr(rets, sr_star=0.0)
        p_dsr = dsr(rets, n_trials=n_trials_dsr)
        headline_rows.append({
            "variant": variant,
            "mode": row["mode"],
            "label_or_k": label,
            "best_prob_th": prob_th,
            "best_return_730d": row["best_return"],
            "max_drawdown": mdd,
            "sharpe_annual": s,
            "PSR": p_psr,
            "DSR": p_dsr,
            "n_trades": bt.n_trades,
        })

        # Quarterly
        q = quarterly_returns(eq, bh)
        if not q.empty:
            q["variant"] = variant
            q["label_or_k"] = label
            quarterly_rows.append(q)

        # Regime
        rg = regime_breakdown(eq, daily_close)
        if not rg.empty:
            rg["variant"] = variant
            rg["label_or_k"] = label
            regime_rows.append(rg)

    headline = pd.DataFrame(headline_rows)
    headline.to_csv(REPORTS / "robustness_all_ebm.csv", index=False)

    if quarterly_rows:
        pd.concat(quarterly_rows, ignore_index=True).to_csv(
            REPORTS / "robustness_quarterly_all_ebm.csv", index=False)
    if regime_rows:
        pd.concat(regime_rows, ignore_index=True).to_csv(
            REPORTS / "robustness_regime_all_ebm.csv", index=False)

    # PBO is computed at the variant level (full label x prob_TH grid each)
    # across ALL unique variants — 48 variants × ~6s/each ≈ 5 minutes
    pbo_rows = []
    prob_th_grid = [round(0.50 + i * 0.01, 2) for i in range(21)]
    all_variants = pool["variant"].drop_duplicates().tolist()
    print(f"\nPBO over {len(all_variants)} unique variants (each = "
          f"label x prob_th grid)...")
    for i, cell in enumerate(all_variants, start=1):
        print(f"  [{i}/{len(all_variants)}] PBO {cell}")
        pbo_rows.append(pbo_for_cell(cell, daily_close, prob_th_grid))
    pd.DataFrame(pbo_rows).to_csv(
        REPORTS / "robustness_pbo_all_ebm.csv", index=False)

    # Console summary (top 20 only — the CSV holds the full table)
    print()
    print("=" * 130)
    print(f"ROBUSTNESS HEADLINE — full {len(headline)} cells (showing top 20 by 730d return)")
    print("=" * 130)
    out = headline.head(20).copy()
    out["best_return_730d"] = (out["best_return_730d"] * 100).round(2).astype(str) + "%"
    out["max_drawdown"] = (out["max_drawdown"] * 100).round(2).astype(str) + "%"
    for c in ("sharpe_annual", "PSR", "DSR"):
        out[c] = out[c].round(3)
    print(out.to_string(index=False))
    print(f"\n(see robustness_all_ebm.csv for all {len(headline)} cells)")

    print()
    print("=" * 130)
    print(f"PBO — all {len(pbo_rows)} variants (S={S_BLOCKS}, N_trials={n_trials_dsr})")
    print("=" * 130)
    pbo_df = pd.DataFrame(pbo_rows).sort_values("PBO")
    for _, r in pbo_df.head(20).iterrows():
        print(f"  {r.get('variant','-'):30s}  PBO={r.get('PBO',float('nan')):.3f}  "
              f"M={int(r.get('n_strategies', 0)):3d}  S={r.get('n_blocks',S_BLOCKS)}")
    if len(pbo_df) > 20:
        print(f"  ... (full {len(pbo_df)} rows in robustness_pbo_all_ebm.csv)")

    print()
    print(f"DSR threshold (N={n_trials_dsr} trials, per-cell): inflates SR* "
          f"~ sqrt(2*ln(N)) ≈ {math.sqrt(2*math.log(n_trials_dsr)):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
