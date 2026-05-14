"""Ensemble-level robustness for the 4 production ensembles (Top1/2/3/5)
at 1095d (3y) and 1460d (4y) windows.

Four complementary lenses on each deployed ensemble:

  A. Stationary block bootstrap 95% CI (Politis & Romano 1994)
     5000 iters × block mean L=7d. Captures uncertainty in headline metrics
     (total_return, Sharpe, Sortino, max_dd, Calmar) given the limited
     trade count.

  B. Regime-conditional breakdown
     Classify each day by 30d-trailing return: bull (>+5%), bear (<-5%),
     sideways. Per-regime: total return, days in market, mean daily ret,
     outperform_pp vs B&H of the same days.

  C. Deflated Sharpe Ratio (Bailey-Lopez de Prado 2014)
     Even though the ensemble is fixed (not selection-best per window), the
     5 deployed cells were selected from a 5×462=2310 trial universe.
     DSR adjusts the ensemble's observed Sharpe for that look-ahead.

  D. Ensemble-level PBO via CSCV
     Universe = all 2^5 − 1 = 31 non-empty subsets of the 5 deployed cells
     (each subset = one split-capital ensemble candidate). Run CSCV with
     S=16 blocks → 12,870 IS/OOS combos. For each combo: IS-best subset →
     check its OOS Sharpe rank. PBO < 0.5 = deployed-style ensembles
     generalise OOS.

Outputs (reports/utc2130_extended/):
  - ensemble_bootstrap_ci.csv
  - ensemble_regime_breakdown.csv
  - ensemble_dsr.csv
  - ensemble_pbo.csv
  - ensemble_pbo_pick_frequency.csv
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import run_backtest
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm_ebm_extended"
OUT_DIR = ROOT / "reports" / "utc2130_extended"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = (1095, 1460)
TRADING_DAYS = 365.0
N_BOOT = 5000
BLOCK_MEAN = 7.0
RNG = np.random.default_rng(42)
EULER_GAMMA = 0.5772156649015329

# Cell universe size for DSR: 22 labels × 21 prob_TH = 462 per cell × 5 cells
N_TRIALS_GLOBAL = 22 * 21 * 5

REGIME_LOOKBACK = 30
BULL_TH = 0.05
BEAR_TH = -0.05

# CSCV PBO parameters
S_BLOCKS = 16

# All 5 deployed cells, with their (label, prob_TH) fixed at the values used
# in production (mirrors coin_service_utc2130/backend/app/config.py SUB_MODELS).
ALL_CELLS = [
    ("utc2130_sm_v5",          "dynk12", 0.63),
    ("utc2130_sm_v3_complete", "th13",   0.68),
    ("utc2130_sm",             "dynk11", 0.64),
    ("utc2130_sm_v3",          "th14",   0.66),
    ("utc2130_sm_v6",          "dynk14", 0.66),
]
DEPLOYED_LABELS = {  # for tagging which production ensemble a subset corresponds to
    frozenset({0}):                "Top1",
    frozenset({0, 1}):             "Top2",
    frozenset({0, 1, 2}):          "Top3",
    frozenset({0, 1, 2, 3, 4}):    "Top5",
}

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


def _split_capital_eq_and_trades(cells, close, window):
    sub_eq = []; trades = []
    for cell, label, prob in cells:
        cache = CACHE_DIR / f"preds_ebm_{label}_{cell}.parquet"
        preds = pd.read_parquet(cache)
        nn = preds.dropna(subset=["prob_up"])
        cutoff = nn.index.max() - pd.Timedelta(days=window)
        p = nn.loc[nn.index >= cutoff]
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


def _stationary_indices(n: int, L_mean: float, rng) -> np.ndarray:
    out = np.empty(n, dtype=np.int64)
    p_new = 1.0 / L_mean
    out[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p_new:
            out[t] = rng.integers(0, n)
        else:
            out[t] = (out[t - 1] + 1) % n
    return out


def _metrics_from_returns(rets: np.ndarray, stake: float = 100.0) -> dict:
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


def _wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (centre - half) / denom, (centre + half) / denom


def _bootstrap(rets: np.ndarray, n_boot: int = N_BOOT) -> dict:
    n = len(rets)
    keys = ("total_return_pct", "sharpe", "sortino", "max_dd_pct", "calmar")
    lists = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = _stationary_indices(n, BLOCK_MEAN, RNG)
        sample = rets[idx]
        m = _metrics_from_returns(sample)
        for k in keys:
            lists[k].append(m[k])
    out = {}
    for k, vals in lists.items():
        arr = np.array([v for v in vals if math.isfinite(v)])
        if len(arr) < 100:
            out[k] = (float("nan"), float("nan"), float("nan"))
            continue
        out[k] = (float(np.percentile(arr, 2.5)),
                  float(np.percentile(arr, 50.0)),
                  float(np.percentile(arr, 97.5)))
    return out


def _regime_classify(close: pd.Series) -> pd.Series:
    ret = close.pct_change(REGIME_LOOKBACK)
    reg = pd.Series(index=close.index, dtype=object)
    reg[ret > BULL_TH] = "bull"
    reg[ret < BEAR_TH] = "bear"
    reg[(ret >= BEAR_TH) & (ret <= BULL_TH)] = "sideways"
    return reg.fillna("sideways")


def _regime_metrics(eq: pd.Series, bh: pd.Series, regime: pd.Series) -> list[dict]:
    daily_eq = eq.pct_change()
    daily_bh = bh.pct_change()
    rows = []
    for reg_name in ("bull", "bear", "sideways"):
        mask = regime.reindex(daily_eq.index).fillna("sideways") == reg_name
        n = int(mask.sum())
        if n == 0:
            rows.append(dict(regime=reg_name, n_days=0,
                             cum_return_pct=0.0, bh_cum_return_pct=0.0,
                             outperform_pp=0.0,
                             mean_daily_pct=float("nan"),
                             sharpe_period=float("nan")))
            continue
        eq_r = daily_eq.loc[mask].fillna(0).to_numpy()
        bh_r = daily_bh.loc[mask].fillna(0).to_numpy()
        cum_eq = float((np.prod(1 + eq_r) - 1) * 100)
        cum_bh = float((np.prod(1 + bh_r) - 1) * 100)
        sharpe = (eq_r.mean() / eq_r.std(ddof=1) * math.sqrt(TRADING_DAYS)
                  if eq_r.std(ddof=1) > 0 else float("nan"))
        rows.append(dict(
            regime=reg_name, n_days=n,
            cum_return_pct=round(cum_eq, 2),
            bh_cum_return_pct=round(cum_bh, 2),
            outperform_pp=round(cum_eq - cum_bh, 2),
            mean_daily_pct=round(eq_r.mean() * 100, 4),
            sharpe_period=round(sharpe, 3) if not math.isnan(sharpe) else None,
        ))
    return rows


def _dsr(rets: np.ndarray, n_trials: int) -> dict:
    """Bailey-Lopez de Prado (2014) Deflated Sharpe Ratio.

    DSR = P(true SR > 0 | observed SR, N trials, moments of returns).
    """
    n = len(rets)
    if n < 5 or rets.std(ddof=1) == 0:
        return dict(sr_obs=float("nan"), dsr=float("nan"), n=n)
    mu = rets.mean(); sd = rets.std(ddof=1)
    sr = mu / sd  # per-period
    skew = float(stats.skew(rets, bias=False))
    kurt = float(stats.kurtosis(rets, fisher=True, bias=False))  # excess kurtosis
    sr_var = (1 - skew * sr + (kurt / 4) * sr ** 2) / (n - 1)
    if sr_var <= 0:
        return dict(sr_obs=sr * math.sqrt(TRADING_DAYS),
                    dsr=float("nan"), n=n)
    sr_std = math.sqrt(sr_var)

    # Expected max SR under null (all N trials have true SR=0):
    # E[SR_max] ≈ Z(1 - 1/N) * sd_SR (large-N approximation with euler-mascheroni)
    if n_trials > 1:
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
        sr_zero_max = sr_std * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    else:
        sr_zero_max = 0.0

    z = (sr - sr_zero_max) / sr_std
    dsr = float(stats.norm.cdf(z))
    return dict(sr_obs=sr * math.sqrt(TRADING_DAYS),
                dsr=dsr, n=n, sr_zero_max_per_period=sr_zero_max,
                sr_var=sr_var)


def _block_stats(returns: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(returns)
    edges = np.linspace(0, n, S + 1, dtype=int)
    sums = np.zeros(S); sqs = np.zeros(S); ns = np.zeros(S)
    for i in range(S):
        seg = returns[edges[i]:edges[i + 1]]
        ns[i] = len(seg)
        if len(seg) > 0:
            sums[i] = seg.sum()
            sqs[i] = (seg * seg).sum()
    return sums, sqs, ns


def _all_subsets(n: int) -> list[tuple[int, ...]]:
    """All non-empty subsets of {0..n-1} as ascending-index tuples."""
    out = []
    for k in range(1, n + 1):
        out.extend(combinations(range(n), k))
    return out


def _ensemble_returns(idx_subset: tuple[int, ...], all_cells, close,
                      window: int) -> np.ndarray:
    cells = [all_cells[i] for i in idx_subset]
    eq, _ = _split_capital_eq_and_trades(cells, close, window)
    if eq.empty:
        return np.array([])
    return eq.pct_change().dropna().to_numpy()


def _cscv_pbo(subsets: list[tuple[int, ...]], all_cells, close, window: int,
              S: int = S_BLOCKS) -> tuple[dict, np.ndarray, list[tuple[int, ...]]]:
    """CSCV across `subsets` candidate ensembles. Returns
    (summary_dict, pick_frequency_per_subset, valid_subsets)."""
    block_sum_list = []
    block_sumsq_list = []
    block_n_list = []
    valid: list[tuple[int, ...]] = []
    for sub in subsets:
        rets = _ensemble_returns(sub, all_cells, close, window)
        if len(rets) < S * 5:
            continue
        s_, q_, n_ = _block_stats(rets, S)
        block_sum_list.append(s_)
        block_sumsq_list.append(q_)
        block_n_list.append(n_)
        valid.append(sub)
    if not valid:
        return {}, np.array([]), valid

    block_sum = np.array(block_sum_list)
    block_sumsq = np.array(block_sumsq_list)
    block_n = np.array(block_n_list)

    M = block_sum.shape[0]
    half = S // 2
    combos = list(combinations(range(S), half))
    K = len(combos)
    indicator = np.zeros((S, K), dtype=np.float64)
    for k, c in enumerate(combos):
        for s in c:
            indicator[s, k] = 1.0
    oos_ind = 1.0 - indicator

    IS_sum = block_sum @ indicator
    IS_sumsq = block_sumsq @ indicator
    IS_n = block_n @ indicator
    OOS_sum = block_sum @ oos_ind
    OOS_sumsq = block_sumsq @ oos_ind
    OOS_n = block_n @ oos_ind

    def sharpe_matrix(sumv, sqv, nv):
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sumv / nv
            var = sqv / nv - mean * mean
            sr = mean / np.sqrt(var)
            sr[~np.isfinite(sr)] = -np.inf
        return sr

    SR_IS = sharpe_matrix(IS_sum, IS_sumsq, IS_n)
    SR_OOS = sharpe_matrix(OOS_sum, OOS_sumsq, OOS_n)

    best_idx = np.argmax(SR_IS, axis=0)
    order = np.argsort(-SR_OOS, axis=0)
    ranks = np.empty_like(order)
    rng = np.arange(M)[:, None]
    ranks[order, np.arange(K)[None, :]] = rng + 1
    rank_best = ranks[best_idx, np.arange(K)]
    omega = (rank_best - 0.5) / M
    omega_clip = np.clip(omega, 1e-9, 1 - 1e-9)
    lam = np.log(omega_clip / (1 - omega_clip))

    pbo = float((lam > 0).mean())
    is_sr_best = SR_IS[best_idx, np.arange(K)]
    oos_sr_best = SR_OOS[best_idx, np.arange(K)]

    # Pick frequency per subset
    pick_freq = np.bincount(best_idx, minlength=M) / K

    return dict(
        pbo=pbo,
        median_omega=float(np.median(omega)),
        median_lambda=float(np.median(lam)),
        mean_lambda=float(np.mean(lam)),
        mean_is_sr=float(np.nanmean(is_sr_best)),
        mean_oos_sr=float(np.nanmean(oos_sr_best)),
        loss_in_translation=float(np.nanmean(is_sr_best) - np.nanmean(oos_sr_best)),
        n_combos=K, n_strategies=M,
    ), pick_freq, valid


def main():
    print("[1/4] loading close (UTC2130) ...", flush=True)
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close = bar.set_index("close_time")["close"]
    regime = _regime_classify(close)

    boot_rows = []
    regime_rows = []
    dsr_rows = []

    for name, cells in ENSEMBLES.items():
        for w in WINDOWS:
            print(f"[2/4] {name} {w}d  bootstrap+regime+DSR ...", flush=True)
            eq, trades = _split_capital_eq_and_trades(cells, close, w)
            if eq.empty:
                continue
            daily = eq.pct_change().dropna().to_numpy()
            obs = _metrics_from_returns(daily)

            # bootstrap CI
            ci = _bootstrap(daily)
            n_trades = len(trades)
            n_wins = sum(1 for t in trades if t.pnl_usd > 0)
            wr_lo, wr_hi = _wilson_ci(n_wins, n_trades)
            boot_rows.append({
                "ensemble": name, "window_days": w,
                "n_days": len(daily), "n_trades": n_trades, "n_wins": n_wins,
                "win_rate_pct": round(n_wins / n_trades * 100, 2) if n_trades else 0.0,
                "win_rate_lo_pct": round(wr_lo * 100, 2),
                "win_rate_hi_pct": round(wr_hi * 100, 2),
                "obs_total_return_pct": round(obs["total_return_pct"], 2),
                "ci_total_return_lo": round(ci["total_return_pct"][0], 2),
                "ci_total_return_med": round(ci["total_return_pct"][1], 2),
                "ci_total_return_hi": round(ci["total_return_pct"][2], 2),
                "obs_sharpe": round(obs["sharpe"], 3),
                "ci_sharpe_lo": round(ci["sharpe"][0], 3),
                "ci_sharpe_hi": round(ci["sharpe"][2], 3),
                "obs_sortino": round(obs["sortino"], 3) if math.isfinite(obs["sortino"]) else None,
                "ci_sortino_lo": round(ci["sortino"][0], 3),
                "ci_sortino_hi": round(ci["sortino"][2], 3),
                "obs_max_dd_pct": round(obs["max_dd_pct"], 2),
                "ci_max_dd_lo": round(ci["max_dd_pct"][0], 2),
                "ci_max_dd_hi": round(ci["max_dd_pct"][2], 2),
                "obs_calmar": round(obs["calmar"], 3) if math.isfinite(obs["calmar"]) else None,
                "ci_calmar_lo": round(ci["calmar"][0], 3),
                "ci_calmar_hi": round(ci["calmar"][2], 3),
            })
            print(f"    obs SR={obs['sharpe']:.2f}  "
                  f"CI95 [{ci['sharpe'][0]:.2f}, {ci['sharpe'][2]:.2f}]  "
                  f"win {n_wins}/{n_trades} CI [{wr_lo*100:.1f}%, {wr_hi*100:.1f}%]",
                  flush=True)

            # regime breakdown
            bh = (close.reindex(eq.index) / close.reindex(eq.index).iloc[0]) * 100.0
            for row in _regime_metrics(eq, bh, regime):
                row.update({"ensemble": name, "window_days": w})
                regime_rows.append(row)

            # DSR
            d = _dsr(daily, n_trials=N_TRIALS_GLOBAL)
            dsr_rows.append({
                "ensemble": name, "window_days": w,
                "n_days": d["n"],
                "sr_obs_annual": round(d["sr_obs"], 3),
                "dsr": round(d["dsr"], 4) if not math.isnan(d.get("dsr", float("nan"))) else None,
                "n_trials_assumed": N_TRIALS_GLOBAL,
            })

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT_DIR / "ensemble_bootstrap_ci.csv", index=False)
    regime_df = pd.DataFrame(regime_rows)[["ensemble", "window_days", "regime",
                                            "n_days", "cum_return_pct",
                                            "bh_cum_return_pct", "outperform_pp",
                                            "mean_daily_pct", "sharpe_period"]]
    regime_df.to_csv(OUT_DIR / "ensemble_regime_breakdown.csv", index=False)
    dsr_df = pd.DataFrame(dsr_rows)
    dsr_df.to_csv(OUT_DIR / "ensemble_dsr.csv", index=False)

    # ----------------------------------------------------------------------
    # D. Ensemble-level CSCV PBO over all 31 non-empty subsets of 5 cells.
    # ----------------------------------------------------------------------
    print("\n[3/4] Ensemble-level PBO (CSCV over 31 candidate ensembles) ...",
          flush=True)
    subsets = _all_subsets(len(ALL_CELLS))   # 31 subsets
    pbo_rows = []
    pick_rows = []
    for w in WINDOWS:
        print(f"  window={w}d  building 31 candidate ensembles ...", flush=True)
        summary, pick_freq, valid = _cscv_pbo(subsets, ALL_CELLS, close, w)
        if not summary:
            print(f"    [skip] no valid subsets at {w}d", flush=True)
            continue
        summary["window_days"] = w
        pbo_rows.append(summary)
        for sub, f in zip(valid, pick_freq):
            label = DEPLOYED_LABELS.get(frozenset(sub), "")
            cells_repr = "+".join(ALL_CELLS[i][0].replace("utc2130_", "") for i in sub)
            pick_rows.append({
                "window_days": w,
                "subset_size": len(sub),
                "subset_cells": cells_repr,
                "deployed_as": label,
                "pick_freq_pct": round(float(f) * 100, 2),
            })
        print(f"    PBO={summary['pbo']:.4f}  median ω={summary['median_omega']:.4f}  "
              f"IS SR={summary['mean_is_sr']:.4f}  OOS SR={summary['mean_oos_sr']:.4f}",
              flush=True)

    pbo_df = pd.DataFrame(pbo_rows)[["window_days", "pbo", "median_omega",
                                      "median_lambda", "mean_lambda",
                                      "mean_is_sr", "mean_oos_sr",
                                      "loss_in_translation",
                                      "n_strategies", "n_combos"]]
    pbo_df.to_csv(OUT_DIR / "ensemble_pbo.csv", index=False)
    pick_df = pd.DataFrame(pick_rows).sort_values(
        ["window_days", "pick_freq_pct"], ascending=[True, False])
    pick_df.to_csv(OUT_DIR / "ensemble_pbo_pick_frequency.csv", index=False)

    print("\n" + "=" * 78)
    print("Bootstrap CI")
    print("=" * 78)
    print(boot_df[[c for c in boot_df.columns
                   if c in ("ensemble", "window_days", "n_trades",
                            "win_rate_pct", "win_rate_lo_pct", "win_rate_hi_pct",
                            "obs_total_return_pct", "ci_total_return_lo",
                            "ci_total_return_hi", "obs_sharpe",
                            "ci_sharpe_lo", "ci_sharpe_hi",
                            "obs_max_dd_pct", "ci_max_dd_lo", "ci_max_dd_hi",
                            "obs_calmar", "ci_calmar_lo", "ci_calmar_hi")]
                  ].to_string(index=False))

    print("\n" + "=" * 78)
    print("Regime breakdown")
    print("=" * 78)
    print(regime_df.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"Deflated Sharpe Ratio (N={N_TRIALS_GLOBAL} trials)")
    print("=" * 78)
    print(dsr_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("Ensemble-level CSCV PBO (31 candidate subsets of 5 cells)")
    print("=" * 78)
    with pd.option_context("display.max_columns", None, "display.width", 220,
                           "display.float_format", "{:.4f}".format):
        print(pbo_df.to_string(index=False))
        print("\nTop pick frequencies (highest IS-Sharpe-best across CSCV folds):")
        for w in WINDOWS:
            sub = pick_df.loc[pick_df["window_days"] == w].head(8)
            if sub.empty:
                continue
            print(f"\n  -- window={w}d --")
            print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
