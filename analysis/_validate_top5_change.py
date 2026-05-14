"""Statistical validation of the proposed Top5 change:
   sm_v6/dynk14:  (buy=0.66, sell=0.50)  →  (buy=0.675, sell=0.510)

The proposed combo was found by maximising 730d return over a 81×81 (buy,sell)
grid that also satisfied 'DD must not worsen in either window'. That selection
introduces look-ahead bias. This script runs four standard checks:

  1. Sub-period (temporal-fold) stability across 4 consecutive 180-day blocks.
  2. Extended-history out-of-sample check: full 4y record (1522 bars).
  3. Block-bootstrap 95% CI on the return-difference (proposed − production).
  4. CSCV PBO (Probability of Backtest Overfitting) on the (buy,sell) grid.
  5. Deflated Sharpe Ratio adjustment for the grid trial count.

Computed both at the cell level (sm_v6/dynk14) AND at the Top5 ensemble level.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from math import sqrt, exp, log

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "preds_cache_ebm_ebm_extended"

CELLS = [
    ("sm_v5_dynk12",         "utc2130_sm_v5",          "dynk12", 0.63, 0.50),
    ("sm_v3_complete_th13",  "utc2130_sm_v3_complete", "th13",   0.68, 0.50),
    ("sm_dynk11",            "utc2130_sm",             "dynk11", 0.64, 0.50),
    ("sm_v3_th14",           "utc2130_sm_v3",          "th14",   0.66, 0.50),
    ("sm_v6_dynk14",         "utc2130_sm_v6",          "dynk14", 0.66, 0.50),
]
TARGET_CELL = 4
PROD = (0.66, 0.50)
PROPOSED = (0.675, 0.510)
FEE = 0.001
STAKE_TOTAL = 100.0

BUYS  = np.round(np.arange(0.300, 0.7001, 0.005), 4)
SELLS = np.round(np.arange(0.300, 0.7001, 0.005), 4)
N_TRIALS = (BUYS[:, None] >= SELLS[None, :]).sum()  # well-formed combos
print(f"Grid trial count (well-formed only) = {int(N_TRIALS)}")

RNG = np.random.default_rng(42)


# ============================================================================
# Backtest engine
# ============================================================================
def cell_eq_one(prob, close, buy_TH, sell_TH, stake=1.0):
    """Single (buy, sell) backtest, returns equity curve and daily return series."""
    T = len(prob)
    in_pos = False
    qty = 0.0
    eq = stake
    eq_curve = np.empty(T)
    daily_ret = np.zeros(T)
    prev_eq = stake
    for t in range(T):
        p = prob[t]; px = close[t]
        if in_pos and p < sell_TH:
            eq = qty * px * (1 - FEE)
            in_pos = False; qty = 0.0
        if (not in_pos) and p >= buy_TH:
            qty = eq * (1 - FEE) / px
            eq = qty * px
            in_pos = True
        if in_pos:
            eq = qty * px
        eq_curve[t] = eq
        daily_ret[t] = eq / prev_eq - 1.0 if prev_eq > 0 else 0.0
        prev_eq = eq
    if in_pos:
        eq_curve[-1] = qty * close[-1] * (1 - FEE)
        daily_ret[-1] = eq_curve[-1] / prev_eq - 1.0
    return eq_curve, daily_ret


def cell_eq_grid(prob, close, buys, sells, stake=1.0):
    Nb, Ns, T = len(buys), len(sells), len(prob)
    in_pos = np.zeros((Nb, Ns), dtype=bool)
    qty = np.zeros((Nb, Ns))
    eq = np.full((Nb, Ns), stake, dtype=np.float64)
    eq_curves = np.empty((Nb, Ns, T), dtype=np.float64)
    bs = buys[:, None]; ss = sells[None, :]
    for t in range(T):
        p = prob[t]; px = close[t]
        sell_sig = p < ss; buy_sig = p >= bs
        sell_trans = in_pos & sell_sig
        if sell_trans.any():
            eq = np.where(sell_trans, qty*px*(1-FEE), eq)
            in_pos = np.where(sell_trans, False, in_pos)
            qty = np.where(sell_trans, 0.0, qty)
        buy_trans = (~in_pos) & buy_sig
        if buy_trans.any():
            new_qty = eq*(1-FEE)/px
            qty = np.where(buy_trans, new_qty, qty)
            eq = np.where(buy_trans, new_qty*px, eq)
            in_pos = np.where(buy_trans, True, in_pos)
        eq = np.where(in_pos, qty*px, eq)
        eq_curves[:, :, t] = eq
    if in_pos.any():
        final_eq = np.where(in_pos, qty*close[-1]*(1-FEE), eq)
        eq_curves[:, :, -1] = np.where(in_pos, final_eq, eq_curves[:, :, -1])
    return eq_curves


def metrics_from_eq(eq, init):
    final = eq[-1] if eq.ndim == 1 else eq[..., -1]
    ret = (final / init - 1) * 100
    peak = np.maximum.accumulate(eq, axis=-1)
    dd = (eq - peak) / peak
    mdd = dd.min(axis=-1) * 100
    daily_ret = np.diff(eq, axis=-1) / eq[..., :-1]
    sharpe_ann = (daily_ret.mean(axis=-1) / (daily_ret.std(axis=-1) + 1e-12)) * sqrt(365)
    return ret, mdd, sharpe_ann


# ============================================================================
# Load cell + ensemble data
# ============================================================================
def load_cells():
    btc = pd.read_parquet(ROOT/"data"/"btc_1d.parquet").set_index("open_time")["close"]
    btc.index = pd.to_datetime(btc.index, utc=True)
    raw = []
    common = None
    for _, variant, label, _, _ in CELLS:
        s = pd.read_parquet(CACHE/f"preds_ebm_{label}_{variant}.parquet")["prob_up"].dropna()
        s.index = pd.to_datetime(s.index, utc=True)
        cl = btc.reindex(s.index, method="nearest")
        s = s[cl.notna()]; cl = cl[s.index]
        raw.append((s, cl))
        common = s.index if common is None else common.intersection(s.index)
    out = []
    for s, cl in raw:
        s = s.reindex(common); cl = cl.reindex(common)
        out.append((s.to_numpy(np.float64), cl.to_numpy(np.float64)))
    return out, common


def slice_window(aligned, idx, days):
    cutoff = idx.max() - pd.Timedelta(days=days)
    mask = idx >= cutoff
    return [(p[mask], c[mask]) for p, c in aligned], idx[mask]


# ============================================================================
# 1. Temporal stability: 4 consecutive 180-day blocks within 730d
# ============================================================================
def test_temporal_stability(aligned, idx):
    print(f"\n{'='*100}\n[1] Temporal stability — 4 consecutive 180-day blocks within 730d\n{'='*100}")
    win730, idx730 = slice_window(aligned, idx, 730)
    T = len(idx730)
    BLOCK = T // 4

    print(f"\n  Block edges:")
    blocks = []
    for k in range(4):
        a = k * BLOCK
        b = (k+1) * BLOCK if k < 3 else T
        d_a, d_b = idx730[a].date(), idx730[b-1].date()
        print(f"    block{k+1}: idx[{a}:{b}]  {d_a} → {d_b}")
        blocks.append((a, b))

    # Cell sm_v6/dynk14 sub-block returns at PROD vs PROPOSED
    prob, close = win730[TARGET_CELL]

    print(f"\n  --- Cell sm_v6/dynk14 ---")
    print(f"  {'block':<8} {'PROD ret':>10} {'PROD DD':>9} {'PROP ret':>10} {'PROP DD':>9} {'Δret':>9} {'winner'}")
    cell_wins = []
    for k, (a, b) in enumerate(blocks):
        eq_p, _ = cell_eq_one(prob[a:b], close[a:b], *PROD)
        eq_q, _ = cell_eq_one(prob[a:b], close[a:b], *PROPOSED)
        rp, ddp, _ = metrics_from_eq(eq_p, 1.0)
        rq, ddq, _ = metrics_from_eq(eq_q, 1.0)
        winner = "PROPOSED" if rq > rp else "prod"
        cell_wins.append(rq > rp)
        print(f"  block{k+1:<3} {rp:>+9.1f}% {ddp:>+8.1f}% {rq:>+9.1f}% {ddq:>+8.1f}% {rq-rp:>+8.1f}pp  {winner}")
    n_wins = sum(cell_wins)
    print(f"  → PROPOSED wins {n_wins}/4 blocks at the cell level")

    # Top5 ensemble sub-block: each cell at its own (prod or proposed for sm_v6)
    print(f"\n  --- Top5 ensemble (only sm_v6 changes; others = prod) ---")
    print(f"  {'block':<8} {'PROD ret':>10} {'PROD DD':>9} {'PROP ret':>10} {'PROP DD':>9} {'Δret':>9} {'winner'}")
    ens_wins = []
    for k, (a, b) in enumerate(blocks):
        prod_eq = np.zeros(b - a)
        prop_eq = np.zeros(b - a)
        stake = STAKE_TOTAL / 5
        for ci in range(5):
            cprob, cclose = win730[ci][0][a:b], win730[ci][1][a:b]
            cb, cs = CELLS[ci][3], CELLS[ci][4]
            ep, _ = cell_eq_one(cprob, cclose, cb, cs, stake=stake)
            prod_eq += ep
            if ci == TARGET_CELL:
                eq_qq, _ = cell_eq_one(cprob, cclose, *PROPOSED, stake=stake)
                prop_eq += eq_qq
            else:
                prop_eq += ep
        rp, ddp, _ = metrics_from_eq(prod_eq, STAKE_TOTAL)
        rq, ddq, _ = metrics_from_eq(prop_eq, STAKE_TOTAL)
        winner = "PROPOSED" if rq > rp else "prod"
        ens_wins.append(rq > rp)
        print(f"  block{k+1:<3} {rp:>+9.1f}% {ddp:>+8.1f}% {rq:>+9.1f}% {ddq:>+8.1f}% {rq-rp:>+8.1f}pp  {winner}")
    print(f"  → PROPOSED wins {sum(ens_wins)}/4 blocks at the ensemble level")
    return n_wins, sum(ens_wins)


# ============================================================================
# 2. Extended history: 4y vs 730d
# ============================================================================
def test_extended_history(aligned, idx):
    print(f"\n{'='*100}\n[2] Extended history — does the proposed combo persist over 4y (1522 bars)?\n{'='*100}")
    full_win = aligned  # full data (1522 bars)
    print(f"  Full history: {idx[0].date()} → {idx[-1].date()}  ({len(idx)} bars)")

    prob, close = full_win[TARGET_CELL]
    eq_p, _ = cell_eq_one(prob, close, *PROD)
    eq_q, _ = cell_eq_one(prob, close, *PROPOSED)
    rp, ddp, sp = metrics_from_eq(eq_p, 1.0)
    rq, ddq, sq = metrics_from_eq(eq_q, 1.0)
    print(f"\n  Cell sm_v6/dynk14:")
    print(f"    PROD     ({PROD[0]}, {PROD[1]}):  ret {rp:+.1f}%  DD {ddp:+.1f}%  Sharpe {sp:.2f}")
    print(f"    PROPOSED ({PROPOSED[0]}, {PROPOSED[1]}):  ret {rq:+.1f}%  DD {ddq:+.1f}%  Sharpe {sq:.2f}")
    print(f"    → Δret = {rq-rp:+.1f}pp,  Δdd = {ddq-ddp:+.1f}pp,  Δsharpe = {sq-sp:+.2f}")

    # Top5
    print(f"\n  Top5 ensemble (only sm_v6 changes):")
    stake = STAKE_TOTAL / 5
    prod_eq = np.zeros(len(idx))
    prop_eq = np.zeros(len(idx))
    for ci in range(5):
        cprob, cclose = full_win[ci]
        cb, cs = CELLS[ci][3], CELLS[ci][4]
        ep, _ = cell_eq_one(cprob, cclose, cb, cs, stake=stake)
        prod_eq += ep
        if ci == TARGET_CELL:
            eq_qq, _ = cell_eq_one(cprob, cclose, *PROPOSED, stake=stake)
            prop_eq += eq_qq
        else:
            prop_eq += ep
    rp, ddp, sp = metrics_from_eq(prod_eq, STAKE_TOTAL)
    rq, ddq, sq = metrics_from_eq(prop_eq, STAKE_TOTAL)
    print(f"    PROD     ret {rp:+.1f}%  DD {ddp:+.1f}%  Sharpe {sp:.2f}")
    print(f"    PROPOSED ret {rq:+.1f}%  DD {ddq:+.1f}%  Sharpe {sq:.2f}")
    print(f"    → Δret = {rq-rp:+.1f}pp,  Δdd = {ddq-ddp:+.1f}pp,  Δsharpe = {sq-sp:+.2f}")
    return rq - rp


# ============================================================================
# 3. Block bootstrap CI on Δreturn
# ============================================================================
def stationary_block_bootstrap(daily_diff, n_iter=5000, mean_block=7):
    """Politis-Romano stationary bootstrap on a 1D series.
    Returns array of bootstrap-resampled means."""
    n = len(daily_diff)
    p = 1.0 / mean_block
    rng = np.random.default_rng(123)
    means = np.empty(n_iter)
    for k in range(n_iter):
        starts = rng.integers(0, n, size=n)
        # build series of blocks until length n; here we sample with random
        # block lengths ~ Geom(p) glued from random starts
        out = np.empty(n)
        i = 0
        while i < n:
            s = rng.integers(0, n)
            blen = rng.geometric(p)
            for j in range(blen):
                if i >= n: break
                out[i] = daily_diff[(s + j) % n]
                i += 1
        means[k] = out.mean()
    return means


def test_bootstrap(aligned, idx):
    print(f"\n{'='*100}\n[3] Stationary block bootstrap — 95% CI on daily Δreturn (proposed - prod)\n{'='*100}")
    win730, idx730 = slice_window(aligned, idx, 730)
    prob, close = win730[TARGET_CELL]

    eq_p, _ = cell_eq_one(prob, close, *PROD)
    eq_q, _ = cell_eq_one(prob, close, *PROPOSED)
    daily_p = np.diff(eq_p) / eq_p[:-1]
    daily_q = np.diff(eq_q) / eq_q[:-1]
    diff = daily_q - daily_p
    obs_mean_pp_per_day = diff.mean() * 100
    obs_total_pp = (eq_q[-1]/eq_p[-1] - 1) * 100  # not exactly but close

    print(f"\n  Cell sm_v6/dynk14, 730d:")
    print(f"    Daily Δret (proposed − prod): mean = {obs_mean_pp_per_day:+.4f}%/day  "
          f"std = {diff.std()*100:.3f}%  n = {len(diff)}")
    means = stationary_block_bootstrap(diff, n_iter=5000, mean_block=7)
    lo, hi = np.percentile(means*100, [2.5, 97.5])
    p_value = (means <= 0).mean() if obs_mean_pp_per_day > 0 else (means >= 0).mean()
    print(f"    Bootstrap 95% CI on daily mean Δret: [{lo:+.4f}, {hi:+.4f}] %/day")
    print(f"    p-value (one-sided, H0: no improvement): {p_value:.4f}")
    print(f"    → CI {'EXCLUDES' if (lo > 0 or hi < 0) else 'INCLUDES'} 0  "
          f"({'significant' if (lo > 0 or hi < 0) else 'NOT significant'})")

    # Now ensemble level
    stake = STAKE_TOTAL / 5
    win730_full = win730
    prod_eq = np.zeros(len(idx730))
    prop_eq = np.zeros(len(idx730))
    for ci in range(5):
        cprob, cclose = win730_full[ci]
        cb, cs = CELLS[ci][3], CELLS[ci][4]
        ep, _ = cell_eq_one(cprob, cclose, cb, cs, stake=stake)
        prod_eq += ep
        if ci == TARGET_CELL:
            eqq, _ = cell_eq_one(cprob, cclose, *PROPOSED, stake=stake)
            prop_eq += eqq
        else:
            prop_eq += ep
    daily_p = np.diff(prod_eq) / prod_eq[:-1]
    daily_q = np.diff(prop_eq) / prop_eq[:-1]
    diff = daily_q - daily_p
    obs_mean_pp_per_day = diff.mean() * 100

    print(f"\n  Top5 ensemble, 730d:")
    print(f"    Daily Δret: mean = {obs_mean_pp_per_day:+.4f}%/day")
    means = stationary_block_bootstrap(diff, n_iter=5000, mean_block=7)
    lo, hi = np.percentile(means*100, [2.5, 97.5])
    p_value = (means <= 0).mean() if obs_mean_pp_per_day > 0 else (means >= 0).mean()
    print(f"    Bootstrap 95% CI: [{lo:+.4f}, {hi:+.4f}] %/day")
    print(f"    p-value (one-sided): {p_value:.4f}")
    print(f"    → CI {'EXCLUDES' if (lo > 0 or hi < 0) else 'INCLUDES'} 0")
    return p_value


# ============================================================================
# 4. Deflated Sharpe Ratio
# ============================================================================
def deflated_sharpe(daily_returns, n_trials):
    """Bailey & Lopez de Prado 2014. Returns DSR z-score and p-value."""
    n = len(daily_returns)
    sr_obs = (daily_returns.mean() / (daily_returns.std() + 1e-12)) * sqrt(365)
    # Higher moments
    skew = stats.skew(daily_returns)
    kurt = stats.kurtosis(daily_returns) + 3  # excess → raw
    # Variance of SR estimator (annualised)
    var_sr = (1 - skew * sr_obs / sqrt(365) +
              ((kurt - 1)/4) * (sr_obs / sqrt(365))**2) / (n - 1)
    sr_var_annual = var_sr * 365
    sr_std_annual = sqrt(sr_var_annual)

    # Expected max SR under null across n_trials
    EULER = 0.5772156649015329
    if n_trials > 1:
        emax = sqrt(2 * log(n_trials)) - (EULER + log(log(n_trials))) / sqrt(2 * log(n_trials))
    else:
        emax = 0.0
    sr_threshold_annual = emax * sr_std_annual
    dsr_z = (sr_obs - sr_threshold_annual) / sr_std_annual
    p = 1 - stats.norm.cdf(dsr_z)
    return sr_obs, sr_threshold_annual, dsr_z, p


def test_dsr(aligned, idx):
    print(f"\n{'='*100}\n[4] Deflated Sharpe Ratio — adjust for 6,561-trial selection bias\n{'='*100}")
    win730, _ = slice_window(aligned, idx, 730)
    prob, close = win730[TARGET_CELL]
    n_trials_well = int(N_TRIALS)  # well-formed half of 81×81

    for label, (b, s) in [("PROD", PROD), ("PROPOSED", PROPOSED)]:
        eq, _ = cell_eq_one(prob, close, b, s)
        daily = np.diff(eq) / eq[:-1]
        # for PROD use n_trials=1 (not selected); for PROPOSED use grid trial count
        nt = n_trials_well if label == "PROPOSED" else 1
        sr, thr, z, p = deflated_sharpe(daily, n_trials=nt)
        print(f"\n  Cell sm_v6/dynk14 — {label} (b={b},s={s})  n_trials_for_DSR={nt}")
        print(f"    Observed Sharpe (annualised): {sr:.3f}")
        print(f"    Expected-max Sharpe under null across {nt} trials: {thr:.3f}")
        print(f"    DSR z-score: {z:.3f}    p-value: {p:.4f}")
        print(f"    → {'PASS (genuinely better than chance)' if z > 0 and p < 0.05 else 'FAIL or marginal'}")


# ============================================================================
# 5. CSCV PBO
# ============================================================================
def test_pbo_cscv(aligned, idx):
    print(f"\n{'='*100}\n[5] CSCV PBO — Probability of Backtest Overfitting on the (buy,sell) grid\n{'='*100}")
    print("    S=16 blocks, all C(16,8)=12,870 IS/OOS splits, performance metric = total return.")
    win730, idx730 = slice_window(aligned, idx, 730)
    prob, close = win730[TARGET_CELL]
    T = len(prob)
    grid = cell_eq_grid(prob, close, BUYS, SELLS)  # (Nb, Ns, T)
    # Restrict to well-formed combos
    Nb, Ns, _ = grid.shape
    well = (BUYS[:, None] >= SELLS[None, :])
    valid_idx = np.argwhere(well)  # (M, 2)
    M = len(valid_idx)
    # Daily log-returns per combo
    daily_log = np.diff(np.log(grid), axis=-1)  # (Nb, Ns, T-1)

    S = 16
    block_len = (T - 1) // S  # operate on daily returns
    blocks = [(k*block_len, (k+1)*block_len if k < S-1 else (T-1)) for k in range(S)]
    splits = list(combinations(range(S), S//2))
    print(f"    Block length ≈ {block_len} bars per block; {len(splits)} IS/OOS splits.")

    pbo_count = 0
    rank_logits = []
    for split in splits:
        is_blocks = set(split)
        oos_blocks = set(range(S)) - is_blocks
        is_idx = np.concatenate([np.arange(*blocks[k]) for k in is_blocks])
        oos_idx = np.concatenate([np.arange(*blocks[k]) for k in oos_blocks])
        # Cum return per combo on each fold (across valid combos only)
        is_returns = np.exp(daily_log[:, :, is_idx].sum(axis=-1)) - 1
        oos_returns = np.exp(daily_log[:, :, oos_idx].sum(axis=-1)) - 1
        # mask to valid
        is_valid = np.where(well, is_returns, -np.inf)
        # best combo on IS
        best_idx = np.unravel_index(np.argmax(is_valid), is_valid.shape)
        # rank that combo in OOS
        oos_flat = oos_returns[well]  # M-length
        # find position in oos_flat that corresponds to (best_idx[0], best_idx[1])
        # need to map (i,j) to a position in valid_idx order
        match = (valid_idx[:, 0] == best_idx[0]) & (valid_idx[:, 1] == best_idx[1])
        if not match.any():
            continue
        pos = np.where(match)[0][0]
        # rank: 0 = worst, M-1 = best
        rank = (oos_flat <= oos_flat[pos]).sum() - 1  # zero-based
        relative = rank / (M - 1)  # 0 to 1
        # Logit-transform: log(rel / (1-rel)). PBO measures fraction with logit < 0
        eps = 1e-6
        rel = max(eps, min(1 - eps, relative))
        logit = log(rel / (1 - rel))
        rank_logits.append(logit)
        if logit < 0:
            pbo_count += 1
    pbo = pbo_count / len(rank_logits) if rank_logits else 1.0
    print(f"\n    PBO = {pbo:.3f}  (best-IS combo lands in OOS bottom-half on {pbo_count}/{len(rank_logits)} splits)")
    print(f"    → {'PASS (PBO < 0.5: deployed-style ensembles generalise)' if pbo < 0.5 else 'FAIL (overfit)'}")

    # Also report PBO for the SPECIFIC pair (PROD, PROPOSED): how often does PROPOSED beat PROD on OOS
    ib_p = int(round((PROD[0] - 0.300)/0.005))
    jb_p = int(round((PROD[1] - 0.300)/0.005))
    ib_q = int(round((PROPOSED[0] - 0.300)/0.005))
    jb_q = int(round((PROPOSED[1] - 0.300)/0.005))
    wins = 0; total = 0
    for split in splits:
        is_blocks = set(split)
        oos_blocks = set(range(S)) - is_blocks
        oos_idx = np.concatenate([np.arange(*blocks[k]) for k in oos_blocks])
        oos_p = np.exp(daily_log[ib_p, jb_p, oos_idx].sum()) - 1
        oos_q = np.exp(daily_log[ib_q, jb_q, oos_idx].sum()) - 1
        if oos_q > oos_p:
            wins += 1
        total += 1
    print(f"\n    Pairwise PBO check: PROPOSED beats PROD on OOS in {wins}/{total} splits = {wins/total:.3f}")


# ============================================================================
# Main
# ============================================================================
def main():
    aligned, idx = load_cells()
    print(f"\nLoaded {len(CELLS)} cells. Common idx: {idx[0].date()} → {idx[-1].date()}  "
          f"({len(idx)} bars)")
    print(f"Production for sm_v6/dynk14: (buy={PROD[0]}, sell={PROD[1]})")
    print(f"Proposed   for sm_v6/dynk14: (buy={PROPOSED[0]}, sell={PROPOSED[1]})")

    test_temporal_stability(aligned, idx)
    test_extended_history(aligned, idx)
    test_bootstrap(aligned, idx)
    test_dsr(aligned, idx)
    test_pbo_cscv(aligned, idx)


if __name__ == "__main__":
    main()
