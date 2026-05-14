"""Re-run the entire (buy, sell) sweep + ensemble + statistical validation
with the PAPER-CORRECT UTC 21:30-aligned close (build_utc2130_daily). The
previous analysis used data/btc_1d.parquet which is UTC 00:00 daily, and
nearest-reindex onto the prediction's 21:30 timestamp introduced look-ahead.

Steps:
  A. Per-cell DD-safe sweep over all 5 cells, both 730d and 365d.
  B. Identify joint-safe optimum per cell (DD must not worsen in either).
  C. Compute ensemble metrics (Top1/Top2/Top3/Top5) for:
       (i) production thresholds (baseline)
      (ii) per-cell joint-safe optimum
  D. Statistical validation for the changes that survive C.
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


from pathlib import Path
from itertools import combinations
from math import sqrt, log
import time

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import run_backtest
from src.utc2130 import build_utc2130_daily

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "preds_cache_ebm_ebm_extended"
OUT = ROOT / "reports" / "utc2130_threshold_sweep"
OUT.mkdir(parents=True, exist_ok=True)

# 5 production cells
CELLS = [
    ("sm_v5_dynk12",         "utc2130_sm_v5",          "dynk12", 0.63, 0.50),
    ("sm_v3_complete_th13",  "utc2130_sm_v3_complete", "th13",   0.68, 0.50),
    ("sm_dynk11",            "utc2130_sm",             "dynk11", 0.64, 0.50),
    ("sm_v3_th14",           "utc2130_sm_v3",          "th14",   0.66, 0.50),
    ("sm_v6_dynk14",         "utc2130_sm_v6",          "dynk14", 0.66, 0.50),
]
ENSEMBLES = {"Top1": [0], "Top2": [0, 1], "Top3": [0, 1, 2], "Top5": [0, 1, 2, 3, 4]}
WINDOWS = (730, 365)
FEE = 0.001
STAKE_TOTAL = 100.0

BUYS  = np.round(np.arange(0.300, 0.7001, 0.005), 4)
SELLS = np.round(np.arange(0.300, 0.7001, 0.005), 4)


def load():
    bar = build_utc2130_daily(ROOT/"data"/"btc_15m.parquet")
    close = bar.set_index("close_time")["close"]
    cells_data = []
    common = None
    for _, variant, label, _, _ in CELLS:
        s = pd.read_parquet(CACHE/f"preds_ebm_{label}_{variant}.parquet")["prob_up"].dropna()
        s.index = pd.to_datetime(s.index, utc=True)
        cl = close.reindex(s.index)
        s = s[cl.notna()]; cl = cl[s.index]
        cells_data.append((s, cl))
        common = s.index if common is None else common.intersection(s.index)
    aligned = []
    for s, cl in cells_data:
        s = s.reindex(common); cl = cl.reindex(common)
        aligned.append((s.to_numpy(np.float64), cl.to_numpy(np.float64)))
    return aligned, common, close


def cell_grid(prob, close, buys, sells, stake=1.0):
    Nb, Ns, T = len(buys), len(sells), len(prob)
    in_pos = np.zeros((Nb, Ns), dtype=bool)
    qty = np.zeros((Nb, Ns))
    eq = np.full((Nb, Ns), stake, dtype=np.float64)
    eqc = np.empty((Nb, Ns, T), dtype=np.float64)
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
            new_q = eq*(1-FEE)/px
            qty = np.where(buy_trans, new_q, qty)
            eq = np.where(buy_trans, new_q*px, eq)
            in_pos = np.where(buy_trans, True, in_pos)
        eq = np.where(in_pos, qty*px, eq)
        eqc[:, :, t] = eq
    if in_pos.any():
        feq = np.where(in_pos, qty*close[-1]*(1-FEE), eq)
        eqc[:, :, -1] = np.where(in_pos, feq, eqc[:, :, -1])
    return eqc


def metrics(eq, init):
    final = eq[..., -1]
    ret = (final/init - 1) * 100
    peak = np.maximum.accumulate(eq, axis=-1)
    dd = (eq - peak) / peak
    mdd = dd.min(axis=-1) * 100
    return ret, mdd


# ===========================================================================
# A. Per-cell DD-safe sweep
# ===========================================================================
def sweep_per_cell(aligned, idx):
    """For each cell, compute (Nb,Ns) return & DD on 730d and 365d windows
    using the same data slicing as the paper. Find joint-safe optima."""
    print(f"\n{'='*120}")
    print("A. Per-cell DD-safe sweep (paper-correct UTC2130 close)")
    print(f"{'='*120}")
    results = []  # one row per cell with all info
    grids = {}    # grids[cell_idx][win] = (ret, mdd)

    for win in WINDOWS:
        cutoff = idx.max() - pd.Timedelta(days=win)
        mask = idx >= cutoff
        for ci, (prob_full, close_full) in enumerate(aligned):
            prob = prob_full[mask]; close = close_full[mask]
            t0 = time.time()
            g = cell_grid(prob, close, BUYS, SELLS)
            r, dd = metrics(g, 1.0)
            grids.setdefault(ci, {})[win] = (r, dd)

    print(f"\n  {'cell':<24} {'730 prod ret/dd':>20} {'365 prod ret/dd':>20} "
          f"{'best joint-safe':>32} {'730/365 Δret pp':>22}")
    cell_picks = []
    for ci, (key, _, _, pb, ps) in enumerate(CELLS):
        r730, d730 = grids[ci][730]; r365, d365 = grids[ci][365]
        ib = int(round((pb-0.3)/0.005)); jb = int(round((ps-0.3)/0.005))
        prod_r730, prod_d730 = r730[ib,jb], d730[ib,jb]
        prod_r365, prod_d365 = r365[ib,jb], d365[ib,jb]
        well = (BUYS[:,None] >= SELLS[None,:])
        # joint safe: DD not worse than prod in BOTH windows (with 1bp tolerance)
        safe = well & (d730 >= prod_d730 - 0.01) & (d365 >= prod_d365 - 0.01)
        # both windows must improve (>0) for a "robust" pick
        improve_both = safe & (r730 > prod_r730) & (r365 > prod_r365)
        n_safe = int(safe.sum()); n_improve = int(improve_both.sum())
        # joint score
        joint = np.minimum(r730 - prod_r730, r365 - prod_r365)
        joint = np.where(improve_both, joint, -np.inf)
        if improve_both.any():
            i, j = np.unravel_index(np.argmax(joint), joint.shape)
            best_b, best_s = float(BUYS[i]), float(SELLS[j])
            best_r730, best_d730 = float(r730[i,j]), float(d730[i,j])
            best_r365, best_d365 = float(r365[i,j]), float(d365[i,j])
            tag = f"({best_b:.3f},{best_s:.3f})"
        else:
            best_b, best_s = pb, ps
            best_r730, best_d730 = float(prod_r730), float(prod_d730)
            best_r365, best_d365 = float(prod_r365), float(prod_d365)
            tag = "= PROD (no improve)"
        d730_pp = best_r730 - prod_r730
        d365_pp = best_r365 - prod_r365
        print(f"  {key:<24} {prod_r730:>+9.1f}/{prod_d730:>+5.1f}%  "
              f"{prod_r365:>+9.1f}/{prod_d365:>+5.1f}%  "
              f"{tag:>32} {d730_pp:>+8.1f} / {d365_pp:>+5.1f}")
        cell_picks.append({
            "cell": key, "prod_buy": pb, "prod_sell": ps,
            "best_buy": best_b, "best_sell": best_s,
            "prod_r730": prod_r730, "prod_d730": prod_d730,
            "prod_r365": prod_r365, "prod_d365": prod_d365,
            "best_r730": best_r730, "best_d730": best_d730,
            "best_r365": best_r365, "best_d365": best_d365,
            "n_safe": n_safe, "n_improve": n_improve,
        })
    return cell_picks, grids


# ===========================================================================
# B. Ensemble metrics with the picks
# ===========================================================================
def ensemble_with_picks(cell_picks, btc_close):
    """Build ensemble equity for production AND tuned thresholds, both windows."""
    print(f"\n{'='*120}")
    print("B. Ensemble metrics: production vs DD-safe per-cell tuning")
    print(f"{'='*120}")
    rows = []
    for ens_name, idxs in ENSEMBLES.items():
        N = len(idxs); stake = STAKE_TOTAL/N
        for win in WINDOWS:
            sub_eqs_prod = []; sub_eqs_tune = []
            trades_prod = []; trades_tune = []
            for ci in idxs:
                p = cell_picks[ci]
                lab = CELLS[ci][2]; var = CELLS[ci][1]
                preds = pd.read_parquet(CACHE/f"preds_ebm_{lab}_{var}.parquet")
                preds.index = pd.to_datetime(preds.index, utc=True)
                nn = preds.dropna(subset=["prob_up"])
                cutoff = nn.index.max() - pd.Timedelta(days=win)
                pp = nn.loc[nn.index >= cutoff]
                cl = btc_close.reindex(pp.index)
                # prod
                resp = run_backtest(pp, cl, stake_usd=stake,
                                    up_threshold=p["prod_buy"], down_threshold=p["prod_sell"])
                sub_eqs_prod.append(resp.equity_curve); trades_prod.extend(resp.trades)
                # tuned
                rest = run_backtest(pp, cl, stake_usd=stake,
                                    up_threshold=p["best_buy"], down_threshold=p["best_sell"])
                sub_eqs_tune.append(rest.equity_curve); trades_tune.extend(rest.trades)
            def agg(sub_eqs, trades):
                eq = sub_eqs[0]
                for s in sub_eqs[1:]: eq, s2 = eq.align(s, join="inner"); eq = eq + s2
                final = float(eq.iloc[-1])
                ret = (final/STAKE_TOTAL - 1) * 100
                peak = eq.cummax(); dd = (eq-peak)/peak
                mdd = float(dd.min()) * 100
                daily = eq.pct_change().dropna()
                sharpe = float(daily.mean()/daily.std()) * np.sqrt(365)
                sortino = (float(daily.mean()/daily[daily<0].std()) * np.sqrt(365)
                           if (daily<0).any() else float("nan"))
                calmar = ret/abs(mdd) if mdd<0 else float("nan")
                n_t = len(trades); n_w = sum(1 for t in trades if t.pnl_usd>0)
                sw = sum(t.pnl_usd for t in trades if t.pnl_usd>0)
                sl_ = sum(-t.pnl_usd for t in trades if t.pnl_usd<0)
                wr = n_w/max(n_t,1)*100
                pf = sw/sl_ if sl_>0 else float("inf")
                return dict(ret=ret, mdd=mdd, sharpe=sharpe, sortino=sortino,
                            calmar=calmar, n_trd=n_t, win_rate=wr, pf=pf)
            mp = agg(sub_eqs_prod, trades_prod); mt = agg(sub_eqs_tune, trades_tune)
            print(f"\n  {ens_name} {win}d  PROD: ret={mp['ret']:+.1f}% dd={mp['mdd']:.1f}% "
                  f"sharpe={mp['sharpe']:.2f} cal={mp['calmar']:.1f} wr={mp['win_rate']:.1f}% pf={mp['pf']:.1f} n={mp['n_trd']}")
            print(f"             TUNED: ret={mt['ret']:+.1f}% (Δ{mt['ret']-mp['ret']:+.1f}pp) "
                  f"dd={mt['mdd']:.1f}% (Δ{mt['mdd']-mp['mdd']:+.1f}pp) "
                  f"sharpe={mt['sharpe']:.2f} cal={mt['calmar']:.1f} wr={mt['win_rate']:.1f}% pf={mt['pf']:.1f} n={mt['n_trd']}")
            improved = (mt['ret'] > mp['ret']) and (mt['mdd'] >= mp['mdd'] - 0.01)
            if improved: print(f"             ✓ DD-safe improvement")
            rows.append({"ensemble":ens_name, "window":win,
                         "prod_ret":mp['ret'], "prod_dd":mp['mdd'],
                         "tuned_ret":mt['ret'], "tuned_dd":mt['mdd'],
                         "delta_ret":mt['ret']-mp['ret'], "delta_dd":mt['mdd']-mp['mdd']})
    return rows


# ===========================================================================
# C. Statistical validation for the change applied to Top5 (with paper close)
# ===========================================================================
def emax_factor(N):
    EULER = 0.5772156649015329
    if N <= 1: return 0.0
    return sqrt(2*log(N)) - (EULER + log(log(N))) / sqrt(2*log(N))


def sigma_sr(daily, periods_per_year=365):
    n = len(daily)
    sr_pp = daily.mean()/daily.std()
    sk = stats.skew(daily); kt = stats.kurtosis(daily) + 3
    var_pp = (1 - sk*sr_pp + ((kt-1)/4)*sr_pp**2)/(n-1)
    return sqrt(var_pp * periods_per_year)


def stationary_block_bootstrap(diff, n_iter=5000, mean_block=7, seed=123):
    n = len(diff); p = 1.0/mean_block; rng = np.random.default_rng(seed)
    means = np.empty(n_iter)
    for k in range(n_iter):
        out = np.empty(n); i = 0
        while i < n:
            s = rng.integers(0, n); blen = rng.geometric(p)
            for j in range(blen):
                if i >= n: break
                out[i] = diff[(s+j) % n]; i += 1
        means[k] = out.mean()
    return means


def stat_validate_top5(cell_picks, btc_close):
    """Run 5-test statistical validation on the new Top5 recommendation."""
    print(f"\n{'='*120}")
    print("C. Statistical validation of Top5 change (paper-correct close)")
    print(f"{'='*120}")
    sm_v6 = cell_picks[4]
    print(f"\n  sm_v6/dynk14: PROD ({sm_v6['prod_buy']:.3f}, {sm_v6['prod_sell']:.3f}) "
          f"→ TUNED ({sm_v6['best_buy']:.3f}, {sm_v6['best_sell']:.3f})")

    # Build Top5 ensemble equity for both
    def build_top5_eq(use_tuned: bool, win_days: int):
        N = 5; stake = STAKE_TOTAL/N
        sub_eqs = []
        for ci in range(5):
            p = cell_picks[ci]
            lab = CELLS[ci][2]; var = CELLS[ci][1]
            preds = pd.read_parquet(CACHE/f"preds_ebm_{lab}_{var}.parquet")
            preds.index = pd.to_datetime(preds.index, utc=True)
            nn = preds.dropna(subset=["prob_up"])
            cutoff = nn.index.max() - pd.Timedelta(days=win_days)
            pp = nn.loc[nn.index >= cutoff]
            cl = btc_close.reindex(pp.index)
            if use_tuned and ci == 4:
                b, s = p["best_buy"], p["best_sell"]
            else:
                b, s = p["prod_buy"], p["prod_sell"]
            res = run_backtest(pp, cl, stake_usd=stake, up_threshold=b, down_threshold=s)
            sub_eqs.append(res.equity_curve)
        eq = sub_eqs[0]
        for s_ in sub_eqs[1:]:
            eq, s2 = eq.align(s_, join="inner"); eq = eq + s2
        return eq

    # 1. Temporal stability: 730d split in 4 blocks
    print(f"\n  [1] Temporal stability — 730d split in 4 × 180d blocks (Top5)")
    eq_p = build_top5_eq(False, 730)
    eq_t = build_top5_eq(True,  730)
    common = eq_p.index.intersection(eq_t.index); eq_p = eq_p.reindex(common); eq_t = eq_t.reindex(common)
    T = len(common); BL = T//4
    blocks = [(k*BL, (k+1)*BL if k<3 else T) for k in range(4)]
    n_wins_block = 0
    for k,(a,b) in enumerate(blocks):
        rp = (float(eq_p.iloc[b-1])/float(eq_p.iloc[a]) - 1)*100
        rq = (float(eq_t.iloc[b-1])/float(eq_t.iloc[a]) - 1)*100
        winner = "TUNED" if rq > rp else "prod"
        n_wins_block += (rq > rp)
        print(f"      block{k+1} {common[a].date()}→{common[b-1].date()}: "
              f"PROD {rp:+5.1f}% TUNED {rq:+5.1f}% Δ{rq-rp:+.1f}pp [{winner}]")
    print(f"      → TUNED wins {n_wins_block}/4 blocks")

    # 2. Extended history (full 4y)
    print(f"\n  [2] Extended history — full 4y range (~{1520} bars)")
    eq_p_full = build_top5_eq(False, 365*5)
    eq_t_full = build_top5_eq(True,  365*5)
    common = eq_p_full.index.intersection(eq_t_full.index)
    eq_p_full = eq_p_full.reindex(common); eq_t_full = eq_t_full.reindex(common)
    rp = (float(eq_p_full.iloc[-1])/100 - 1)*100; rt = (float(eq_t_full.iloc[-1])/100 - 1)*100
    dp = ((eq_p_full - eq_p_full.cummax())/eq_p_full.cummax()).min()*100
    dt = ((eq_t_full - eq_t_full.cummax())/eq_t_full.cummax()).min()*100
    daily_p = eq_p_full.pct_change().dropna(); daily_t = eq_t_full.pct_change().dropna()
    sp = float(daily_p.mean()/daily_p.std())*sqrt(365)
    st = float(daily_t.mean()/daily_t.std())*sqrt(365)
    print(f"      PROD  ret={rp:+.1f}% dd={dp:.1f}% sharpe={sp:.2f}")
    print(f"      TUNED ret={rt:+.1f}% dd={dt:.1f}% sharpe={st:.2f}")
    print(f"      Δret={rt-rp:+.1f}pp Δdd={dt-dp:+.1f}pp Δsharpe={st-sp:+.2f}")

    # 3. Block bootstrap CI on daily diff
    print(f"\n  [3] Stationary block bootstrap on daily Δret (730d)")
    diff = (eq_t.pct_change() - eq_p.pct_change()).dropna().to_numpy()
    means = stationary_block_bootstrap(diff)
    lo, hi = np.percentile(means*100, [2.5, 97.5])
    obs = diff.mean()*100
    p_one = (means <= 0).mean() if obs > 0 else (means >= 0).mean()
    sig = "EXCLUDES 0 (significant)" if (lo > 0 or hi < 0) else "INCLUDES 0 (not sig)"
    print(f"      Daily mean Δret: {obs:+.4f}%/day, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"      one-sided p={p_one:.4f}  → CI {sig}")

    # 4. DSR apples-to-apples (both at same N)
    print(f"\n  [4] Deflated Sharpe — apples-to-apples DSR for PROD vs TUNED")
    daily_p = eq_p.pct_change().dropna()
    daily_t = eq_t.pct_change().dropna()
    sr_p = float(daily_p.mean()/daily_p.std()) * sqrt(365)
    sr_t = float(daily_t.mean()/daily_t.std()) * sqrt(365)
    sig_p = sigma_sr(daily_p.to_numpy()); sig_t = sigma_sr(daily_t.to_numpy())
    print(f"      PROD  Sharpe={sr_p:.3f}  σ_SR={sig_p:.3f}")
    print(f"      TUNED Sharpe={sr_t:.3f}  σ_SR={sig_t:.3f}  ΔSharpe={sr_t-sr_p:+.3f}")
    print(f"      {'N':<8} {'PROD z':>10} {'PROD p':>10} {'TUNED z':>10} {'TUNED p':>10}")
    for N in [1, 462, 2310, 3321, 6561]:
        zp = (sr_p - emax_factor(N)*sig_p)/sig_p; pp = 1 - stats.norm.cdf(zp)
        zt = (sr_t - emax_factor(N)*sig_t)/sig_t; pt = 1 - stats.norm.cdf(zt)
        print(f"      {N:<8} {zp:>+10.3f} {pp:>10.4f} {zt:>+10.3f} {pt:>10.4f}")

    # 5. CSCV pairwise PBO using cell-level grid (sm_v6 only — the cell that changes)
    print(f"\n  [5] CSCV pairwise PBO at sm_v6/dynk14 cell level")
    # We need cell-level prob+close for sm_v6, run grid, then test pairwise
    preds = pd.read_parquet(CACHE/"preds_ebm_dynk14_utc2130_sm_v6.parquet")
    preds.index = pd.to_datetime(preds.index, utc=True)
    s = preds["prob_up"].dropna()
    cl = btc_close.reindex(s.index); s = s[cl.notna()]; cl = cl[s.index]
    cutoff = s.index.max() - pd.Timedelta(days=730)
    s = s[s.index >= cutoff]; cl = cl[s.index]
    prob = s.to_numpy(); close = cl.to_numpy()
    g = cell_grid(prob, close, BUYS, SELLS)
    daily_log = np.diff(np.log(g), axis=-1)
    T = daily_log.shape[-1]
    S = 16; BL = T // S
    blocks = [(k*BL, (k+1)*BL if k<S-1 else T) for k in range(S)]
    splits = list(combinations(range(S), S//2))
    ib_p = int(round((sm_v6["prod_buy"]-0.3)/0.005)); jb_p = int(round((sm_v6["prod_sell"]-0.3)/0.005))
    ib_t = int(round((sm_v6["best_buy"]-0.3)/0.005)); jb_t = int(round((sm_v6["best_sell"]-0.3)/0.005))
    wins = 0
    for split in splits:
        oos_blocks = set(range(S)) - set(split)
        oos_idx = np.concatenate([np.arange(*blocks[k]) for k in oos_blocks])
        rp = np.exp(daily_log[ib_p, jb_p, oos_idx].sum()) - 1
        rt = np.exp(daily_log[ib_t, jb_t, oos_idx].sum()) - 1
        if rt > rp: wins += 1
    print(f"      TUNED beats PROD on OOS in {wins}/{len(splits)} splits = {wins/len(splits):.3f}")


def main():
    aligned, idx, btc_close = load()
    print(f"Loaded 5 cells, common idx {idx[0].date()} → {idx[-1].date()} ({len(idx)} bars)")

    cell_picks, grids = sweep_per_cell(aligned, idx)
    ensemble_rows = ensemble_with_picks(cell_picks, btc_close)
    pd.DataFrame(cell_picks).to_csv(OUT/"_per_cell_picks_correct_close.csv", index=False)
    pd.DataFrame(ensemble_rows).to_csv(OUT/"_ensemble_correct_close.csv", index=False)
    print(f"\nSaved: {OUT/'_per_cell_picks_correct_close.csv'}")
    print(f"Saved: {OUT/'_ensemble_correct_close.csv'}")

    stat_validate_top5(cell_picks, btc_close)


if __name__ == "__main__":
    main()
