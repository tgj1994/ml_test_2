"""Per-cell threshold tuning subject to 'don't hurt MaxDD' constraint, then
aggregate to ensemble. Tests whether independent per-cell improvements
(found in the previous sweep) actually translate to ensemble-level gains
under the user's constraint that MaxDD must not get worse.

Strategy:
  1. For each cell, compute equity grid for (buy, sell) ∈ [0.30, 0.70] × [0.30, 0.70].
  2. For each cell, identify the production point and its (ret_prod, dd_prod).
  3. Pick the (buy, sell) that maximises that cell's RETURN
     subject to the cell's max_dd >= dd_prod (i.e., DD must not deteriorate)
     and considered separately on the 730d and 365d windows.
  4. Aggregate per-cell tuned equities into Top2/Top3/Top5 and report the
     ensemble's resulting return and DD vs the production ensemble.
  5. Also test 'jointly safe': pick threshold that does NOT hurt DD in
     EITHER 730d or 365d (intersection of safe regions across both windows).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "preds_cache_ebm_ebm_extended"
OUT = ROOT / "reports" / "utc2130_threshold_sweep"

CELLS = [
    ("sm_v5_dynk12",         "utc2130_sm_v5",          "dynk12", 0.63, 0.50),
    ("sm_v3_complete_th13",  "utc2130_sm_v3_complete", "th13",   0.68, 0.50),
    ("sm_dynk11",            "utc2130_sm",             "dynk11", 0.64, 0.50),
    ("sm_v3_th14",           "utc2130_sm_v3",          "th14",   0.66, 0.50),
    ("sm_v6_dynk14",         "utc2130_sm_v6",          "dynk14", 0.66, 0.50),
]
ENSEMBLES = {"Top2": [0,1], "Top3": [0,1,2], "Top5": [0,1,2,3,4]}
WINDOWS = (730, 365)
FEE = 0.001
STAKE_TOTAL = 100.0

BUYS  = np.round(np.arange(0.300, 0.7001, 0.005), 4)
SELLS = np.round(np.arange(0.300, 0.7001, 0.005), 4)


def load_aligned():
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
    aligned = []
    for s, cl in raw:
        s = s.reindex(common); cl = cl.reindex(common)
        aligned.append((s.to_numpy(np.float64), cl.to_numpy(np.float64)))
    return aligned, common


def cell_equity_grid(prob, close, buys, sells, stake=1.0):
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


def metrics(curve, init):
    final = curve[..., -1]
    ret = (final/init - 1) * 100
    peak = np.maximum.accumulate(curve, axis=-1)
    dd = (curve - peak) / peak
    mdd = dd.min(axis=-1) * 100
    return ret, mdd


def main():
    aligned, idx = load_aligned()
    summary_rows = []

    # Compute full equity grid for each cell on both windows once
    cell_grids = {730: [], 365: []}
    cell_prod_curve = {730: [], 365: []}
    cell_prod_metrics = {730: [], 365: []}
    bh = {}

    for win in WINDOWS:
        cutoff = idx.max() - pd.Timedelta(days=win)
        mask = idx >= cutoff
        sliced = [(p[mask], c[mask]) for p, c in aligned]
        bh[win] = (sliced[0][1][-1]/sliced[0][1][0] - 1) * 100

        print(f"\n=== Window {win}d (B&H={bh[win]:+.1f}%) ===")
        for ci, (prob, close) in enumerate(sliced):
            t0 = time.time()
            grid = cell_equity_grid(prob, close, BUYS, SELLS)
            cell_grids[win].append(grid)
            # production curve at exact (prod_buy, prod_sell)
            pb = CELLS[ci][3]; ps = CELLS[ci][4]
            ib = int(round((pb - 0.300)/0.005))
            jb = int(round((ps - 0.300)/0.005))
            prod_curve = grid[ib, jb, :].copy()
            cell_prod_curve[win].append(prod_curve)
            r, d = metrics(prod_curve[None, :], 1.0)
            cell_prod_metrics[win].append((float(r[0]), float(d[0])))
            print(f"   cell {ci} {CELLS[ci][0]:<22} prod ret={r[0]:+6.1f}% dd={d[0]:5.1f}%  "
                  f"[grid {time.time()-t0:.2f}s]")

    # ---- SCENARIO A: per-cell best return, single-window safety (DD must not worsen for that window) ----
    # ---- SCENARIO B: per-cell best return, JOINT safety (DD must not worsen in BOTH windows) ----
    # ---- SCENARIO C: per-cell best max-DD, with return >= prod_return ----

    print(f"\n{'='*100}")
    print("SCENARIO B (recommended): per-cell tune that does NOT hurt MaxDD in EITHER window")
    print(f"{'='*100}")

    # For each cell, find combos such that DD_730 >= prod_DD_730 AND DD_365 >= prod_DD_365.
    # Among those, pick the one that maximises some objective. We'll prepare 3 picks:
    #   (i)  max RETURN_730 (bull-favouring)
    #   (ii) max RETURN_365 (bear-favouring)
    #   (iii) max min(RETURN_730 - prod_730, RETURN_365 - prod_365) (jointly improving)
    cell_picks = {"max730":[], "max365":[], "joint":[], "tightDD":[]}
    for ci in range(len(CELLS)):
        ret_730, dd_730 = metrics(cell_grids[730][ci], 1.0)  # (Nb, Ns)
        ret_365, dd_365 = metrics(cell_grids[365][ci], 1.0)
        prod_ret_730, prod_dd_730 = cell_prod_metrics[730][ci]
        prod_ret_365, prod_dd_365 = cell_prod_metrics[365][ci]
        well = (BUYS[:,None] >= SELLS[None,:])
        safe = well & (dd_730 >= prod_dd_730) & (dd_365 >= prod_dd_365)
        n_safe = int(safe.sum())

        # (i) max return on 730d among safe
        r_v = np.where(safe, ret_730, -np.inf)
        ib, jb = np.unravel_index(np.argmax(r_v), r_v.shape)
        cell_picks["max730"].append((float(BUYS[ib]), float(SELLS[jb])))
        # (ii) max return on 365d among safe
        r_v = np.where(safe, ret_365, -np.inf)
        ib, jb = np.unravel_index(np.argmax(r_v), r_v.shape)
        cell_picks["max365"].append((float(BUYS[ib]), float(SELLS[jb])))
        # (iii) joint
        joint = np.minimum(ret_730 - prod_ret_730, ret_365 - prod_ret_365)
        joint = np.where(safe, joint, -np.inf)
        ib, jb = np.unravel_index(np.argmax(joint), joint.shape)
        cell_picks["joint"].append((float(BUYS[ib]), float(SELLS[jb])))
        # (iv) tightest DD (max DD value, i.e., smallest |DD|) among combos that don't lose return
        ret_safe = safe & (ret_730 >= prod_ret_730) & (ret_365 >= prod_ret_365)
        dd_min = np.maximum(dd_730, dd_365)  # worst-of dd across windows
        v = np.where(ret_safe, dd_min, -np.inf)
        if ret_safe.any():
            ib, jb = np.unravel_index(np.argmax(v), v.shape)
            cell_picks["tightDD"].append((float(BUYS[ib]), float(SELLS[jb])))
        else:
            # fall back to prod
            cell_picks["tightDD"].append((CELLS[ci][3], CELLS[ci][4]))

        print(f"  cell {ci} {CELLS[ci][0]:<22}  safe-region size = {n_safe:>4} / 6561  "
              f"(prod 730d ret/dd={prod_ret_730:+.0f}/{prod_dd_730:.0f},  "
              f"365d ret/dd={prod_ret_365:+.0f}/{prod_dd_365:.0f})")
        for label, picks in cell_picks.items():
            b, s = picks[ci]
            r730, d730 = metrics(cell_grids[730][ci][int(round((b-0.3)/0.005)),
                                                      int(round((s-0.3)/0.005)), :][None,:], 1.0)
            r365, d365 = metrics(cell_grids[365][ci][int(round((b-0.3)/0.005)),
                                                      int(round((s-0.3)/0.005)), :][None,:], 1.0)
            print(f"      pick[{label:7}]  buy={b:.3f} sell={s:.3f}  "
                  f"730d ret={float(r730[0]):+6.1f}%(Δ{float(r730[0])-prod_ret_730:+5.1f}) dd={float(d730[0]):5.1f}%  "
                  f"365d ret={float(r365[0]):+6.1f}%(Δ{float(r365[0])-prod_ret_365:+5.1f}) dd={float(d365[0]):5.1f}%")

    # Aggregate ensembles for each pick variant
    print(f"\n{'='*100}")
    print("ENSEMBLE result with per-cell tuned thresholds")
    print(f"{'='*100}")
    for ens_name, idxs in ENSEMBLES.items():
        N = len(idxs)
        stake = STAKE_TOTAL / N
        # production ensemble baseline
        for win in WINDOWS:
            prod_ens_curve = sum(cell_prod_curve[win][i] for i in idxs) * stake
            r, d = metrics(prod_ens_curve[None, :], STAKE_TOTAL)
            print(f"\n  {ens_name} {win}d  PROD ensemble  ret={float(r[0]):+.1f}%  dd={float(d[0]):.1f}%  "
                  f"(B&H {bh[win]:+.1f}%)")
            for variant in ("max730", "max365", "joint", "tightDD"):
                # Build ensemble with each cell at its picked (buy, sell)
                tuned_curves = []
                picks_str = []
                for ci in idxs:
                    b, s = cell_picks[variant][ci]
                    ib = int(round((b - 0.3)/0.005))
                    jb = int(round((s - 0.3)/0.005))
                    tuned_curves.append(cell_grids[win][ci][ib, jb, :])
                    picks_str.append(f"{CELLS[ci][0][:8]}({b:.3f}/{s:.3f})")
                ens_curve = sum(tuned_curves) * stake
                rr, dd = metrics(ens_curve[None, :], STAKE_TOTAL)
                rr = float(rr[0]); dd = float(dd[0])
                marker = " ✓" if (rr > float(r[0]) and dd >= float(d[0])) else ""
                print(f"     [{variant:7}] ret={rr:+.1f}% (Δ{rr-float(r[0]):+5.1f}pp)  "
                      f"dd={dd:.1f}% (Δ{dd-float(d[0]):+4.1f}pp){marker}")
                summary_rows.append({
                    "ensemble":ens_name, "window_d":win, "scenario":variant,
                    "prod_ret":round(float(r[0]),2), "prod_dd":round(float(d[0]),2),
                    "tuned_ret":round(rr,2), "tuned_dd":round(dd,2),
                    "delta_ret_pp":round(rr-float(r[0]),2), "delta_dd_pp":round(dd-float(d[0]),2),
                    "picks":" | ".join(picks_str),
                })

    pd.DataFrame(summary_rows).to_csv(OUT/"_ensemble_per_cell_tuned.csv", index=False)
    print(f"\nSaved: {OUT/'_ensemble_per_cell_tuned.csv'}")


if __name__ == "__main__":
    main()
