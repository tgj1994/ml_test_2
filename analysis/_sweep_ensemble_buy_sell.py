"""Ensemble (Top2/Top3/Top5) buy_TH × sell_TH sweep.

Mirrors the split-capital backtest convention: each component cell trades on
1/N of capital independently, ensemble equity = sum of sub-account equities.

Two sweep variants:
  A) UNIFORM   — same (buy_TH, sell_TH) applied to all cells in the ensemble
                 buy ∈ [0.30, 0.70] step 0.005 ; sell ∈ [0.30, 0.70] step 0.005
  B) DELTA     — each cell uses its production threshold + a uniform shift
                 (buy_delta, sell_delta) ∈ [-0.10, +0.10] step 0.005

For each combo we compute total_return_pct and max_dd_pct on the SUMMED equity
curve. Compared against PRODUCTION (each cell at its own prod_buy/prod_sell).

Outputs reports/utc2130_threshold_sweep/:
  ensemble_uniform_<top>_<win>d.csv
  ensemble_delta_<top>_<win>d.csv
  _ensemble_summary.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "preds_cache_ebm_ebm_extended"
OUT = ROOT / "reports" / "utc2130_threshold_sweep"
OUT.mkdir(parents=True, exist_ok=True)

CELLS = [
    # (key, variant, label, prod_buy, prod_sell)
    ("sm_v5_dynk12",         "utc2130_sm_v5",          "dynk12", 0.63, 0.50),
    ("sm_v3_complete_th13",  "utc2130_sm_v3_complete", "th13",   0.68, 0.50),
    ("sm_dynk11",            "utc2130_sm",             "dynk11", 0.64, 0.50),
    ("sm_v3_th14",           "utc2130_sm_v3",          "th14",   0.66, 0.50),
    ("sm_v6_dynk14",         "utc2130_sm_v6",          "dynk14", 0.66, 0.50),
]
ENSEMBLES = {
    "Top2": [0, 1],
    "Top3": [0, 1, 2],
    "Top5": [0, 1, 2, 3, 4],
}
WINDOWS = (730, 365)
FEE = 0.001
STAKE_TOTAL = 100.0  # ensemble notional; per-cell stake = STAKE_TOTAL / N

BUYS  = np.round(np.arange(0.300, 0.7001, 0.005), 4)
SELLS = np.round(np.arange(0.300, 0.7001, 0.005), 4)
BUY_DELTAS  = np.round(np.arange(-0.100, 0.1001, 0.005), 4)
SELL_DELTAS = np.round(np.arange(-0.100, 0.1001, 0.005), 4)


def load_all_cells():
    btc = pd.read_parquet(ROOT / "data" / "btc_1d.parquet").set_index("open_time")["close"]
    btc.index = pd.to_datetime(btc.index, utc=True)

    cell_data = []
    common_idx = None
    for key, variant, label, _, _ in CELLS:
        s = pd.read_parquet(CACHE / f"preds_ebm_{label}_{variant}.parquet")["prob_up"].dropna()
        s.index = pd.to_datetime(s.index, utc=True)
        cl = btc.reindex(s.index, method="nearest")
        s = s[cl.notna()]; cl = cl[s.index]
        cell_data.append((s, cl))
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)

    # restrict every cell to the same date set so ensemble summing aligns
    aligned = []
    for s, cl in cell_data:
        s = s.reindex(common_idx); cl = cl.reindex(common_idx)
        aligned.append((s.to_numpy(dtype=np.float64), cl.to_numpy(dtype=np.float64)))
    return aligned, common_idx


def cell_equity_grid(prob: np.ndarray, close: np.ndarray,
                     buys: np.ndarray, sells: np.ndarray,
                     stake: float) -> np.ndarray:
    """Vectorised: returns equity curves of shape (Nb, Ns, T) for all combos."""
    Nb, Ns, T = len(buys), len(sells), len(prob)
    in_pos = np.zeros((Nb, Ns), dtype=bool)
    qty = np.zeros((Nb, Ns))
    eq = np.full((Nb, Ns), stake, dtype=np.float64)
    eq_curves = np.empty((Nb, Ns, T), dtype=np.float64)

    buys_col = buys[:, None]
    sells_row = sells[None, :]

    for t in range(T):
        p = prob[t]; px = close[t]
        sell_sig = p < sells_row
        buy_sig  = p >= buys_col

        # SELL first
        sell_trans = in_pos & sell_sig
        if sell_trans.any():
            new_eq = qty * px * (1 - FEE)
            eq = np.where(sell_trans, new_eq, eq)
            in_pos = np.where(sell_trans, False, in_pos)
            qty = np.where(sell_trans, 0.0, qty)
        # BUY
        buy_trans = (~in_pos) & buy_sig
        if buy_trans.any():
            new_qty = eq * (1 - FEE) / px
            qty = np.where(buy_trans, new_qty, qty)
            eq = np.where(buy_trans, new_qty * px, eq)
            in_pos = np.where(buy_trans, True, in_pos)
        # mark to market
        eq = np.where(in_pos, qty * px, eq)
        eq_curves[:, :, t] = eq

    # final force-close (fee on last bar)
    if in_pos.any():
        final_eq = np.where(in_pos, qty * close[-1] * (1 - FEE), eq)
        eq_curves[:, :, -1] = np.where(in_pos, final_eq, eq_curves[:, :, -1])
    return eq_curves


def run_window(cells_aligned, idx, window_days: int):
    """Slice cell prob/close to last `window_days` and sweep."""
    cutoff = idx.max() - pd.Timedelta(days=window_days)
    mask = idx >= cutoff
    sliced = [(p[mask], c[mask]) for p, c in cells_aligned]
    win_idx = idx[mask]

    print(f"  window={window_days}d  T={mask.sum()} bars  "
          f"{win_idx.min().date()} → {win_idx.max().date()}")

    bh = (sliced[0][1][-1] / sliced[0][1][0] - 1) * 100  # all cells share same close

    # ---- UNIFORM sweep: all cells use same (buy, sell) ----
    print(f"  -- UNIFORM sweep across all cells ...", flush=True)
    t0 = time.time()
    cell_grids_uniform = []
    for cell_idx, (prob, close) in enumerate(sliced):
        N = 5  # always divide by max ensemble size? No: per-ensemble stake
        # we'll compute stake later by re-scaling per ensemble
        # For now use stake=1.0 per cell, then multiply by per-ensemble allocation
        grid = cell_equity_grid(prob, close, BUYS, SELLS, stake=1.0)
        cell_grids_uniform.append(grid)
    print(f"     {time.time()-t0:.1f}s", flush=True)

    # ---- DELTA sweep: each cell uses prod + delta ----
    print(f"  -- DELTA sweep around production ...", flush=True)
    t0 = time.time()
    cell_grids_delta = []
    for cell_idx, (prob, close) in enumerate(sliced):
        pb, ps = CELLS[cell_idx][3], CELLS[cell_idx][4]
        cell_buys  = np.round(pb + BUY_DELTAS, 4)
        cell_sells = np.round(ps + SELL_DELTAS, 4)
        grid = cell_equity_grid(prob, close, cell_buys, cell_sells, stake=1.0)
        cell_grids_delta.append(grid)
    print(f"     {time.time()-t0:.1f}s", flush=True)

    # ---- Production reference (each cell at its own prod) ----
    cell_eq_prod = []
    for cell_idx, (prob, close) in enumerate(sliced):
        pb, ps = CELLS[cell_idx][3], CELLS[cell_idx][4]
        bs = np.array([pb]); ss = np.array([ps])
        g = cell_equity_grid(prob, close, bs, ss, stake=1.0)
        cell_eq_prod.append(g[0, 0, :])  # 1D equity curve

    return (sliced, win_idx, bh, cell_grids_uniform, cell_grids_delta, cell_eq_prod)


def metrics_from_curve(curve: np.ndarray, init: float):
    final = curve[..., -1]
    total_ret = (final / init - 1) * 100
    peak = np.maximum.accumulate(curve, axis=-1)
    dd = (curve - peak) / peak
    max_dd = dd.min(axis=-1) * 100
    return total_ret, max_dd


def aggregate_ensemble(cell_grids, cell_indices, n_total_cells_for_stake=None):
    """Sum (Nb,Ns,T) grids across selected cells; per-cell stake = STAKE_TOTAL/N_ens."""
    selected = [cell_grids[i] for i in cell_indices]
    N = len(selected)
    stake = STAKE_TOTAL / N
    total_init = STAKE_TOTAL
    total_curve = sum(g for g in selected) * stake  # (Nb,Ns,T)
    return total_curve, total_init


def aggregate_prod(cell_eq_prod, cell_indices):
    selected = [cell_eq_prod[i] for i in cell_indices]
    N = len(selected)
    stake = STAKE_TOTAL / N
    total_curve = sum(g for g in selected) * stake
    return total_curve, STAKE_TOTAL


def main():
    aligned, idx = load_all_cells()
    summary_rows = []

    for win in WINDOWS:
        print(f"\n=== Window {win}d ===")
        sliced, win_idx, bh, grids_u, grids_d, eq_prod = run_window(aligned, idx, win)

        for ens_name, cell_idxs in ENSEMBLES.items():
            N = len(cell_idxs)
            # production ensemble metrics
            prod_curve, init = aggregate_prod(eq_prod, cell_idxs)
            prod_ret, prod_dd = metrics_from_curve(prod_curve[None, :], init)
            prod_ret = float(prod_ret[0]); prod_dd = float(prod_dd[0])

            # === UNIFORM ===
            uni_total, _ = aggregate_ensemble(grids_u, cell_idxs)
            uni_ret, uni_dd = metrics_from_curve(uni_total, STAKE_TOTAL)
            # save full grid
            rows = []
            for i, b in enumerate(BUYS):
                for j, s in enumerate(SELLS):
                    rows.append((round(float(b),4), round(float(s),4),
                                 float(uni_ret[i,j]), float(uni_dd[i,j]),
                                 float(uni_ret[i,j]) - bh))
            df_u = pd.DataFrame(rows, columns=["buy_TH","sell_TH","total_return_pct",
                                                "max_dd_pct","alpha_pp"])
            df_u.to_csv(OUT / f"ensemble_uniform_{ens_name}_{win}d.csv", index=False)

            # best return / best calmar / dominate-prod
            valid_u = (BUYS[:,None] >= SELLS[None,:])
            r_v = np.where(valid_u, uri := uni_ret, -np.inf)
            ib, jb = np.unravel_index(np.argmax(r_v), r_v.shape)
            best_u_b = float(BUYS[ib]); best_u_s = float(SELLS[jb])
            best_u_r = float(uni_ret[ib,jb]); best_u_d = float(uni_dd[ib,jb])

            # combos that strictly dominate prod (ret > prod_ret AND dd > prod_dd, less negative)
            dom_mask_u = valid_u & (uni_ret > prod_ret) & (uni_dd > prod_dd)
            n_dom_u = int(dom_mask_u.sum())
            if n_dom_u > 0:
                # pick the one with maximum (return - prod) + (dd - prod_dd) (joint improvement)
                joint = np.where(dom_mask_u, (uni_ret - prod_ret) + (uni_dd - prod_dd), -np.inf)
                ib2, jb2 = np.unravel_index(np.argmax(joint), joint.shape)
                dom_u_b = float(BUYS[ib2]); dom_u_s = float(SELLS[jb2])
                dom_u_r = float(uni_ret[ib2,jb2]); dom_u_d = float(uni_dd[ib2,jb2])
            else:
                dom_u_b = dom_u_s = dom_u_r = dom_u_d = float("nan")

            # === DELTA ===
            del_total, _ = aggregate_ensemble(grids_d, cell_idxs)
            del_ret, del_dd = metrics_from_curve(del_total, STAKE_TOTAL)
            rows = []
            for i, db in enumerate(BUY_DELTAS):
                for j, ds in enumerate(SELL_DELTAS):
                    rows.append((round(float(db),4), round(float(ds),4),
                                 float(del_ret[i,j]), float(del_dd[i,j]),
                                 float(del_ret[i,j]) - bh))
            df_d = pd.DataFrame(rows, columns=["buy_delta","sell_delta","total_return_pct",
                                                "max_dd_pct","alpha_pp"])
            df_d.to_csv(OUT / f"ensemble_delta_{ens_name}_{win}d.csv", index=False)

            # well-formedness: each cell must have prod_buy+db >= prod_sell+ds.
            # Conservative: db >= ds + max(ps_i - pb_i over selected cells).
            ds_lim = max((CELLS[i][4] - CELLS[i][3]) for i in cell_idxs)  # negative
            valid_d = (BUY_DELTAS[:,None] >= SELL_DELTAS[None,:] + ds_lim)
            r_vd = np.where(valid_d, del_ret, -np.inf)
            ib, jb = np.unravel_index(np.argmax(r_vd), r_vd.shape)
            best_d_db = float(BUY_DELTAS[ib]); best_d_ds = float(SELL_DELTAS[jb])
            best_d_r = float(del_ret[ib,jb]); best_d_d = float(del_dd[ib,jb])

            dom_mask_d = valid_d & (del_ret > prod_ret) & (del_dd > prod_dd)
            n_dom_d = int(dom_mask_d.sum())
            if n_dom_d > 0:
                joint = np.where(dom_mask_d, (del_ret - prod_ret) + (del_dd - prod_dd), -np.inf)
                ib2, jb2 = np.unravel_index(np.argmax(joint), joint.shape)
                dom_d_db = float(BUY_DELTAS[ib2]); dom_d_ds = float(SELL_DELTAS[jb2])
                dom_d_r = float(del_ret[ib2,jb2]); dom_d_d = float(del_dd[ib2,jb2])
            else:
                dom_d_db = dom_d_ds = dom_d_r = dom_d_d = float("nan")

            print(f"\n  {ens_name} {win}d  B&H={bh:+.1f}%  PROD: ret={prod_ret:+.1f}% dd={prod_dd:.1f}%")
            print(f"    UNIFORM  best_ret  ({best_u_b:.3f}, {best_u_s:.3f}) → {best_u_r:+.1f}% dd={best_u_d:.1f}% "
                  f"| dominate-prod combos: {n_dom_u}")
            if n_dom_u > 0:
                print(f"             best_joint ({dom_u_b:.3f}, {dom_u_s:.3f}) → {dom_u_r:+.1f}% (Δret={dom_u_r-prod_ret:+.1f}pp) "
                      f"dd={dom_u_d:.1f}% (Δdd={dom_u_d-prod_dd:+.1f}pp)")
            print(f"    DELTA    best_ret  (Δb={best_d_db:+.3f}, Δs={best_d_ds:+.3f}) → {best_d_r:+.1f}% dd={best_d_d:.1f}% "
                  f"| dominate-prod combos: {n_dom_d}")
            if n_dom_d > 0:
                print(f"             best_joint (Δb={dom_d_db:+.3f}, Δs={dom_d_ds:+.3f}) → {dom_d_r:+.1f}% (Δret={dom_d_r-prod_ret:+.1f}pp) "
                      f"dd={dom_d_d:.1f}% (Δdd={dom_d_d-prod_dd:+.1f}pp)")

            summary_rows.append({
                "ensemble": ens_name, "window_d": win, "bh_pct": round(bh,2),
                "prod_ret": round(prod_ret,2), "prod_dd": round(prod_dd,2),
                "uni_best_buy": best_u_b, "uni_best_sell": best_u_s,
                "uni_best_ret": round(best_u_r,2), "uni_best_dd": round(best_u_d,2),
                "uni_n_dominate": n_dom_u,
                "uni_dom_buy": dom_u_b, "uni_dom_sell": dom_u_s,
                "uni_dom_ret": round(dom_u_r,2) if n_dom_u else float("nan"),
                "uni_dom_dd": round(dom_u_d,2) if n_dom_u else float("nan"),
                "del_best_db": best_d_db, "del_best_ds": best_d_ds,
                "del_best_ret": round(best_d_r,2), "del_best_dd": round(best_d_d,2),
                "del_n_dominate": n_dom_d,
                "del_dom_db": dom_d_db, "del_dom_ds": dom_d_ds,
                "del_dom_ret": round(dom_d_r,2) if n_dom_d else float("nan"),
                "del_dom_dd": round(dom_d_d,2) if n_dom_d else float("nan"),
            })

    pd.DataFrame(summary_rows).to_csv(OUT / "_ensemble_summary.csv", index=False)
    print(f"\nSummary: {OUT/'_ensemble_summary.csv'}")


if __name__ == "__main__":
    main()
