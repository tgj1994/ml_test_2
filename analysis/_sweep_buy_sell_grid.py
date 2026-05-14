"""Fine-grained 2D (buy_TH, sell_TH) sweep over the 5 production cells used in
coin_service_utc2130. Buy and sell thresholds each step from 0.30 to 0.70
in 0.005 increments → 81 × 81 = 6,561 combos per (cell × window).

For each combo, vectorised state-machine backtest over the cell's daily
prob_up cache and BTC close, using the same 3-state semantics as
src/backtest.run_backtest:
    prob >= buy_TH  → BUY  (enter if flat)
    prob <  sell_TH → SELL (exit if in pos)
    else            → HOLD

When buy_TH < sell_TH (HOLD inverts), the original convention is preserved:
sell fires first, then buy on the same bar → round-trip every active day.
The summary tables filter for the well-formed half (buy_TH ≥ sell_TH).

Outputs:
  reports/utc2130_threshold_sweep/<cell>_<win>.csv  — full grid
  reports/utc2130_threshold_sweep/_top_per_cell.csv — top combos per cell
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

# 5 production cells from coin_service_utc2130/backend/app/config.py
CELLS = [
    # (cell key, variant, label_slug, prod_buy, prod_sell)
    ("sm_v5_dynk12",         "utc2130_sm_v5",          "dynk12", 0.63, 0.50),
    ("sm_v3_complete_th13",  "utc2130_sm_v3_complete", "th13",   0.68, 0.50),
    ("sm_dynk11",            "utc2130_sm",             "dynk11", 0.64, 0.50),
    ("sm_v3_th14",           "utc2130_sm_v3",          "th14",   0.66, 0.50),
    ("sm_v6_dynk14",         "utc2130_sm_v6",          "dynk14", 0.66, 0.50),
]
WINDOWS = (730, 365)
FEE = 0.001  # 10 bps per side, matches backtest.py

BUYS  = np.round(np.arange(0.300, 0.7001, 0.005), 4)
SELLS = np.round(np.arange(0.300, 0.7001, 0.005), 4)


def load_btc() -> pd.Series:
    s = pd.read_parquet(ROOT / "data" / "btc_1d.parquet").set_index("open_time")["close"]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def sweep_cell(prob: np.ndarray, close: np.ndarray, buys: np.ndarray, sells: np.ndarray):
    """Vectorised 3-state backtest across the (Nb, Ns) threshold grid.
    Returns four (Nb, Ns) arrays: total_return_%, n_trades, win_rate, max_dd_%.
    """
    Nb, Ns = len(buys), len(sells)
    T = len(prob)
    in_pos = np.zeros((Nb, Ns), dtype=bool)
    qty = np.zeros((Nb, Ns))
    eq = np.ones((Nb, Ns))
    entry_eq = np.zeros((Nb, Ns))
    n_trades = np.zeros((Nb, Ns), dtype=np.int32)
    n_wins = np.zeros((Nb, Ns), dtype=np.int32)
    days_im = np.zeros((Nb, Ns), dtype=np.int32)
    peak = np.ones((Nb, Ns))
    max_dd = np.zeros((Nb, Ns))

    buys_col = buys[:, None]   # (Nb,1)
    sells_row = sells[None, :]  # (1,Ns)

    for t in range(T):
        p = prob[t]; px = close[t]
        sell_sig = p < sells_row    # (1,Ns) → broadcasts to (Nb,Ns)
        buy_sig  = p >= buys_col    # (Nb,1) → broadcasts to (Nb,Ns)

        # SELL first (only those currently in position)
        sell_trans = in_pos & sell_sig
        if sell_trans.any():
            new_eq = qty * px * (1 - FEE)
            won = sell_trans & (new_eq > entry_eq)
            n_wins += won.astype(np.int32)
            n_trades += sell_trans.astype(np.int32)
            eq = np.where(sell_trans, new_eq, eq)
            in_pos = np.where(sell_trans, False, in_pos)
            qty = np.where(sell_trans, 0.0, qty)

        # BUY (only those now flat)
        buy_trans = (~in_pos) & buy_sig
        if buy_trans.any():
            new_qty = eq * (1 - FEE) / px
            new_entry = eq * (1 - FEE)
            qty = np.where(buy_trans, new_qty, qty)
            entry_eq = np.where(buy_trans, new_entry, entry_eq)
            eq = np.where(buy_trans, new_qty * px, eq)
            in_pos = np.where(buy_trans, True, in_pos)

        # mark-to-market: open positions track price
        eq = np.where(in_pos, qty * px, eq)

        days_im += in_pos.astype(np.int32)
        peak = np.maximum(peak, eq)
        dd = (eq - peak) / peak
        max_dd = np.minimum(max_dd, dd)

    # force-close remaining positions at the last bar (already mark-to-marketed,
    # apply exit fee and book trade)
    if in_pos.any():
        final_eq = np.where(in_pos, qty * close[-1] * (1 - FEE), eq)
        won = in_pos & (final_eq > entry_eq)
        n_wins += won.astype(np.int32)
        n_trades += in_pos.astype(np.int32)
        eq = final_eq
        peak = np.maximum(peak, eq)
        max_dd = np.minimum(max_dd, (eq - peak) / peak)

    total_ret = (eq - 1.0) * 100
    with np.errstate(invalid="ignore", divide="ignore"):
        win_rate = np.where(n_trades > 0, n_wins / np.maximum(n_trades, 1), np.nan)
    return total_ret, n_trades, win_rate, max_dd * 100, days_im


def main():
    btc = load_btc()
    rows_summary = []

    for cell_key, variant, label, prod_buy, prod_sell in CELLS:
        preds = pd.read_parquet(CACHE / f"preds_ebm_{label}_{variant}.parquet")
        s = preds["prob_up"].dropna()
        s.index = pd.to_datetime(s.index, utc=True)

        for win_days in WINDOWS:
            cutoff = s.index.max() - pd.Timedelta(days=win_days)
            sw = s.loc[s.index >= cutoff]
            cl = btc.reindex(sw.index, method="nearest")
            mask = cl.notna()
            sw = sw[mask]; cl = cl[mask]
            prob = sw.to_numpy(dtype=float)
            close = cl.to_numpy(dtype=float)
            bh_pct = (close[-1] / close[0] - 1) * 100

            t0 = time.time()
            ret, ntrd, wrt, mdd, dim = sweep_cell(prob, close, BUYS, SELLS)
            elapsed = time.time() - t0

            # Long-form CSV
            grid = []
            for i, b in enumerate(BUYS):
                for j, s_th in enumerate(SELLS):
                    grid.append((round(float(b), 4), round(float(s_th), 4),
                                 float(ret[i, j]), int(ntrd[i, j]),
                                 float(wrt[i, j]), float(mdd[i, j]),
                                 int(dim[i, j])))
            df = pd.DataFrame(grid, columns=["buy_TH", "sell_TH", "total_return_pct",
                                              "n_trades", "win_rate", "max_dd_pct",
                                              "days_in_market"])
            df["bh_return_pct"] = bh_pct
            df["alpha_pp"] = df["total_return_pct"] - bh_pct
            df["calmar"] = df["total_return_pct"] / df["max_dd_pct"].abs().clip(lower=1e-9)
            out = OUT / f"{cell_key}_{win_days}d.csv"
            df.to_csv(out, index=False)

            # production combo lookup
            i_p = int(round((prod_buy - 0.300) / 0.005))
            j_p = int(round((prod_sell - 0.300) / 0.005))
            prod_ret = float(ret[i_p, j_p])
            prod_alpha = prod_ret - bh_pct
            prod_dd = float(mdd[i_p, j_p])

            # sweep best (within well-formed half: buy_TH >= sell_TH)
            valid = (BUYS[:, None] >= SELLS[None, :])
            ret_v = np.where(valid, ret, -np.inf)
            i_b, j_b = np.unravel_index(np.argmax(ret_v), ret_v.shape)
            best_buy = float(BUYS[i_b]); best_sell = float(SELLS[j_b])
            best_ret = float(ret[i_b, j_b]); best_dd = float(mdd[i_b, j_b])
            best_trd = int(ntrd[i_b, j_b]); best_wr = float(wrt[i_b, j_b])

            # best by Calmar (return / |max_dd|), only if |dd| > 1%
            cal = np.where(valid & (np.abs(mdd) > 1.0), ret / np.maximum(np.abs(mdd), 1e-9), -np.inf)
            i_c, j_c = np.unravel_index(np.argmax(cal), cal.shape)
            best_cal_buy = float(BUYS[i_c]); best_cal_sell = float(SELLS[j_c])
            best_cal_ret = float(ret[i_c, j_c]); best_cal_dd = float(mdd[i_c, j_c])
            best_cal = float(cal[i_c, j_c])

            rows_summary.append({
                "cell": cell_key, "window_d": win_days, "bh_pct": round(bh_pct, 2),
                "prod_buy": prod_buy, "prod_sell": prod_sell,
                "prod_ret_pct": round(prod_ret, 2), "prod_alpha_pp": round(prod_alpha, 2),
                "prod_max_dd_pct": round(prod_dd, 2),
                "best_ret_buy": best_buy, "best_ret_sell": best_sell,
                "best_ret_pct": round(best_ret, 2), "best_ret_alpha_pp": round(best_ret - bh_pct, 2),
                "best_ret_max_dd_pct": round(best_dd, 2),
                "best_ret_n_trades": best_trd, "best_ret_win_rate": round(best_wr, 3),
                "best_calmar_buy": best_cal_buy, "best_calmar_sell": best_cal_sell,
                "best_calmar_ret_pct": round(best_cal_ret, 2),
                "best_calmar_max_dd_pct": round(best_cal_dd, 2),
                "best_calmar": round(best_cal, 2),
                "elapsed_s": round(elapsed, 2),
            })

            print(f"  {cell_key} {win_days}d  B&H={bh_pct:+.1f}%  "
                  f"prod=({prod_buy},{prod_sell})→{prod_ret:+.1f}% (DD {prod_dd:.1f}%)  "
                  f"best_ret=({best_buy},{best_sell})→{best_ret:+.1f}%  "
                  f"best_calmar=({best_cal_buy},{best_cal_sell})→{best_cal_ret:+.1f}% "
                  f"DD {best_cal_dd:.1f}% Calmar {best_cal:.1f}  "
                  f"[{elapsed:.1f}s]")

    summ = pd.DataFrame(rows_summary)
    summ.to_csv(OUT / "_top_per_cell.csv", index=False)
    print(f"\nSummary saved: {OUT/'_top_per_cell.csv'}")


if __name__ == "__main__":
    main()
