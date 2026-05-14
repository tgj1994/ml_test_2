"""Slippage sensitivity sweep — addresses reviewer critique #2 (execution
realism / 10 bps optimism, especially at UTC 00:00 funding settlement).

Method:
  1. For each (cell, window), pick the (label, prob_TH) trajectory the paper
     highlights as best-at-10bps (read from existing threshold_sweep CSVs).
  2. Re-run THAT trajectory with fee_bps ∈ {5, 10, 15, 20, 25, 30}.
  3. Report total_return, Sharpe, n_trades, breakeven_fee per cell.

The cached predictions are reused — no XGBoost retraining.

Outputs (under reports/slippage_sensitivity/):
  - per_cell_slippage.csv   : long-format (cell, window, fee_bps, return, sharpe...)
  - matrix_return.csv       : wide matrix [cell × window × fee_bps] of returns
  - breakeven_fees.csv      : breakeven fee bps per cell × window
  - sensitivity_plot.png    : line chart for top cells across fee_bps
  - summary.txt             : key numbers for the paper
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.utc0000 import build_utc0000_daily
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm"
REPORTS_DIR = ROOT / "reports"
OUT_DIR = REPORTS_DIR / "slippage_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = (730, 365)
FEE_LEVELS_BPS = (5, 10, 15, 20, 25, 30)
TRADING_DAYS = 365.0
EXCLUDE_TOKENS = ("smoke",)


def _utc_kind(suffix: str) -> str | None:
    if "utc0000" in suffix:
        return "utc0000"
    if "utc2130" in suffix:
        return "utc2130"
    return None


def _all_cells() -> list[str]:
    out = set()
    for f in os.listdir(CACHE_DIR):
        m = re.match(r"preds_ebm_(th\d+|dynk\d+)_(.+)\.parquet$", f)
        if m and ("utc0000" in m.group(2) or "utc2130" in m.group(2)):
            if not any(t in m.group(2) for t in EXCLUDE_TOKENS):
                out.add(m.group(2))
    return sorted(out)


def _label_to_value(label: str) -> tuple[str, float]:
    """th15 -> ('static', 0.015); dynk15 -> ('dynamic', 1.5)."""
    if label.startswith("th"):
        return "static", int(label[2:]) / 1000.0
    if label.startswith("dynk"):
        return "dynamic", int(label[4:]) / 10.0
    raise ValueError(f"unknown label: {label}")


def find_paper_best(cell: str, window: int) -> tuple[str, float] | None:
    """Read the paper's published best (label, prob_TH) for this cell-window
    from threshold_sweep CSVs. Returns None if not available."""
    rep_root = REPORTS_DIR / _utc_kind(cell)
    rep_dir = rep_root / f"{cell}_{window}d"
    if not rep_dir.exists():
        return None
    best = None  # (return, label, prob_TH)
    for csv in rep_dir.glob("threshold_sweep_*.csv"):
        m = re.match(r"threshold_sweep_(th\d+|dynk\d+)_.+\.csv", csv.name)
        if not m:
            continue
        label = m.group(1)
        df = pd.read_csv(csv)
        if df.empty:
            continue
        idx = df["total_return_pct"].idxmax()
        ret = float(df.loc[idx, "total_return_pct"])
        prob = float(df.loc[idx, "threshold"])
        if best is None or ret > best[0]:
            best = (ret, label, prob)
    if best is None:
        return None
    return best[1], best[2]


def run_slippage_sweep(cell: str, window: int,
                       label: str, prob_TH: float,
                       close_series: pd.Series) -> list[dict]:
    """Re-run the SAME (label, prob_TH) trajectory at multiple fee levels."""
    cache_path = CACHE_DIR / f"preds_ebm_{label}_{cell}.parquet"
    if not cache_path.exists():
        return []
    preds_full = pd.read_parquet(cache_path)
    cutoff = preds_full.index.max() - pd.Timedelta(days=window)
    preds = preds_full.dropna(subset=["prob_up"])
    preds = preds.loc[preds.index >= cutoff]
    if len(preds) < 30:
        return []
    close_w = close_series.reindex(preds.index)
    rows: list[dict] = []
    for fee in FEE_LEVELS_BPS:
        res = run_backtest(preds, close_w,
                           fee_bps=float(fee),
                           up_threshold=prob_TH, down_threshold=0.5)
        # Sharpe from equity curve
        eq = res.equity_curve
        if len(eq) < 5:
            sr = float("nan")
        else:
            r = eq.pct_change().dropna().to_numpy()
            if r.std(ddof=1) == 0:
                sr = 0.0
            else:
                sr = (r.mean() / r.std(ddof=1)) * math.sqrt(TRADING_DAYS)
        rows.append({
            "cell": cell,
            "kind": _utc_kind(cell),
            "window_days": window,
            "label": label,
            "prob_TH": prob_TH,
            "fee_bps": fee,
            "n_trades": res.n_trades,
            "win_rate": res.win_rate,
            "total_return_pct": res.total_return_pct,
            "buy_hold_pct": res.buy_and_hold_return_pct,
            "final_equity": res.final_equity,
            "sharpe": sr,
        })
    return rows


def breakeven_fee_bps(rows: list[dict]) -> float:
    """Linearly interpolate the fee_bps at which total_return_pct hits 0
    (fall to break-even). Returns +inf if never reaches 0 within the swept
    range; returns 0 if already <0 at the cheapest fee."""
    fees = np.array([r["fee_bps"] for r in rows], dtype=float)
    rets = np.array([r["total_return_pct"] for r in rows], dtype=float)
    order = np.argsort(fees)
    fees, rets = fees[order], rets[order]
    if rets[0] <= 0:
        return float(fees[0])
    if rets[-1] > 0:
        return float("inf")
    # find first crossing
    for i in range(1, len(fees)):
        if rets[i] <= 0:
            x0, x1 = fees[i - 1], fees[i]
            y0, y1 = rets[i - 1], rets[i]
            if y1 == y0:
                return float(x1)
            return float(x0 + (x1 - x0) * (-y0) / (y1 - y0))
    return float("inf")


def main() -> None:
    print("[1/4] building close series ...", flush=True)
    bar0 = build_utc0000_daily(DATA_DIR / "btc_15m.parquet")
    bar21 = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close_utc0000 = bar0.set_index("close_time")["close"]
    close_utc2130 = bar21.set_index("close_time")["close"]

    cells = _all_cells()
    print(f"[2/4] running fee sweep on {len(cells)} cells × {len(WINDOWS)} windows × "
          f"{len(FEE_LEVELS_BPS)} fees ...", flush=True)

    all_rows: list[dict] = []
    breakeven_rows: list[dict] = []
    for i, cell in enumerate(cells, 1):
        kind = _utc_kind(cell)
        cs = close_utc0000 if kind == "utc0000" else close_utc2130
        for window in WINDOWS:
            best = find_paper_best(cell, window)
            if best is None:
                continue
            label, prob_TH = best
            rows = run_slippage_sweep(cell, window, label, prob_TH, cs)
            if not rows:
                continue
            all_rows.extend(rows)
            be = breakeven_fee_bps(rows)
            base = next(r for r in rows if r["fee_bps"] == 10)
            high = next(r for r in rows if r["fee_bps"] == 20)
            breakeven_rows.append({
                "cell": cell,
                "kind": kind,
                "window_days": window,
                "label": label,
                "prob_TH": prob_TH,
                "n_trades_at_10bps": base["n_trades"],
                "return_at_10bps_pct": base["total_return_pct"],
                "return_at_20bps_pct": high["total_return_pct"],
                "delta_pct_pts": high["total_return_pct"] - base["total_return_pct"],
                "delta_relative": ((high["total_return_pct"] - base["total_return_pct"])
                                   / abs(base["total_return_pct"])
                                   if base["total_return_pct"] != 0 else float("nan")),
                "breakeven_fee_bps": be,
            })
        if i % 10 == 0 or i == len(cells):
            print(f"  [{i:>3}/{len(cells)}] done", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "per_cell_slippage.csv", index=False)
    print(f"  long → {OUT_DIR/'per_cell_slippage.csv'} ({len(df)} rows)", flush=True)

    # wide return matrix
    print("[3/4] writing matrices + breakeven ...", flush=True)
    pivot_ret = df.pivot_table(index=["cell", "window_days", "kind", "label", "prob_TH"],
                               columns="fee_bps",
                               values="total_return_pct").reset_index()
    pivot_ret.to_csv(OUT_DIR / "matrix_return.csv", index=False)

    pivot_sr = df.pivot_table(index=["cell", "window_days", "kind", "label", "prob_TH"],
                              columns="fee_bps",
                              values="sharpe").reset_index()
    pivot_sr.to_csv(OUT_DIR / "matrix_sharpe.csv", index=False)

    be_df = pd.DataFrame(breakeven_rows).sort_values(
        ["window_days", "return_at_10bps_pct"], ascending=[True, False])
    be_df.to_csv(OUT_DIR / "breakeven_fees.csv", index=False)

    # plots
    print("[4/4] plots ...", flush=True)
    try:
        import matplotlib.pyplot as plt
        for window in WINDOWS:
            sub = df.loc[df["window_days"] == window].copy()
            top_cells = (sub.loc[sub["fee_bps"] == 10]
                         .sort_values("total_return_pct", ascending=False)
                         .head(8)["cell"].tolist())
            fig, ax = plt.subplots(figsize=(11, 6))
            for cell in top_cells:
                cs = sub.loc[sub["cell"] == cell].sort_values("fee_bps")
                ax.plot(cs["fee_bps"], cs["total_return_pct"] * 100,
                        marker="o", label=cell)
            ax.axhline(0, color="black", linestyle=":", alpha=0.5)
            ax.set_xlabel("fee_bps (per side)")
            ax.set_ylabel("total return (%)")
            ax.set_title(f"Slippage sensitivity — top-8 cells (window={window}d)")
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"sensitivity_plot_{window}d.png", dpi=130)
            plt.close(fig)
    except Exception as e:
        print(f"  ! plot failed: {e}")

    # summary
    lines = []
    lines.append("# Slippage sensitivity summary")
    lines.append("")
    lines.append(f"Cells × windows evaluated: {len(be_df)}")
    lines.append(f"Fee levels (bps per side): {list(FEE_LEVELS_BPS)}")
    lines.append("")
    for window in WINDOWS:
        sub = be_df.loc[be_df["window_days"] == window]
        positive_at_10 = int((sub["return_at_10bps_pct"] > 0).sum())
        positive_at_20 = int((sub["return_at_20bps_pct"] > 0).sum())
        survives_25 = int((sub["breakeven_fee_bps"] >= 25).sum())
        survives_inf = int(np.isinf(sub["breakeven_fee_bps"]).sum())
        lines.append(f"## window = {window}d")
        lines.append(f"  cells with return>0 @ 10bps : {positive_at_10}/{len(sub)}")
        lines.append(f"  cells with return>0 @ 20bps : {positive_at_20}/{len(sub)}")
        lines.append(f"  cells with breakeven ≥ 25bps : {survives_25}/{len(sub)}")
        lines.append(f"  cells still profitable @ 30bps: {survives_inf}/{len(sub)}")
        worst = sub.sort_values("delta_pct_pts").head(5)
        lines.append("  largest 10→20bps degradation (pct points):")
        for _, r in worst.iterrows():
            lines.append(f"    {r['cell']:<40s} label={r['label']:<7s} "
                         f"trades={int(r['n_trades_at_10bps']):>3d} "
                         f"10bps={r['return_at_10bps_pct']*100:+7.2f}% "
                         f"→ 20bps={r['return_at_20bps_pct']*100:+7.2f}% "
                         f"(Δ={r['delta_pct_pts']*100:+.2f} pp)")
        top_robust = sub.sort_values("breakeven_fee_bps", ascending=False).head(5)
        lines.append("  most slippage-robust (highest breakeven):")
        for _, r in top_robust.iterrows():
            be = r["breakeven_fee_bps"]
            be_str = ">30" if math.isinf(be) else f"{be:.1f}"
            lines.append(f"    {r['cell']:<40s} label={r['label']:<7s} "
                         f"trades={int(r['n_trades_at_10bps']):>3d} "
                         f"breakeven_bps={be_str}")
        lines.append("")

    txt = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(txt)
    print(f"  summary → {OUT_DIR/'summary.txt'}")
    print(txt)


if __name__ == "__main__":
    main()
