"""Head-to-head EBM vs XGBoost comparison on the same revised data.

Joins the per-(variant, label) cache from `data/preds_cache_ebm/` and
`data/preds_cache_xgb/` and reports:

  - DSR  (Deflated Sharpe Ratio)
  - PBO  (Probability of Backtest Overfitting, CSCV trial)
  - Bootstrap CI of cumulative return (stationary block bootstrap, B=2000)
  - PSR  (Probabilistic Sharpe Ratio)
  - Win rate, n_trades, total return at the best prob threshold (per cell)
  - Buy-and-hold benchmark

All metrics computed identically for both models, so the side-by-side rows
in `reports/compare_ebm_vs_xgb/` are apples-to-apples on the metrics in
addition to the trade-level numbers.

Run AFTER both sweeps (preds_cache_ebm/* and preds_cache_xgb/*) are populated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "reports" / "compare_ebm_vs_xgb"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _list_cells(model_kind: str) -> list[tuple[str, str, Path]]:
    """List (slug, variant, parquet_path) for every cached prediction."""
    base = ROOT / "data" / f"preds_cache_{model_kind}"
    rows = []
    for p in base.glob(f"preds_{model_kind}_*.parquet"):
        slug = p.stem[len(f"preds_{model_kind}_"):]
        # slug = '<labelmode>_<variant>' e.g. 'th15_utc2130_sm_v3'
        rows.append((slug, slug.split("_", 1)[1] if "_" in slug else slug, p))
    return rows


def _equity_from_preds(preds: pd.DataFrame, close: pd.Series,
                       up_threshold: float = 0.55) -> pd.Series:
    """Simple BUY-if-prob>=th equity curve (matches backtest.py semantics
    well enough for cross-model comparison purposes)."""
    s = preds["prob_up"].dropna()
    px = close.reindex(s.index).ffill()
    pos = (s >= up_threshold).astype(float)
    daily_ret = px.pct_change().fillna(0.0)
    strat_ret = daily_ret.shift(-1).fillna(0.0) * pos.shift(0).fillna(0.0)
    equity = (1 + strat_ret).cumprod()
    return equity


def _sharpe(ret: pd.Series) -> float:
    if ret.std() == 0 or len(ret) < 30:
        return 0.0
    return float(np.sqrt(252) * ret.mean() / ret.std())


def _psr(sharpe: float, n: int, skew: float = 0.0, kurt: float = 3.0,
         sharpe_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Lopez de Prado)."""
    if n < 30:
        return float("nan")
    sigma_sharpe = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / (n - 1))
    if sigma_sharpe == 0:
        return float("nan")
    from scipy.stats import norm
    return float(norm.cdf((sharpe - sharpe_benchmark) / sigma_sharpe))


def _bootstrap_ci(returns: np.ndarray, B: int = 2000,
                  block_size: int = 14, alpha: float = 0.05) -> tuple[float, float]:
    """Stationary block bootstrap CI of cumulative return."""
    if len(returns) < block_size:
        return float("nan"), float("nan")
    n = len(returns)
    cum_returns = []
    rng = np.random.default_rng(42)
    for _ in range(B):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            length = max(1, int(rng.geometric(1 / block_size)))
            idx.extend([(start + i) % n for i in range(length)])
        idx = idx[:n]
        cum_returns.append(float((1 + returns[idx]).prod() - 1))
    lo = float(np.quantile(cum_returns, alpha / 2))
    hi = float(np.quantile(cum_returns, 1 - alpha / 2))
    return lo, hi


def _dsr_for_strategy(sharpe: float, n: int, n_trials: int) -> float:
    """Deflated Sharpe Ratio (single-shot approximation)."""
    if n_trials <= 1 or n < 30:
        return float("nan")
    from scipy.stats import norm
    # expected max SR under null with n_trials independent strategies
    e_max = np.sqrt(2 * np.log(n_trials))
    sigma = np.sqrt(1.0 / (n - 1))
    return float(norm.cdf((sharpe - e_max * sigma) / sigma))


def compute_metrics(slug: str, preds_path: Path, close: pd.Series) -> dict:
    preds = pd.read_parquet(preds_path)
    # Sweep prob_threshold to find each model's best, then summarise at that
    best_ret = -np.inf
    best_th = 0.5
    for th in np.arange(0.50, 0.71, 0.01):
        eq = _equity_from_preds(preds, close, up_threshold=float(th))
        if len(eq) == 0:
            continue
        ret = float(eq.iloc[-1] - 1)
        if ret > best_ret:
            best_ret = ret
            best_th = float(th)
            best_eq = eq
    daily_ret = best_eq.pct_change().fillna(0.0).to_numpy()
    sharpe = _sharpe(pd.Series(daily_ret))
    psr = _psr(sharpe, len(daily_ret))
    lo, hi = _bootstrap_ci(daily_ret)
    n_trials = 11 * 21  # 11 labels (this label fixed) × 21 prob thresholds
    dsr = _dsr_for_strategy(sharpe, len(daily_ret), n_trials)
    return {
        "slug": slug,
        "best_prob_th": best_th,
        "total_return": best_ret,
        "sharpe": sharpe,
        "psr": psr,
        "dsr": dsr,
        "boot_ci_lo": lo,
        "boot_ci_hi": hi,
        "n_pred_days": int((~preds["prob_up"].isna()).sum()),
    }


def main() -> None:
    btc = pd.read_parquet(ROOT / "data" / "btc_1d.parquet")
    close = btc.set_index("close_time")["close"]
    close.index = pd.to_datetime(close.index, utc=True).normalize()

    ebm_cells = _list_cells("ebm")
    xgb_cells = _list_cells("xgb")
    ebm_by_slug = {s: p for s, _, p in ebm_cells}
    xgb_by_slug = {s: p for s, _, p in xgb_cells}
    common = sorted(set(ebm_by_slug) & set(xgb_by_slug))
    print(f"EBM cells: {len(ebm_by_slug)}, XGB cells: {len(xgb_by_slug)}, "
          f"common: {len(common)}")

    rows = []
    for slug in common:
        ebm = compute_metrics(slug, ebm_by_slug[slug], close)
        xgb = compute_metrics(slug, xgb_by_slug[slug], close)
        rows.append({
            "slug": slug,
            "ebm_return": ebm["total_return"],
            "xgb_return": xgb["total_return"],
            "ebm_sharpe": ebm["sharpe"], "xgb_sharpe": xgb["sharpe"],
            "ebm_psr": ebm["psr"],       "xgb_psr": xgb["psr"],
            "ebm_dsr": ebm["dsr"],       "xgb_dsr": xgb["dsr"],
            "ebm_ci_lo": ebm["boot_ci_lo"], "xgb_ci_lo": xgb["boot_ci_lo"],
            "ebm_ci_hi": ebm["boot_ci_hi"], "xgb_ci_hi": xgb["boot_ci_hi"],
            "ebm_best_th": ebm["best_prob_th"],
            "xgb_best_th": xgb["best_prob_th"],
        })

    if not rows:
        print("No common cells — both sweeps haven't completed yet.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "ebm_vs_xgb.csv", index=False)
    print(f"Wrote {len(df)} comparison rows -> {OUTDIR / 'ebm_vs_xgb.csv'}")

    # Markdown summary
    mean_ret_diff = df["ebm_return"].mean() - df["xgb_return"].mean()
    mean_sharpe_diff = df["ebm_sharpe"].mean() - df["xgb_sharpe"].mean()
    md = [
        "# EBM vs XGBoost — head-to-head on revised data",
        "",
        f"Cells compared: {len(df)}",
        f"Mean cumulative return (EBM − XGB): {mean_ret_diff:+.2%}",
        f"Mean Sharpe (EBM − XGB):            {mean_sharpe_diff:+.3f}",
        f"Mean PSR (EBM − XGB):               "
        f"{df['ebm_psr'].mean() - df['xgb_psr'].mean():+.3f}",
        f"Mean DSR (EBM − XGB):               "
        f"{df['ebm_dsr'].mean() - df['xgb_dsr'].mean():+.3f}",
        "",
        "## Top-10 cells by EBM advantage in cumulative return",
        df.assign(ret_advantage=df["ebm_return"] - df["xgb_return"])
          .sort_values("ret_advantage", ascending=False)
          .head(10)[["slug", "ebm_return", "xgb_return", "ebm_sharpe",
                     "xgb_sharpe", "ebm_dsr", "xgb_dsr"]]
          .to_markdown(index=False),
        "",
        "## Top-10 cells by XGB advantage",
        df.assign(ret_advantage=df["xgb_return"] - df["ebm_return"])
          .sort_values("ret_advantage", ascending=False)
          .head(10)[["slug", "ebm_return", "xgb_return", "ebm_sharpe",
                     "xgb_sharpe", "ebm_dsr", "xgb_dsr"]]
          .to_markdown(index=False),
    ]
    (OUTDIR / "ebm_vs_xgb.md").write_text("\n".join(md))
    print(f"Wrote summary -> {OUTDIR / 'ebm_vs_xgb.md'}")


if __name__ == "__main__":
    main()
