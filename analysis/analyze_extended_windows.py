"""Extended-window backtest for the 4 ensembles surfaced in coin_service_utc2130.

Re-runs walk-forward predictions for the 5 sub-model cells with cold_start
adjusted so the predictions cover at least 1500 days, then reports
split-capital ensemble metrics (Top1/Top2/Top3/Top5) on 1095d (~3y) and
1460d (~4y) windows.

Cell predictions are cached to data/preds_cache_ebm_ebm_extended/ to avoid clobbering
the 730d-window cache used by the existing pipeline.

Run:
    uv run python analyze_extended_windows.py            # use cache if present
    uv run python analyze_extended_windows.py --retrain  # force walk-forward
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))


import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import run_backtest
from src.features import build_features, build_labels, compute_dynamic_threshold
from src.model import WFConfig, walk_forward_predict
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "preds_cache_ebm_ebm_extended"
OUT_DIR = ROOT / "reports" / "utc2130_extended"

TRAINING_START = pd.Timestamp("2019-12-01", tz="UTC")
WINDOWS_DAYS = (1095, 1460)  # 3y, 4y
PREDICT_SPAN_DAYS = max(WINDOWS_DAYS) + 60  # extra slack so cutoffs fit
DYNAMIC_N_DAYS = 30
TRADING_DAYS = 365
STAKE_USD = 100.0


V6_REGULARIZATION = {
    "reg_alpha":         0.8,
    "reg_lambda":        3.0,
    "colsample_bytree":  0.5,
    "max_depth":         3,
    "min_child_weight":  10.0,
}


def _ffill_bfill(X: pd.DataFrame) -> pd.DataFrame:
    return X.sort_index().ffill().bfill()


@dataclass
class CellSpec:
    """One walk-forward cell. Mirrors a SUB_MODELS entry from the service config."""
    cell_name: str          # e.g. "utc2130_sm_v5"
    label_slug: str         # e.g. "dynk12" or "th13"
    label_mode: str         # "static" | "dynamic"
    label_value: float
    prob_TH: float
    refit_calendar: str = "SM"
    use_features_v3: bool = False
    use_features_v4: bool = False
    use_features_v5: bool = False
    x_modifier_name: str | None = None  # "ffill_bfill" or None
    wfconfig_overrides: dict | None = None


CELLS: list[CellSpec] = [
    CellSpec(
        cell_name="utc2130_sm_v5", label_slug="dynk12",
        label_mode="dynamic", label_value=1.2, prob_TH=0.63,
        use_features_v5=True,
    ),
    CellSpec(
        cell_name="utc2130_sm_v3_complete", label_slug="th13",
        label_mode="static", label_value=0.013, prob_TH=0.68,
        use_features_v3=True, x_modifier_name="ffill_bfill",
    ),
    CellSpec(
        cell_name="utc2130_sm", label_slug="dynk11",
        label_mode="dynamic", label_value=1.1, prob_TH=0.64,
    ),
    CellSpec(
        cell_name="utc2130_sm_v3", label_slug="th14",
        label_mode="static", label_value=0.014, prob_TH=0.66,
        use_features_v3=True,
    ),
    CellSpec(
        cell_name="utc2130_sm_v6", label_slug="dynk14",
        label_mode="dynamic", label_value=1.4, prob_TH=0.66,
        use_features_v4=True, wfconfig_overrides=V6_REGULARIZATION,
    ),
]


# Mirror of MODELS (top-level ensembles) from coin_service_utc2130/backend/app/config.py
ENSEMBLES = {
    "top1": ["utc2130_sm_v5/dynk12"],
    "top2": ["utc2130_sm_v5/dynk12", "utc2130_sm_v3_complete/th13"],
    "top3": ["utc2130_sm_v5/dynk12", "utc2130_sm_v3_complete/th13",
             "utc2130_sm/dynk11"],
    "top5": ["utc2130_sm_v5/dynk12", "utc2130_sm_v3_complete/th13",
             "utc2130_sm/dynk11", "utc2130_sm_v3/th14",
             "utc2130_sm_v6/dynk14"],
}


def _cell_key(c: CellSpec) -> str:
    return f"{c.cell_name}/{c.label_slug}"


def _cache_path(c: CellSpec) -> Path:
    return CACHE_DIR / f"preds_ebm_{c.label_slug}_{c.cell_name}.parquet"


def _build_X_close_for_cell(c: CellSpec, daily_bar: pd.DataFrame
                            ) -> tuple[pd.DataFrame, pd.Series]:
    X_full, _, close_full = build_features(
        DATA_DIR, label_threshold=0.015,
        daily_df=daily_bar, external_lag_days=0,
        use_mvrv_v2=False,
        use_features_v3=c.use_features_v3,
        use_features_v4=c.use_features_v4,
        use_features_v5=c.use_features_v5,
    )
    if c.x_modifier_name == "ffill_bfill":
        X_full = _ffill_bfill(X_full)
    train_mask = X_full.index >= TRAINING_START
    X = X_full.loc[train_mask].copy()
    close = close_full.loc[X.index].copy()
    return X, close


def _build_y(close: pd.Series, c: CellSpec) -> pd.Series:
    if c.label_mode == "static":
        return build_labels(close, c.label_value)
    threshold_series = compute_dynamic_threshold(close, k=c.label_value, n=DYNAMIC_N_DAYS)
    return build_labels(close, threshold_series)


def _walk_forward_for_cell(c: CellSpec, retrain: bool) -> pd.DataFrame:
    cache = _cache_path(c)
    if cache.exists() and not retrain:
        df = pd.read_parquet(cache)
        nn = df.dropna(subset=["prob_up"])
        span = (nn.index.max() - nn.index.min()).days if len(nn) else 0
        if span >= max(WINDOWS_DAYS):
            print(f"[cache HIT] {_cell_key(c)} non_null={len(nn)} span={span}d", flush=True)
            return df
        print(f"[cache stale] {_cell_key(c)} span={span}d < {max(WINDOWS_DAYS)}d — retraining", flush=True)

    print(f"[fetch] daily bar for {_cell_key(c)} ...", flush=True)
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    print(f"[features] {_cell_key(c)} v3={c.use_features_v3} v4={c.use_features_v4} v5={c.use_features_v5}", flush=True)
    X, close = _build_X_close_for_cell(c, bar)
    y = _build_y(close, c)
    n_up = int((y == 1.0).sum())
    n_dn = int((y == 0.0).sum())
    n_ig = int(y.isna().sum())

    backtest_cutoff = X.index.max() - pd.Timedelta(days=PREDICT_SPAN_DAYS)
    cold_start = max(60, int((X.index < backtest_cutoff).sum()))

    cfg_kwargs = dict(initial_train_days=cold_start, refit_calendar=c.refit_calendar)
    if c.wfconfig_overrides:
        cfg_kwargs.update(c.wfconfig_overrides)
    cfg = WFConfig(**cfg_kwargs)

    print(f"  X={X.shape}  cold_start={cold_start}  "
          f"predict_from={X.index[cold_start].date()}  "
          f"UP={n_up} DN={n_dn} NaN={n_ig}", flush=True)
    t0 = time.time()
    preds_full = walk_forward_predict(
        X, y, cfg,
        progress_every=200,
        progress_label=_cell_key(c),
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed/60:.1f}m", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    preds_full.to_parquet(cache)
    return preds_full


def _drawdown(eq: pd.Series) -> tuple[float, int]:
    cummax = eq.cummax()
    dd = eq / cummax - 1
    if dd.isna().all():
        return float("nan"), 0
    trough = dd.idxmin()
    max_dd = float(dd.min())
    peak_date = eq.loc[:trough].idxmax()
    after = eq.loc[trough:]
    peak_value = eq.loc[peak_date]
    recovered = after[after >= peak_value]
    end_date = recovered.index[0] if len(recovered) else after.index[-1]
    return max_dd, (end_date - peak_date).days


def _longest_inactive(eq: pd.Series) -> int:
    flat = (eq.diff().abs() < 1e-9).astype(int)
    runs = []; cur = 0
    for f in flat.values:
        if f:
            cur += 1
        else:
            if cur:
                runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return max(runs) if runs else 0


def _sharpe_psr(rets: np.ndarray) -> dict:
    n = len(rets)
    if n < 5 or rets.std(ddof=1) == 0:
        return dict(sharpe=float("nan"), psr=float("nan"), sortino=float("nan"))
    mu = rets.mean(); sd = rets.std(ddof=1)
    sr_pp = mu / sd
    skew = float(stats.skew(rets, bias=False))
    kurt = float(stats.kurtosis(rets, fisher=False, bias=False))
    sr_var = (1 - skew * sr_pp + (kurt - 1) / 4 * sr_pp ** 2) / (n - 1)
    psr = float(stats.norm.cdf(sr_pp / math.sqrt(sr_var))) if sr_var > 0 else float("nan")
    downside = rets[rets < 0]
    if len(downside) > 1 and downside.std(ddof=1) > 0:
        sortino = mu / downside.std(ddof=1) * math.sqrt(TRADING_DAYS)
    else:
        sortino = float("nan")
    return dict(
        sharpe=sr_pp * math.sqrt(TRADING_DAYS),
        psr=psr,
        sortino=sortino,
    )


def _calmar(total_return: float, max_dd: float, years: float) -> float:
    if years <= 0 or max_dd >= 0 or pd.isna(max_dd):
        return float("nan")
    cagr = (1 + total_return) ** (1 / years) - 1
    return cagr / abs(max_dd)


def _profit_factor(trades_pnl: list[float]) -> float:
    wins = [p for p in trades_pnl if p > 0]
    loss = [-p for p in trades_pnl if p < 0]
    if not loss:
        return float("inf") if wins else float("nan")
    return sum(wins) / sum(loss)


def _quarterly_outperform(eq: pd.Series, bh: pd.Series) -> dict:
    eq_q = eq.resample("QE").last()
    bh_q = bh.resample("QE").last()
    s_ret = eq_q.pct_change().dropna() * 100
    b_ret = bh_q.pct_change().dropna() * 100
    s_ret, b_ret = s_ret.align(b_ret, join="inner")
    out = (s_ret - b_ret)
    return dict(
        n_quarters=len(out),
        n_outperform=int((out > 0).sum()),
        pct_outperform=float((out > 0).mean() * 100) if len(out) else 0.0,
    )


def _bear_outperform_pp(eq: pd.Series, bh: pd.Series) -> float:
    """B&H total return in regime where B&H drawdown >= 10%, vs strategy in same window."""
    bh_dd = bh / bh.cummax() - 1
    bear_mask = bh_dd <= -0.10
    if bear_mask.sum() < 5:
        return float("nan")
    eq_b = eq.loc[bear_mask]
    bh_b = bh.loc[bear_mask]
    eq_ret = float(eq_b.iloc[-1] / eq_b.iloc[0]) - 1
    bh_ret = float(bh_b.iloc[-1] / bh_b.iloc[0]) - 1
    return (eq_ret - bh_ret) * 100


def _split_capital_metrics(picks: list[CellSpec], window_days: int,
                           preds_by_cell: dict[str, pd.DataFrame],
                           close: pd.Series) -> dict:
    N = len(picks)
    sub_eq: list[pd.Series] = []
    bh_curve = None
    sub_trades_total = 0
    sub_wins_total = 0
    all_trade_pnls: list[float] = []

    for c in picks:
        preds = preds_by_cell[_cell_key(c)]
        nn = preds.dropna(subset=["prob_up"])
        cutoff = nn.index.max() - pd.Timedelta(days=window_days)
        p = nn.loc[nn.index >= cutoff]
        if len(p) < 30:
            return {}
        cl = close.reindex(p.index)
        res = run_backtest(p, cl, stake_usd=STAKE_USD / N,
                           up_threshold=c.prob_TH, down_threshold=0.50)
        sub_eq.append(res.equity_curve)
        sub_trades_total += res.n_trades
        sub_wins_total += sum(1 for t in res.trades if t.pnl_usd > 0)
        all_trade_pnls.extend(t.pnl_usd for t in res.trades)
        if bh_curve is None:
            bh_curve = res.buy_and_hold_curve

    aligned = sub_eq[0]
    for s in sub_eq[1:]:
        a, s2 = aligned.align(s, join="inner")
        aligned = a + s2
    eq = aligned
    bh = bh_curve.reindex(eq.index)
    if eq.empty:
        return {}

    final_eq = float(eq.iloc[-1])
    total_return = final_eq / STAKE_USD - 1
    drets = eq.pct_change().dropna().to_numpy()
    sp = _sharpe_psr(drets)
    max_dd, dd_dur = _drawdown(eq)
    lng_flat = _longest_inactive(eq)
    bh_ret = float(bh.iloc[-1] / bh.iloc[0]) - 1
    years = window_days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1
    calmar = _calmar(total_return, max_dd, years)
    pf = _profit_factor(all_trade_pnls)
    qb = _quarterly_outperform(eq, bh)
    bear_pp = _bear_outperform_pp(eq, bh)

    win_rate = (sub_wins_total / sub_trades_total) if sub_trades_total else 0.0
    return dict(
        N=N,
        window_days=window_days,
        years=round(years, 2),
        total_return_pct=round(total_return * 100, 2),
        bh_pct=round(bh_ret * 100, 2),
        outperform_pp=round((total_return - bh_ret) * 100, 2),
        cagr_pct=round(cagr * 100, 2),
        n_trades=sub_trades_total,
        win_rate_pct=round(win_rate * 100, 2),
        max_dd_pct=round(max_dd * 100, 2),
        max_dd_dur_days=dd_dur,
        longest_inactive_days=lng_flat,
        sharpe=round(sp["sharpe"], 3) if not pd.isna(sp["sharpe"]) else None,
        sortino=round(sp["sortino"], 3) if not pd.isna(sp["sortino"]) else None,
        psr=round(sp["psr"], 3) if not pd.isna(sp["psr"]) else None,
        calmar=round(calmar, 3) if not pd.isna(calmar) else None,
        profit_factor=round(pf, 3) if not pd.isna(pf) and pf != float("inf") else pf,
        bear_outperform_pp=round(bear_pp, 2) if not pd.isna(bear_pp) else None,
        q_n_quarters=qb["n_quarters"],
        q_n_outperform=qb["n_outperform"],
        q_pct_outperform=round(qb["pct_outperform"], 2),
    )


def main():
    retrain = "--retrain" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Extended-window backtest — windows: {WINDOWS_DAYS}d "
          f"({[round(w/365.25, 1) for w in WINDOWS_DAYS]}y)")
    print(f"  cells: {[_cell_key(c) for c in CELLS]}")
    print(f"  ensembles: {list(ENSEMBLES)}")
    print(f"  cache: {CACHE_DIR.relative_to(ROOT)}")
    print(f"  output: {OUT_DIR.relative_to(ROOT)}")
    print("=" * 78, flush=True)

    by_key: dict[str, CellSpec] = {_cell_key(c): c for c in CELLS}

    # Step 1: produce preds for each cell
    preds_by_cell: dict[str, pd.DataFrame] = {}
    for c in CELLS:
        preds_by_cell[_cell_key(c)] = _walk_forward_for_cell(c, retrain)

    # Step 2: get a 21:30-aligned close once for backtests
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    close = bar.set_index("close_time")["close"]

    # Step 3: ensemble metrics per (model, window)
    summary_rows = []
    for model_name, cell_keys in ENSEMBLES.items():
        picks = [by_key[k] for k in cell_keys]
        for w in WINDOWS_DAYS:
            metrics = _split_capital_metrics(picks, w, preds_by_cell, close)
            if not metrics:
                continue
            metrics["model"] = model_name
            summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)
    cols_first = ["model", "window_days", "years", "N", "total_return_pct",
                  "bh_pct", "outperform_pp", "cagr_pct", "n_trades",
                  "win_rate_pct", "max_dd_pct", "calmar", "sortino", "sharpe",
                  "psr", "profit_factor", "longest_inactive_days",
                  "max_dd_dur_days", "bear_outperform_pp",
                  "q_pct_outperform"]
    summary = summary[[c for c in cols_first if c in summary.columns]
                      + [c for c in summary.columns if c not in cols_first]]
    out_csv = OUT_DIR / "topN_ensemble_summary_extended.csv"
    summary.to_csv(out_csv, index=False)
    print("\n" + "=" * 78)
    print(f"Saved → {out_csv.relative_to(ROOT)}")
    print("=" * 78)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
