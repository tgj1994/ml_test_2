"""Shared runner for the KST10 (10:00 KST decision time) backtest variants.

This module is the workhorse behind:
  - main_th_sweep_kst10.py           (raw)
  - main_th_sweep_kst10_close.py     (+ is_macro_offday flag)
  - main_th_sweep_kst10_complete.py  (+ ffill/bfill on NaNs)

What this guarantees:
  1. Daily bars are 10:00 KST aligned (= 01:00 UTC) and built from 15m data.
  2. Walk-forward retrains every day (refit_every=1).
  3. External feeds (macro/funding/basis/fng) lagged by 1 day so unpublished
     same-day NYSE closes / funding settlements are not leaked.
  4. One walk-forward per label threshold; both 730d and 365d windows are
     produced from the same prediction series (slicing).
  5. Per-window reports written under reports/kst10/<suffix>_<window>d/.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest import BacktestResult, run_backtest
from src.features import build_features
from src.kst10 import build_kst10_daily
from src.model import WFConfig, feature_importance, walk_forward_predict


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_ROOT = ROOT / "reports" / "kst10"
CACHE_DIR = ROOT / "data" / "preds_cache_ebm"

TRAINING_START = pd.Timestamp("2019-12-01", tz="UTC")
WINDOWS_DAYS = (730, 365)
LABEL_THRESHOLDS = [round(0.010 + i * 0.001, 3) for i in range(11)]
PROB_THRESHOLDS  = [round(0.50 + i * 0.01, 2) for i in range(21)]


@dataclass
class VariantConfig:
    """Per-script variant settings."""
    suffix: str                          # e.g. "kst10", "kst10_close", "kst10_complete"
    title: str                           # human description for plots
    x_modifier: Callable[[pd.DataFrame], pd.DataFrame] | None = None


def _slug(label_threshold: float, suffix: str) -> str:
    return f"th{int(round(label_threshold * 1000))}_{suffix}"


def _walk_forward_for_label(
    label_threshold: float,
    cfg_variant: VariantConfig,
    kst10_df: pd.DataFrame,
    retrain: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, str, dict]:
    """Run walk-forward for one label threshold (with caching).

    Returns (preds_full, feature_importance_df, close_series, slug, stats).
    """
    slug = _slug(label_threshold, cfg_variant.suffix)
    pred_cache = CACHE_DIR / f"preds_ebm_{slug}.parquet"
    fi_cache = CACHE_DIR / f"fi_ebm_{slug}.parquet"
    close_cache = CACHE_DIR / f"close_{slug}.parquet"

    print(f"\n--- ±{label_threshold*100:.1f}%  ({slug}) ---", flush=True)

    X_full, y_full, close_full = build_features(
        DATA_DIR, label_threshold=label_threshold,
        daily_df=kst10_df, external_lag_days=1,
    )
    if cfg_variant.x_modifier is not None:
        X_full = cfg_variant.x_modifier(X_full)

    train_mask = X_full.index >= TRAINING_START
    X = X_full.loc[train_mask].copy()
    y = y_full.loc[X.index].copy()
    close = close_full.loc[X.index].copy()
    n_up = int((y == 1.0).sum()); n_dn = int((y == 0.0).sum())
    n_ig = int(y.isna().sum())

    # Cold-start: anchor on the larger of the two windows so the same prediction
    # series can be sliced for any smaller window.
    backtest_days = max(WINDOWS_DAYS)
    backtest_cutoff = X.index.max() - pd.Timedelta(days=backtest_days)
    cold_start = max(60, int((X.index < backtest_cutoff).sum()))
    cfg = WFConfig(initial_train_days=cold_start, refit_every=1)

    cache_ready = (not retrain
                   and pred_cache.exists()
                   and fi_cache.exists()
                   and close_cache.exists())
    if cache_ready:
        preds_full = pd.read_parquet(pred_cache)
        fi = pd.read_parquet(fi_cache)
        close = pd.read_parquet(close_cache)["close"]
        print(f"    cache HIT  (cold_start={cold_start}, "
              f"{n_up}/{n_dn}/{n_ig} labels, X={X.shape})",
              flush=True)
    else:
        print(f"    training... cold_start={cold_start}  refit_every=1  "
              f"({n_up}/{n_dn}/{n_ig} labels, X={X.shape})",
              flush=True)
        preds_full = walk_forward_predict(
            X, y, cfg,
            progress_every=50,
            progress_label=slug,
        )
        fi = feature_importance(
            X.loc[X.index < backtest_cutoff],
            y.loc[X.index < backtest_cutoff],
            cfg,
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        preds_full.to_parquet(pred_cache)
        fi.to_parquet(fi_cache)
        pd.DataFrame({"close": close}).to_parquet(close_cache)

    stats = {
        "n_up": n_up, "n_dn": n_dn, "n_ig": n_ig,
        "ignore_pct": n_ig / max(1, len(y)),
        "cold_start": cold_start,
    }
    return preds_full, fi, close, slug, stats


def _backtest_for_window(
    preds_full: pd.DataFrame,
    close: pd.Series,
    window_days: int,
) -> tuple[dict[float, BacktestResult], dict]:
    cutoff = preds_full.index.max() - pd.Timedelta(days=window_days)
    preds = preds_full.dropna(subset=["prob_up"])
    preds = preds.loc[preds.index >= cutoff]
    close_w = close.loc[preds.index]

    actual_lbl = preds["actual"].dropna()
    pred_lbl = preds.loc[actual_lbl.index]
    hit_05 = ((pred_lbl["prob_up"] >= 0.5).astype(float) == pred_lbl["actual"]).mean() \
              if len(actual_lbl) else float("nan")

    results: dict[float, BacktestResult] = {}
    for th in PROB_THRESHOLDS:
        results[th] = run_backtest(preds, close_w,
                                   up_threshold=th, down_threshold=0.5)

    bt_stats = {
        "n_pred_labels_in_window": int(len(actual_lbl)),
        "acc_at_05": float(hit_05),
        "prob_mean": float(preds["prob_up"].mean()),
        "buy_hold": next(iter(results.values())).buy_and_hold_return_pct,
    }
    return results, bt_stats


def _plot_one(results, out_path, label_threshold, window_days, title_extra=""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    ths = sorted(results.keys())
    cmap = plt.get_cmap("viridis")
    show_idx = list(range(0, len(ths), 4))
    if (len(ths) - 1) not in show_idx:
        show_idx.append(len(ths) - 1)
    best_th = max(results, key=lambda k: results[k].total_return_pct)
    best_idx = ths.index(best_th)
    if best_idx not in show_idx:
        show_idx.append(best_idx)
    for i in sorted(show_idx):
        th = ths[i]; r = results[th]
        ax.plot(r.equity_curve.index, r.equity_curve.values,
                label=f"prob_TH={th:.2f}", color=cmap(i / max(1, len(ths) - 1)))
    any_r = next(iter(results.values()))
    ax.plot(any_r.buy_and_hold_curve.index, any_r.buy_and_hold_curve.values,
            label="Buy & Hold", linestyle="--", color="black", alpha=0.6)
    ax.set_title(f"label ±{label_threshold*100:.1f}% — {window_days}d backtest "
                 f"— equity by BUY threshold (KST10{title_extra})")
    ax.set_ylabel("USD")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_heatmap(matrix: pd.DataFrame, out_path: Path, window_days: int,
                  title_extra: str = "") -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 6))
    arr = matrix.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([f"±{idx*100:.1f}%" for idx in matrix.index])
    ax.set_xlabel("BUY probability threshold")
    ax.set_ylabel("Label threshold (training)")
    ax.set_title(f"{window_days}-day backtest return — label × prob (KST10{title_extra})")
    cbar = plt.colorbar(im, ax=ax, format="%.2f")
    cbar.set_label("Total return")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                    color="black" if abs(v) < 0.30 else "white", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _generate_markdown(runs: list[dict], matrix: pd.DataFrame,
                       out_path: Path, window_days: int, title_extra: str,
                       report_dir: Path) -> None:
    lines = []
    lines.append(f"# KST10 — {window_days}-day backtest{title_extra}")
    lines.append("")
    lines.append(f"**Setup**: 10:00 KST aligned daily bar (= 01:00 UTC), "
                 f"daily refit (refit_every=1), external feeds (macro/funding/"
                 f"basis/fng) lagged 1 day to avoid same-day publication leakage. "
                 f"Training from 2019-12-01. Backtest = last {window_days} days.")
    lines.append("")
    lines.append("## Summary (best prob_TH per label)")
    lines.append("")
    bh = runs[0]["buy_hold"]
    lines.append(f"Buy & Hold (window): {bh*100:+.2f}%")
    lines.append("")
    lines.append("| label | best_prob | trades | win% | return | final $ | acc@.5 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        lines.append(
            f"| ±{r['label_threshold']*100:.1f}% "
            f"| {r['best_prob_th']:.2f} "
            f"| {r['best_trades']} "
            f"| {r['best_winrate']*100:.1f}% "
            f"| **{r['best_return']*100:+.2f}%** "
            f"| ${r['best_equity']:.2f} "
            f"| {r['acc_at_05']*100:.1f}% |"
        )
    lines.append("")
    for r in runs:
        lt = r["label_threshold"]; slug = r["slug"]
        df = pd.read_csv(report_dir / f"threshold_sweep_{slug}.csv") \
                .sort_values("threshold").reset_index(drop=True)
        best_idx = df["total_return_pct"].idxmax()
        bh_local = df["buy_and_hold_pct"].iloc[0]
        lines.append(f"## ±{lt*100:.1f}% labels  ({slug})")
        lines.append("")
        lines.append(f"Buy & Hold (window): {bh_local*100:+.2f}%")
        lines.append("")
        lines.append("| prob_TH | trades | win% | BUY days | HOLD days | SELL days | "
                     "in-market | return | final $ |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for i, row in df.iterrows():
            cells = (
                f"{row['threshold']:.2f}",
                f"{int(row['trades'])}",
                f"{row['win_rate']*100:.1f}%" if row['trades'] else "—",
                f"{int(row['days_buy'])}",
                f"{int(row['days_hold'])}",
                f"{int(row['days_sell'])}",
                f"{int(row['days_in_market'])}",
                f"{row['total_return_pct']*100:+.2f}%",
                f"${row['final_equity_usd']:.2f}",
            )
            if i == best_idx:
                lines.append("| " + " | ".join(f"**{c}**" for c in cells) + " |")
            else:
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    overall = matrix.stack().idxmax()
    overall_val = matrix.stack().max()
    lines.append("## OVERALL BEST")
    lines.append("")
    lines.append(f"**label ±{overall[0]*100:.1f}% × prob {overall[1]:.2f} → "
                 f"{overall_val*100:+.2f}%** (Buy & Hold: {bh*100:+.2f}%)")
    lines.append("")
    out_path.write_text("\n".join(lines))


def run_variant(variant: VariantConfig, retrain: bool = False,
                label_thresholds: list[float] | None = None) -> None:
    """Run one variant end-to-end (walk-forward + both windows)."""
    label_thresholds = label_thresholds or LABEL_THRESHOLDS
    title_extra = f" — {variant.title}" if variant.title else ""

    print("=" * 78)
    print(f"KST10 LABEL × PROB sweep — variant={variant.suffix}")
    print(f"  refit_every=1, external_lag_days=1, training from 2019-12-01")
    print(f"  windows: {WINDOWS_DAYS} days each")
    print(f"  label thresholds: {[round(t*100,1) for t in label_thresholds]} %")
    print(f"  prob thresholds : 0.50..0.70 step 0.01 ({len(PROB_THRESHOLDS)})")
    print("=" * 78, flush=True)

    print("[fetch] building KST10 daily bar from 15m...", flush=True)
    kst10 = build_kst10_daily(DATA_DIR / "btc_15m.parquet")
    print(f"  {len(kst10)} rows, {kst10['close_time'].iloc[0].date()} -> "
          f"{kst10['close_time'].iloc[-1].date()}", flush=True)

    # Walk-forward once per label threshold (caches predictions).
    label_runs: list[dict] = []
    for lt in label_thresholds:
        preds_full, fi, close, slug, stats = _walk_forward_for_label(
            lt, variant, kst10, retrain=retrain)
        label_runs.append({
            "label_threshold": lt, "slug": slug,
            "preds_full": preds_full, "fi": fi, "close": close, **stats,
        })

    # For each window, run all backtests on the cached predictions.
    for window_days in WINDOWS_DAYS:
        suffix_w = f"{variant.suffix}_{window_days}d"
        report_dir = REPORT_ROOT / suffix_w
        report_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== window={window_days}d  → {report_dir.relative_to(ROOT)}",
              flush=True)

        runs = []
        for lr in label_runs:
            results, bt_stats = _backtest_for_window(
                lr["preds_full"], lr["close"], window_days)

            # per-label per-window outputs
            slug_w = f"th{int(round(lr['label_threshold']*1000))}_{suffix_w}"
            sweep_rows = [{
                "threshold": th, "trades": r.n_trades, "win_rate": r.win_rate,
                "days_buy": r.days_buy, "days_hold": r.days_sideways,
                "days_sell": r.days_sell, "days_in_market": r.days_in_market,
                "total_return_pct": r.total_return_pct,
                "final_equity_usd": r.final_equity,
                "buy_and_hold_pct": r.buy_and_hold_return_pct,
            } for th, r in results.items()]
            pd.DataFrame(sweep_rows).to_csv(
                report_dir / f"threshold_sweep_{slug_w}.csv", index=False)
            lr["fi"].to_csv(report_dir / f"feature_importance_ebm_{slug_w}.csv",
                            index=False)
            _plot_one(results, report_dir / f"equity_curve_{slug_w}.png",
                      lr["label_threshold"], window_days,
                      title_extra=" — " + variant.title if variant.title else "")

            best_th, best_res = max(results.items(),
                                    key=lambda kv: kv[1].total_return_pct)
            if best_res.trades:
                pd.DataFrame([t.__dict__ for t in best_res.trades]).to_csv(
                    report_dir / f"trades_ebm_{slug_w}_th{best_th:.2f}.csv",
                    index=False)

            print(f"    [{window_days}d] ±{lr['label_threshold']*100:.1f}% → "
                  f"BEST prob_TH={best_th:.2f}  ret={best_res.total_return_pct:+.2%}  "
                  f"trades={best_res.n_trades}  win={best_res.win_rate:.1%}  "
                  f"acc@0.5={bt_stats['acc_at_05']:.1%}", flush=True)

            runs.append({
                "label_threshold": lr["label_threshold"], "slug": slug_w,
                "n_train_labels_up": lr["n_up"], "n_train_labels_dn": lr["n_dn"],
                "ignore_pct": lr["ignore_pct"],
                "best_prob_th": best_th,
                "best_return": best_res.total_return_pct,
                "best_equity": best_res.final_equity,
                "best_trades": best_res.n_trades,
                "best_winrate": best_res.win_rate,
                "buy_hold": best_res.buy_and_hold_return_pct,
                "acc_at_05": bt_stats["acc_at_05"],
                "prob_mean": bt_stats["prob_mean"],
                "results": results,
            })

        # window-level matrix / heatmap / markdown
        matrix = pd.DataFrame(
            index=[r["label_threshold"] for r in runs],
            columns=PROB_THRESHOLDS, dtype=float)
        for r in runs:
            for th, res in r["results"].items():
                matrix.loc[r["label_threshold"], th] = res.total_return_pct
        matrix.to_csv(report_dir / f"label_x_prob_return_heatmap_{suffix_w}.csv")
        _plot_heatmap(matrix, report_dir / f"label_x_prob_heatmap_{suffix_w}.png",
                      window_days,
                      title_extra=" — " + variant.title if variant.title else "")

        pd.DataFrame([{k: v for k, v in r.items() if k != "results"}
                      for r in runs]).to_csv(
            report_dir / f"label_threshold_sweep_summary_{suffix_w}.csv",
            index=False)

        md_path = report_dir / f"label_prob_threshold_tables_{suffix_w}.md"
        _generate_markdown(runs, matrix, md_path, window_days,
                           title_extra=" — " + variant.title if variant.title else "",
                           report_dir=report_dir)

        bh = runs[0]["buy_hold"]
        overall = matrix.stack().idxmax()
        overall_val = matrix.stack().max()
        print(f"    [{window_days}d] B&H={bh:+.2%}  "
              f"OVERALL BEST: label ±{overall[0]*100:.1f}% × prob {overall[1]:.2f} "
              f"→ {overall_val:+.2%}", flush=True)
        print(f"    Markdown: {md_path}", flush=True)
