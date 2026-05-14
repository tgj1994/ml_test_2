"""Shared runner for the UTC2130 (21:30 UTC = post macro-release decision)
backtest variants.

What this guarantees:
  1. Daily bars are 21:30 UTC aligned — built from 15m data.
  2. Walk-forward refits on the 1st of each calendar month
     (refit_calendar='M').
  3. External feeds (macro/funding/basis/fng) joined with lag = 0 —
     21:30 UTC is past every same-day publication, so no leakage.
  4. Labels run in TWO modes per variant:
       - STATIC : 11 fixed thresholds (1.0% .. 2.0% in 0.1% steps)
       - DYNAMIC: 11 multipliers k (1.0 .. 2.0 in 0.1 steps) of the
                  rolling 30-day sigma of daily returns
                  (tau_t = k * sigma_30(r))
  5. One walk-forward per label setting; both 730d and 365d backtest
     windows produced from the same prediction series (slicing).
  6. Per-window reports under reports/utc2130/<suffix>_<window>d/.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest import BacktestResult, run_backtest
from src.features import build_features, build_labels, compute_dynamic_threshold
from src.model import WFConfig, _xgb_to_ebm, feature_importance, walk_forward_predict
from src.utc2130 import build_utc2130_daily


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_BASE = ROOT / "reports"
REPORT_ROOT = REPORT_BASE / "utc2130"   # legacy default (overridden per-variant)


def _cache_dir_for(model_kind: str) -> Path:
    return ROOT / "data" / f"preds_cache_{model_kind}"


def _model_kind() -> str:
    """Resolve the active model kind for output-path tagging.

    All report-directory artifacts that used to be written without a model
    suffix (threshold_sweep_*, label_threshold_sweep_summary_*,
    label_prob_threshold_tables_*, equity_curve_*, label_x_prob_*_heatmap_*)
    now include {model_kind} so EBM and XGB results coexist in the same
    variant directory instead of clobbering each other.
    """
    return os.environ.get("MODEL_KIND", "ebm").lower()


def _report_root_for_variant(variant) -> Path:
    """Where to write a variant's report tree, model-aware.

    EBM keeps the original `reports/<variant.report_subdir>/...` layout
    (e.g. `reports/utc2130/utc2130_v3_close_365d/`). XGB results route
    into a sibling `reports/xgb/<variant.report_subdir>/...` so the two
    model trees never share a directory — EBM analysis scripts and
    v5_keep_selector_ebm.py can keep globbing `reports/<subdir>/...`
    without ever picking up XGB CSVs by accident, and `reports/xgb/...`
    becomes the one place to look for the XGB baseline.
    """
    base = REPORT_BASE / "xgb" if _model_kind() == "xgb" else REPORT_BASE
    return base / variant.report_subdir


# legacy alias — kept so external imports (analysis scripts) that grep the
# constant still resolve. Always points at the EBM cache by default; the
# walk-forward function uses `_cache_dir_for(cfg.model_kind)` internally.
CACHE_DIR = _cache_dir_for("ebm")

TRAINING_START = pd.Timestamp("2019-12-01", tz="UTC")
WINDOWS_DAYS = (730, 365)
STATIC_THRESHOLDS = [round(0.010 + i * 0.001, 3) for i in range(11)]
DYNAMIC_K_VALUES  = [round(1.0 + i * 0.1, 1) for i in range(11)]
DYNAMIC_N_DAYS = 30
PROB_THRESHOLDS  = [round(0.50 + i * 0.01, 2) for i in range(21)]


@dataclass
class VariantConfig:
    suffix: str
    title: str
    x_modifier: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    refit_calendar: str = "M"   # 'M' = monthly (1st), 'SM' = semi-monthly (1st & 16th), 'W-SUN' = weekly Sunday
    use_mvrv_v2: bool = False   # if True, use the improved synthetic MVRV-Z features (incl. free Coin Metrics)
    use_features_v3: bool = False  # if True, v3 mode = v2 + cross-asset corr + funding skew + hashrate ribbon + ... and drop redundant cols
    use_features_v4: bool = False  # if True, v4 mode = v3 + FRED M2 (US Money Stock) + restore the 4 macro z_60d that v3 dropped
    use_features_v5: bool = False  # if True, v5 mode = v4 features pruned to top-60 by aggregated importance
    use_features_v5_2: bool = False  # if True, v5_2 mode = looser prune (~cum 0.92 vs v5's 0.80)
    use_features_v7: bool = False    # if True, v7 = v4 + FRED Tier-1 expansion + GDELT news features
    use_features_v5_3: bool = False  # if True, v5_3 = pruned v7 keep (~cum 0.93)
    wfconfig_overrides: dict | None = None  # additional XGBoost hyperparam overrides for WFConfig (e.g. {"reg_alpha": 0.8, "max_depth": 3})
    bar_builder: Callable[[Path], pd.DataFrame] | None = None  # daily-bar builder (None → utc2130). e.g. build_utc0000_daily
    report_subdir: str = "utc2130"  # results land under reports/<report_subdir>/


def _label_slug(mode: str, value: float, suffix: str) -> str:
    """Per-label cache/report slug.

    mode='static' value=0.015 -> 'th15_<suffix>'
    mode='dynamic' value=1.5  -> 'dynk15_<suffix>'
    """
    if mode == "static":
        tag = f"th{int(round(value * 1000))}"
    else:
        tag = f"dynk{int(round(value * 10))}"
    return f"{tag}_{suffix}"


def _walk_forward_one(
    mode: str,
    value: float,
    cfg_variant: VariantConfig,
    X: pd.DataFrame,
    close: pd.Series,
    cold_start: int,
    retrain: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict]:
    """Run walk-forward for one (mode, value) label setting (with caching)."""
    slug = _label_slug(mode, value, cfg_variant.suffix)
    model_kind = (cfg_variant.wfconfig_overrides or {}).get("model_kind") \
                  or os.environ.get("MODEL_KIND", "ebm")
    cache_dir = _cache_dir_for(model_kind)
    pred_cache = cache_dir / f"preds_{model_kind}_{slug}.parquet"
    fi_cache = cache_dir / f"fi_{model_kind}_{slug}.parquet"

    if mode == "static":
        threshold = value
        y = build_labels(close, threshold)
    else:
        threshold_series = compute_dynamic_threshold(close, k=value, n=DYNAMIC_N_DAYS)
        y = build_labels(close, threshold_series)

    n_up = int((y == 1.0).sum())
    n_dn = int((y == 0.0).sum())
    n_ig = int(y.isna().sum())

    backtest_cutoff = X.index.max() - pd.Timedelta(days=max(WINDOWS_DAYS))
    cfg_kwargs = dict(initial_train_days=cold_start,
                      refit_calendar=cfg_variant.refit_calendar)
    if cfg_variant.wfconfig_overrides:
        cfg_kwargs.update(_xgb_to_ebm(cfg_variant.wfconfig_overrides))
    cfg = WFConfig(**cfg_kwargs)

    print(f"\n--- {mode} {value:.3f}  ({slug}) "
          f"UP={n_up} DN={n_dn} NaN={n_ig} "
          f"refit_calendar={cfg_variant.refit_calendar} "
          f"model={cfg.model_kind} cal_method={cfg.cal_method!r} cal_cv={cfg.cal_cv} ---",
          flush=True)

    cache_ready = (not retrain and pred_cache.exists() and fi_cache.exists())
    if cache_ready:
        preds_full = pd.read_parquet(pred_cache)
        fi = pd.read_parquet(fi_cache)
        print(f"    cache HIT  (cold_start={cold_start})", flush=True)
    else:
        print(f"    training... cold_start={cold_start}  "
              f"refit_calendar={cfg_variant.refit_calendar}", flush=True)
        preds_full = walk_forward_predict(
            X, y, cfg,
            progress_every=100,
            progress_label=slug,
        )
        fi = feature_importance(
            X.loc[X.index < backtest_cutoff],
            y.loc[X.index < backtest_cutoff],
            cfg,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        preds_full.to_parquet(pred_cache)
        fi.to_parquet(fi_cache)

    stats = {"n_up": n_up, "n_dn": n_dn, "n_ig": n_ig,
             "ignore_pct": n_ig / max(1, len(y))}
    return preds_full, fi, slug, stats


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


def _plot_one(results, out_path, label_str, window_days, title_extra=""):
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
    ax.set_title(f"{label_str} — {window_days}d backtest "
                 f"— equity by BUY threshold (UTC2130{title_extra})")
    ax.set_ylabel("USD")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_heatmap(matrix: pd.DataFrame, out_path: Path, window_days: int,
                  mode: str, title_extra: str = "") -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 6))
    arr = matrix.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    if mode == "static":
        ax.set_yticklabels([f"±{idx*100:.1f}%" for idx in matrix.index])
        ylabel = "Static label threshold"
    else:
        ax.set_yticklabels([f"k={idx:.1f}" for idx in matrix.index])
        ylabel = f"Dynamic k (× sigma_{DYNAMIC_N_DAYS}d)"
    ax.set_xlabel("BUY probability threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{window_days}-day backtest return — {mode} label × prob "
                 f"(UTC2130{title_extra})")
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


def _generate_markdown(static_runs: list[dict], dynamic_runs: list[dict],
                       static_matrix: pd.DataFrame, dynamic_matrix: pd.DataFrame,
                       out_path: Path, window_days: int, title_extra: str,
                       report_dir: Path,
                       refit_calendar: str = "M") -> None:
    cadence_desc = {
        "M":  "calendar-monthly refit (refit on 1st of each month)",
        "SM": "semi-monthly refit (refit on 1st AND 16th of each month)",
    }.get(refit_calendar, f"refit_calendar={refit_calendar}")
    lines = []
    lines.append(f"# UTC2130 — {window_days}-day backtest{title_extra}")
    lines.append("")
    lines.append(f"**Setup**: 21:30 UTC aligned daily bar (= 06:30 KST next day), "
                 f"{cadence_desc}, "
                 f"external feeds joined with lag = 0 (all same-day macro/funding/"
                 f"fng publications complete by 21:30 UTC). "
                 f"Training from 2019-12-01. Backtest = last {window_days} days.")
    lines.append("")
    lines.append("Two label modes:")
    lines.append(f"  - **Static**: y=±1 if next-day return crosses ±threshold "
                 f"({STATIC_THRESHOLDS[0]*100:.1f}% .. {STATIC_THRESHOLDS[-1]*100:.1f}%)")
    lines.append(f"  - **Dynamic**: tau_t = k · sigma_{DYNAMIC_N_DAYS}d(r), "
                 f"k ∈ {DYNAMIC_K_VALUES}")
    lines.append("")

    bh = static_runs[0]["buy_hold"] if static_runs else float("nan")
    lines.append(f"Buy & Hold ({window_days}d window): {bh*100:+.2f}%")
    lines.append("")

    # Section 1: Static summary
    lines.append("## STATIC labels — best prob_TH per label")
    lines.append("")
    lines.append("| label | best_prob | trades | win% | return | final $ | acc@.5 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in static_runs:
        lines.append(
            f"| ±{r['value']*100:.1f}% "
            f"| {r['best_prob_th']:.2f} "
            f"| {r['best_trades']} "
            f"| {r['best_winrate']*100:.1f}% "
            f"| **{r['best_return']*100:+.2f}%** "
            f"| ${r['best_equity']:.2f} "
            f"| {r['acc_at_05']*100:.1f}% |"
        )
    lines.append("")

    # Section 2: Dynamic summary
    lines.append("## DYNAMIC labels — best prob_TH per k")
    lines.append("")
    lines.append("| k | best_prob | trades | win% | return | final $ | acc@.5 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in dynamic_runs:
        lines.append(
            f"| {r['value']:.1f} "
            f"| {r['best_prob_th']:.2f} "
            f"| {r['best_trades']} "
            f"| {r['best_winrate']*100:.1f}% "
            f"| **{r['best_return']*100:+.2f}%** "
            f"| ${r['best_equity']:.2f} "
            f"| {r['acc_at_05']*100:.1f}% |"
        )
    lines.append("")

    # Overall best
    if not static_matrix.empty:
        s_best = static_matrix.stack().idxmax()
        s_val = static_matrix.stack().max()
        lines.append("## OVERALL BEST")
        lines.append(f"  - **Static**: label ±{s_best[0]*100:.1f}% × prob {s_best[1]:.2f} "
                     f"→ {s_val*100:+.2f}%")
    if not dynamic_matrix.empty:
        d_best = dynamic_matrix.stack().idxmax()
        d_val = dynamic_matrix.stack().max()
        lines.append(f"  - **Dynamic**: k={d_best[0]:.1f} × prob {d_best[1]:.2f} "
                     f"→ {d_val*100:+.2f}%")
    lines.append(f"  - **Buy & Hold**: {bh*100:+.2f}%")
    lines.append("")

    # Per-label full prob-threshold tables (STATIC)
    if static_runs:
        lines.append("---")
        lines.append("")
        lines.append("## Full prob-threshold sweep — STATIC labels")
        lines.append("")
        for r in static_runs:
            slug = r["slug"]
            csv_path = report_dir / f"threshold_sweep_{_model_kind()}_{slug}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path).sort_values("threshold").reset_index(drop=True)
            best_idx = int(df["total_return_pct"].idxmax())
            bh_local = df["buy_and_hold_pct"].iloc[0]
            lines.append(f"### Static label ±{r['value']*100:.1f}%  ({slug})")
            lines.append("")
            lines.append(f"Buy & Hold (window): {bh_local*100:+.2f}% · "
                         f"acc@0.5: {r['acc_at_05']*100:.1f}% · "
                         f"prob mean: {r['prob_mean']:.3f} · "
                         f"train labels: UP={r['n_train_labels_up']} / "
                         f"DN={r['n_train_labels_dn']} / "
                         f"NaN={r['ignore_pct']*100:.1f}%")
            lines.append("")
            lines.append("| prob_TH | trades | win% | BUY days | HOLD days | "
                         "SELL days | in-market | return | final $ |")
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

    # Per-k full prob-threshold tables (DYNAMIC)
    if dynamic_runs:
        lines.append("---")
        lines.append("")
        lines.append("## Full prob-threshold sweep — DYNAMIC labels  "
                     f"(τ = k × σ_{DYNAMIC_N_DAYS}d)")
        lines.append("")
        for r in dynamic_runs:
            slug = r["slug"]
            csv_path = report_dir / f"threshold_sweep_{_model_kind()}_{slug}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path).sort_values("threshold").reset_index(drop=True)
            best_idx = int(df["total_return_pct"].idxmax())
            bh_local = df["buy_and_hold_pct"].iloc[0]
            lines.append(f"### Dynamic k={r['value']:.1f}  ({slug})")
            lines.append("")
            lines.append(f"Buy & Hold (window): {bh_local*100:+.2f}% · "
                         f"acc@0.5: {r['acc_at_05']*100:.1f}% · "
                         f"prob mean: {r['prob_mean']:.3f} · "
                         f"train labels: UP={r['n_train_labels_up']} / "
                         f"DN={r['n_train_labels_dn']} / "
                         f"NaN={r['ignore_pct']*100:.1f}%")
            lines.append("")
            lines.append("| prob_TH | trades | win% | BUY days | HOLD days | "
                         "SELL days | in-market | return | final $ |")
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

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _process_label_runs(
    mode: str,
    label_runs: list[dict],
    window_days: int,
    report_dir: Path,
    title_extra: str,
) -> tuple[list[dict], pd.DataFrame]:
    """Run backtests for a window and emit per-label outputs. Returns the
    per-label summary list + the (label × prob) return matrix for heatmap."""
    suffix_w_label = f"{label_runs[0]['variant_suffix']}_{window_days}d"
    runs = []
    for lr in label_runs:
        results, bt_stats = _backtest_for_window(
            lr["preds_full"], lr["close"], window_days)

        slug_w = f"{lr['slug']}_{window_days}d"
        sweep_rows = [{
            "threshold": th, "trades": r.n_trades, "win_rate": r.win_rate,
            "days_buy": r.days_buy, "days_hold": r.days_sideways,
            "days_sell": r.days_sell, "days_in_market": r.days_in_market,
            "total_return_pct": r.total_return_pct,
            "final_equity_usd": r.final_equity,
            "buy_and_hold_pct": r.buy_and_hold_return_pct,
        } for th, r in results.items()]
        model_kind = _model_kind()
        pd.DataFrame(sweep_rows).to_csv(
            report_dir / f"threshold_sweep_{model_kind}_{slug_w}.csv",
            index=False)
        lr["fi"].to_csv(
            report_dir / f"feature_importance_{model_kind}_{slug_w}.csv",
            index=False)
        label_str = (f"label ±{lr['value']*100:.1f}%" if mode == "static"
                     else f"k={lr['value']:.1f}")
        _plot_one(results,
                  report_dir / f"equity_curve_{model_kind}_{slug_w}.png",
                  label_str, window_days,
                  title_extra=" — " + lr["variant_title"] if lr["variant_title"] else "")

        best_th, best_res = max(results.items(),
                                key=lambda kv: kv[1].total_return_pct)
        if best_res.trades:
            pd.DataFrame([t.__dict__ for t in best_res.trades]).to_csv(
                report_dir / f"trades_{model_kind}_{slug_w}_th{best_th:.2f}.csv",
                index=False)

        prefix = "[" + str(window_days) + "d " + mode + "]"
        print(f"    {prefix} {label_str:>15s} → BEST prob_TH={best_th:.2f}  "
              f"ret={best_res.total_return_pct:+.2%}  trades={best_res.n_trades}  "
              f"win={best_res.win_rate:.1%}  acc@0.5={bt_stats['acc_at_05']:.1%}",
              flush=True)

        runs.append({
            "value": lr["value"], "slug": slug_w,
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

    matrix = pd.DataFrame(
        index=[r["value"] for r in runs],
        columns=PROB_THRESHOLDS, dtype=float)
    for r in runs:
        for th, res in r["results"].items():
            matrix.loc[r["value"], th] = res.total_return_pct
    return runs, matrix


def run_variant(variant: VariantConfig, retrain: bool = False) -> None:
    title_extra = f" — {variant.title}" if variant.title else ""

    print("=" * 78)
    print(f"UTC2130 STATIC + DYNAMIC sweep — variant={variant.suffix}")
    print(f"  refit_calendar={variant.refit_calendar}, external_lag_days=0,")
    print(f"  training from 2019-12-01, windows: {WINDOWS_DAYS}d")
    print(f"  STATIC  label thresholds: "
          f"{[round(t*100,1) for t in STATIC_THRESHOLDS]} %")
    print(f"  DYNAMIC k values        : {DYNAMIC_K_VALUES} "
          f"(× sigma_{DYNAMIC_N_DAYS}d)")
    print(f"  prob thresholds         : 0.50..0.70 step 0.01 "
          f"({len(PROB_THRESHOLDS)})")
    print("=" * 78, flush=True)

    bar_fn = variant.bar_builder or build_utc2130_daily
    print(f"[fetch] building daily bar via {bar_fn.__name__} ...", flush=True)
    bar = bar_fn(DATA_DIR / "btc_15m.parquet")
    print(f"  {len(bar)} rows, {bar['close_time'].iloc[0].date()} -> "
          f"{bar['close_time'].iloc[-1].date()}", flush=True)

    print(f"[features] building feature matrix once (lag=0, "
          f"use_mvrv_v2={variant.use_mvrv_v2}, "
          f"use_features_v3={variant.use_features_v3}, "
          f"use_features_v4={variant.use_features_v4}, "
          f"use_features_v5={variant.use_features_v5}, "
          f"use_features_v5_2={variant.use_features_v5_2}, "
          f"use_features_v7={variant.use_features_v7}, "
          f"use_features_v5_3={variant.use_features_v5_3}, "
          f"wfconfig_overrides={variant.wfconfig_overrides})...", flush=True)
    # Use a sentinel label_threshold; we'll override with build_labels later.
    X_full, _, close_full = build_features(
        DATA_DIR, label_threshold=0.015,
        daily_df=bar, external_lag_days=0,
        use_mvrv_v2=variant.use_mvrv_v2,
        use_features_v3=variant.use_features_v3,
        use_features_v4=variant.use_features_v4,
        use_features_v5=variant.use_features_v5,
        use_features_v5_2=variant.use_features_v5_2,
        use_features_v7=variant.use_features_v7,
        use_features_v5_3=variant.use_features_v5_3,
    )
    if variant.x_modifier is not None:
        X_full = variant.x_modifier(X_full)

    train_mask = X_full.index >= TRAINING_START
    X = X_full.loc[train_mask].copy()
    close = close_full.loc[X.index].copy()
    print(f"  X: {X.shape}  close: {len(close)} rows  "
          f"({X.index.min().date()} -> {X.index.max().date()})",
          flush=True)

    backtest_cutoff = X.index.max() - pd.Timedelta(days=max(WINDOWS_DAYS))
    cold_start = max(60, int((X.index < backtest_cutoff).sum()))

    # Walk-forward — STATIC labels
    static_label_runs = []
    for th in STATIC_THRESHOLDS:
        preds_full, fi, slug, stats = _walk_forward_one(
            "static", th, variant, X, close, cold_start, retrain)
        static_label_runs.append({
            "value": th, "slug": slug, "preds_full": preds_full,
            "fi": fi, "close": close,
            "variant_suffix": variant.suffix,
            "variant_title": variant.title, **stats,
        })

    # Walk-forward — DYNAMIC labels
    dynamic_label_runs = []
    for k in DYNAMIC_K_VALUES:
        preds_full, fi, slug, stats = _walk_forward_one(
            "dynamic", k, variant, X, close, cold_start, retrain)
        dynamic_label_runs.append({
            "value": k, "slug": slug, "preds_full": preds_full,
            "fi": fi, "close": close,
            "variant_suffix": variant.suffix,
            "variant_title": variant.title, **stats,
        })

    report_root_for_variant = _report_root_for_variant(variant)
    # For each window, run all backtests + per-window outputs.
    for window_days in WINDOWS_DAYS:
        suffix_w = f"{variant.suffix}_{window_days}d"
        report_dir = report_root_for_variant / suffix_w
        report_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== window={window_days}d  → {report_dir.relative_to(ROOT)}",
              flush=True)

        static_runs, static_matrix = _process_label_runs(
            "static", static_label_runs, window_days, report_dir, title_extra)
        dynamic_runs, dynamic_matrix = _process_label_runs(
            "dynamic", dynamic_label_runs, window_days, report_dir, title_extra)

        # heatmaps (model-suffixed so EBM/XGB coexist)
        mk = _model_kind()
        static_matrix.to_csv(
            report_dir / f"label_x_prob_static_heatmap_{mk}_{suffix_w}.csv")
        _plot_heatmap(static_matrix,
                      report_dir / f"label_x_prob_static_heatmap_{mk}_{suffix_w}.png",
                      window_days, "static",
                      title_extra=" — " + variant.title if variant.title else "")

        dynamic_matrix.to_csv(
            report_dir / f"label_x_prob_dynamic_heatmap_{mk}_{suffix_w}.csv")
        _plot_heatmap(dynamic_matrix,
                      report_dir / f"label_x_prob_dynamic_heatmap_{mk}_{suffix_w}.png",
                      window_days, "dynamic",
                      title_extra=" — " + variant.title if variant.title else "")

        # summary CSVs (model-suffixed)
        pd.DataFrame([{k: v for k, v in r.items() if k != "results"}
                      for r in static_runs]).to_csv(
            report_dir / f"label_threshold_sweep_summary_static_{mk}_{suffix_w}.csv",
            index=False)
        pd.DataFrame([{k: v for k, v in r.items() if k != "results"}
                      for r in dynamic_runs]).to_csv(
            report_dir / f"label_threshold_sweep_summary_dynamic_{mk}_{suffix_w}.csv",
            index=False)

        # markdown (model-suffixed)
        md_path = report_dir / f"label_prob_threshold_tables_{mk}_{suffix_w}.md"
        _generate_markdown(static_runs, dynamic_runs,
                           static_matrix, dynamic_matrix,
                           md_path, window_days,
                           title_extra=" — " + variant.title if variant.title else "",
                           report_dir=report_dir,
                           refit_calendar=variant.refit_calendar)

        bh = static_runs[0]["buy_hold"]
        s_overall = static_matrix.stack().idxmax()
        s_overall_val = static_matrix.stack().max()
        d_overall = dynamic_matrix.stack().idxmax()
        d_overall_val = dynamic_matrix.stack().max()
        print(f"    [{window_days}d] B&H={bh:+.2%}", flush=True)
        print(f"    [{window_days}d] STATIC  BEST: ±{s_overall[0]*100:.1f}% × "
              f"prob {s_overall[1]:.2f} → {s_overall_val:+.2%}", flush=True)
        print(f"    [{window_days}d] DYNAMIC BEST: k={d_overall[0]:.1f} × "
              f"prob {d_overall[1]:.2f} → {d_overall_val:+.2%}", flush=True)
        print(f"    Markdown: {md_path}", flush=True)
