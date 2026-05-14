"""Walk-forward classifier (EBM by default, XGBoost as the comparison baseline).

The model is chosen by `WFConfig.model_kind`:
  - 'ebm' (default): `interpret.glassbox.ExplainableBoostingClassifier`
  - 'xgb'          : `xgboost.XGBClassifier` matching the legacy coin_analysis
                     hyperparameters so the XGBoost run on this revised data
                     gives a true apples-to-apples baseline alongside the EBM.

Both kinds share the same `walk_forward_predict` / `feature_importance`
public interface so `utc2130_runner` / `kst10_runner` / `main_th_sweep` /
`analysis` work transparently — the only thing that changes is the cache
subdirectory and the `_xgb`/`_ebm` filename prefix, both wired off the
`model_kind` field.

This is the EBM rewrite of the original XGBoost+CalibratedClassifierCV pipeline.
Public API (`WFConfig`, `walk_forward_predict`, `feature_importance`) is kept
identical so that `src/utc2130_runner.py`, `src/kst10_runner.py`,
`main_th_sweep/`, `retrain/`, and `analysis/` need only a path/prefix change,
not a structural rewrite.

Pipeline at every refit:
  1. Drop training rows with NaN labels (filtered "strong move" target).
  2. Fit an `interpret.glassbox.ExplainableBoostingClassifier` on the labelled
     subset. EBM is a GA^2M (additive main effects + selected pairwise
     interactions), trained by cyclical boosting on round-robin shape
     functions. It produces probability scores that are already
     well-calibrated in the additive logistic-regression sense.
  3. Wrap in `CalibratedClassifierCV(method='isotonic', cv=3)` to apply an
     additional monotone calibration. This is optional; we keep it for parity
     with the XGBoost pipeline so that probability semantics match between
     model swaps. If `WFConfig.cal_method='none'` is passed, the raw EBM is
     returned.
  4. Predict next-day prob_up for every row (including unlabeled IGNORE days)
     so the threshold-based backtest can run on the full daily series.

XGBoost <-> EBM hyperparameter conceptual mapping (used in v6 tight reg):
  - reg_alpha / reg_lambda  -> no direct equivalent. We approximate "tight reg"
    by forcing main-effects-only (interactions=0) and shallow shape functions
    (max_leaves=4, min_samples_leaf=20).
  - colsample_bytree=0.5    -> EBM does not subsample features; we instead
    drop interactions (interactions=0) which is the dominant overfit channel.
  - max_depth=3             -> approx. max_leaves=4 on each main-effect.
  - min_child_weight=10     -> min_samples_leaf=20.

`v6` variants pass these via `VariantConfig.wfconfig_overrides`. The XGBoost
keys (`reg_alpha` etc.) in v6 configs are translated below in `_xgb_to_ebm`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


_N_JOBS_DEFAULT = int(os.environ.get("EBM_N_JOBS",
                                     max(1, (os.cpu_count() or 2) - 2)))
# `MODEL_KIND` env var lets a sweep override the WFConfig default without
# touching every main_th_sweep variant. Accepted values: 'ebm' | 'xgb'.
_MODEL_KIND_DEFAULT = os.environ.get("MODEL_KIND", "ebm").lower()


@dataclass
class WFConfig:
    initial_train_days: int = 365
    refit_every: int = 14
    refit_calendar: str | None = None
    seed: int = 42

    # 'ebm' (default) or 'xgb' for the apples-to-apples baseline.
    model_kind: str = _MODEL_KIND_DEFAULT

    # --- EBM hyperparameters (model_kind='ebm') --------------------------
    # Tuned for ~1-2s/fit at 2000 rows × 120 cols so the 30-variant ×
    # 22-label sweep completes in ~10-15 hours on this 6c/12t / 32 GB box,
    # leaving headroom for other services and bitcoind IBD.
    max_bins: int = 256
    max_interaction_bins: int = 16
    interactions: int = 5
    outer_bags: int = 4
    learning_rate: float = 0.01
    max_leaves: int = 3
    min_samples_leaf: int = 4
    max_rounds: int = 2000
    n_jobs: int = _N_JOBS_DEFAULT

    # --- XGBoost hyperparameters (model_kind='xgb') ----------------------
    # Mirror the legacy coin_analysis defaults so the baseline is a faithful
    # rerun on this revised data, not a new tuning experiment.
    xgb_n_estimators: int = 400
    xgb_learning_rate: float = 0.04
    xgb_max_depth: int = 4
    xgb_min_child_weight: float = 6.0
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.2
    xgb_reg_lambda: float = 1.5
    xgb_gamma: float = 0.2

    # Calibration. EBM is a GA^2M trained with a logistic link so its raw
    # predict_proba is already well-calibrated in the additive sense. We
    # default to no extra wrap to keep the full 30-variant EBM sweep
    # tractable. For the XGBoost baseline we re-enable isotonic with cv=3
    # to match the original coin_analysis pipeline exactly.
    cal_cv: int = 3
    cal_method: str = "none"

    # Minimum labelled samples required before fitting
    min_train_samples: int = 60

    def __post_init__(self):
        # XGB: default cal_method='isotonic' (legacy coin_analysis baseline)
        # unless XGB_NO_CALIB=1 explicitly disables it.
        if (self.model_kind == "xgb" and self.cal_method == "none"
                and os.environ.get("XGB_NO_CALIB", "").lower()
                    not in ("1", "true", "yes")):
            self.cal_method = "isotonic"

        # EBM: original codepath left cal_method='none' on the theory that
        # GA^2M's logistic link is already well-calibrated. Threshold-based
        # buy/sell/hold decisions need TRUE probability calibration, so
        # setting EBM_CAL_METHOD=isotonic (or =sigmoid) in the sweep env
        # opts into CalibratedClassifierCV(cv=3) wrapping for EBM too.
        if self.model_kind == "ebm" and self.cal_method == "none":
            ebm_cal = os.environ.get("EBM_CAL_METHOD", "").lower()
            if ebm_cal in ("isotonic", "sigmoid"):
                self.cal_method = ebm_cal


# Map legacy XGBoost hyperparam names (used in v6 wfconfig_overrides) into
# the closest EBM equivalents. Unknown keys pass through unchanged so
# misconfigurations surface immediately at WFConfig() time.
_XGB_TO_EBM = {
    "reg_alpha":        ("interactions",         lambda v: 0 if v >= 0.5 else 10),
    "reg_lambda":       ("max_leaves",           lambda v: max(2, int(round(6 - v)))),
    "colsample_bytree": ("max_interaction_bins", lambda v: max(8, int(round(32 * v)))),
    "max_depth":        ("max_leaves",           lambda v: max(2, 1 << max(1, int(v) - 1))),
    "min_child_weight": ("min_samples_leaf",     lambda v: max(4, int(round(2 * v)))),
    "subsample":        (None, None),
    "gamma":            (None, None),
    "n_estimators":     ("max_rounds",           lambda v: int(v) * 5),
    # learning_rate has a direct name match; EBM lr is ~4x smaller than XGB.
    "learning_rate":    ("learning_rate",        lambda v: float(v) * 0.25),
}


def _xgb_to_ebm(overrides: dict | None) -> dict:
    if not overrides:
        return {}
    out: dict = {}
    for k, v in overrides.items():
        if k in _XGB_TO_EBM:
            mapped_k, fn = _XGB_TO_EBM[k]
            if mapped_k is None:
                continue
            out[mapped_k] = fn(v)
        else:
            out[k] = v
    return out


def _refit_group_series(index: pd.DatetimeIndex, refit_calendar: str) -> list:
    if refit_calendar == "M":
        return [(ts.year, ts.month) for ts in index]
    if refit_calendar == "SM":
        return [(ts.year, ts.month, 1 if ts.day < 16 else 2) for ts in index]
    if refit_calendar == "W-SUN":
        out = []
        for ts in index:
            back = (ts.dayofweek + 1) % 7
            sun = ts.normalize() - pd.Timedelta(days=back)
            out.append(sun.date())
        return out
    raise ValueError(f"unsupported refit_calendar: {refit_calendar!r}")


def _make_ebm(cfg: WFConfig) -> ExplainableBoostingClassifier:
    return ExplainableBoostingClassifier(
        max_bins=cfg.max_bins,
        max_interaction_bins=cfg.max_interaction_bins,
        interactions=cfg.interactions,
        outer_bags=cfg.outer_bags,
        learning_rate=cfg.learning_rate,
        max_leaves=cfg.max_leaves,
        min_samples_leaf=cfg.min_samples_leaf,
        max_rounds=cfg.max_rounds,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
    )


def _make_xgb(cfg: WFConfig):
    """Build the legacy-equivalent XGBoost classifier for the baseline run.

    device: read from XGB_DEVICE env var ('cuda' or 'cpu'). Defaults to 'cuda'
    so the RTX 2060 (CUDA 12.x build) accelerates each fit; set XGB_DEVICE=cpu
    to fall back to CPU hist when GPU is unavailable.
    """
    import xgboost as xgb
    device = os.environ.get("XGB_DEVICE", "cuda").lower()
    return xgb.XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        learning_rate=cfg.xgb_learning_rate,
        max_depth=cfg.xgb_max_depth,
        min_child_weight=cfg.xgb_min_child_weight,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        reg_alpha=cfg.xgb_reg_alpha,
        reg_lambda=cfg.xgb_reg_lambda,
        gamma=cfg.xgb_gamma,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        verbosity=0,
    )


def _make_base(cfg: WFConfig):
    if cfg.model_kind == "ebm":
        return _make_ebm(cfg)
    if cfg.model_kind == "xgb":
        return _make_xgb(cfg)
    raise ValueError(f"unsupported model_kind: {cfg.model_kind!r}")


def _fit_calibrated(X: pd.DataFrame, y: pd.Series, cfg: WFConfig) -> Any:
    mask = y.notna()
    Xl = X.loc[mask]
    yl = y.loc[mask].astype(int)
    if len(Xl) < cfg.min_train_samples or yl.nunique() < 2:
        return None
    if cfg.model_kind == "xgb":
        # XGB's QuantileDMatrix (used by tree_method='hist') rejects inf values
        # outright unless missing=inf, while EBM tolerates them via its
        # histogram binning. Coerce +/-inf to NaN so the XGB baseline can fit
        # on the same feature matrix as EBM without changing features.py.
        Xl = Xl.replace([np.inf, -np.inf], np.nan)
    base = _make_base(cfg)
    if cfg.cal_method == "none":
        base.fit(Xl, yl)
        return base
    cal = CalibratedClassifierCV(estimator=base, method=cfg.cal_method,
                                 cv=cfg.cal_cv)
    cal.fit(Xl, yl)
    return cal


def walk_forward_predict(X: pd.DataFrame,
                         y: pd.Series,
                         cfg: WFConfig | None = None,
                         progress_every: int = 0,
                         progress_label: str = "") -> pd.DataFrame:
    import time
    cfg = cfg or WFConfig()
    if cfg.model_kind == "xgb":
        # See _fit_calibrated: XGB rejects inf at predict time too (the model
        # caches its quantile cuts from training). Replace once at entry so
        # both the rolling fits and the per-row predict_proba calls below see
        # NaN where features.py emits inf.
        X = X.replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(index=X.index, columns=["prob_up", "actual"], dtype=float)
    out["actual"] = y.astype(float)

    n = len(X)
    if cfg.initial_train_days >= n:
        raise ValueError("Not enough data for cold-start window")

    n_steps = n - cfg.initial_train_days
    t0 = time.time()

    use_calendar = cfg.refit_calendar is not None
    refit_groups = _refit_group_series(X.index, cfg.refit_calendar) if use_calendar else None

    model: Any = None
    days_since_fit = 0
    prev_group = None
    n_refits = 0
    for i in range(cfg.initial_train_days, n):
        if use_calendar:
            cur_group = refit_groups[i]
            should_refit = (model is None) or (cur_group != prev_group)
            if should_refit:
                model = _fit_calibrated(X.iloc[:i], y.iloc[:i], cfg)
                prev_group = cur_group
                n_refits += 1
        else:
            if model is None or days_since_fit >= cfg.refit_every:
                model = _fit_calibrated(X.iloc[:i], y.iloc[:i], cfg)
                days_since_fit = 0
                n_refits += 1
        if model is not None:
            prob = float(model.predict_proba(X.iloc[[i]])[0, 1])
            out.iloc[i, out.columns.get_loc("prob_up")] = prob
        days_since_fit += 1

        if progress_every and ((i - cfg.initial_train_days + 1) % progress_every == 0
                               or i == n - 1):
            now = time.time()
            done = i - cfg.initial_train_days + 1
            elapsed = now - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (n_steps - done) / rate if rate > 0 else float("inf")
            tag = f"[{progress_label}] " if progress_label else ""
            print(f"      {tag}wf {done}/{n_steps}  "
                  f"({100*done/n_steps:.1f}%)  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m  "
                  f"rate={rate:.2f}/s  fits={n_refits}",
                  flush=True)
    return out


def _ebm_term_importances(ebm: ExplainableBoostingClassifier) -> dict[str, float]:
    """Return a {term_name: importance_score} dict from a fitted EBM.

    For univariate terms the name is the original column. For pairwise
    interaction terms the name is 'feat_a & feat_b'.
    """
    importances = ebm.term_importances(importance_type="avg_weight")
    names = []
    for term_features in ebm.term_features_:
        if len(term_features) == 1:
            names.append(ebm.feature_names_in_[term_features[0]])
        else:
            names.append(" & ".join(ebm.feature_names_in_[f] for f in term_features))
    return dict(zip(names, importances))


def feature_importance(X: pd.DataFrame, y: pd.Series,
                       cfg: WFConfig | None = None,
                       top_k: int = 20) -> pd.DataFrame:
    """Top-k feature importance for the active model_kind, returned in a
    (feature, gain) DataFrame.

    EBM rows include interaction terms named 'feat_a & feat_b'; XGBoost rows
    are univariate split-gain scores.
    """
    cfg = cfg or WFConfig()
    mask = y.notna()
    Xl, yl = X.loc[mask], y.loc[mask].astype(int)
    if cfg.model_kind == "xgb":
        # Same rationale as _fit_calibrated / walk_forward_predict — XGB's
        # QuantileDMatrix rejects inf at fit time; coerce to NaN so XGB's
        # missing-value path handles it the same way EBM silently does.
        Xl = Xl.replace([np.inf, -np.inf], np.nan)
    base = _make_base(cfg)
    base.fit(Xl, yl)
    if cfg.model_kind == "ebm":
        imp = _ebm_term_importances(base)
        rows = [{"feature": name, "gain": float(score)}
                for name, score in imp.items()]
    else:
        score = base.get_booster().get_score(importance_type="gain")
        rows = [{"feature": col, "gain": float(score.get(col, 0.0))}
                for col in X.columns]
    return (pd.DataFrame(rows)
            .sort_values("gain", ascending=False)
            .head(top_k)
            .reset_index(drop=True))
