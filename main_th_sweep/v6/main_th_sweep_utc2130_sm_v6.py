"""UTC2130 + v4 features + tighter regularization (v6), SM cadence.

v6 keeps the full v4 feature set (116 features) but tightens XGBoost
regularization to control overfitting from the high feature/sample ratio.
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


from src.utc2130_runner import VariantConfig, run_variant


V6_REGULARIZATION = {
    "reg_alpha":         0.8,   # was 0.2 — stronger L1, push features to 0
    "reg_lambda":        3.0,   # was 1.5 — stronger L2 weight smoothing
    "colsample_bytree":  0.5,   # was 0.8 — force tree diversity
    "max_depth":         3,     # was 4 — simpler trees
    "min_child_weight":  10.0,  # was 6.0 — require more samples per leaf
}


VARIANT = VariantConfig(suffix="utc2130_sm_v6", title="SM 1st&16th, v6 (tight reg)",
                        refit_calendar="SM", use_features_v4=True,
                        wfconfig_overrides=V6_REGULARIZATION)


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
