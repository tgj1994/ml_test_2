"""UTC2130 + v4 features + tighter regularization (v6), monthly cadence."""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


from src.utc2130_runner import VariantConfig, run_variant


V6_REGULARIZATION = {
    "reg_alpha": 0.8, "reg_lambda": 3.0,
    "colsample_bytree": 0.5, "max_depth": 3, "min_child_weight": 10.0,
}


VARIANT = VariantConfig(suffix="utc2130_v6", title="v6 (tight reg)",
                        refit_calendar="M", use_features_v4=True,
                        wfconfig_overrides=V6_REGULARIZATION)


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
