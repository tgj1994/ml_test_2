"""UTC2130 + v4 features + offday-flag + tighter regularization (v6), SM."""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.utc2130_runner import VariantConfig, run_variant


def _add_macro_offday_feature(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    idx = pd.DatetimeIndex(X.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    is_we = idx.dayofweek >= 5
    cal = USFederalHolidayCalendar()
    hols = cal.holidays(start=idx.min(), end=idx.max())
    is_hol = idx.normalize().isin(hols)
    out["is_macro_offday"] = (is_we | is_hol).astype(float)
    return out


V6_REGULARIZATION = {
    "reg_alpha": 0.8, "reg_lambda": 3.0,
    "colsample_bytree": 0.5, "max_depth": 3, "min_child_weight": 10.0,
}


VARIANT = VariantConfig(
    suffix="utc2130_sm_v6_close",
    title="SM 1st&16th, v6 (tight reg), offday-flag",
    x_modifier=_add_macro_offday_feature,
    refit_calendar="SM",
    use_features_v4=True,
    wfconfig_overrides=V6_REGULARIZATION,
)


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
