"""UTC2130 backtest — `is_macro_offday` flag variant.

Same as main_th_sweep_utc2130.py but adds a single binary feature
`is_macro_offday` (= weekend OR US federal holiday). Macros stay NaN on
those rows; the flag lets the model learn to discount them.

Run:
    uv run python main_th_sweep_utc2130_close.py            # cached predictions
    uv run python main_th_sweep_utc2130_close.py --retrain  # force refit
"""

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


VARIANT = VariantConfig(
    suffix="utc2130_close",
    title="offday-flag",
    x_modifier=_add_macro_offday_feature,
)


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
