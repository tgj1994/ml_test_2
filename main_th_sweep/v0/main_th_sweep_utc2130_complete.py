"""UTC2130 backtest — ffill/bfill NaN bridge variant.

Same as main_th_sweep_utc2130.py but applies a forward-then-backward fill on
the feature matrix so weekend/holiday gaps in macro/futures/sentiment series
are bridged with the previous available value.

Run:
    uv run python main_th_sweep_utc2130_complete.py            # cached predictions
    uv run python main_th_sweep_utc2130_complete.py --retrain  # force refit
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


import pandas as pd

from src.utc2130_runner import VariantConfig, run_variant


def _fill_gaps(X: pd.DataFrame) -> pd.DataFrame:
    return X.sort_index().ffill().bfill()


VARIANT = VariantConfig(
    suffix="utc2130_complete",
    title="ffill+bfill",
    x_modifier=_fill_gaps,
)


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
