"""UTC2130 + v3 features + ffill+bfill, monthly refit."""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


import pandas as pd

from src.utc2130_runner import VariantConfig, run_variant


def _fill_gaps(X: pd.DataFrame) -> pd.DataFrame:
    return X.sort_index().ffill().bfill()


VARIANT = VariantConfig(
    suffix="utc2130_v3_complete",
    title="v3 features, ffill+bfill",
    x_modifier=_fill_gaps,
    refit_calendar="M",
    use_features_v3=True,
)


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
