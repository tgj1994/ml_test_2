"""UTC2130 backtest — base variant with SEMI-MONTHLY retraining.

Same as main_th_sweep_utc2130.py but model is refit on the 1st AND 16th of
each calendar month (refit_calendar='SM') instead of only the 1st. About 2x
the refit frequency vs the monthly variant.

Run:
    uv run python main_th_sweep_utc2130_sm.py            # cached predictions
    uv run python main_th_sweep_utc2130_sm.py --retrain  # force refit
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


from src.utc2130_runner import VariantConfig, run_variant


VARIANT = VariantConfig(suffix="utc2130_sm", title="SM 1st&16th",
                        refit_calendar="SM")


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
