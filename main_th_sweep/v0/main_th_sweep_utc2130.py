"""UTC2130 backtest — base variant.

Setup:
  - Daily bars closing at 21:30 UTC (= 06:30 KST next day)
  - Calendar-monthly retraining (refit on the 1st of each month)
  - External feeds joined with lag = 0 (all same-day macro/funding/fng
    publications are complete by 21:30 UTC; no lookahead leakage)
  - Two label modes per run:
      STATIC : 11 fixed thresholds (1.0% .. 2.0%)
      DYNAMIC: tau_t = k * sigma_30(r), k = 1.0 .. 2.0
  - Both 730d and 365d windows produced from one walk-forward

Outputs land under reports/utc2130/utc2130_730d/ and reports/utc2130/utc2130_365d/.

Run:
    uv run python main_th_sweep_utc2130.py            # use cached predictions
    uv run python main_th_sweep_utc2130.py --retrain  # force walk-forward refit
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


from src.utc2130_runner import VariantConfig, run_variant


VARIANT = VariantConfig(suffix="utc2130", title="")


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
