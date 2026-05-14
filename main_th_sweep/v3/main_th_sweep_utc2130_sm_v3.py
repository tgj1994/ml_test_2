"""UTC2130 + MVRV v2 + v3 feature additions, semi-monthly refit (1st & 16th).

v3 = v2 (mvrv2_* MVRV improvements) PLUS:
  + 4 cross-asset 30d rolling correlations (BTC vs SPX/DXY/gold/ETH)
  + 2 funding-rate features (skew, velocity)
  + 2 hash-rate features (ribbon, 7d change)
  + 2 active-address features (7d change, tx-per-address)
  + 2 vol-regime features (vol-of-vol, vol z-score vs 1y)
  - 8 redundant columns dropped (d_close, cal_dow, macro z_60d × 4, sw_chop_er extremes)
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


from src.utc2130_runner import VariantConfig, run_variant


VARIANT = VariantConfig(suffix="utc2130_sm_v3", title="SM 1st&16th, v3 features",
                        refit_calendar="SM", use_features_v3=True)


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
