"""UTC2130 + v5_3 (pruned v7, ~cum 0.93), monthly cadence."""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


from src.utc2130_runner import VariantConfig, run_variant


VARIANT = VariantConfig(suffix="utc2130_v5_3",
                        title="v5_3 (~93% pruned, v7 universe)",
                        refit_calendar="M", use_features_v5_3=True)


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
