"""UTC2130 + v5 (top-60 pruned), monthly cadence."""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


from src.utc2130_runner import VariantConfig, run_variant


VARIANT = VariantConfig(suffix="utc2130_v5", title="v5 (60-pruned)",
                        refit_calendar="M", use_features_v5=True)


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
