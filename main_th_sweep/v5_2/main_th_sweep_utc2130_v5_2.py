"""UTC2130 + v5_2 (looser cumulative-~92% keep set), monthly cadence."""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


from src.utc2130_runner import VariantConfig, run_variant


VARIANT = VariantConfig(suffix="utc2130_v5_2", title="v5_2 (~92% pruned)",
                        refit_calendar="M", use_features_v5_2=True)


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
