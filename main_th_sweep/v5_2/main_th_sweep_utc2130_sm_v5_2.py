"""UTC2130 + v5_2 (looser ~92% prune), SM cadence."""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))


from src.utc2130_runner import VariantConfig, run_variant


VARIANT = VariantConfig(suffix="utc2130_sm_v5_2",
                        title="SM 1st&16th, v5_2 (~92% pruned)",
                        refit_calendar="SM", use_features_v5_2=True)


if __name__ == "__main__":
    import sys
    retrain = "--retrain" in sys.argv
    run_variant(VARIANT, retrain=retrain)
