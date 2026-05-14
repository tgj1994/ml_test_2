"""Dispatcher — run any utc2130 variant config but with the UTC midnight
daily bar (= legacy btc_1d.parquet semantics) instead of the 21:30 UTC bar.

Usage:
    uv run python main_th_sweep_utc0000.py <utc2130_variant_name>  [--retrain]

Example:
    uv run python main_th_sweep_utc0000.py utc2130_sm_v5  --retrain
        → reads main_th_sweep_utc2130_sm_v5.py's VARIANT,
          swaps bar_builder=build_utc0000_daily,
          rewrites suffix utc2130_sm_v5 → utc0000_sm_v5,
          saves results under reports/utc0000/.

This avoids creating 45 near-duplicate main scripts; instead one dispatcher
mutates the imported variant config at launch time.
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))


import importlib
import sys

from src.utc0000 import build_utc0000_daily
from src.utc2130_runner import run_variant


def _main() -> None:
    if len(sys.argv) < 2:
        print("usage: main_th_sweep_utc0000.py <utc2130_variant_name> [--retrain]")
        sys.exit(2)
    src_name = sys.argv[1]
    if not src_name.startswith("utc2130"):
        print(f"!! variant must start with 'utc2130': {src_name!r}")
        sys.exit(2)

    mod = importlib.import_module(f"main_th_sweep_{src_name}")
    cfg = mod.VARIANT

    # Override the three things that distinguish utc0000 from utc2130:
    cfg.bar_builder = build_utc0000_daily
    cfg.report_subdir = "utc0000"
    cfg.suffix = cfg.suffix.replace("utc2130", "utc0000", 1)
    cfg.title = (cfg.title + " — UTC0000").strip(" —")

    retrain = "--retrain" in sys.argv
    run_variant(cfg, retrain=retrain)


if __name__ == "__main__":
    _main()
