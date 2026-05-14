"""Live-service daily inference using a local Bitcoin Core full node.

The research/backtest stage uses the same data sources as coin_analysis
(yfinance / Binance / Coin Metrics community / FRED / alternative.me /
CFTC). At service time we cannot redistribute those feeds in a commercial
product, so the live inference path swaps:

  - market data (BTC OHLC, ETH OHLC, macro) -> still fetched from the
    same daily-update sources for the operator's own internal use; the
    license issue is about redistribution, not internal use
  - on-chain data -> Bitcoin Core RPC running on W:\\onchain\\bitcoind
    (HashRate, Supply, TxCnt, AdrActCnt — see src/btc_core_rpc.py and
    src/btc_core_chain_scan.py)

This module loads the most recent EBM model from `data/live_models_ebm/`,
fetches the latest day's features (using the same `build_features` pipeline,
with the on-chain parquet replaced by the freshly-extracted Bitcoin Core
data), and prints a daily prob_up for the relevant variant.

Run as part of a daily cron (Windows Task Scheduler) AFTER Bitcoin Core
IBD is complete:

    uv run python -m src.live_inference_btc_core --variant utc2130_sm_v5
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.btc_core_rpc import snapshot as rpc_snapshot, ibd_done
from src.btc_core_chain_scan import run as chain_scan_run
from src.features import build_features
from src.utc2130 import build_utc2130_daily

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "live_models_ebm"


def _load_latest_model(variant: str):
    pkl = MODELS_DIR / f"{variant}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"no pickle for variant {variant!r} at {pkl}. "
            "Run the corresponding main_th_sweep script with --retrain first "
            "and save the final-refit EBM there.")
    with open(pkl, "rb") as f:
        return pickle.load(f)


def _refresh_onchain() -> None:
    """Update the local Bitcoin Core derived parquet files."""
    if not ibd_done():
        print("  Bitcoin Core IBD still in progress — onchain features will "
              "fall back to the cached Coin Metrics parquet for this run.")
        return
    rpc_snapshot(DATA_DIR / "onchain_btc_rpc.parquet")
    chain_scan_run(DATA_DIR / "onchain_btc_chain.parquet")
    # Replace the legacy coinmetrics.parquet symbolic path used by
    # features.py with a unified onchain frame, only if both feeds exist.
    rpc_path = DATA_DIR / "onchain_btc_rpc.parquet"
    chain_path = DATA_DIR / "onchain_btc_chain.parquet"
    if rpc_path.exists() and chain_path.exists():
        rpc = pd.read_parquet(rpc_path)
        chain = pd.read_parquet(chain_path).drop(columns=["last_block_height"],
                                                  errors="ignore")
        unified = rpc.join(chain, how="outer").sort_index()
        # features.py expects 'CapMrktCurUSD', 'HashRate', 'AdrActCnt', 'TxCnt'
        # at minimum. Market cap is HashRate-independent — set to NaN here so
        # downstream MVRV proxy uses the price-based fallback.
        if "CapMrktCurUSD" not in unified.columns:
            unified["CapMrktCurUSD"] = float("nan")
        unified.to_parquet(DATA_DIR / "coinmetrics.parquet")
        print("  refreshed coinmetrics.parquet from Bitcoin Core RPC + chain scan")


def predict_for_today(variant: str) -> dict:
    _refresh_onchain()
    bar = build_utc2130_daily(DATA_DIR / "btc_15m.parquet")
    # variant -> feature-flag bundle
    flags = dict(use_features_v3=False, use_features_v4=False, use_features_v5=False)
    if "v3" in variant:
        flags["use_features_v3"] = True
    if "v4" in variant or "v6" in variant:
        flags["use_features_v4"] = True
    if "v5" in variant:
        flags["use_features_v5"] = True

    X, _, _ = build_features(DATA_DIR, label_threshold=0.015,
                              daily_df=bar, external_lag_days=0,
                              **flags)
    model = _load_latest_model(variant)
    today_row = X.iloc[[-1]]
    prob_up = float(model.predict_proba(today_row)[0, 1])
    out = {
        "variant": variant,
        "ts": today_row.index[-1].isoformat(),
        "prob_up": prob_up,
        "model_pkl": str(MODELS_DIR / f"{variant}.pkl"),
        "n_features": int(X.shape[1]),
    }
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="utc2130_sm_v5",
                        help="variant suffix matching a saved EBM pickle")
    args = parser.parse_args()
    predict_for_today(args.variant)


if __name__ == "__main__":
    main()
