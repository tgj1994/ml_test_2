"""Bitcoin Core RPC daily snapshot fetcher.

Pulls metrics from a local bitcoind running under W:\\onchain\\bitcoind\\bitcoin.conf:

  HashRate  : getnetworkhashps(2016)        — last 2 weeks of difficulty
  Supply    : gettxoutsetinfo  -> total_amount   (slow: minutes; daily snapshot only)

For block-level series like TxCnt and AdrActCnt see btc_core_chain_scan.py.

This script is intended for daily cron / manual invocation AFTER initial block
download (IBD) is complete. While IBD is running, the parquet stays empty and
the v3/v4/v5/v6 onchain features fall back to NaN (EBM handles missing values
natively via a 'missing' bin).

Output:
  data/onchain_btc_rpc.parquet  — date | HashRate | Supply
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BITCOIN_CLI = Path(os.environ.get(
    "BITCOIN_CLI", r"W:\onchain\bitcoind\bitcoin-31.0\bin\bitcoin-cli.exe"))
BITCOIN_CONF = Path(os.environ.get(
    "BITCOIN_CONF", r"W:\onchain\bitcoind\bitcoin.conf"))


def _cli(*args: str) -> str:
    cmd = [str(BITCOIN_CLI), f"-conf={BITCOIN_CONF}", *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"bitcoin-cli {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def ibd_done() -> bool:
    info = json.loads(_cli("getblockchaininfo"))
    return not info.get("initialblockdownload", True)


def snapshot(out_path: Path) -> None:
    """Append today's HashRate + Supply to onchain_btc_rpc.parquet.

    If IBD is still running, prints a notice and exits without writing.
    """
    if not ibd_done():
        info = json.loads(_cli("getblockchaininfo"))
        prog = info.get("verificationprogress", 0.0)
        print(f"  IBD still in progress ({prog*100:.2f}%) — skipping snapshot")
        return

    # last 2016 blocks ≈ 2 weeks
    hashrate = float(_cli("getnetworkhashps", "2016"))
    txout = json.loads(_cli("gettxoutsetinfo"))
    supply = float(txout["total_amount"])

    today = pd.Timestamp.now(tz="UTC").normalize()
    row = pd.DataFrame(
        {"HashRate": [hashrate], "Supply": [supply]},
        index=pd.DatetimeIndex([today], name="date"),
    )
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        merged = pd.concat([prev, row])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = row
    merged.to_parquet(out_path)
    print(f"  onchain_btc_rpc.parquet : {len(merged)} rows  (latest {today.date()})")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    snapshot(here / "data" / "onchain_btc_rpc.parquet")
