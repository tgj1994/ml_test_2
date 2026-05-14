"""Binance Spot OHLCV fetcher — drop-in replacement for data_fetcher.py.

Produces the same parquet schema (open_time, close_time, open, high, low,
close, volume) so features.py and the rest of the pipeline keep working
without any code change.

Endpoint:
  GET https://api.binance.com/api/v3/klines
       ?symbol={sym}&interval={iv}&startTime={ms}&endTime={ms}&limit={N}
  intervals used: 1d, 1w, 1M, 15m
  limit: max 1000 per request — we paginate by setting startTime to the
  last open_time + interval_ms.

History:
  BTCUSDT listed 2017-08-17 ~13:00 UTC
  ETHUSDT listed 2017-08-17 ~04:00 UTC
  So both pairs give ~9 years of history versus Bitstamp's ~14 years.
  This trade-off was explicitly accepted (user decision) to unify the
  spot/derivatives stack on Binance.

Outputs (under data/):
  btc_1d.parquet   — BTCUSDT daily klines
  btc_1w.parquet   — BTCUSDT weekly klines (1w interval native, NOT resampled)
  btc_1M.parquet   — BTCUSDT monthly klines (1M interval native)
  btc_15m.parquet  — BTCUSDT 15-minute klines
  eth_1d.parquet   — ETHUSDT daily klines
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


KLINES_URL = "https://api.binance.com/api/v3/klines"
LIMIT = 1000

_INTERVAL_MS = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
    # Months are variable length; Binance uses native calendar months, so we
    # advance the cursor by 28d (minimum) to be safe and let dedupe handle
    # overlap.
    "1M": 28 * 24 * 60 * 60 * 1000,
}


def _request_klines(symbol: str, interval: str, start_ms: int,
                    end_ms: int) -> list[list]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }
    for attempt in range(6):
        try:
            r = requests.get(KLINES_URL, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"Binance klines failed: {symbol} {interval} {start_ms}-{end_ms}")


def fetch_binance_klines(symbol: str, interval: str,
                          start: datetime, end: datetime) -> pd.DataFrame:
    """Paginate Binance klines over [start, end)."""
    step_ms = _INTERVAL_MS[interval]
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _request_klines(symbol, interval, cursor, end_ms)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        # Stay well under Binance's 1200 weight/min spot limit. klines = 1
        # weight each up to limit=500, 2 weight above. We're at limit=1000
        # so ~2 weight; pausing 0.15s keeps us at ~13 req/s = 26 weight/s
        # = ~1560 weight/min, but bursts are allowed.
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(
            columns=["open_time", "close_time", "open", "high", "low",
                     "close", "volume"])

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"),
                                     unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"),
                                       unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df = (df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]
          .drop_duplicates(subset=["open_time"], keep="last")
          .sort_values("open_time")
          .reset_index(drop=True))
    return df


# Binance launch dates (BTCUSDT/ETHUSDT both listed 2017-08-17). Using 17th
# 00:00 UTC as a safe lower bound — Binance returns nothing before listing
# and we de-dup by open_time anyway.
BTC_INCEPTION = datetime(2017, 8, 17, tzinfo=timezone.utc)
ETH_INCEPTION = datetime(2017, 8, 17, tzinfo=timezone.utc)


def fetch_all(out_dir: Path, end: datetime | None = None) -> dict[str, pd.DataFrame]:
    end = end or datetime.now(tz=timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}

    jobs = [
        ("btc_1d.parquet",  "BTCUSDT", "1d",  BTC_INCEPTION),
        ("btc_1w.parquet",  "BTCUSDT", "1w",  BTC_INCEPTION),
        ("btc_1M.parquet",  "BTCUSDT", "1M",  BTC_INCEPTION),
        ("btc_15m.parquet", "BTCUSDT", "15m", BTC_INCEPTION),
        ("eth_1d.parquet",  "ETHUSDT", "1d",  ETH_INCEPTION),
    ]
    for fname, symbol, interval, inception in jobs:
        print(f"  [Binance] {symbol} {interval}   "
              f"{inception.date()} -> {end.date()}")
        df = fetch_binance_klines(symbol, interval, inception, end)
        df.to_parquet(out_dir / fname, index=False)
        frames[fname.replace(".parquet", "")] = df
        print(f"    -> {len(df):>7} rows written to {fname}")

    return frames


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data"
    fetch_all(out)
