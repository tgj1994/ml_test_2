"""Bitstamp-only OHLCV fetcher (commercial-licensed source).

Replaces the legacy Binance + Bitstamp combined fetcher. The user holds a
Bitstamp commercial Data License Agreement, which permits commercial use
and redistribution of Bitstamp's exchange data (partners@bitstamp.net).

Bitstamp spot OHLC coverage:
  - BTC/USD : 2011-08-18 -> today  (daily + 15m)
  - ETH/USD : 2017-08-08 -> today  (daily)

Endpoint:
  GET https://www.bitstamp.net/api/v2/ohlc/{market_symbol}/
       ?step={seconds}&limit={N}&start={unix_s}&end={unix_s}
  step values used: 86400 (1d), 900 (15m)

Weekly and monthly bars are resampled from the daily series (W-MON for
weekly start, MS for monthly start), keeping the entire price pipeline on
a single source.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


BITSTAMP_OHLC = "https://www.bitstamp.net/api/v2/ohlc/{market_symbol}/"
DEFAULT_LIMIT = 1000


def _request_ohlc(market_symbol: str, step_s: int, start_s: int,
                  limit: int = DEFAULT_LIMIT) -> list[dict]:
    params = {"step": step_s, "limit": limit, "start": start_s}
    url = BITSTAMP_OHLC.format(market_symbol=market_symbol)
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json().get("data", {}).get("ohlc", [])
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"Bitstamp OHLC failed: {market_symbol} step={step_s} start={start_s}")


def fetch_bitstamp_ohlc(market_symbol: str, step_s: int,
                        start: datetime, end: datetime) -> pd.DataFrame:
    """Paginate Bitstamp OHLC over the [start, end) range."""
    rows: list[dict] = []
    cursor = int(start.timestamp())
    end_s = int(end.timestamp())
    while cursor < end_s:
        batch = _request_ohlc(market_symbol, step_s, cursor)
        if not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1]["timestamp"])
        next_cursor = last_ts + step_s
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.12)
        if cursor >= end_s:
            break

    if not rows:
        return pd.DataFrame(
            columns=["open_time", "close_time", "open", "high", "low",
                     "close", "volume"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
    df = (df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True))
    df = df[df["timestamp"] < end_s]
    df["open_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["close_time"] = (df["open_time"]
                         + pd.Timedelta(seconds=step_s)
                         - pd.Timedelta(milliseconds=1))
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


def _resample(df_daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df_daily.empty:
        return df_daily
    g = df_daily.set_index("open_time")
    agg = (g.resample(rule, label="left", closed="left")
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"})
             .dropna(subset=["open"])
             .reset_index())
    if rule == "W-MON":
        agg["close_time"] = (agg["open_time"] + pd.Timedelta(days=7)
                             - pd.Timedelta(milliseconds=1))
    elif rule == "MS":
        agg["close_time"] = (agg["open_time"] + pd.offsets.MonthEnd(1)
                             + pd.Timedelta(days=1)
                             - pd.Timedelta(milliseconds=1))
    return agg[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


BTC_INCEPTION = datetime(2011, 8, 18, tzinfo=timezone.utc)
ETH_INCEPTION = datetime(2017, 8, 8, tzinfo=timezone.utc)
BTC_15M_INCEPTION = datetime(2014, 1, 1, tzinfo=timezone.utc)


def fetch_all(out_dir: Path, end: datetime | None = None) -> dict[str, pd.DataFrame]:
    end = end or datetime.now(tz=timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}

    print(f"  [Bitstamp] BTC/USD daily   {BTC_INCEPTION.date()} -> {end.date()}")
    btc_d = fetch_bitstamp_ohlc("btcusd", 86400, BTC_INCEPTION, end)
    btc_d.to_parquet(out_dir / "btc_1d.parquet", index=False)
    frames["btc_1d"] = btc_d
    print(f"    -> {len(btc_d):>6} daily rows")

    print(f"  [Bitstamp] BTC/USD 15m     {BTC_15M_INCEPTION.date()} -> {end.date()}")
    btc_15 = fetch_bitstamp_ohlc("btcusd", 900, BTC_15M_INCEPTION, end)
    btc_15.to_parquet(out_dir / "btc_15m.parquet", index=False)
    frames["btc_15m"] = btc_15
    print(f"    -> {len(btc_15):>6} 15m rows")

    print(f"  [Bitstamp] ETH/USD daily   {ETH_INCEPTION.date()} -> {end.date()}")
    eth_d = fetch_bitstamp_ohlc("ethusd", 86400, ETH_INCEPTION, end)
    eth_d.to_parquet(out_dir / "eth_1d.parquet", index=False)
    frames["eth_1d"] = eth_d
    print(f"    -> {len(eth_d):>6} daily rows")

    print("  [resample] BTC daily -> weekly + monthly")
    btc_w = _resample(btc_d, "W-MON")
    btc_M = _resample(btc_d, "MS")
    btc_w.to_parquet(out_dir / "btc_1w.parquet", index=False)
    btc_M.to_parquet(out_dir / "btc_1M.parquet", index=False)
    frames["btc_1w"] = btc_w
    frames["btc_1M"] = btc_M
    print(f"    -> 1w: {len(btc_w)} rows, 1M: {len(btc_M)} rows")

    return frames


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data"
    fetch_all(out)
