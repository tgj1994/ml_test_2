"""Fetch BTC on-chain metrics from Coin Metrics community API.

The community endpoint is free and unauthenticated (CC BY-NC 4.0 license,
non-commercial use only — see COIN_METRICS_EXCHANGE_FLOW.md). It serves a
curated subset of network metrics on a roughly 1-day publish lag.

Metrics actually pulled (must match DEFAULT_METRICS below):
  - PriceUSD         BTC USD price
  - CapMrktCurUSD    market cap, USD
  - SplyCur          current circulating supply
  - AdrActCnt        active address count
  - TxCnt            transaction count
  - HashRate         network hash rate

(Earlier drafts of this docstring also listed CapMVRVCur, CapRealUSD,
FlowInExUSD, FlowOutExUSD. These were never pulled by DEFAULT_METRICS
and never reached features.py — the labelled exchange-flow product is
Network Data Pro-only and intentionally avoided here.)

Output: data/coinmetrics.parquet, indexed by UTC-midnight date.

Note: at ~21:30 UTC of day X, Coin Metrics has typically published values
for day X-1 (sometimes day X-2). features.py applies a +1d lag to this
feed at consumption time to avoid lookahead.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


COMMUNITY_API = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
DEFAULT_METRICS = [
    # All available in the free community API (verified 2026-05).
    "PriceUSD",
    "CapMrktCurUSD",   # market cap, USD
    "SplyCur",         # current circulating supply
    "AdrActCnt",       # active address count
    "TxCnt",           # transaction count
    "HashRate",        # network hash rate
]


def fetch_coinmetrics(
    out_path: Path,
    asset: str = "btc",
    metrics: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    page_size: int = 10000,
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    metrics = metrics or DEFAULT_METRICS
    start = start or datetime(2018, 1, 1, tzinfo=timezone.utc)
    end = end or datetime.now(tz=timezone.utc)

    params_base = {
        "assets": asset,
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "start_time": start.strftime("%Y-%m-%d"),
        "end_time": end.strftime("%Y-%m-%d"),
        "page_size": page_size,
    }
    rows: list[dict] = []
    next_token: str | None = None
    page = 0
    while True:
        params = dict(params_base)
        if next_token:
            params["next_page_token"] = next_token
        page += 1
        r = requests.get(COMMUNITY_API, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(
                f"coinmetrics community API: HTTP {r.status_code} {r.text[:200]}"
            )
        payload = r.json()
        data = payload.get("data", [])
        rows.extend(data)
        next_token = payload.get("next_page_token")
        print(f"  page {page}: +{len(data)} rows (total {len(rows)})  "
              f"next={'yes' if next_token else 'no'}", flush=True)
        if not next_token:
            break
        time.sleep(sleep_s)

    if not rows:
        raise RuntimeError("coinmetrics community API returned no rows")

    df = pd.DataFrame(rows)
    # Coin Metrics returns 'time' as ISO timestamp at start of period (UTC midnight).
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.normalize()
    # Cast every requested metric to float (API returns strings).
    for col in metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("time").sort_index()[metrics]
    df.index.name = "date"
    df.to_parquet(out_path)
    print(f"  -> {len(df)} rows  ({df.index.min().date()} -> "
          f"{df.index.max().date()})  cols={list(df.columns)}",
          flush=True)
    return df


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    fetch_coinmetrics(here / "data" / "coinmetrics.parquet")
