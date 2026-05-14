"""CoinGecko free Demo API fetcher — last 365 days of supplementary
crypto-market context.

Strategy (per user decision):
  - Training/backtest: Bitstamp is primary (full history 2011+). CoinGecko
    free is the 365d-limited supplementary feed used to enrich the recent
    portion of the dataset with cross-market features that Bitstamp does
    not provide (BTC dominance, total crypto market cap, 24h global vol).
  - Live inference (bit-on-wave): CoinGecko is the primary feed at decision
    time. The training distribution therefore should *not* differ from the
    live feed for the most recent year — that's what this fetcher anchors.

Output (under data/):
  data/cg_btc_market.parquet   -- BTC daily {price, market_cap, volume} from
                                  /coins/bitcoin/market_chart, last 365d
  data/cg_eth_market.parquet   -- ETH equivalent
  data/cg_global.parquet       -- /global daily snapshot of total_market_cap_usd,
                                  total_volume_usd, btc_dominance_pct,
                                  eth_dominance_pct (single point per call;
                                  this script appends one daily row)

The /global endpoint returns the current snapshot only, so this script is
intended to run daily under Task Scheduler — each run appends one row to
cg_global.parquet. The /market_chart 365d call re-fetches the full window
on every run (cheap) and overwrites cg_btc_market / cg_eth_market.

Free Demo API:
  - Up to 365 days of historical data
  - Rate limit ~30 calls/min
  - Set COINGECKO_DEMO_API_KEY env var for the demo header (optional)
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


API_BASE = "https://api.coingecko.com/api/v3"
DEMO_HEADER = "x-cg-demo-api-key"


def _headers() -> dict[str, str]:
    key = os.environ.get("COINGECKO_DEMO_API_KEY")
    return {DEMO_HEADER: key} if key else {}


def _get(url: str, params: dict, retries: int = 6) -> dict | list:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=_headers(), timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"CoinGecko GET failed: {url} {params}")


def fetch_coin_market(coin_id: str, days: int = 365) -> pd.DataFrame:
    """Daily price + market_cap + total_volume for the last N days."""
    data = _get(f"{API_BASE}/coins/{coin_id}/market_chart",
                {"vs_currency": "usd", "days": days})
    prices = data.get("prices", [])
    caps = data.get("market_caps", [])
    vols = data.get("total_volumes", [])
    if not prices:
        return pd.DataFrame()
    df = pd.DataFrame({
        "ts_ms": [p[0] for p in prices],
        "price": [p[1] for p in prices],
        "market_cap": [c[1] for c in caps],
        "volume_24h": [v[1] for v in vols],
    })
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.normalize()
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return df[["date", "price", "market_cap", "volume_24h"]].set_index("date")


def fetch_global_snapshot() -> dict:
    """Current /global snapshot — call daily, append to cg_global.parquet."""
    payload = _get(f"{API_BASE}/global", {})
    d = payload.get("data", {})
    today = pd.Timestamp.now(tz="UTC").normalize()
    return {
        "date": today,
        "total_market_cap_usd": float(d.get("total_market_cap", {}).get("usd", float("nan"))),
        "total_volume_usd":     float(d.get("total_volume", {}).get("usd", float("nan"))),
        "btc_dominance_pct":    float(d.get("market_cap_percentage", {}).get("btc", float("nan"))),
        "eth_dominance_pct":    float(d.get("market_cap_percentage", {}).get("eth", float("nan"))),
        "n_active_cryptos":     int(d.get("active_cryptocurrencies", 0)),
        "n_markets":            int(d.get("markets", 0)),
    }


def update_global_history(out_path: Path) -> pd.DataFrame:
    row = fetch_global_snapshot()
    new = pd.DataFrame([row]).set_index("date")
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        merged = pd.concat([prev, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = new
    merged.to_parquet(out_path)
    print(f"  cg_global.parquet : {len(merged)} rows "
          f"(latest {merged.index.max().date()}: btc_dom={row['btc_dominance_pct']:.2f}%)")
    return merged


def fetch_all(data_dir: Path, days: int = 365) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CoinGecko] BTC market_chart last {days}d ...")
    btc = fetch_coin_market("bitcoin", days=days)
    btc.to_parquet(data_dir / "cg_btc_market.parquet")
    print(f"  cg_btc_market.parquet : {len(btc)} rows "
          f"({btc.index.min().date()} -> {btc.index.max().date()})")
    time.sleep(2.5)

    print(f"[CoinGecko] ETH market_chart last {days}d ...")
    eth = fetch_coin_market("ethereum", days=days)
    eth.to_parquet(data_dir / "cg_eth_market.parquet")
    print(f"  cg_eth_market.parquet : {len(eth)} rows")
    time.sleep(2.5)

    print("[CoinGecko] /global snapshot ...")
    update_global_history(data_dir / "cg_global.parquet")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    fetch_all(here / "data")
