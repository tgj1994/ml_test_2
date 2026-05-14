"""Fetch external context data: macro (yfinance), Binance futures (public),
Fear & Greed Index (alternative.me).

Data availability windows:
  - DXY / Gold / S&P / VIX / 10Y yield  : multi-decade (yfinance)
  - ETH-USD                             : 2017-08+
  - Binance futures funding / OI        : 2019-09+ (BTC perp launch)
  - Fear & Greed Index                  : 2018-02-01+

XGBoost handles NaN natively, so missing-history rows in the older portion
of the daily index are fine.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


# ---------- yfinance macros ----------

YF_TICKERS = {
    "dxy":   "DX-Y.NYB",
    "gold":  "GC=F",
    "spx":   "^GSPC",
    "vix":   "^VIX",
    "tnx":   "^TNX",        # 10Y treasury yield, in *percent* × 10
    "eth":   "ETH-USD",
}


def fetch_yfinance(out_path: Path, start: datetime | None = None) -> pd.DataFrame:
    import yfinance as yf
    start = start or datetime(2014, 1, 1, tzinfo=timezone.utc)
    frames = []
    for name, ticker in YF_TICKERS.items():
        print(f"  yfinance {name:<5} ({ticker})")
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
        except Exception as exc:
            print(f"    !! failed: {exc}")
            continue
        if df is None or df.empty:
            print("    !! empty")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].rename(columns={"Close": f"{name}_close"})
        df.index = pd.to_datetime(df.index, utc=True).normalize()
        frames.append(df)
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame()
    macro = pd.concat(frames, axis=1).sort_index()
    macro.index.name = "date"
    macro.to_parquet(out_path)
    print(f"  -> {len(macro)} rows, cols={list(macro.columns)}")
    return macro


# ---------- Binance futures funding ----------

BINANCE_FAPI = "https://fapi.binance.com"
SYMBOL_PERP = "BTCUSDT"


def _get(url: str, params: dict, retries: int = 5) -> list | dict:
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            time.sleep(1.0 * (i + 1))
        except requests.RequestException:
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"Failed: {url} {params}")


def fetch_funding_rate(out_path: Path,
                       start: datetime | None = None,
                       end: datetime | None = None) -> pd.DataFrame:
    """Daily-aggregated funding rate from Binance USDT-M perpetual."""
    end = end or datetime.now(tz=timezone.utc)
    start = start or datetime(2019, 9, 8, tzinfo=timezone.utc)  # BTCUSDT perp listing
    rows: list[dict] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        batch = _get(f"{BINANCE_FAPI}/fapi/v1/fundingRate",
                     {"symbol": SYMBOL_PERP, "startTime": cursor,
                      "endTime": end_ms, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1]["fundingTime"])
        if last <= cursor:
            break
        cursor = last + 1
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"])
    daily = (df.groupby(df["fundingTime"].dt.normalize())
               .agg(funding_mean=("fundingRate", "mean"),
                    funding_max=("fundingRate", "max"),
                    funding_min=("fundingRate", "min")))
    daily.index.name = "date"
    daily.to_parquet(out_path)
    print(f"  funding: {len(daily)} daily rows ({daily.index.min().date()} -> {daily.index.max().date()})")
    return daily


def fetch_open_interest(out_path: Path,
                        start: datetime | None = None,
                        end: datetime | None = None) -> pd.DataFrame:
    """Daily Open Interest. Binance only serves the last ~30 days of OI history,
    so older rows will be NaN — XGBoost handles this natively.
    """
    end = end or datetime.now(tz=timezone.utc)
    start = start or (end - timedelta(days=30))
    rows: list[dict] = []
    try:
        batch = _get(f"{BINANCE_FAPI}/futures/data/openInterestHist",
                     {"symbol": SYMBOL_PERP, "period": "1d",
                      "startTime": int(start.timestamp() * 1000),
                      "endTime": int(end.timestamp() * 1000),
                      "limit": 500})
        rows = batch or []
    except Exception as exc:
        print(f"  OI fetch failed: {exc}; skipping")
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms", utc=True)
    df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"])
    df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"])
    df = df.set_index(df["timestamp"].dt.normalize())[
        ["sumOpenInterest", "sumOpenInterestValue"]]
    df = df.rename(columns={"sumOpenInterest": "oi_btc",
                            "sumOpenInterestValue": "oi_usd"})
    df.index.name = "date"
    df.to_parquet(out_path)
    print(f"  OI: {len(df)} daily rows ({df.index.min().date()} -> {df.index.max().date()})")
    return df


def fetch_premium_basis(out_path: Path,
                        start: datetime | None = None,
                        end: datetime | None = None) -> pd.DataFrame:
    """Daily perp-vs-spot basis = (perp_close / spot_close - 1).

    Perp marked from /fapi/v1/klines (interval=1d), spot from /api/v3/klines.
    """
    end = end or datetime.now(tz=timezone.utc)
    start = start or datetime(2019, 9, 8, tzinfo=timezone.utc)

    def fetch_klines(host: str, path: str) -> pd.DataFrame:
        rows: list[list] = []
        cursor = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        step = 1000 * 24 * 60 * 60 * 1000  # 1000 days per request
        while cursor < end_ms:
            batch_end = min(cursor + step, end_ms)
            batch = _get(host + path,
                         {"symbol": SYMBOL_PERP, "interval": "1d",
                          "startTime": cursor, "endTime": batch_end, "limit": 1000})
            if not batch:
                cursor = batch_end + 1
                continue
            rows.extend(batch)
            last = int(batch[-1][0])
            if last <= cursor:
                break
            cursor = last + 24 * 60 * 60 * 1000
            time.sleep(0.12)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(pd.to_numeric(df[0]), unit="ms", utc=True).dt.normalize()
        df["close"] = pd.to_numeric(df[4])
        return df.set_index("date")[["close"]]

    perp = fetch_klines(BINANCE_FAPI, "/fapi/v1/klines")
    spot = fetch_klines("https://api.binance.com", "/api/v3/klines")
    if perp.empty or spot.empty:
        return pd.DataFrame()
    perp = perp.rename(columns={"close": "perp_close"})
    spot = spot.rename(columns={"close": "spot_close"})
    df = perp.join(spot, how="inner")
    df["basis"] = df["perp_close"] / df["spot_close"] - 1.0
    df = df[["basis"]]
    df.to_parquet(out_path)
    print(f"  basis: {len(df)} daily rows")
    return df


# ---------- FRED (free, no API key needed via CSV download) ----------

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES = {
    "m2": "M2SL",      # M2 Money Stock, Seasonally Adjusted, monthly (Bn USD)
    "walcl": "WALCL",  # Fed total assets, weekly (Wed)
}


def fetch_fred(out_path: Path,
               series_id: str,
               col_name: str,
               start: datetime | None = None) -> pd.DataFrame:
    """Download a FRED series via the public fredgraph CSV endpoint.

    Returns a single-column DataFrame indexed by observation date (UTC,
    midnight-normalised), saved to out_path as parquet.
    """
    start = start or datetime(2010, 1, 1, tzinfo=timezone.utc)
    params = {"id": series_id,
              "cosd": start.strftime("%Y-%m-%d")}
    r = requests.get(FRED_CSV_URL, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"FRED CSV: HTTP {r.status_code} for {series_id}")
    import io
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        raise RuntimeError(f"FRED CSV: empty result for {series_id}")
    # Columns are 'observation_date' and the series_id (or DATE for older API).
    date_col = next(c for c in df.columns
                    if c.lower() in ("observation_date", "date"))
    val_col = next(c for c in df.columns if c != date_col)
    df["date"] = pd.to_datetime(df[date_col], utc=True).dt.normalize()
    df[col_name] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.set_index("date").sort_index()[[col_name]]
    df.index.name = "date"
    df.to_parquet(out_path)
    print(f"  FRED {series_id} -> {col_name}: {len(df)} rows  "
          f"({df.index.min().date()} -> {df.index.max().date()})")
    return df


# ---------- Fear & Greed Index ----------

def fetch_fear_greed(out_path: Path) -> pd.DataFrame:
    r = requests.get("https://api.alternative.me/fng/?limit=0", timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="s", utc=True).dt.normalize()
    df["fng"] = pd.to_numeric(df["value"])
    df = df.set_index("date").sort_index()[["fng"]]
    df.to_parquet(out_path)
    print(f"  F&G: {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})")
    return df


# ---------- top-level ----------

def fetch_all_external(data_dir: Path) -> dict[str, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    print("\n[macro] yfinance ...")
    out["macro"] = fetch_yfinance(data_dir / "macro.parquet")
    print("\n[futures] Binance funding rate ...")
    out["funding"] = fetch_funding_rate(data_dir / "funding.parquet")
    print("\n[futures] Binance Open Interest history ...")
    out["oi"] = fetch_open_interest(data_dir / "oi.parquet")
    print("\n[futures] perp-vs-spot basis ...")
    out["basis"] = fetch_premium_basis(data_dir / "basis.parquet")
    print("\n[sentiment] Fear & Greed Index ...")
    out["fng"] = fetch_fear_greed(data_dir / "fng.parquet")
    print("\n[macro] FRED M2 ...")
    out["m2"] = fetch_fred(data_dir / "m2.parquet", "M2SL", "m2")
    return out


if __name__ == "__main__":
    fetch_all_external(Path(__file__).resolve().parent.parent / "data")
