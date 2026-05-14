"""FRED fetcher for free-licensed series (Federal Reserve / U.S. Treasury / BLS only).

We explicitly avoid third-party copyrighted series on FRED (S&P 500, CBOE VIX,
LBMA Gold) per user policy. The seven series we pull are all Federal Reserve,
U.S. Treasury, or BLS originals:

  DTWEXBGS  — Trade Weighted U.S. Dollar Index: Broad, Goods and Services (daily, Fed)
  DGS10     — 10-Year Treasury Constant Maturity Rate (daily, Treasury)
  M2SL      — M2 Money Stock, SA (monthly, Fed)
  T10Y2Y    — 10Y minus 2Y Treasury yield spread (daily, Treasury computed)
  T10YIE    — 10-Year Breakeven Inflation Rate (daily, Treasury computed)
  INDPRO    — Industrial Production: Total Index (monthly, Fed)
  UNRATE    — Civilian Unemployment Rate (monthly, BLS)

Output:
  data/macro_fred.parquet — daily index, columns: dxy, tnx, t10y2y, t10yie, indpro, unrate
                            (weekly/monthly series are ffill'd to daily)
  data/m2.parquet         — single column 'm2', daily ffill from monthly
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

SERIES_TO_COLUMN = {
    # Existing core series (Fed / Treasury / BLS — all public domain)
    "DTWEXBGS": "dxy",
    "DGS10":    "tnx",
    "T10Y2Y":   "t10y2y",
    "T10YIE":   "t10yie",
    "INDPRO":   "indpro",
    "UNRATE":   "unrate",

    # ----- v7 Tier-1 expansion (US Treasury / Fed Board / BLS only) -----
    # Treasury yields (curve completeness)
    "DGS2":     "tnx2y",     # 2Y Treasury constant maturity
    "DGS5":     "tnx5y",     # 5Y Treasury
    "DGS30":    "tnx30y",    # 30Y Treasury
    # TIPS / real yields (store-of-value narrative for BTC)
    "DFII5":    "tips5y",    # 5Y TIPS real yield
    "DFII10":   "tips10y",   # 10Y TIPS real yield
    # Policy / overnight rates
    "DFF":      "fedfunds",  # Effective Fed Funds Rate
    "SOFR":     "sofr",      # Secured Overnight Financing Rate
    # Fed balance sheet + liquidity tools (weekly cadence, ffill to daily)
    "WALCL":    "fed_assets",   # Fed total assets (QE/QT)
    "WRESBAL":  "fed_reserves", # Bank reserves at Fed
    "RRPONTSYD": "rrp_overnight",  # Overnight reverse repo (liquidity sink)
    # FX cross-rates (Fed published)
    "DEXKOUS":  "fx_krw",    # KRW per USD (KR retail proxy)
    "DEXJPUS":  "fx_jpy",    # JPY per USD (JPY carry)
    "DEXCHUS":  "fx_cny",    # CNY per USD (CN macro)
    # Labour leading indicators
    "ICSA":     "jobless_initial",  # Weekly initial jobless claims (BLS)
    "PAYEMS":   "payems",    # Monthly nonfarm payrolls (BLS)
}

M2_SERIES = ("M2SL", "m2")


def fetch_series(series_id: str, col_name: str, start: datetime) -> pd.DataFrame:
    """Download a FRED series CSV, return DataFrame indexed by date (UTC normalised)."""
    params = {"id": series_id, "cosd": start.strftime("%Y-%m-%d")}
    r = requests.get(FRED_CSV_URL, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"FRED {series_id}: HTTP {r.status_code}")
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        raise RuntimeError(f"FRED {series_id}: empty CSV")
    date_col = next(c for c in df.columns if c.lower() in ("observation_date", "date"))
    val_col = next(c for c in df.columns if c != date_col)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], utc=True).dt.normalize(),
        col_name: pd.to_numeric(df[val_col], errors="coerce"),
    }).dropna(subset=[col_name])
    out = out.set_index("date").sort_index()
    return out


def fetch_all(data_dir: Path,
              start: datetime = datetime(2015, 1, 1, tzinfo=timezone.utc)) -> None:
    """Pull all FRED series and write parquet outputs.

    Daily series (DTWEXBGS, DGS10, T10Y2Y, T10YIE) are joined onto a continuous
    daily calendar with forward-fill so weekends/holidays inherit the previous
    trading-day value. Monthly series (INDPRO, UNRATE, M2SL) are similarly ffill'd
    onto the daily index.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for sid, col in SERIES_TO_COLUMN.items():
        print(f"[FRED] {sid} -> {col}")
        frames[col] = fetch_series(sid, col, start)
        time.sleep(0.3)

    # Daily calendar from earliest available to today, UTC
    end = pd.Timestamp.now(tz="UTC").normalize()
    daily_idx = pd.date_range(
        min(f.index.min() for f in frames.values()),
        end, freq="D", tz="UTC", name="date",
    )
    macro = pd.DataFrame(index=daily_idx)
    for col, df in frames.items():
        macro[col] = df[col].reindex(daily_idx).ffill()
    out_path = data_dir / "macro_fred.parquet"
    macro.to_parquet(out_path)
    print(f"  macro_fred.parquet : {macro.shape}  "
          f"({macro.index.min().date()} -> {macro.index.max().date()})")

    # M2 (monthly, ffill'd to daily)
    print(f"[FRED] {M2_SERIES[0]} -> {M2_SERIES[1]}")
    m2 = fetch_series(*M2_SERIES, start=start)
    m2_daily = m2.reindex(daily_idx).ffill()
    m2_daily.to_parquet(data_dir / "m2.parquet")
    print(f"  m2.parquet : {m2_daily.shape}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    fetch_all(here / "data")
