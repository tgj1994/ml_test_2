"""Build a 21:30 UTC aligned daily bar from 15-minute OHLCV.

21:30 UTC is ~30 minutes after the major macro release window (NYSE close
21:00 UTC, plus the 21:00 UTC funding settlement on Binance). At 21:30 UTC
all of:
  - same-day NYSE close prints (DXY/Gold/SPX/VIX/TNX/ETH-USD)
  - same-day Binance funding settlements (00:00, 08:00, 16:00 UTC)
  - same-day Fear & Greed daily print
are published, so external feeds can be joined with **lag = 0** without
introducing lookahead.

Each "UTC2130 daily" bar represents the 24-hour window ending at 21:30 UTC
of its label date X:

    open_time  = 21:30 UTC on date X-1
    close_time = 21:30 UTC on date X
    close      = price at 21:30 UTC on date X (the decision-time snapshot)

The forward-return label is from one 21:30 UTC close to the next.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_utc2130_daily(m15_path: Path) -> pd.DataFrame:
    """Resample btc_15m.parquet to daily bars closing at 21:30 UTC.

    Returns columns matching btc_1d.parquet: open_time, close_time,
    open, high, low, close, volume.
    """
    m15 = pd.read_parquet(m15_path)
    g = m15.set_index("open_time").sort_index()
    # Group [21:30 UTC X-1, 21:30 UTC X) → labelled at right edge = 21:30 UTC X.
    # 15m bars sit on :00/:15/:30/:45, so 21:30 UTC is a clean boundary.
    rs = g.resample("24h", origin="start_day", offset="21h30min",
                    label="right", closed="left")
    agg = rs.agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
    agg["bar_count"] = rs.size()
    # Keep only complete 96-bar (24h × 15m) days.
    agg = agg.loc[agg["bar_count"] == 96].drop(columns=["bar_count"]).copy()
    agg.index.name = "close_time"
    agg = agg.reset_index()
    agg["open_time"] = agg["close_time"] - pd.Timedelta(days=1)
    # Normalise datetime resolution to nanoseconds so merge_asof against the
    # weekly/monthly bars (which use [us, UTC]) does not raise MergeError.
    agg["close_time"] = agg["close_time"].astype("datetime64[ns, UTC]")
    agg["open_time"] = agg["open_time"].astype("datetime64[ns, UTC]")
    return agg[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    df = build_utc2130_daily(here / "data" / "btc_15m.parquet")
    print(f"shape: {df.shape}")
    print("head:"); print(df.head(3))
    print("tail:"); print(df.tail(3))
    print(f"close_time hours seen: "
          f"{sorted(df['close_time'].dt.hour.unique().tolist())}")
    print(f"close_time minutes seen: "
          f"{sorted(df['close_time'].dt.minute.unique().tolist())}")
