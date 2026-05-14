"""Build a 10:00 KST aligned daily bar from 15-minute OHLCV.

10:00 KST = 01:00 UTC. Each "KST10 daily" bar represents the 24-hour
window ending at 10:00 KST (= 01:00 UTC) of its label date X:

    open_time  = 01:00 UTC on date X-1
    close_time = 01:00 UTC on date X
    close      = price at 10:00 KST on date X (the decision-time snapshot)

Rolling features computed off this index use information up to the decision
time exactly — no leakage. The forward-return label is computed from one
KST10 close to the next.

External feeds (macro / funding / basis / fng) need a +1-day lag in this
mode: their daily index represents data published *after* 10:00 KST of
that index date, so an as-of join without lag would peek at unpublished
values.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


KST10_OFFSET = pd.Timedelta(hours=1)  # 01:00 UTC = 10:00 KST


def build_kst10_daily(m15_path: Path) -> pd.DataFrame:
    """Resample btc_15m.parquet to daily bars closing at 01:00 UTC.

    Returns columns matching btc_1d.parquet: open_time, close_time,
    open, high, low, close, volume.
    """
    m15 = pd.read_parquet(m15_path)
    g = m15.set_index("open_time").sort_index()
    # Group [01:00 UTC X-1, 01:00 UTC X) → labelled at right edge = 01:00 UTC X.
    # Use '24h' (tick-like) instead of '1D' so the offset='1h' actually takes effect.
    rs = g.resample("24h", origin="start_day", offset="1h",
                    label="right", closed="left")
    agg = rs.agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
    agg["bar_count"] = rs.size()
    # Keep only complete 96-bar (24h × 15m) days. This drops the partial bar
    # at the very start (15m data begins mid-day) and any partial bar at the
    # very end (data may not yet span the next 01:00 UTC boundary).
    agg = agg.loc[agg["bar_count"] == 96].drop(columns=["bar_count"]).copy()
    agg.index.name = "close_time"
    agg = agg.reset_index()
    agg["open_time"] = agg["close_time"] - pd.Timedelta(days=1)
    # Normalise datetime resolution to nanoseconds so merge_asof against the
    # existing weekly/monthly bars (which use [us, UTC]) does not fail with a
    # MergeError on dtype mismatch.
    agg["close_time"] = agg["close_time"].astype("datetime64[ns, UTC]")
    agg["open_time"] = agg["open_time"].astype("datetime64[ns, UTC]")
    return agg[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    df = build_kst10_daily(here / "data" / "btc_15m.parquet")
    print(f"shape: {df.shape}")
    print("head:")
    print(df.head(3))
    print("tail:")
    print(df.tail(3))
    print(f"\nclose_time hours seen: "
          f"{sorted(df['close_time'].dt.hour.unique().tolist())}")
    print(f"close_time minutes seen: "
          f"{sorted(df['close_time'].dt.minute.unique().tolist())}")
