"""UTC midnight (00:00 UTC) aligned daily bar.

Equivalent to btc_1d.parquet (which already closes at 23:59:59.999 UTC of
each calendar day — i.e. the bar's close price IS the price at UTC midnight,
shifted by 1ms before the boundary so as-of merges against external data
indexed at 00:00 UTC of the SAME day correctly pick that day's NYSE close
(published 21:00 UTC same day, ~3 hours before the bar close).

We expose this as a thin wrapper so the utc2130_runner can swap bar
sources via the `bar_builder` field on VariantConfig — no other code path
needs to change. Reports go under reports/utc0000/ instead of utc2130/.

Decision time semantics:
  open_time  = 00:00:00.000 UTC on date X
  close_time = 23:59:59.999 UTC on date X  (≈ 00:00 UTC of date X+1, ~09:00 KST X+1)
  close      = price at the end of UTC day X (= 23:59:59 UTC ≈ 09:00 KST next day)

Forward-return label: from close[t] (23:59 UTC X) to close[t+1] (23:59 UTC X+1).
External data (macro/funding/fng) at 00:00 UTC of day X represents that day's
close (published 21:00 UTC X), available 3 hours before our 23:59 UTC X
decision time → external_lag_days=0 is correct (no leakage).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_utc0000_daily(arg: Path) -> pd.DataFrame:
    """Return the UTC midnight aligned daily bar.

    Accepts either a path to btc_15m.parquet (called by the runner) or to a
    directory; resolves btc_1d.parquet from the same data folder.
    """
    if arg.is_dir():
        path = arg / "btc_1d.parquet"
    elif arg.suffix == ".parquet" and arg.name == "btc_15m.parquet":
        path = arg.parent / "btc_1d.parquet"
    else:
        path = arg
    df = pd.read_parquet(path)
    # Match utc2130's history start (15m data begins 2014-01) for fair compare.
    cutoff = pd.Timestamp("2014-01-01", tz="UTC")
    df = df.loc[df["open_time"] >= cutoff].reset_index(drop=True)
    # Normalise datetime resolution to nanoseconds — consistency with the
    # weekly/monthly/15m bars that build_features later as-of-merges against.
    df["close_time"] = df["close_time"].astype("datetime64[ns, UTC]")
    df["open_time"] = df["open_time"].astype("datetime64[ns, UTC]")
    return df


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    df = build_utc0000_daily(here / "data" / "btc_15m.parquet")
    print(f"shape: {df.shape}")
    print("head:"); print(df.head(3))
    print("tail:"); print(df.tail(3))
    print(f"close_time hours: {sorted(df['close_time'].dt.hour.unique().tolist())}")
