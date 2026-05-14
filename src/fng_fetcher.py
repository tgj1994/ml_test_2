"""Fear & Greed Index fetcher (alternative.me) — full daily history."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


def fetch_all(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get("https://api.alternative.me/fng/?limit=0", timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise RuntimeError("alternative.me F&G returned no data")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="s", utc=True).dt.normalize()
    df["fng"] = pd.to_numeric(df["value"])
    df = df.set_index("date").sort_index()[["fng"]]
    df.to_parquet(data_dir / "fng.parquet")
    print(f"  fng.parquet : {len(df)} rows  "
          f"({df.index.min().date()} -> {df.index.max().date()})")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    fetch_all(here / "data")
