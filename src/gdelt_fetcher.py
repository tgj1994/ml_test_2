"""GDELT 2.0 DOC API fetcher — daily Bitcoin news volume + sentiment.

License: The GDELT Project explicitly states all datasets are
"available for unlimited and unrestricted use for any academic, commercial,
or governmental use of any kind without fee." (gdeltproject.org)

Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
  - No authentication required.
  - Rate-limited per IP; daily aggregates over 8 years fit in a few requests.
  - Two relevant timeline modes:
      * TimelineVolRaw — raw daily article-mention count for the query
      * TimelineTone   — daily volume-weighted average article tone in
                          [-100, +100], where 0 ≈ neutral
                          (GDELT tone is heuristic but stable over time)

Query used: `(bitcoin OR "BTC") sourcelang:eng`
  - English-language news only — non-English would inflate the count for
    headlines we cannot reason about and adds different distributional
    characteristics than the EN-language corpus.

Output: data/gdelt.parquet, indexed by UTC-midnight date:
    n_articles    raw article-mention count (TimelineVolRaw)
    avg_tone      volume-weighted mean tone (TimelineTone)
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests


API = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = "bitcoin sourcelang:eng"  # GDELT rejects multi-token OR phrases as "too short"


def _hit(mode: str, start: datetime, end: datetime,
         retries: int = 4) -> pd.DataFrame:
    """One GDELT timeline call → DataFrame indexed by date.

    Note on headers: GDELT's edge tier resets python-requests' default
    User-Agent on Windows (observed empirically). Set a browser-like UA
    to avoid the TCP RST.
    """
    params = {
        "query": QUERY,
        "mode": mode,
        "format": "CSV",
        "timelinesmooth": 0,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/121.0.0.0 Safari/537.36"),
        "Accept": "text/csv,*/*;q=0.5",
    }
    url = f"{API}?{urlencode(params)}"
    last_err: Exception | None = None
    for attempt in range(retries):
        # GDELT enforces "one request per 5 seconds". The first attempt also
        # respects this since fetch_all may call us back-to-back.
        time.sleep(6 if attempt == 0 else 10 * attempt)
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 200 and r.text.strip() and "," in r.text:
                # GDELT returns a 3-column long-format CSV:
                #   Date, Series, Value
                # where one mode may emit multiple series rows per date
                # (e.g. TimelineVolRaw emits both "Article Count" and
                # "Total Monitored Articles"). We keep all of them and
                # pivot to wide form so downstream feature code can pick
                # the columns it cares about.
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip().lstrip("﻿") for c in df.columns]
                df["Date"] = pd.to_datetime(df["Date"], utc=True,
                                            errors="coerce").dt.normalize()
                df = df.dropna(subset=["Date"])
                df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
                wide = (df.pivot_table(index="Date", columns="Series",
                                        values="Value", aggfunc="first")
                        .sort_index())
                # Normalize column names — lowercase + underscore
                wide.columns = [
                    str(c).strip().lower()
                          .replace(" ", "_").replace("-", "_")
                    for c in wide.columns
                ]
                return wide
            last_err = RuntimeError(
                f"GDELT {mode} HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GDELT {mode} failed after retries: {last_err}")


def _hit_chunked(mode: str, start: datetime, end: datetime,
                  chunk_years: int = 2) -> pd.DataFrame:
    """Fetch a long range in `chunk_years`-sized pieces and concatenate.

    Empirically, asking GDELT for an 8-year daily timeline in one shot
    times out about 1/3 of the time. Chunking is more reliable and the
    rate-limit delay between hits keeps us within their published 1 req
    per 5 seconds policy.
    """
    pieces: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor.replace(year=cursor.year + chunk_years), end)
        print(f"    chunk {cursor.date()} -> {nxt.date()}")
        pieces.append(_hit(mode, cursor, nxt))
        cursor = nxt
    return pd.concat(pieces).sort_index()[~pd.concat(pieces).index.duplicated()]


def fetch_all(data_dir: Path,
              start: datetime = datetime(2018, 1, 1, tzinfo=timezone.utc),
              end: datetime | None = None) -> pd.DataFrame:
    end = end or datetime.now(tz=timezone.utc)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GDELT] TimelineVolRaw   {start.date()} -> {end.date()}")
    vol_wide = _hit_chunked("TimelineVolRaw", start, end)
    print(f"  vol columns: {list(vol_wide.columns)}  rows={len(vol_wide)}")

    print(f"[GDELT] TimelineTone     {start.date()} -> {end.date()}")
    try:
        tone_wide = _hit_chunked("TimelineTone", start, end)
        print(f"  tone columns: {list(tone_wide.columns)}  rows={len(tone_wide)}")
    except Exception as e:
        # Volume signal alone is still useful. Tone has been flaky from
        # GDELT's edge in the past; degrade gracefully instead of failing
        # the whole fetch.
        print(f"  tone fetch failed: {e}")
        print(f"  proceeding with vol-only gdelt.parquet (tone columns NaN)")
        tone_wide = pd.DataFrame(index=vol_wide.index,
                                  columns=["average_tone"], dtype=float)

    # Vol returns "article_count" (matches our query) and
    # "total_monitored_articles" (denominator for normalised attention).
    # Tone returns "average_tone" (volume-weighted mean tone).
    out = pd.DataFrame(index=vol_wide.index.union(tone_wide.index))
    out["n_articles"] = vol_wide.get("article_count")
    out["total_monitored"] = vol_wide.get("total_monitored_articles")
    out["avg_tone"] = tone_wide.get("average_tone")
    # Mention rate per 10,000 monitored articles — normalises away the
    # secular growth of GDELT's monitored-source pool.
    import numpy as np
    denom = (out["total_monitored"] / 1e4).replace(0, np.nan)
    rate = out["n_articles"] / denom
    out["mention_rate"] = rate.replace([np.inf, -np.inf], np.nan)

    rows = len(out)
    print(f"  gdelt rows: {rows}   "
          f"n_articles coverage: {int(out['n_articles'].notna().sum())}/{rows}   "
          f"avg_tone coverage: {int(out['avg_tone'].notna().sum())}/{rows}")
    out.to_parquet(data_dir / "gdelt.parquet")
    print(f"  gdelt.parquet : {out.shape}  "
          f"({out.index.min().date()} -> {out.index.max().date()})")
    return out


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    fetch_all(here / "data")
