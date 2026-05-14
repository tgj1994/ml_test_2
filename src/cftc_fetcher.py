"""CFTC TFF (Traders in Financial Futures) fetcher for CME Bitcoin futures.

Source : CFTC Public Reporting Environment (Socrata)
Dataset: gpe5-46if  — Traders in Financial Futures - Futures Only Reports
Contract: CME Bitcoin futures (contract_market_code 133741, contract size 5 BTC).

Report is published every Friday, reflecting position dates as of the previous
Tuesday. We tag each row with `report_date_as_yyyy_mm_dd` and apply a lag of
4 days at feature-build time (Tue position -> available Saturday at the earliest).

Derived features (computed at feature time, not stored raw here):
  cot_levmoney_net_long_ratio    (lev_money_long - lev_money_short) / OI
  cot_levmoney_net_long_chg_1w   1-week diff of the above
  cot_levmoney_z_52w             52-week rolling z-score of the above
  cot_dealer_net_long_ratio      (dealer_long - dealer_short) / OI
  cot_money_manager_net_long_ratio
  cot_top4_concentration_long    conc_gross_le_4_tdr_long_all (%)
  cot_top4_concentration_short
  cot_ma_4w                      4-week MA of cot_levmoney_net_long_ratio

Output:
  data/cot.parquet — indexed by report_date_as_yyyy_mm_dd (Tuesday), weekly cadence
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sodapy import Socrata


CFTC_DOMAIN = "publicreporting.cftc.gov"
TFF_DATASET = "gpe5-46if"  # Traders in Financial Futures - Futures Only Reports
BITCOIN_CONTRACT_CODES = ("133741",)  # CME Bitcoin futures (regular)


def fetch_all(data_dir: Path,
              start: datetime = datetime(2017, 12, 1, tzinfo=timezone.utc)) -> None:
    """Pull CFTC TFF rows filtered to CME Bitcoin futures contracts.

    CME Bitcoin futures launched 2017-12-17 so we start the query from
    2017-12-01 to be safe.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    client = Socrata(CFTC_DOMAIN, None, timeout=60)
    rows: list[dict] = []
    page = 1000
    offset = 0
    where = (
        f"cftc_contract_market_code in ({','.join(repr(c) for c in BITCOIN_CONTRACT_CODES)}) "
        f"AND report_date_as_yyyy_mm_dd >= '{start.strftime('%Y-%m-%dT00:00:00')}'"
    )
    while True:
        batch = client.get(TFF_DATASET, where=where, limit=page, offset=offset,
                            order="report_date_as_yyyy_mm_dd")
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
    client.close()

    if not rows:
        raise RuntimeError("CFTC TFF: no rows for CME Bitcoin futures")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], utc=True).dt.normalize()

    keep_num = [
        "open_interest_all",
        "dealer_positions_long_all", "dealer_positions_short_all",
        "lev_money_positions_long", "lev_money_positions_short",
        "asset_mgr_positions_long", "asset_mgr_positions_short",
        "conc_gross_le_4_tdr_long_all", "conc_gross_le_4_tdr_short_all",
    ]
    for c in keep_num:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.set_index("date").sort_index()[keep_num]
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(data_dir / "cot.parquet")
    print(f"  cot.parquet : {len(df)} rows  "
          f"({df.index.min().date()} -> {df.index.max().date()})")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    fetch_all(here / "data")
