"""Build per-day feature matrix from the production data sources.

Inputs (all written to data/ by the matching fetcher modules):
  data/btc_1d.parquet, btc_1w.parquet, btc_1M.parquet, btc_15m.parquet
                              -- Binance BTCUSDT klines via binance_fetcher.py
  data/eth_1d.parquet         -- Binance ETHUSDT daily klines via binance_fetcher.py
  data/macro_fred.parquet     -- FRED via fred_fetcher.py. v0~v6 use 6 core
                                 series (dxy, tnx, t10y2y, t10yie, indpro,
                                 unrate); v7 expands to 21 series (Tier-1
                                 yield curve, TIPS, policy rates, Fed balance
                                 sheet, FX, jobless/payrolls). See _FRED_COLS
                                 for the whitelist. No SPX/VIX/Gold — third-
                                 party copyright.
  data/m2.parquet             -- FRED M2SL (used from v4 onward)
  data/cot.parquet            -- CFTC TFF (CME BTC futures, contract 133741)
                                 via cftc_fetcher.py. Coexists with Binance
                                 funding/basis below — TFF captures large-
                                 trader directional positioning while funding/
                                 basis captures perp/futures crowding.
  data/funding.parquet,
  data/basis.parquet          -- Binance perpetual funding rate + futures-spot
                                 basis (eight microstructure features). Restored
                                 from the legacy coin_analysis pipeline once the
                                 user confirmed Binance use is acceptable.
  data/fng.parquet            -- alternative.me Fear & Greed (free public API)
  data/gdelt.parquet          -- GDELT BTC news attention + tone via
                                 gdelt_fetcher.py (loaded only when
                                 use_features_v7 or use_features_v5_3).
  data/coinmetrics.parquet    -- Coin Metrics community on-chain snapshot
                                 (HashRate/AdrActCnt/TxCnt/CapMrktCurUSD).
                                 Used as-is for both training and inference
                                 today. The planned swap to Bitcoin Core RPC
                                 + block scan output happens after the
                                 W:\\onchain IBD finishes (see
                                 live_inference_btc_core.py); the swap keeps
                                 the same column schema so features.py needs
                                 no change. If the file is absent on-chain
                                 features fall back to NaN, which EBM/XGB
                                 handle natively.

Bitstamp (legacy `data_fetcher.py`) is retired. Original Bitstamp parquets
live under `data/_bitstamp_backup/` for reference but no longer feed the
pipeline.

Layers (final per-day matrix):
  1.  Multi-timeframe TA snapshots        (daily / weekly / monthly)
  2.  MVRV-Z proxies                       (price-based + 2 on-chain via coinmetrics)
  3.  v3 add-ons                           (cross-asset rho, CFTC velocity,
                                            hashrate ribbon, vol regime)
  4.  Daily short-window + microstructure  (RV, skew, kurt, up-frac, vol surge)
  5.  Sideways / regime features
  6.  Macro context                        (FRED daily series, v7-aware)
  7.  ETH features + ETH/BTC ratio
  8.  Futures positioning                  (CFTC TFF + Binance funding/basis)
  9.  15-minute intraday summary           (Binance 15m → 6 daily features)
  10. Sentiment                            (Fear & Greed Index)
  11. GDELT news attention                 (v7 / v5_3 only)
  12. Calendar                             (DOW / month / halving distance)

Target: filtered binary direction (next-day return crossing +-threshold).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABEL_THRESHOLD = 0.010


def build_labels(close: pd.Series, threshold) -> pd.Series:
    fwd_ret = close.pct_change().shift(-1)
    y = pd.Series(np.nan, index=close.index, dtype=float)
    if isinstance(threshold, pd.Series):
        th = threshold.reindex(close.index)
        y[fwd_ret > th] = 1.0
        y[fwd_ret < -th] = 0.0
    else:
        y[fwd_ret > threshold] = 1.0
        y[fwd_ret < -threshold] = 0.0
    return y


def compute_dynamic_threshold(close: pd.Series, k: float, n: int = 30) -> pd.Series:
    daily_ret = close.pct_change()
    sigma_n = daily_ret.rolling(n, min_periods=n).std()
    return k * sigma_n


# ---------- technical-indicator primitives ----------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - 100 / (1 + rs)


def _macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def _bb_position(close: pd.Series, window: int = 20) -> pd.Series:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (close - ma) / (2 * sd)


def _slope(series: pd.Series) -> float:
    y = series.dropna().to_numpy()
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    return np.polyfit(x, y, 1)[0] / (y.mean() + 1e-9)


def _mvrv_z_proxy(close: pd.Series, window: int = 365) -> pd.Series:
    win = min(window, max(30, len(close) // 2))
    realized = close.rolling(win, min_periods=win // 3).mean()
    mv_rv = close - realized
    std = mv_rv.rolling(win, min_periods=win // 3).std()
    return mv_rv / (std + 1e-9)


# v3 drop list — redundant columns relative to v2/v3 add-ons.
# (SPX/VIX/Gold-related columns are not generated at all in commercial mode,
#  so they are not listed here.)
V3_DROP_COLS = (
    "dxy_z_60d", "tnx_z_60d",
    "d_close",
    "sw_chop_er_5d", "sw_chop_er_60d",
    "cal_dow",
)

V4_DROP_COLS = (
    "d_close",
    "sw_chop_er_5d", "sw_chop_er_60d",
    "cal_dow",
)

# v5 / v5_2 / v5_3 keep lists.
#   - v5     : cumulative ~80% main-effect importance (aggressive prune)
#   - v5_2   : cumulative ~92% main-effect importance (looser prune)
#   - v5_3   : cumulative ~93% from the v7 sweep importance ranking
# All three are auto-generated by analysis/v5_keep_selector_ebm.py with
# different --cumulative / --output-file settings. The fallback tuple at
# the bottom (V5_KEEP_COLS literal) is only used if the v5_keep module is
# missing (e.g. first run before any selector pass).
try:
    from src.v5_3_keep import V5_3_KEEP_COLS  # type: ignore
except ImportError:
    V5_3_KEEP_COLS = ()   # generated after v7 sweep completes

try:
    from src.v5_2_keep import V5_2_KEEP_COLS  # type: ignore
except ImportError:
    V5_2_KEEP_COLS = ()   # forces use_features_v5_2 to fall back to no-op

try:
    from src.v5_keep import V5_KEEP_COLS  # type: ignore
except ImportError:
    V5_KEEP_COLS = (
        # daily TA
        "d_rsi14", "d_bb_pos", "d_macd_hist", "d_vol", "d_range_pct",
        # weekly / monthly TA
        "w_range_pct", "w_w26_dd", "w_close", "w_w26_slope", "w_w26_cum_ret",
        "w_vol", "w_ma_ratio", "w_bb_pos", "w_macd_hist",
        "M_close", "M_vol", "M_macd_hist", "M_w6_cum_ret", "M_w6_dd",
        "M_bb_pos", "M_rsi14", "M_w6_slope", "M_vol_z", "M_range_pct", "M_ret_1",
        # window summary daily
        "d_w180_dd", "d_w180_slope", "d_w180_cum_ret",
        # daily-rolling replacements for m15
        "d_5d_dd", "d_5d_vol",
        # sideways
        "sw_ci_14d", "sw_ci_30d", "sw_range_pct_30d", "sw_chop_er_14d",
        "sw_bb_width_20d",
        # macro (FRED, commercial)
        "dxy_ret_5d", "dxy_vol_20d", "tnx_ret_5d", "tnx_vol_20d",
        "t10y2y", "t10yie", "indpro", "unrate",
        # ETH (Binance ETHUSDT)
        "eth_close", "ethbtc_ratio", "eth_close_vol_20d",
        "eth_close_ret_1d", "eth_close_ret_5d",
        # CFTC positioning (CME futures; complements Binance funding/basis)
        "cot_levmoney_net_long_ratio", "cot_levmoney_net_long_chg_1w",
        "cot_dealer_net_long_ratio", "cot_money_manager_net_long_ratio",
        "cot_top4_long_pct", "cot_levmoney_z_52w", "cot_ma_4w",
        # sentiment
        "fng", "fng_ma_7d", "fng_chg_3d",
        # MVRV proxies (price-based, no paid API needed)
        "mvrv2_z_200w", "mvrv2_z_ema900", "mvrv2_mayer",
        # on-chain (coinmetrics.parquet today; Bitcoin Core after IBD)
        "hashrate_ribbon", "adr_chg_7d", "tx_per_adr",
        "mvrv2_nvt_z", "mvrv2_hashrate_z",
        # vol regime
        "vol_of_vol_30d", "vol_regime_z",
        # calendar
        "cal_month", "cal_days_to_halving", "cal_days_since_halving",
    )


def _m2_features(data_dir: Path,
                 daily_idx: pd.DatetimeIndex) -> pd.DataFrame:
    p = data_dir / "m2.parquet"
    if not p.exists():
        return pd.DataFrame(index=daily_idx)
    m = pd.read_parquet(p)
    m.index = pd.to_datetime(m.index, utc=True).normalize()
    m.index = m.index + pd.Timedelta(days=21)

    daily_range = pd.date_range(
        m.index.min(),
        max(m.index.max(), daily_idx.max()) + pd.Timedelta(days=1),
        freq="D", tz="UTC")
    m_daily = m.reindex(daily_range).interpolate(method="linear", limit_direction="both")
    s = m_daily["m2"]
    out = pd.DataFrame(index=m_daily.index)
    out["m2_chg_1d"] = s.pct_change(1)
    out["m2_chg_1w"] = s.pct_change(7)
    out["m2_chg_1m"] = s.pct_change(30)
    out["m2_chg_3m"] = s.pct_change(90)
    z_win = 720
    out["m2_z_24m"] = (s - s.rolling(z_win, min_periods=180).mean()) \
                       / (s.rolling(z_win, min_periods=180).std() + 1e-9)
    return _align_to_daily(out, daily_idx)


def _cot_features(data_dir: Path,
                  daily_idx: pd.DatetimeIndex,
                  external_lag_days: int) -> pd.DataFrame:
    """CFTC TFF positioning features (CME BTC futures, contract 133741).

    Coexists with the restored Binance funding/basis in _futures_features:
    funding/basis captures perpetual/futures crowding pressure while TFF
    captures large-trader directional positioning, so they're complementary
    rather than substitutes.

    CFTC publishes Tuesday positions on Friday. We apply a 4-day lag to be
    safe (Tue position -> available Saturday onwards), then ffill weekly
    values to the daily index.
    """
    p = data_dir / "cot.parquet"
    if not p.exists():
        return pd.DataFrame(index=daily_idx)
    cot = pd.read_parquet(p)
    cot.index = pd.to_datetime(cot.index, utc=True).normalize()
    lag = max(external_lag_days, 4)
    cot.index = cot.index + pd.Timedelta(days=lag)

    oi = cot["open_interest_all"].replace(0, np.nan)
    lev_long = cot["lev_money_positions_long"]
    lev_short = cot["lev_money_positions_short"]
    dealer_long = cot["dealer_positions_long_all"]
    dealer_short = cot["dealer_positions_short_all"]
    mm_long = cot["asset_mgr_positions_long"]
    mm_short = cot["asset_mgr_positions_short"]
    # NB: `conc_gross_le_4_tdr_long/short_all` are 100% NaN in the CFTC TFF
    # feed for CME BTC futures (133741). They were emitted as features
    # previously, contributed zero importance, and only added noise.
    # Dropped here; cot_top4_skew_4w in _features_v3_additions is removed
    # for the same reason.

    lev_net_ratio = (lev_long - lev_short) / oi
    out = pd.DataFrame(index=cot.index)
    out["cot_levmoney_net_long_ratio"] = lev_net_ratio
    out["cot_levmoney_net_long_chg_1w"] = lev_net_ratio.diff(1)
    out["cot_dealer_net_long_ratio"] = (dealer_long - dealer_short) / oi
    out["cot_money_manager_net_long_ratio"] = (mm_long - mm_short) / oi
    out["cot_levmoney_z_52w"] = (
        (lev_net_ratio - lev_net_ratio.rolling(52, min_periods=12).mean())
        / (lev_net_ratio.rolling(52, min_periods=12).std() + 1e-9)
    )
    out["cot_ma_4w"] = lev_net_ratio.rolling(4, min_periods=2).mean()

    # weekly -> daily ffill
    daily_calendar = pd.date_range(out.index.min(), daily_idx.max(),
                                    freq="D", tz="UTC")
    out_daily = out.reindex(daily_calendar).ffill()
    return _align_to_daily(out_daily, daily_idx)


def _features_v3_additions(d1: pd.DataFrame, data_dir: Path,
                            daily_idx: pd.DatetimeIndex,
                            external_lag_days: int) -> pd.DataFrame:
    """v3 add-on features.

      A. BTC cross-asset 30d log-return correlation vs ETH/DXY only
         (SPX/Gold dropped per third-party copyright)
      B. CFTC positioning velocity (top4_skew was dropped — its inputs are
         100% NaN in the CME BTC TFF feed)
      C. Hash rate ribbon and 7d change   (from coinmetrics today; Bitcoin
                                            Core after IBD)
      D. Active address momentum + tx/adr (same source as C)
      E. Volatility regime
    """
    out = pd.DataFrame(index=daily_idx)

    # ---- A. Cross-asset correlations (FRED dxy + Binance eth) -------------
    g = d1.set_index("close_time").sort_index()
    btc_log_ret = np.log(g["close"]).diff().reindex(daily_idx)

    macro_path = data_dir / "macro_fred.parquet"
    if macro_path.exists():
        m = pd.read_parquet(macro_path)
        m.index = pd.to_datetime(m.index, utc=True).normalize()
        if external_lag_days:
            m.index = m.index + pd.Timedelta(days=external_lag_days)
        m_aligned = _align_to_daily(m, daily_idx)
        if "dxy" in m_aligned.columns:
            dxy_ret = np.log(m_aligned["dxy"]).diff().reindex(btc_log_ret.index)
            out["rho30_btc_dxy"] = btc_log_ret.rolling(30, min_periods=15).corr(dxy_ret)

    eth_path = data_dir / "eth_1d.parquet"
    if eth_path.exists():
        eth = pd.read_parquet(eth_path)
        # BUG fix 2026-05-13: previously the eth index was .normalize()-d to
        # midnight while btc_log_ret kept the raw close_time (23:59:59.999),
        # so reindex() produced 100% NaN and rho30_btc_eth had zero signal.
        eth_close = (eth.set_index(pd.to_datetime(eth["close_time"], utc=True))
                       ["close"])
        eth_close = eth_close.reindex(btc_log_ret.index)
        eth_ret = np.log(eth_close).diff()
        out["rho30_btc_eth"] = btc_log_ret.rolling(30, min_periods=15).corr(eth_ret)

    # ---- B. CFTC positioning velocity (replaces Binance funding skew) ------
    # NB: `cot_top4_skew_4w` removed — its inputs (conc_gross_le_4_tdr_*) are
    # 100% NaN in the CME BTC TFF feed.
    cot_path = data_dir / "cot.parquet"
    if cot_path.exists():
        c = pd.read_parquet(cot_path)
        c.index = pd.to_datetime(c.index, utc=True).normalize() + pd.Timedelta(
            days=max(external_lag_days, 4))
        velocity_num = c["lev_money_positions_long"].diff(1)
        velocity_den = c["lev_money_positions_long"].rolling(8, min_periods=3).std() + 1e-9
        cot_extra = pd.DataFrame({
            "cot_levmoney_velocity": velocity_num / velocity_den,
        }, index=c.index)
        daily_calendar = pd.date_range(cot_extra.index.min(), daily_idx.max(),
                                        freq="D", tz="UTC")
        cot_daily = cot_extra.reindex(daily_calendar).ffill()
        out = out.join(_align_to_daily(cot_daily, daily_idx), how="left")

    # ---- C/D. On-chain (data/coinmetrics.parquet today; Bitcoin Core later) -
    cm_path = data_dir / "coinmetrics.parquet"
    if cm_path.exists():
        cm = pd.read_parquet(cm_path)
        cm.index = pd.to_datetime(cm.index, utc=True).normalize()
        cm_lag = max(external_lag_days, 1)
        cm.index = cm.index + pd.Timedelta(days=cm_lag)

        cm_extra = pd.DataFrame(index=cm.index)
        if "HashRate" in cm.columns:
            hr = cm["HashRate"]
            cm_extra["hashrate_ribbon"] = (hr.rolling(30, min_periods=10).mean()
                                            / hr.rolling(60, min_periods=20).mean()
                                            - 1.0)
            cm_extra["hashrate_chg_7d"] = hr.pct_change(7)
        if "AdrActCnt" in cm.columns:
            cm_extra["adr_chg_7d"] = cm["AdrActCnt"].pct_change(7)
        if {"AdrActCnt", "TxCnt"}.issubset(cm.columns):
            cm_extra["tx_per_adr"] = cm["TxCnt"] / (cm["AdrActCnt"] + 1.0)
        if not cm_extra.empty:
            out = out.join(_align_to_daily(cm_extra, daily_idx), how="left")

    # ---- E. Volatility regime -----------------------------------------------
    log_ret = np.log(g["close"]).diff()
    daily_vol = log_ret.rolling(14, min_periods=5).std()
    vov = daily_vol.rolling(30, min_periods=10).std()
    vol_long_mean = daily_vol.rolling(365, min_periods=120).mean()
    vol_long_std = daily_vol.rolling(365, min_periods=120).std()
    vol_regime_z = (daily_vol - vol_long_mean) / (vol_long_std + 1e-9)
    e_extra = pd.DataFrame({
        "vol_of_vol_30d": vov,
        "vol_regime_z": vol_regime_z,
    })
    out = out.join(_align_to_daily(e_extra, daily_idx), how="left")
    return out


def _mvrv_v2_features(d1: pd.DataFrame, data_dir: Path,
                      daily_idx: pd.DatetimeIndex,
                      external_lag_days: int) -> pd.DataFrame:
    g = d1.set_index("close_time").sort_index()
    close = g["close"]

    out = pd.DataFrame(index=close.index)
    ma200w = close.rolling(1400, min_periods=200).mean()
    diff = close - ma200w
    sigma = diff.rolling(1400, min_periods=200).std()
    out["mvrv2_z_200w"] = diff / (sigma + 1e-9)

    log_close = np.log(close)
    real_ema = np.exp(log_close.ewm(halflife=900, adjust=False).mean())
    diff_ema = close - real_ema
    sigma_ema = diff_ema.rolling(900, min_periods=180).std()
    out["mvrv2_z_ema900"] = diff_ema / (sigma_ema + 1e-9)

    ma90 = close.rolling(90, min_periods=30).mean()
    ma365 = close.rolling(365, min_periods=120).mean()
    ma1000 = close.rolling(1000, min_periods=200).mean()
    ma1500 = close.rolling(1500, min_periods=200).mean()
    real_multi = 0.2 * ma90 + 0.2 * ma365 + 0.3 * ma1000 + 0.3 * ma1500
    diff_multi = close - real_multi
    sigma_multi = diff_multi.rolling(365, min_periods=120).std()
    out["mvrv2_z_multi"] = diff_multi / (sigma_multi + 1e-9)

    ma200 = close.rolling(200, min_periods=50).mean()
    out["mvrv2_mayer"] = close / ma200 - 1.0

    out_aligned = _align_to_daily(out, daily_idx)

    # NVT and HashRate z from on-chain (data/coinmetrics.parquet today)
    cm_path = data_dir / "coinmetrics.parquet"
    if cm_path.exists():
        cm = pd.read_parquet(cm_path)
        cm.index = pd.to_datetime(cm.index, utc=True).normalize()
        cm_lag = max(external_lag_days, 1)
        cm.index = cm.index + pd.Timedelta(days=cm_lag)

        cm_feat = pd.DataFrame(index=cm.index)
        if {"CapMrktCurUSD", "AdrActCnt"}.issubset(cm.columns):
            mcap = cm["CapMrktCurUSD"]
            adr = cm["AdrActCnt"]
            nvt = mcap / (adr + 1.0)
            cm_feat["mvrv2_nvt_z"] = ((nvt - nvt.rolling(180, min_periods=60).mean())
                                       / (nvt.rolling(180, min_periods=60).std() + 1e-9))
        if "HashRate" in cm.columns:
            hr = cm["HashRate"]
            cm_feat["mvrv2_hashrate_z"] = ((hr - hr.rolling(30, min_periods=15).mean())
                                            / (hr.rolling(30, min_periods=15).std() + 1e-9))
        if not cm_feat.empty:
            cm_aligned = _align_to_daily(cm_feat, daily_idx)
            out_aligned = out_aligned.join(cm_aligned, how="left")
    return out_aligned


# ---------- per-timeframe feature pack ----------

def _tf_features(df: pd.DataFrame, prefix: str, lookback: int) -> pd.DataFrame:
    g = df.copy().set_index("close_time")
    g["ret_1"] = g["close"].pct_change()
    g["log_ret"] = np.log(g["close"]).diff()
    g["rsi14"] = _rsi(g["close"], 14)
    g["macd_hist"] = _macd_hist(g["close"])
    g["bb_pos"] = _bb_position(g["close"], min(20, max(5, lookback // 2)))
    g["ma_fast"] = g["close"].rolling(min(7, lookback)).mean()
    g["ma_slow"] = g["close"].rolling(min(25, lookback)).mean()
    g["ma_ratio"] = g["ma_fast"] / g["ma_slow"] - 1
    g["vol"] = g["log_ret"].rolling(min(14, lookback)).std()
    g["range_pct"] = (g["high"] - g["low"]) / g["close"]
    g["vol_z"] = (g["volume"] - g["volume"].rolling(lookback).mean()) / (
        g["volume"].rolling(lookback).std() + 1e-9)
    keep = ["close", "ret_1", "rsi14", "macd_hist", "bb_pos", "ma_ratio",
            "vol", "range_pct", "vol_z"]
    out = g[keep].copy()
    out.columns = [f"{prefix}_{c}" for c in out.columns]
    return out


def _asof_join(daily_idx: pd.DatetimeIndex, htf: pd.DataFrame) -> pd.DataFrame:
    target_ts = pd.to_datetime(pd.DatetimeIndex(daily_idx), utc=True).astype(
        "datetime64[ns, UTC]")
    target = pd.DataFrame({"ts": target_ts}).sort_values("ts")
    src = htf.copy()
    src.index.name = "ts"
    src = src.reset_index().sort_values("ts")
    src["ts"] = pd.to_datetime(src["ts"], utc=True).astype("datetime64[ns, UTC]")
    merged = pd.merge_asof(target, src, on="ts", direction="backward")
    return merged.set_index("ts")


def _window_summary(daily_idx: pd.DatetimeIndex,
                    htf: pd.DataFrame,
                    prefix: str,
                    window: int) -> pd.DataFrame:
    closes = htf[f"{prefix}_close"].dropna()
    rows = []
    for t in daily_idx:
        s = closes[closes.index <= t]
        if len(s) < window:
            rows.append({"slope": np.nan, "cum_ret": np.nan, "dd": np.nan})
            continue
        w = s.iloc[-window:]
        cum_ret = w.iloc[-1] / w.iloc[0] - 1
        roll_max = w.cummax()
        dd = (w / roll_max - 1).min()
        rows.append({"slope": _slope(w), "cum_ret": cum_ret, "dd": dd})
    df = pd.DataFrame(rows, index=daily_idx)
    df.columns = [f"{prefix}_w{window}_{c}" for c in df.columns]
    return df


def _daily_short_window(d1: pd.DataFrame, daily_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Replace legacy m15_5d_dd / m15_5d_vol with daily-based equivalents."""
    g = d1.copy().set_index("close_time").sort_index()
    close = g["close"]
    log_ret = np.log(close).diff()
    roll_max_5d = close.rolling(5, min_periods=3).max()
    dd_5d = close / roll_max_5d - 1.0
    vol_5d = log_ret.rolling(5, min_periods=3).std()
    out = pd.DataFrame({"d_5d_dd": dd_5d, "d_5d_vol": vol_5d}, index=close.index)
    return _align_to_daily(out, daily_idx)


def _microstructure_features(d1: pd.DataFrame,
                              daily_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Close- and volume-only proxies that complement the restored Binance
    funding/basis microstructure signal — these capture realized-vol regime
    and momentum/mean-reversion shape on the spot series itself, which
    funding/basis (term-structure premium) does not see.

    Outputs:
      d_rv_30d / d_rv_60d        realized vol over longer regimes
                                  (complements existing d_vol = 14d, d_5d_vol)
      d_ret_acorr_5d / _20d      lag-1 autocorrelation — momentum/mean-reversion
      d_ret_skew_30d             distributional asymmetry
      d_ret_kurt_30d             tail-fat indicator
      d_up_frac_20d              fraction of up-days, simple momentum
      d_vol_surge_5d             short-window volume z vs 60d baseline
    """
    g = d1.copy().set_index("close_time").sort_index()
    close = g["close"]
    volume = g["volume"]
    log_ret = np.log(close).diff()

    out = pd.DataFrame(index=close.index)
    out["d_rv_30d"] = log_ret.rolling(30, min_periods=15).std()
    out["d_rv_60d"] = log_ret.rolling(60, min_periods=30).std()
    out["d_ret_acorr_5d"] = log_ret.rolling(5, min_periods=4).apply(
        lambda s: s.autocorr(lag=1), raw=False)
    out["d_ret_acorr_20d"] = log_ret.rolling(20, min_periods=10).apply(
        lambda s: s.autocorr(lag=1), raw=False)
    out["d_ret_skew_30d"] = log_ret.rolling(30, min_periods=15).skew()
    out["d_ret_kurt_30d"] = log_ret.rolling(30, min_periods=15).kurt()
    is_up = (log_ret > 0).astype(float)
    out["d_up_frac_20d"] = is_up.rolling(20, min_periods=10).mean()
    out["d_vol_surge_5d"] = (
        (volume.rolling(5, min_periods=3).mean()
         - volume.rolling(60, min_periods=30).mean())
        / (volume.rolling(60, min_periods=30).std() + 1e-9)
    )
    # Replace any inf that slip through (e.g. constant-volume windows / zero std)
    out = out.replace([np.inf, -np.inf], np.nan)
    return _align_to_daily(out, daily_idx)


# ---------- sideways / regime features ----------

def _efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return net / (path + 1e-12)


def _choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(n).sum()
    rng = high.rolling(n).max() - low.rolling(n).min()
    ci = 100.0 * np.log10(atr_sum / (rng + 1e-12)) / np.log10(n)
    return ci


def _consecutive_streak(s: pd.Series) -> pd.Series:
    grp = (s != s.shift()).cumsum()
    return s.astype(int).groupby(grp).cumsum().where(s, 0)


def _sideways_features(d1: pd.DataFrame) -> pd.DataFrame:
    g = d1.copy().set_index("close_time").sort_index()
    close = g["close"]; high = g["high"]; low = g["low"]
    ret = close.pct_change()
    sigma_30 = ret.rolling(30, min_periods=20).std()

    out = pd.DataFrame(index=close.index)
    for n in (5, 14, 30, 60):
        out[f"sw_chop_er_{n}d"] = 1.0 - _efficiency_ratio(close, n)
    is_sw = (ret.abs() < (0.4 * sigma_30)).fillna(False)
    out["sw_ratio_14d"] = is_sw.rolling(14).mean()
    out["sw_ratio_30d"] = is_sw.rolling(30).mean()
    out["sw_ci_14d"] = _choppiness_index(high, low, close, 14)
    out["sw_ci_30d"] = _choppiness_index(high, low, close, 30)
    out["sw_bb_width_20d"] = (close.rolling(20).std() * 4.0) / close
    out["sw_range_pct_30d"] = (high.rolling(30).max() - low.rolling(30).min()) / close
    out["sw_streak_med"] = _consecutive_streak(is_sw).astype(float)
    return out


# ---------- macro / sentiment / calendar ----------

# FRED-only macro context.
#
# Daily-cadence US-gov series only. indpro / unrate / payems are monthly
# with multi-week release lags so they don't drive next-day BTC; we still
# whitelist them because v7 wants the broader macro envelope (the model
# can learn to ignore them if they truly add nothing). All listed series
# are US Treasury / Fed Reserve / BLS publications, public-domain.
_FRED_COLS = (
    # legacy core (always on)
    "dxy", "tnx", "t10y2y", "t10yie",
    # v7 Tier-1 expansion
    "tnx2y", "tnx5y", "tnx30y",          # Treasury yield curve
    "tips5y", "tips10y",                 # TIPS real yields
    "fedfunds", "sofr",                  # policy / overnight
    "fed_assets", "fed_reserves",        # Fed balance sheet
    "rrp_overnight",                     # reverse repo liquidity sink
    "fx_krw", "fx_jpy", "fx_cny",        # cross-rates (Fed published)
    "jobless_initial", "payems",         # labour leading indicators
)


def _macro_features(daily_idx: pd.DatetimeIndex, data_dir: Path,
                    external_lag_days: int = 0) -> pd.DataFrame:
    """FRED daily macro context restricted to _FRED_COLS.

    pct_change can produce +/-inf when a spread series (t10y2y) crosses zero
    — XGB rejects inf at fit/predict time and EBM otherwise lumps inf into
    the topmost bin (inconsistent across models). Coerce inf -> NaN here so
    every downstream consumer sees the same value.
    """
    path = data_dir / "macro_fred.parquet"
    if not path.exists():
        return pd.DataFrame(index=daily_idx)
    m = pd.read_parquet(path)
    m.index = pd.to_datetime(m.index, utc=True).normalize()
    if external_lag_days:
        m.index = m.index + pd.Timedelta(days=external_lag_days)

    out = pd.DataFrame(index=m.index)
    for col in m.columns:
        if col not in _FRED_COLS:
            continue
        c = m[col].dropna()
        ret = c.pct_change().replace([np.inf, -np.inf], np.nan)
        ret5 = c.pct_change(5).replace([np.inf, -np.inf], np.nan)
        out[col + "_ret_1d"] = ret
        out[col + "_ret_5d"] = ret5
        out[col + "_vol_20d"] = ret.rolling(20).std()
        out[col + "_z_60d"] = (c - c.rolling(60).mean()) / (c.rolling(60).std() + 1e-9)
        out[col] = c  # raw level
    return _align_to_daily(out, daily_idx)


def _eth_features(daily_idx: pd.DatetimeIndex, data_dir: Path,
                  external_lag_days: int = 0) -> pd.DataFrame:
    """ETH/USD daily features from Binance ETHUSDT (data/eth_1d.parquet, built
    by binance_fetcher.py). Returns eth_close raw + ret_1d/ret_5d/vol_20d/z_60d."""
    path = data_dir / "eth_1d.parquet"
    if not path.exists():
        return pd.DataFrame(index=daily_idx)
    eth = pd.read_parquet(path)
    eth.index = pd.to_datetime(eth["close_time"], utc=True).dt.normalize()
    c = eth["close"]
    if external_lag_days:
        c.index = c.index + pd.Timedelta(days=external_lag_days)
    ret = c.pct_change()
    out = pd.DataFrame({
        "eth_close": c,
        "eth_close_ret_1d": ret,
        "eth_close_ret_5d": c.pct_change(5),
        "eth_close_vol_20d": ret.rolling(20).std(),
        "eth_close_z_60d": (c - c.rolling(60).mean()) / (c.rolling(60).std() + 1e-9),
    })
    return _align_to_daily(out, daily_idx)


def _futures_features(daily_idx: pd.DatetimeIndex, data_dir: Path,
                      external_lag_days: int = 0) -> pd.DataFrame:
    """Binance perpetual funding rate + futures-spot basis features.

    Restored from the legacy coin_analysis pipeline (commit history shows
    the same 8 columns there). funding/basis are Binance-derived; the user
    confirmed Binance is acceptable for both training and inference, so
    this function ships eight microstructure features that capture the
    crowding pressure signal that pure spot OHLCV cannot.

    Columns produced (8 total):
      funding_mean, funding_max, funding_min  — raw daily aggregates
      funding_ma_7d, funding_z_30d            — smoothed level + std z
      basis                                   — raw daily basis
      basis_ma_5d, basis_z_30d                — smoothed level + std z
    """
    out = pd.DataFrame(index=daily_idx)

    fpath = data_dir / "funding.parquet"
    if fpath.exists():
        f = pd.read_parquet(fpath)
        f.index = pd.to_datetime(f.index, utc=True).normalize()
        f["funding_ma_7d"] = f["funding_mean"].rolling(7).mean()
        denom = f["funding_mean"].rolling(30).std() + 1e-9
        f["funding_z_30d"] = (f["funding_mean"]
                              - f["funding_mean"].rolling(30).mean()) / denom
        if external_lag_days:
            f.index = f.index + pd.Timedelta(days=external_lag_days)
        out = out.join(_align_to_daily(f, daily_idx), how="left")

    bpath = data_dir / "basis.parquet"
    if bpath.exists():
        b = pd.read_parquet(bpath)
        b.index = pd.to_datetime(b.index, utc=True).normalize()
        b["basis_ma_5d"] = b["basis"].rolling(5).mean()
        denom = b["basis"].rolling(30).std() + 1e-9
        b["basis_z_30d"] = (b["basis"] - b["basis"].rolling(30).mean()) / denom
        if external_lag_days:
            b.index = b.index + pd.Timedelta(days=external_lag_days)
        out = out.join(_align_to_daily(b, daily_idx), how="left")
    return out


def _intraday_summary(daily_idx: pd.DatetimeIndex,
                      m15: pd.DataFrame) -> pd.DataFrame:
    """Compress Binance 15-minute klines into 6 daily summary features.

    Restored from the legacy pipeline; uses asof-backward join so that on
    each daily timestamp we see only intraday data that closed BEFORE the
    daily bar (i.e. no leakage from the current day's still-forming bar).

    Columns produced (6):
      m15_24h_ret   — 96-bar close pct_change (last 24h log-ish return)
      m15_24h_vol   — 96-bar realized vol of 15m log returns
      m15_5d_vol    — 480-bar realized vol (5d intraday)
      m15_24h_rsi   — RSI(14) on 15m close
      m15_5d_dd     — 5d intraday drawdown from 480-bar rolling max
      m15_24h_slope — 24h linear regression slope on log close
    """
    g = m15.copy().set_index("close_time").sort_index()
    g["log_ret"] = np.log(g["close"]).diff()
    w24, w5d = 96, 96 * 5  # 24h and 5d in 15-minute bars
    g["m15_24h_ret"] = g["close"].pct_change(w24)
    g["m15_24h_vol"] = g["log_ret"].rolling(w24).std()
    g["m15_5d_vol"] = g["log_ret"].rolling(w5d).std()
    g["m15_24h_rsi"] = _rsi(g["close"], 14)
    roll_max_5d = g["close"].rolling(w5d).max()
    g["m15_5d_dd"] = g["close"] / roll_max_5d - 1

    log_close = np.log(g["close"].to_numpy())
    if len(log_close) >= w24:
        from numpy.lib.stride_tricks import sliding_window_view
        x = np.arange(w24, dtype=float) - (w24 - 1) / 2.0
        sx2 = float((x ** 2).sum())
        windows = sliding_window_view(log_close, w24)
        slope = (windows * x).sum(axis=1) / sx2
        slope_full = np.full(len(log_close), np.nan)
        slope_full[w24 - 1:] = slope
        g["m15_24h_slope"] = slope_full
    else:
        g["m15_24h_slope"] = np.nan

    keep = ["m15_24h_ret", "m15_24h_vol", "m15_24h_rsi",
            "m15_5d_vol", "m15_5d_dd", "m15_24h_slope"]
    src = (g[keep].reset_index()
            .rename(columns={"close_time": "ts"})
            .sort_values("ts"))
    src["ts"] = pd.to_datetime(src["ts"], utc=True).astype("datetime64[ns, UTC]")
    target_ts = pd.to_datetime(pd.DatetimeIndex(daily_idx), utc=True).astype(
        "datetime64[ns, UTC]")
    target = pd.DataFrame({"ts": target_ts}).sort_values("ts")
    merged = pd.merge_asof(target, src, on="ts", direction="backward")
    return merged.set_index("ts")


def _gdelt_features(daily_idx: pd.DatetimeIndex,
                    data_dir: Path,
                    external_lag_days: int = 0) -> pd.DataFrame:
    """GDELT-derived news-attention + sentiment features (v7 add-on).

    Reads data/gdelt.parquet (built by src.gdelt_fetcher) which exposes:
      n_articles      raw Bitcoin article-mention count per day
      total_monitored GDELT's daily monitored-article denominator
      mention_rate    n_articles / (total_monitored / 1e4)
      avg_tone        volume-weighted mean tone (-100..+100), may be NaN
                      if the GDELT TimelineTone endpoint was rate-limited
                      at fetch time

    Emits 7-9 daily features (varies by tone availability):
      gdelt_n_articles, gdelt_mention_rate                — raw level
      gdelt_n_log, gdelt_mention_log                      — log(1+x)
      gdelt_n_z_30d, gdelt_mention_z_30d                  — 30d z-score
      gdelt_n_chg_3d                                       — 3d log-diff
      gdelt_tone, gdelt_tone_chg_3d                       — optional
    """
    p = data_dir / "gdelt.parquet"
    if not p.exists():
        return pd.DataFrame(index=daily_idx)
    g = pd.read_parquet(p)
    g.index = pd.to_datetime(g.index, utc=True).normalize()
    if external_lag_days:
        g.index = g.index + pd.Timedelta(days=external_lag_days)

    out = pd.DataFrame(index=g.index)
    if "n_articles" in g.columns:
        n = g["n_articles"].astype(float)
        out["gdelt_n_articles"] = n
        out["gdelt_n_log"] = np.log1p(n)
        mu = n.rolling(30, min_periods=15).mean()
        sd = n.rolling(30, min_periods=15).std()
        out["gdelt_n_z_30d"] = (n - mu) / (sd + 1e-9)
        out["gdelt_n_chg_3d"] = np.log1p(n) - np.log1p(n.shift(3))
    if "mention_rate" in g.columns:
        m = g["mention_rate"].astype(float)
        out["gdelt_mention_rate"] = m
        out["gdelt_mention_log"] = np.log1p(m)
        mu = m.rolling(30, min_periods=15).mean()
        sd = m.rolling(30, min_periods=15).std()
        out["gdelt_mention_z_30d"] = (m - mu) / (sd + 1e-9)
    if "avg_tone" in g.columns and g["avg_tone"].notna().any():
        t = g["avg_tone"].astype(float)
        out["gdelt_tone"] = t
        out["gdelt_tone_chg_3d"] = t - t.shift(3)
    out = out.replace([np.inf, -np.inf], np.nan)
    return _align_to_daily(out, daily_idx)


def _sentiment_features(daily_idx: pd.DatetimeIndex, data_dir: Path,
                        external_lag_days: int = 0) -> pd.DataFrame:
    path = data_dir / "fng.parquet"
    if not path.exists():
        return pd.DataFrame(index=daily_idx)
    f = pd.read_parquet(path)
    f.index = pd.to_datetime(f.index, utc=True).normalize()
    if external_lag_days:
        f.index = f.index + pd.Timedelta(days=external_lag_days)
    f["fng_ma_7d"] = f["fng"].rolling(7).mean()
    f["fng_chg_3d"] = f["fng"].diff(3)
    f["fng_extreme_low"] = (f["fng"] <= 25).astype(float)
    f["fng_extreme_high"] = (f["fng"] >= 75).astype(float)
    return _align_to_daily(f, daily_idx)


def _calendar_features(daily_idx: pd.DatetimeIndex) -> pd.DataFrame:
    halvings = pd.to_datetime([
        "2012-11-28", "2016-07-09", "2020-05-11",
        "2024-04-19", "2028-04-19",
    ], utc=True)
    df = pd.DataFrame(index=daily_idx)
    idx_naive = df.index.tz_convert("UTC")
    df["cal_dow"] = idx_naive.dayofweek.astype(float)
    df["cal_month"] = idx_naive.month.astype(float)
    days_to_next = []
    days_since_last = []
    for ts in idx_naive:
        future = halvings[halvings > ts]
        past = halvings[halvings <= ts]
        days_to_next.append((future[0] - ts).days if len(future) else np.nan)
        days_since_last.append((ts - past[-1]).days if len(past) else np.nan)
    df["cal_days_to_halving"] = days_to_next
    df["cal_days_since_halving"] = days_since_last
    return df


def _align_to_daily(df: pd.DataFrame, daily_idx: pd.DatetimeIndex) -> pd.DataFrame:
    src = df.copy()
    src.index.name = "ts"
    src = src.reset_index().sort_values("ts")
    src["ts"] = pd.to_datetime(src["ts"], utc=True).astype("datetime64[ns, UTC]")
    target_ts = pd.to_datetime(pd.DatetimeIndex(daily_idx), utc=True).astype(
        "datetime64[ns, UTC]")
    target = pd.DataFrame({"ts": target_ts}).sort_values("ts")
    out = pd.merge_asof(target, src, on="ts", direction="backward").set_index("ts")
    out.index = pd.DatetimeIndex(daily_idx, tz="UTC", name="ts")
    return out


# ---------- main builder ----------

def build_features(data_dir: Path,
                   label_threshold: float = LABEL_THRESHOLD,
                   daily_df: pd.DataFrame | None = None,
                   external_lag_days: int = 0,
                   use_mvrv_v2: bool = False,
                   use_features_v3: bool = False,
                   use_features_v4: bool = False,
                   use_features_v5: bool = False,
                   use_features_v5_2: bool = False,
                   use_features_v7: bool = False,
                   use_features_v5_3: bool = False,
                   ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if daily_df is None:
        d1 = pd.read_parquet(data_dir / "btc_1d.parquet")
    else:
        d1 = daily_df
    w1 = pd.read_parquet(data_dir / "btc_1w.parquet")
    M1 = pd.read_parquet(data_dir / "btc_1M.parquet")
    m15_path = data_dir / "btc_15m.parquet"
    m15 = pd.read_parquet(m15_path) if m15_path.exists() else None

    d_feat = _tf_features(d1, "d", lookback=180)
    w_feat = _tf_features(w1, "w", lookback=26)
    M_feat = _tf_features(M1, "M", lookback=6)

    daily_idx = d_feat.index
    base = d_feat.copy()

    # v5_2 / v7 / v5_3 are all v4-or-broader supersets:
    #   - v5_2 = looser v5 keep set, same input universe
    #   - v7   = v4 + FRED Tier-1 expansion + GDELT news features
    #   - v5_3 = pruned v7 (cum ~0.93 keep)
    use_v2_effective = (use_mvrv_v2 or use_features_v3 or use_features_v4
                        or use_features_v5 or use_features_v5_2
                        or use_features_v7 or use_features_v5_3)
    use_v3_addons = (use_features_v3 or use_features_v4 or use_features_v5
                     or use_features_v5_2 or use_features_v7
                     or use_features_v5_3)
    use_m2 = (use_features_v4 or use_features_v5 or use_features_v5_2
              or use_features_v7 or use_features_v5_3)
    use_gdelt = use_features_v7 or use_features_v5_3

    if use_v2_effective:
        v2 = _mvrv_v2_features(d1, data_dir, daily_idx, external_lag_days)
        base = base.join(v2, how="left")
    else:
        base["d_mvrv_z"] = _mvrv_z_proxy(
            d1.set_index("close_time")["close"], window=365)
        base["d_mvrv_z_chg7"] = base["d_mvrv_z"].diff(7)
    if use_v3_addons:
        v3 = _features_v3_additions(d1, data_dir, daily_idx, external_lag_days)
        base = base.join(v3, how="left")
    if use_m2:
        m2 = _m2_features(data_dir, daily_idx)
        base = base.join(m2, how="left")

    w_snap = _asof_join(daily_idx, w_feat)
    M_snap = _asof_join(daily_idx, M_feat)
    d_win = _window_summary(daily_idx, d_feat, "d", window=180)
    w_win = _window_summary(daily_idx, w_feat, "w", window=26)
    M_win = _window_summary(daily_idx, M_feat, "M", window=6)
    daily_short = _daily_short_window(d1, daily_idx)
    micro    = _microstructure_features(d1, daily_idx)

    sideways = _sideways_features(d1).reindex(daily_idx)
    macro    = _macro_features(daily_idx, data_dir, external_lag_days=external_lag_days)
    eth      = _eth_features(daily_idx, data_dir, external_lag_days=external_lag_days)
    cot      = _cot_features(data_dir, daily_idx, external_lag_days=external_lag_days)
    futures  = _futures_features(daily_idx, data_dir, external_lag_days=external_lag_days)
    intraday = (_intraday_summary(daily_idx, m15) if m15 is not None
                else pd.DataFrame(index=daily_idx))
    sentiment = _sentiment_features(daily_idx, data_dir, external_lag_days=external_lag_days)
    gdelt    = (_gdelt_features(daily_idx, data_dir,
                                external_lag_days=external_lag_days)
                if use_gdelt else pd.DataFrame(index=daily_idx))
    calendar = _calendar_features(daily_idx)

    # ETH/BTC ratio
    if "eth_close" in eth.columns and len(eth) == len(base):
        eth_vals = eth["eth_close"].to_numpy()
        btc_vals = base["d_close"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(btc_vals > 0, eth_vals / btc_vals, np.nan)
        eth["ethbtc_ratio"] = ratio
        eth["ethbtc_ret_5d"] = pd.Series(ratio, index=eth.index).pct_change(5).to_numpy()

    X = pd.concat(
        [base, w_snap, M_snap, d_win, w_win, M_win, daily_short, micro,
         sideways, macro, eth, cot, futures, intraday, sentiment, gdelt,
         calendar],
        axis=1,
    )

    close = d1.set_index("close_time")["close"]
    next_ret = close.pct_change().shift(-1)
    y = pd.Series(np.nan, index=close.index, dtype=float)
    y[next_ret >  label_threshold] = 1.0
    y[next_ret < -label_threshold] = 0.0
    y = y.reindex(X.index)

    valid_idx = X["d_w180_cum_ret"].notna()
    X = X.loc[valid_idx]
    y = y.loc[X.index]
    close = close.loc[X.index]
    X = X.loc[:, ~X.columns.duplicated()]

    if use_features_v5_3:
        if not V5_3_KEEP_COLS:
            # selector hasn't been run yet — fall back to v5_2 if available,
            # otherwise to v5 keep
            if V5_2_KEEP_COLS:
                keep = [c for c in X.columns if c in V5_2_KEEP_COLS]
            else:
                keep = [c for c in X.columns if c in V5_KEEP_COLS]
        else:
            keep = [c for c in X.columns if c in V5_3_KEEP_COLS]
        X = X[keep]
    elif use_features_v7:
        # v7 is the broadest set — keep everything that build produces.
        pass
    elif use_features_v5_2:
        if not V5_2_KEEP_COLS:
            keep = [c for c in X.columns if c in V5_KEEP_COLS]
        else:
            keep = [c for c in X.columns if c in V5_2_KEEP_COLS]
        X = X[keep]
    elif use_features_v5:
        keep = [c for c in X.columns if c in V5_KEEP_COLS]
        X = X[keep]
    elif use_features_v4:
        X = X.drop(columns=list(V4_DROP_COLS), errors="ignore")
    elif use_features_v3:
        X = X.drop(columns=list(V3_DROP_COLS), errors="ignore")
    return X, y, close


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    X, y, close = build_features(here / "data")
    n_up = int((y == 1.0).sum())
    n_dn = int((y == 0.0).sum())
    n_ig = int(y.isna().sum())
    print(f"matrix : {X.shape}")
    print(f"span   : {X.index.min().date()} -> {X.index.max().date()}")
    print(f"labels : UP={n_up}  DOWN={n_dn}  IGNORE={n_ig}")
    print(f"\n{X.shape[1]} features:")
    for c in X.columns:
        nan_pct = X[c].isna().mean()
        print(f"  {c:<30} nan={nan_pct:>5.1%}")
