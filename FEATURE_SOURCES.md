# Feature ↔ Data Source 매핑

`src/features.py::build_features()` 가 생성하는 모든 feature와 그 출처. 데이터
파일(`data/*.parquet`)을 만드는 fetcher 모듈도 함께 표기.

## 데이터 source 요약

| Parquet 파일 | Fetcher 모듈 | 외부 source | 비고 |
|---|---|---|---|
| `btc_1d.parquet`, `btc_1w.parquet`, `btc_1M.parquet`, `btc_15m.parquet` | `src/binance_fetcher.py` | **Binance BTCUSDT** klines (`fapi`/`api`) | 사용자 confirmed 사용 허용. 옛 `data_fetcher.py` (Bitstamp)는 retired, raw는 `data/_bitstamp_backup/`에 보존 |
| `eth_1d.parquet` | `src/binance_fetcher.py` | **Binance ETHUSDT** daily klines | 동일 |
| `macro_fred.parquet` | `src/fred_fetcher.py` | **FRED** (자유이용 series만, 21 컬럼) | v0~v6 = 6 core, v7 expansion = +15 series |
| `m2.parquet` | `src/fred_fetcher.py` | **FRED M2SL** | v4 이상에서 사용 |
| `cot.parquet` | `src/cftc_fetcher.py` | **CFTC TFF (CME BTC futures, contract 133741)** | U.S. government public domain |
| `funding.parquet` | `src/binance_fetcher.py` (또는 동일 보관 흐름) | **Binance perpetual funding rate** | restored from legacy pipeline |
| `basis.parquet` | `src/binance_fetcher.py` (또는 동일 보관 흐름) | **Binance futures-spot basis** | restored from legacy pipeline |
| `fng.parquet` | `src/fng_fetcher.py` | **alternative.me** Fear & Greed | free public API |
| `gdelt.parquet` | `src/gdelt_fetcher.py` | **GDELT 2.0 TimelineRaw + TimelineTone** | v7 / v5_3 only |
| `coinmetrics.parquet` | `src/coinmetrics_fetcher.py` | **Coin Metrics community API** (HashRate / AdrActCnt / TxCnt / CapMrktCurUSD / PriceUSD / SplyCur) | 현재 학습/inference 양쪽 모두 사용. Bitcoin Core RPC + block scan으로의 swap은 `W:\onchain` IBD 완료 후 `src/live_inference_btc_core.py` 가 같은 스키마로 덮어쓰는 방식. 코드 수정 0줄. |
| `cg_btc_market.parquet`, `cg_eth_market.parquet`, `cg_global.parquet` | `src/coingecko_fetcher.py` | **CoinGecko free Demo** (최근 365일) | 라이브 inference 보조 / 가격 cross-check |

> 제외된 source (third-party copyright / 라이센스 미해결): SPX (S&P), VIX
> (CBOE), Gold (LBMA/IBA), yfinance.

---

## Feature 카테고리별 출처

### 1. Daily timeframe TA (9 features)
**출처: Binance BTCUSDT daily klines (`btc_1d.parquet`)**

`d_close`, `d_ret_1`, `d_rsi14`, `d_macd_hist`, `d_bb_pos`, `d_ma_ratio`,
`d_vol`, `d_range_pct`, `d_vol_z`

### 2. Weekly timeframe TA (9 features)
**출처: Binance daily resampled to W-MON (`btc_1w.parquet`)**

`w_close`, `w_ret_1`, `w_rsi14`, `w_macd_hist`, `w_bb_pos`, `w_ma_ratio`,
`w_vol`, `w_range_pct`, `w_vol_z`

### 3. Monthly timeframe TA (9 features)
**출처: Binance daily resampled to MS (`btc_1M.parquet`)**

`M_close`, `M_ret_1`, `M_rsi14`, `M_macd_hist`, `M_bb_pos`, `M_ma_ratio`,
`M_vol`, `M_range_pct`, `M_vol_z`

### 4. Window-summary statistics (9 features)
**출처: Binance BTC daily/weekly/monthly**

| feature | 윈도우 | 출처 |
|---|---|---|
| `d_w180_slope`, `d_w180_cum_ret`, `d_w180_dd` | 180-day | Binance daily |
| `w_w26_slope`, `w_w26_cum_ret`, `w_w26_dd` | 26-week | Binance weekly |
| `M_w6_slope`, `M_w6_cum_ret`, `M_w6_dd` | 6-month | Binance monthly |

### 5. Daily short-window 대체 (2 features)
**출처: Binance BTC daily (`btc_1d.parquet`)**

| feature | 의미 | 비고 |
|---|---|---|
| `d_5d_dd` | 5-day rolling drawdown | 기존 `m15_5d_dd`를 daily 5-bar 근사로 대체. 학습 robust |
| `d_5d_vol` | 5-day log-return std | 기존 `m15_5d_vol` 대체 |

> 15분봉 기반 m15_5d_*는 daily에서도 동일 정보를 얻을 수 있고 fetcher
> 의존성을 줄이므로 daily 근사를 우선 사용. 15분봉 자체는 `_intraday_summary`
> 에서 별도 6개 feature로 활용된다 (섹션 17 참고).

### 6. Microstructure (7 features)
**출처: Binance BTC daily (`btc_1d.parquet`)**

`d_rv_30d`, `d_rv_60d`, `d_ret_acorr_5d`, `d_ret_acorr_20d`,
`d_ret_skew_30d`, `d_ret_kurt_30d`, `d_up_frac_20d`, `d_vol_surge_5d`

> Close- and volume-only proxies that complement the Binance funding/basis
> microstructure signal. Realized-vol regime + momentum/mean-reversion shape.

### 7. MVRV-Z proxies (v0 baseline, 2 features)
**출처: Binance BTC daily**

`d_mvrv_z` (365d MA z-score), `d_mvrv_z_chg7`

### 8. MVRV-Z v2 (v3+ all branches, 6 features)
**출처: 4 price-based + 2 on-chain**

| feature | 출처 | 비고 |
|---|---|---|
| `mvrv2_z_200w` | Binance BTC daily | 1400d MA z-score |
| `mvrv2_z_ema900` | Binance BTC daily | halflife=900d EMA |
| `mvrv2_z_multi` | Binance BTC daily | weighted blend MA90/365/1000/1500 |
| `mvrv2_mayer` | Binance BTC daily | Mayer Multiple |
| `mvrv2_nvt_z` | `coinmetrics.parquet` → (planned: Bitcoin Core) | MarketCap / AdrActCnt 비율의 180d z |
| `mvrv2_hashrate_z` | `coinmetrics.parquet` → (planned: Bitcoin Core) | 30d HashRate z |

### 9. v3 add-ons (모든 v3+ 분기)
| feature | 출처 | 비고 |
|---|---|---|
| `rho30_btc_dxy` | Binance BTC + **FRED DTWEXBGS** | 30d log-return rolling correlation |
| `rho30_btc_eth` | Binance BTC + **Binance ETH** | 동일 |
| `cot_levmoney_velocity` | **CFTC TFF** | leveraged_long 1주 diff / 8주 std |
| `hashrate_ribbon` | `coinmetrics.parquet` → (planned: Bitcoin Core) | 30d/60d HashRate MA 비율 |
| `hashrate_chg_7d` | `coinmetrics.parquet` → (planned: Bitcoin Core) | 7d HashRate 변화율 |
| `adr_chg_7d` | `coinmetrics.parquet` → (planned: Bitcoin Core) | 7d AdrActCnt 변화율 |
| `tx_per_adr` | `coinmetrics.parquet` → (planned: Bitcoin Core) | TxCnt / AdrActCnt |
| `vol_of_vol_30d` | Binance BTC daily | 14d log-return std → 30d std |
| `vol_regime_z` | Binance BTC daily | daily_vol의 365d rolling z |

> 제거: `cot_top4_skew_4w` — `conc_gross_le_4_tdr_*` 컬럼이 CME BTC TFF 피드
> 에서 100% NaN이라 신호가 없음.

### 10. Sideways / regime (11 features)
**출처: Binance BTC daily (close/high/low)**

`sw_chop_er_5d`, `sw_chop_er_14d`, `sw_chop_er_30d`, `sw_chop_er_60d`,
`sw_ratio_14d`, `sw_ratio_30d`, `sw_ci_14d`, `sw_ci_30d`,
`sw_bb_width_20d`, `sw_range_pct_30d`, `sw_streak_med`

### 11. Macro (FRED daily series)
**출처: `macro_fred.parquet`**

각 base series당 5개 derived (raw level + ret_1d/ret_5d/vol_20d/z_60d).

**v0~v6 core (6 base × 5 = 30 features)**

| base | FRED ID | 의미 |
|---|---|---|
| `dxy` | DTWEXBGS | Trade Weighted U.S. Dollar Index: Broad |
| `tnx` | DGS10 | 10-Year Treasury Constant Maturity Rate |
| `t10y2y` | T10Y2Y | 10Y − 2Y Treasury yield spread |
| `t10yie` | T10YIE | 10-Year Breakeven Inflation Rate |
| `indpro` | INDPRO | Industrial Production: Total Index (monthly) |
| `unrate` | UNRATE | Civilian Unemployment Rate (monthly) |

**v7 Tier-1 expansion (+15 base × 5 = +75 features)**

| base | FRED ID | 분류 |
|---|---|---|
| `tnx2y` / `tnx5y` / `tnx30y` | DGS2 / DGS5 / DGS30 | Treasury yield curve |
| `tips5y` / `tips10y` | DFII5 / DFII10 | TIPS real yields |
| `fedfunds` / `sofr` | DFF / SOFR | Policy / overnight rates |
| `fed_assets` / `fed_reserves` | WALCL / WRESBAL | Fed balance sheet |
| `rrp_overnight` | RRPONTSYD | ON RRP take-up (liquidity sink) |
| `fx_krw` / `fx_jpy` / `fx_cny` | DEXKOUS / DEXJPUS / DEXCHUS | Cross-rates |
| `jobless_initial` / `payems` | ICSA / PAYEMS | Labour leading indicators |

> 월간 시리즈(`indpro` / `unrate` / `payems`)는 발표 지연이 커서 next-day BTC
> 시그널과 align하지 않을 수 있지만 v7에서는 macro envelope을 넓혀 모델이
> 스스로 무시할지 학습하도록 한다.

### 12. M2 (5 features, v4 이상)
**출처: FRED M2SL (`m2.parquet`), 21-day publish lag 적용**

`m2_chg_1d`, `m2_chg_1w`, `m2_chg_1m`, `m2_chg_3m`, `m2_z_24m`

### 13. ETH features (5 features)
**출처: Binance ETHUSDT daily (`eth_1d.parquet`)**

`eth_close`, `eth_close_ret_1d`, `eth_close_ret_5d`, `eth_close_vol_20d`,
`eth_close_z_60d`

### 14. ETH/BTC ratio (2 features)
**출처: Binance ETHUSDT / Binance BTCUSDT**

`ethbtc_ratio`, `ethbtc_ret_5d`

### 15. CFTC TFF positioning (8 features)
**출처: CFTC TFF report — CME BTC futures (contract code 133741)**
**Lag: 4-day (Tue position → 안전하게 Saturday 이후 사용)**

| feature | 의미 |
|---|---|
| `cot_levmoney_net_long_ratio` | (leveraged long − short) / open_interest |
| `cot_levmoney_net_long_chg_1w` | 위 1주 변화 |
| `cot_dealer_net_long_ratio` | (dealer long − short) / OI |
| `cot_money_manager_net_long_ratio` | (asset manager long − short) / OI |
| `cot_levmoney_z_52w` | leveraged net의 1년 z-score |
| `cot_ma_4w` | leveraged net의 4-week MA |
| `cot_levmoney_velocity` | leveraged_long 1주 diff / 8주 std (v3+ add-on) |
| `cot_top4_long_pct` | top-4 long concentration (사용되는 변형에서) |

### 16. Binance futures microstructure (8 features)
**출처: `funding.parquet` + `basis.parquet`**

| feature | 의미 |
|---|---|
| `funding_mean`, `funding_max`, `funding_min` | raw 일별 aggregates |
| `funding_ma_7d`, `funding_z_30d` | smoothed level + std z |
| `basis` | raw 일별 basis |
| `basis_ma_5d`, `basis_z_30d` | smoothed level + std z |

> CFTC TFF와 *공존*. TFF는 large-trader 디렉셔널 포지셔닝, funding/basis는
> derivatives crowding pressure — 직교 신호로 본다.

### 17. 15-minute intraday summary (6 features)
**출처: Binance BTCUSDT 15-minute klines (`btc_15m.parquet`)**
**Join: asof-backward (현재 일자 미완성 봉은 leakage 방지로 제외)**

`m15_24h_ret`, `m15_24h_vol`, `m15_24h_rsi`, `m15_5d_vol`, `m15_5d_dd`,
`m15_24h_slope`

### 18. Fear & Greed (5 features)
**출처: alternative.me F&G index (`fng.parquet`)**

`fng`, `fng_ma_7d`, `fng_chg_3d`, `fng_extreme_low`, `fng_extreme_high`

### 19. GDELT news features (7~9 features, v7 / v5_3 only)
**출처: `gdelt.parquet` (built by `src/gdelt_fetcher.py`)**

raw 컬럼: `n_articles`, `total_monitored`, `mention_rate`, `avg_tone`.
파생 features:

| feature | 의미 |
|---|---|
| `gdelt_n_articles`, `gdelt_n_log` | raw level + log(1+x) |
| `gdelt_n_z_30d`, `gdelt_n_chg_3d` | 30d z-score, 3d log-diff |
| `gdelt_mention_rate`, `gdelt_mention_log` | 동일 변환 |
| `gdelt_mention_z_30d` | 30d z-score |
| `gdelt_tone`, `gdelt_tone_chg_3d` | (옵션) tone이 fetch 시 rate-limit으로 NaN인 경우 생략 |

### 20. Calendar (4 features)
**출처: Binance BTC daily의 `close_time` index만 사용**

`cal_dow`, `cal_month`, `cal_days_to_halving`, `cal_days_since_halving`

---

## On-chain features의 라이브 swap 절차

`coinmetrics.parquet` (Coin Metrics community API 캐시)는 *현재* 학습/추론
양쪽에서 그대로 사용된다. 라이브 inference 시점에 `W:\onchain` Bitcoin Core
IBD가 완료되면 `src/live_inference_btc_core.py::_refresh_onchain()` 가 다음을
수행해 같은 파일을 덮어쓰는 방식으로 source를 swap한다:

1. `src/btc_core_rpc.py::snapshot()` — Bitcoin Core RPC로
   - `HashRate` ← `getnetworkhashps(2016)`
   - `Supply` ← `gettxoutsetinfo().total_amount`

2. `src/btc_core_chain_scan.py::run()` — 블록 스캔 (multi-process)
   - `TxCnt` ← 블록당 tx 수 일별 합
   - `AdrActCnt` ← unique scriptPubKey 주소 수

3. 위 두 parquet을 join, `CapMrktCurUSD = Supply × CoinGecko price`로
   채우고, 동일 스키마(`HashRate, AdrActCnt, TxCnt, CapMrktCurUSD`)로
   `coinmetrics.parquet` 덮어쓰기.

→ `features.py`는 컬럼명만 읽으므로 코드 수정 0줄로 source 전환 가능.
W:\onchain bitcoind IBD가 완료(verificationprogress ≈ 1.0)되어야 작동.

---

## 라이브 inference 시점의 source 호출 매트릭스

`bit-on-wave` 라이브 서비스가 매일 호출하는 외부 API:

| 호출 endpoint | 빈도 | source | 갱신 대상 |
|---|---|---|---|
| Binance `/api/v3/klines?symbol=BTCUSDT&interval=1d` (+ 15m/1w/1M) | 매일 | Binance | BTC OHLCV |
| Binance `/api/v3/klines?symbol=ETHUSDT&interval=1d` | 매일 | Binance | ETH daily |
| Binance `/fapi/v1/fundingRate` + 선물/현물 basis 계산 | 매일 | Binance | funding/basis |
| `https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?days=365` | 매일 (보조) | CoinGecko free Demo | BTC daily price/volume cross-check |
| `https://api.coingecko.com/api/v3/coins/ethereum/market_chart?days=365` | 매일 (보조) | CoinGecko free Demo | ETH cross-check |
| `https://api.coingecko.com/api/v3/global` | 매일 | CoinGecko free Demo | BTC dominance / total market cap |
| `https://fred.stlouisfed.org/graph/fredgraph.csv?id=...` | 매일 | FRED | 21 macro series |
| `https://publicreporting.cftc.gov/resource/gpe5-46if.json?cftc_contract_market_code=133741` | 매주 (Friday 발표) | CFTC SODA API | TFF positioning |
| `https://api.alternative.me/fng/?limit=0` | 매일 | alternative.me | F&G |
| GDELT 2.0 TimelineRaw + TimelineTone (`api.gdeltproject.org`) | 매일 | GDELT | v7 news features |
| Bitcoin Core RPC `getblockchaininfo` / `getnetworkhashps` / `getblock` | 매일 (IBD 후) | 본인 `W:\onchain` 노드 | 6 on-chain features (현재는 Coin Metrics) |

---

## 제외된 feature (commercial-clean 정책으로 영구 제거)

기존 coin_analysis (`../coin_analysis/FEATURE_VERSIONS.md`) 에서 사용했지만
본 버전에서 제거된 feature들:

### SPX (S&P Dow Jones Indices copyright) — 5개 제거
`spx_close`, `spx_close_ret_1d/5d`, `spx_close_vol_20d`, `spx_close_z_60d`,
`rho30_btc_spx`

### VIX (CBOE copyright) — 5개 제거
`vix_close`, `vix_close_ret_1d/5d`, `vix_close_vol_20d`, `vix_close_z_60d`

### Gold (LBMA / IBA, FRED에서도 2022-01-31 제거됨) — 5개 제거
`gold_close`, `gold_close_ret_1d/5d`, `gold_close_vol_20d`,
`gold_close_z_60d`, `rho30_btc_gold`

### CFTC top-4 concentration (1개 제거)
`cot_top4_skew_4w` — `conc_gross_le_4_tdr_*` inputs가 CME BTC TFF에서 100% NaN.

총 영구 제거: **16개** (이전 commercial-license-only 시기엔 추가로 Binance
funding/basis도 제거됐으나 사용자 confirmation 후 복원됐다.)

---

## 변경 로그

| 날짜 | 변경 |
|---|---|
| 2026-05-11 | 초기 작성. Bitstamp + FRED + CFTC + alternative.me + Bitcoin Core(라이브) 매핑. SPX/VIX/Gold/Binance perp/m15 영구 제거. |
| 2026-05-12 | Bitstamp → Binance(BTCUSDT/ETHUSDT)로 OHLCV fetcher 교체. 옛 Bitstamp parquet은 `data/_bitstamp_backup/`에 보존. funding/basis 복원. |
| 2026-05-12 | FRED 21 series로 Tier-1 확장(v7). GDELT 뉴스 attention/tone feature 추가(v7/v5_3). v5_2 (looser keep) / v5_3 (v7 prune) variant 추가. |
| 2026-05-13 | 본 문서를 현재 코드와 동기화 — Bitstamp 표기 모두 Binance로 교체, v7/v5_3, microstructure 8 features, intraday 6 features, GDELT 섹션 추가. |
