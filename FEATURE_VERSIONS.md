# Feature 버전 정리 — coin_prediction_revised (v0 / v3 / v4 / v5 / v5_2 / v6 / v7 / v5_3)

`src/features.py::build_features()` 가 여덟 가지 모드로 feature matrix `X`를
생성한다. 모드는 `VariantConfig`의 플래그 (`use_features_v3` / `_v4` / `_v5` /
`_v5_2` / `_v7` / `_v5_3`)로 선택한다.

> 원본 정의(coin_analysis)는 `../coin_analysis/FEATURE_VERSIONS.md` 참고.

## Source 매핑 (실제 현재 fetcher 기준, 2026-05-13)

| 분류 | 기존 (coin_analysis) | 현재 (coin_prediction_revised) |
|---|---|---|
| BTC OHLCV (1d/1w/1M/15m) | Binance + Bitstamp | **Binance BTCUSDT** (`binance_fetcher.py`). Bitstamp는 retired (`data/_bitstamp_backup/`에 보존) |
| ETH-USD | yfinance ETH-USD | **Binance ETHUSDT** (`binance_fetcher.py`) |
| Funding rate | Binance fapi | **Binance fapi** (복원됨, 사용자 confirmed) |
| Perp basis | Binance fapi + spot | **Binance** (복원됨) |
| 15m intraday | Binance + Bitstamp 15m | **Binance 15m → 6 daily summary features** (`_intraday_summary`) + daily 5-bar rolling 2개 (`d_5d_dd`, `d_5d_vol`) |
| DXY | yfinance DX-Y.NYB | **FRED DTWEXBGS** |
| 10Y yield | yfinance ^TNX | **FRED DGS10** |
| M2 | FRED M2SL | (그대로) |
| Yield curve 2Y/5Y/30Y | — | **FRED DGS2/DGS5/DGS30** (v7 신규) |
| TIPS real yields | — | **FRED DFII5/DFII10** (v7 신규) |
| Policy rates | — | **FRED DFF (fedfunds), SOFR** (v7 신규) |
| Fed balance sheet | — | **FRED WALCL/WRESBAL, RRPONTSYD** (v7 신규) |
| FX cross-rates | — | **FRED DEXKOUS/DEXJPUS/DEXCHUS** (v7 신규) |
| Labour | — | **FRED ICSA (jobless), PAYEMS** (v7 신규) |
| Inflation expectations | — | **FRED T10YIE** (v3+ 신규) |
| Yield spread 10Y2Y | — | **FRED T10Y2Y** (v3+ 신규) |
| Industrial Production | — | **FRED INDPRO** (v3+ 신규) |
| Unemployment | — | **FRED UNRATE** (v3+ 신규) |
| SPX / VIX / Gold | yfinance ^GSPC / ^VIX / GC=F | **제외** (S&P / CBOE / LBMA-IBA copyright) |
| Futures positioning | — | **CFTC TFF (CME BTC, contract 133741)** (v3+ 신규, `cftc_fetcher.py`) |
| Fear & Greed | alternative.me | (그대로, `fng_fetcher.py`) |
| News attention / tone | — | **GDELT 2.0** (v7 / v5_3 신규, `gdelt_fetcher.py`) |
| HashRate / AdrActCnt / TxCnt / MarketCap | Coin Metrics community | **Coin Metrics community (`coinmetrics_fetcher.py`)** — 현재 학습/추론 양쪽에서 사용. `W:\onchain` Bitcoin Core IBD 완료 후 `live_inference_btc_core.py`가 같은 파일을 RPC + 블록 스캔 결과로 덮어쓰는 swap이 계획됨 (코드 수정 0줄) |

## 버전별 feature 분기

`features.py::build_features()` 의 동작 차이만 정리:

| 버전 | use_features_* 플래그 | 입력 superset | 후처리 |
|---|---|---|---|
| **v0** | 모두 False | 기본 + `d_mvrv_z`/`d_mvrv_z_chg7` (price-only MVRV proxy) | 없음 |
| **v3** | `_v3=True` | v0 superset + MVRV-Z v2 (6) + v3 add-ons (cross-asset rho, CFTC velocity, on-chain ribbon, vol regime) | `V3_DROP_COLS` 4개 제거 (`dxy_z_60d`, `tnx_z_60d`, `d_close`, `sw_chop_er_5d/60d`, `cal_dow`) |
| **v4** | `_v4=True` | v3 superset + M2 5 features | `V4_DROP_COLS` 3개 제거 (`d_close`, `sw_chop_er_5d/60d`, `cal_dow`) |
| **v5** | `_v5=True` | v4 superset | `V5_KEEP_COLS` 만 유지 (cum ~80%, aggressive prune) |
| **v5_2** | `_v5_2=True` | v4 superset | `V5_2_KEEP_COLS` 만 유지 (cum ~92%, looser prune). fallback = V5_KEEP_COLS |
| **v6** | `_v4=True` + `wfconfig_overrides` (XGB-style hyperparams) | v4와 동일 입력 | drop 동일, **모델 하이퍼파라미터만 tighter** (interactions/leaves/etc.) |
| **v7** | `_v7=True` | v4 superset + **FRED Tier-1 확장 (15 series × 5 derived)** + **GDELT (7~9 features)** | drop 없음 (가장 넓은 feature 세트) |
| **v5_3** | `_v5_3=True` | v7 superset (입력 동일) | `V5_3_KEEP_COLS` 만 유지 (cum ~93% of v7). fallback chain: V5_3 → V5_2 → V5 |

> 버전별 feature 수는 sweep 마지막에 `X.shape[1]`로 정확히 확정된다. 위
> 분기를 사람이 읽고 만든 추정치는 부정확할 수 있어 의도적으로 표에서 제거.

## V5_KEEP / V5_2_KEEP / V5_3_KEEP 생성 절차

세 keep 리스트 모두 `analysis/v5_keep_selector_ebm.py` 가 생성:

```
uv run python analysis/v5_keep_selector_ebm.py \
  --cumulative 0.93 \
  --output-file src/v5_3_keep.py \
  --var-name V5_3_KEEP_COLS
```

`--cumulative` 값 (0.80 / 0.92 / 0.93)이 V5 / V5_2 / V5_3을 결정. 입력은
`data/preds_cache_ebm/fi_ebm_*.parquet` 의 importance ranking (각 모델별
EBM main-effect importance를 모든 cell에서 평균).

각 keep 모듈이 누락된 경우 features.py의 fallback이 동작:
- v5_3_keep 없음 → V5_2_KEEP 사용 → 없으면 V5_KEEP 사용
- v5_2_keep 없음 → V5_KEEP 사용
- v5_keep 없음 → features.py 안의 하드코딩 fallback tuple 사용

## 변종 명명 규칙

```
{decision_time}_{cadence}_{feature_version}_{nan_policy}
```

- `decision_time`: `utc2130` | `utc0000`
- `cadence`: 생략 = M (월별 refit) | `sm` (semi-monthly, 1일 + 16일 refit) |
  `wsun` (weekly Sunday, 후순위)
- `feature_version`: `v0` | `v3` | `v4` | `v5` | `v5_2` | `v6` | `v7` | `v5_3`
- `nan_policy`: 생략 = raw | `close` (offday flag 컬럼 추가) | `complete`
  (ffill + bfill)

## Sweep 범위

- 8 versions × 2 cadences (M, sm) × 3 nan policies = **48 variants**
- `runners/run_all_M_sm.py --only-version <v>` 가 한 version 6 variants를
  한 번에 사용 (`--throttle N --threads M`로 동시성 조정)
- W-SUN cadence 9 variants는 후순위로 보관, 본 sweep에서는 제외

## 변경 로그

| 날짜 | 변경 |
|---|---|
| 2026-05-11 | 초기 작성. v0/v3/v4/v5/v6 정의. Bitstamp + FRED + CFTC + alternative.me + Bitcoin Core(라이브) 매핑. |
| 2026-05-12 | Bitstamp → Binance(BTCUSDT/ETHUSDT) OHLCV fetcher 교체. funding/basis 복원. v5_2 (looser keep) 신설. |
| 2026-05-12 | FRED Tier-1 확장(15 series), GDELT 뉴스 features 추가. v7 (가장 넓은 입력) + v5_3 (v7 prune) 변형 신설. |
| 2026-05-13 | 본 문서를 현재 코드와 동기화 — 모든 OHLCV 소스를 Binance로 표기, v5_2/v7/v5_3 정식 추가, Bitcoin Core swap의 status 명확화. |
