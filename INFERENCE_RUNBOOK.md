# INFERENCE_RUNBOOK — 일일 추론 운영 가이드

학습된 EBM/XGB 모델로 매일 BTC 다음날 방향 확률(`prob_up`)을 산출하고
buy/sell/hold 결정을 내리는 서비스 운영자(또는 운영용 Claude Code)가 보는
가이드입니다. **학습 파이프라인은 절대 손대지 말고**, 본 문서의 흐름만
따라가면 됩니다.

서비스 코드는 `src/live_inference_btc_core.py`에 이미 있고 (`_refresh_onchain()`
+ `predict_for_today()`), 본 문서는 그 코드를 *왜 그렇게 동작하는지*와
*하루치 데이터를 어떻게 준비해야 leakage/lag 없이 예측이 정확한지*를
설명합니다.

---

## 1. 서비스 일일 사이클 (단일 cron 작업)

UTC 기준 매일 한 번 실행. 권장 시각: **UTC 00:10**(전날 daily bar
close_time `23:59:59.999 UTC` 직후 + 10분 안전 마진). 한국시간 09:10.

```
UTC 00:00:00.000  ← 전날 daily bar close
UTC 00:10:00      ← 일일 cron 시작 (안전 마진)
  ├── 0) Bitcoin Core IBD 상태 점검
  ├── 1) 일일 raw 데이터 fetch (Binance, FRED, CFTC, F&G, GDELT, CoinGecko)
  ├── 2) 온체인 데이터 자체 생성 (W:\onchain → coinmetrics.parquet 덮어쓰기)
  ├── 3) build_features() — 학습과 동일 파이프라인
  ├── 4) 모델 pickle load + X.iloc[[-1]].predict_proba → prob_up
  ├── 5) decision (prob_up vs threshold) → buy/sell/hold
  ├── 6) 결과 저장 (DB, parquet, 알림)
  └── 7) sanity check + 알림
```

모든 단계가 **UTC** 시각계로 동작합니다. KST/EST로의 변환은 사용자
표시(UI)에서만, 내부 계산엔 절대 사용하지 않습니다.

---

## 2. 사용할 모델 — sweep 종료 후 확정

sweep(v0~v7 EBM/XGB)이 끝나면 다음 산출물 기준으로 운영 모델을 선정합니다:

- `reports/utc2130/robustness_*_ebm.csv` (EBM headline / quarterly / PBO)
- `reports/xgb/utc2130/robustness_*_xgb.csv` (XGB 쪽 동일 분석을 새로 산출)
- `reports/utc2130/aggregate_*_730d_with_365d_top*.csv`

**선정 기준 (강한 필터부터)**:

| 지표 | 기준 | 의미 |
|---|---|---|
| PBO | `< 0.35` | backtest overfit 확률 낮음 |
| 365d return | `> 0%` | 최근 1년에 살아남음 |
| PSR | `> 0.95` | per-period 통계적 유의성 |
| 분기별 outperform | `>= 4/8` | regime-robust |
| MaxDD | `> -45%` | 운영 가능한 손실폭 |
| n_trades | `>= 20` | 통계가 의미 있을 정도의 거래 빈도 |
| DSR | `> 0.10` | trial inflation 보정 후에도 양의 신호 |

복수 모델을 **앙상블**로 운영하려면 `reports/utc2130/robustness_ensemble_*.csv`
참조. 단일 모델은 위 기준의 top 3-5 중 선택.

**선정된 모델은 `data/live_models_ebm/<variant>.pkl` 또는
`data/live_models_xgb/<variant>.pkl`에 pickle로 저장.** 저장 책임은 학습
파이프라인 쪽 (sweep 끝난 후 마지막 refit 모델을 pickle하는 단계가 필요).

---

## 3. 데이터 source 매트릭스 — 학습 vs 추론

각 raw parquet의 origin이 학습 단계와 운영 단계에서 다른지 같은지 명확화:

| 파일 | 학습 단계 source | 운영 단계 source | swap 시점 |
|---|---|---|---|
| `btc_1d/1w/1M/15m.parquet` | Binance BTCUSDT klines | **동일 (Binance)** | 없음 |
| `eth_1d.parquet` | Binance ETHUSDT klines | **동일 (Binance)** | 없음 |
| `macro_fred.parquet` | FRED (21 series) | **동일 (FRED)** | 없음 |
| `m2.parquet` | FRED M2SL | **동일 (FRED)** | 없음 |
| `cot.parquet` | CFTC TFF SODA API | **동일** | 없음 |
| `funding.parquet` / `basis.parquet` | Binance perp/futures | **동일** | 없음 |
| `fng.parquet` | alternative.me | **동일** | 없음 |
| `gdelt.parquet` | GDELT 2.0 | **동일** | 없음 |
| `coinmetrics.parquet` (HashRate / AdrActCnt / TxCnt / CapMrktCurUSD) | Coin Metrics community API | **🔁 `W:\onchain` Bitcoin Core RPC + block scan으로 자체 생성** | `_refresh_onchain()` 호출 |
| `cg_btc_market.parquet` / `cg_eth_market.parquet` / `cg_global.parquet` | CoinGecko free Demo | **동일 (CoinGecko)** | 없음 (보조용) |

→ 운영 단계에서 **유일한 source 전환은 on-chain feature**입니다. 시장
데이터/매크로/positioning/sentiment는 학습 때 쓰던 외부 API를 그대로 호출.

---

## 4. ⚠️ On-chain swap — 가장 중요한 운영 차이점

**원칙**: 학습 시 사용한 Coin Metrics community 데이터는 운영 서비스에
재배포하기 어렵고, 또 사용자가 자체 노드(`W:\onchain`)에서 직접 산출하는
파이프라인을 가지고 있으므로 **운영 시에는 자체 노드 데이터로 매일
`coinmetrics.parquet`을 덮어씌웁니다**.

`src/live_inference_btc_core.py::_refresh_onchain()`이 하루 한 번 수행하는 절차:

### 4.1 사전조건 — Bitcoin Core IBD 완료 확인

```python
from src.btc_core_rpc import ibd_done
if not ibd_done():
    # verificationprogress < 1.0 — 아직 sync 중
    # 운영은 그래도 진행하되, on-chain 4개 컬럼은 직전 학습용 coinmetrics
    # cache로 fallback. 모델은 NaN을 처리할 수 있으므로 동작은 함.
```

현재 IBD 진행률은 약 63% (2026-05-13 기준). 운영 개시 전 100% 도달
필수. 다음 명령으로 진행률 확인:
```
bitcoin-cli -conf=W:\onchain\bitcoind\bitcoin.conf getblockchaininfo \
  | grep verificationprogress
```

### 4.2 4개 컬럼 산출 방법

| 컬럼 | 산출 함수 | 호출 |
|---|---|---|
| `HashRate` | `src/btc_core_rpc.py::snapshot()` | RPC `getnetworkhashps(2016)` — 2016 블록(~14일) 평균 |
| `Supply` (→ `CapMrktCurUSD`로 변환) | 동 | RPC `gettxoutsetinfo().total_amount` |
| `TxCnt` | `src/btc_core_chain_scan.py::run()` | `getblock verbosity=1`로 블록당 tx 수 집계 |
| `AdrActCnt` | 동 | 블록 스캔, unique scriptPubKey 주소 카운트 (multi-process) |
| `CapMrktCurUSD` | features.py 내부 계산 | `Supply × CoinGecko price` (위 4개 산출 후 features.py가 합침) |

### 4.3 `coinmetrics.parquet` 덮어쓰기 정책

`_refresh_onchain()`은 두 산출 결과를 join한 unified 프레임을 **학습 시와
동일한 스키마**로 `data/coinmetrics.parquet`에 저장합니다:

```
columns: ["HashRate", "AdrActCnt", "TxCnt", "CapMrktCurUSD"]
index  : pd.DatetimeIndex (UTC, daily, normalize=midnight)
```

→ features.py는 컬럼명만 보고 읽기 때문에 **코드 변경 0줄로 source가
바뀝니다**. 다만 *학습/추론 distribution shift* 가능성을 다음 절차로
검증한 뒤에만 production 전환:

### 4.4 학습 ↔ 자체 노드 데이터 일치성 검증 (전환 전 1회)

학습 데이터 = Coin Metrics, 추론 데이터 = 본인 노드. **계산 방법이 미묘하게
달라서 distribution shift가 생기면 모델이 잘못된 prob_up을 내뱉습니다.**
전환 전 다음을 *반드시* 점검:

```python
import pandas as pd
cm = pd.read_parquet("data/_archive_coinmetrics_backup.parquet")  # 학습 직전 백업
own = pd.read_parquet("data/coinmetrics.parquet")                 # 본인 노드 산출
overlap = cm.index.intersection(own.index)
for col in ["HashRate", "AdrActCnt", "TxCnt", "CapMrktCurUSD"]:
    a, b = cm.loc[overlap, col], own.loc[overlap, col]
    corr = a.corr(b)
    ratio = (b / a).median()
    print(f"{col}: corr={corr:.4f}  median(own/cm)={ratio:.3f}")
```

**합격 기준**:
- 각 컬럼 corr `> 0.95` (시계열 동조)
- median ratio `0.9 ~ 1.1` (scale 일치)

불합격 시 옵션:
- A. 본인 노드 산출 코드 보정 (가장 정공법)
- B. 학습 데이터를 본인 노드 산출로 다시 받아 sweep 재학습
- C. AdrActCnt 등 차이 큰 컬럼만 학습 keep list에서 제외 후 재학습

**현재까지 학습된 모델은 Coin Metrics 분포에 fit 됐기 때문에 검증을
건너뛰면 사실상 `bad inputs → bad probability`입니다.** 학습이 calib됐어도
input distribution shift는 calibration이 보정해주지 않습니다.

### 4.5 운영 시점에 매일 호출 흐름

```python
_refresh_onchain()        # 1) 본인 노드에서 4 컬럼 산출 → coinmetrics.parquet 덮어쓰기
                          #    IBD 미완료 시 직전 cache fallback
X, _, _ = build_features(DATA_DIR, ..., daily_df=bar, **flags)
                          # 2) features.py가 새 coinmetrics.parquet 읽고 features 생성
model = pickle.load(open("data/live_models_*/variant.pkl", "rb"))
prob_up = float(model.predict_proba(X.iloc[[-1]])[0, 1])
                          # 3) 마지막 row(=오늘 결정 시점) 한 행만 predict
```

### 4.6 4개 raw → 6개 derived feature 매핑 (학습이 실제로 보는 컬럼)

자체 노드가 산출하는 4개 raw 컬럼이 features.py에서 학습 feature로 어떻게
파생되는지. **자체 노드 산출 값이 Coin Metrics와 scale·분포 다르면 이
6개 derived 전부가 distribution shift됨**.

| derived feature | 함수 | 산출 공식 | 사용 분기 |
|---|---|---|---|
| `hashrate_ribbon` | `_features_v3_additions` | `HashRate.rolling(30).mean() / HashRate.rolling(60).mean() - 1` | v3+ |
| `hashrate_chg_7d` | `_features_v3_additions` | `HashRate.pct_change(7)` | v3+ |
| `adr_chg_7d` | `_features_v3_additions` | `AdrActCnt.pct_change(7)` | v3+ |
| `tx_per_adr` | `_features_v3_additions` | `TxCnt / (AdrActCnt + 1.0)` | v3+ |
| `mvrv2_nvt_z` | `_mvrv_v2_features` | `(CapMrktCurUSD / (AdrActCnt + 1))`의 180d z-score | v3+ MVRV v2 |
| `mvrv2_hashrate_z` | `_mvrv_v2_features` | `HashRate`의 30d z-score | v3+ MVRV v2 |

운영 함의:
- raw 4개 중 *어떤 컬럼이 NaN이면 해당 derived 전부 NaN* (체인됨)
  - `HashRate` NaN → ribbon + chg_7d + mvrv2_hashrate_z 모두 NaN (3개)
  - `AdrActCnt` NaN → adr_chg_7d + tx_per_adr + mvrv2_nvt_z (3개)
  - `TxCnt` NaN → tx_per_adr (1개)
  - `CapMrktCurUSD` NaN → mvrv2_nvt_z (1개)
- **`CapMrktCurUSD`는 자체 노드에서 직접 안 나옴**. `Supply × CoinGecko price`로 합성 (Coin Metrics는 직접 publish). CoinGecko 호출 실패 시 mvrv2_nvt_z만 NaN
- `coinmetrics.parquet`에 `PriceUSD`, `SplyCur` 컬럼이 있어도 features.py는 안 읽음 — Bitcoin Core swap 시 굳이 채울 필요 없음

### 4.7 다른 fetcher들의 raw → derived 매핑 (참고용 cheat sheet)

각 fetcher가 학습 feature를 *몇 개* 생성하는지. 운영 중 fetcher 실패 시
영향 추정에 사용.

| Fetcher | Raw 컬럼 / series 수 | Derived features (총) | 주요 prefix |
|---|---:|---:|---|
| Binance BTC 1d | 5 OHLCV | ~50 | `d_*`, `d_w180_*`, MVRV v0/v2 price 4개, sideways 11개, vol regime 2개, calendar 4개 |
| Binance BTC 1w | 5 OHLCV | 9 + 3 (window) | `w_*` |
| Binance BTC 1M | 5 OHLCV | 9 + 3 | `M_*` |
| Binance BTC 15m | 5 OHLCV | 6 | `m15_*` |
| Binance ETH 1d | 5 OHLCV | 5 + 2 + 1 | `eth_*`, `ethbtc_*`, `rho30_btc_eth` |
| FRED macro | 21 series | 21 × 5 + 1 = **106** | `<base>` / `_ret_1d` / `_ret_5d` / `_vol_20d` / `_z_60d` + `rho30_btc_dxy` |
| FRED M2 | 1 | 5 | `m2_*` |
| CFTC TFF | 7 raw | 8 | `cot_*` |
| Binance funding | 1 | 5 | `funding_*` |
| Binance basis | 1 | 3 | `basis*` |
| F&G | 1 | 5 | `fng*` |
| GDELT (v7/v5_3) | 4 | 7~9 | `gdelt_*` |
| Coin Metrics → 자체 노드 | 4 | **6** | (§4.6 참조) |

**예상 v7 X.shape[1] ≈ 215** (v2 NaN audit 결과와 일치 확인됨).

`gdelt_tone`이 rate-limit로 사라지면 X.shape[1]이 213이 됨 — train cols
fingerprint와 비교해서 reindex 필요 (§7).

---

## 5. ⚠️ "오늘" 데이터 생성 시 주의사항

다음 함정들은 학습 시점엔 자연스럽지만 *추론 시점*에 사람이 매일 다루면
실수하기 매우 쉽습니다.

### 5.1 "오늘" = 가장 최근 *완성된* daily bar

Binance daily kline의 `close_time = 23:59:59.999 UTC`. 추론은 **이미 close된
bar**까지만 사용합니다.

- ✅ UTC 00:10에 fetch한 어제(D-1) bar — close_time이 어제 23:59:59.999 → "오늘 결정용"
- ❌ UTC 12:00에 fetch한 *오늘 진행 중* bar — close_time이 아직 미래 (23:59:59.999)
  → 일부 거래소는 미완성 bar를 반환하므로 *open_time 기준 어제까지* 잘라야 함

`build_utc2130_daily()`(`src/utc2130.py`)는 이미 UTC 21:30에 자르므로 안전
하지만, 직접 `btc_1d.parquet`을 읽으면 마지막 row가 미완성 bar인지 확인:
```python
last_close_time = btc_1d["close_time"].iloc[-1]
assert (pd.Timestamp.now(tz="UTC") - last_close_time).total_seconds() > 60, \
    "last bar may be unclosed"
```

### 5.2 15-minute intraday — asof-backward join

`_intraday_summary()`는 `merge_asof(direction="backward")`로 daily timestamp에
align합니다. 이 시점에 *오늘 진행 중*인 15m bar (예: 15:30 시작, 15:45
close)가 포함되면 미래 정보가 새어 들어옵니다.

→ `btc_15m.parquet`을 fetch할 때 항상 *과거 완성된* 15m bar만 가져오게
요청. Binance API는 미완성 bar도 반환하므로 마지막 row를 검사:
```python
last = btc_15m["close_time"].iloc[-1]
if (pd.Timestamp.now(tz="UTC") - last).total_seconds() < 900:  # 15분 = 900s
    btc_15m = btc_15m.iloc[:-1]  # drop incomplete bar
```

### 5.3 외부 데이터 발표 lag — features.py가 자동 처리

`build_features(..., external_lag_days=N)`의 `external_lag_days`는 *learning
참조용*이고, 운영 시점에 추가로 손댈 필요 없습니다. features.py 내부의
fetcher별 lag 보정:

- CFTC TFF: 4-day lag (`_cot_features`, `_features_v3_additions`에서 `+max(lag, 4)일`)
- coinmetrics(=자체 노드): 1-day lag (`_features_v3_additions`, `_mvrv_v2_features`)
- macro_fred: lag 0 (FRED는 매일 18:30 UTC 무렵 갱신, 안전한 시점에 cron 실행)
- M2: `_m2_features`에서 publish 후 **21일** 자동 가산 (FRED M2SL publish lag)

운영자는 `external_lag_days=0`을 그대로 두고 fetcher만 최신 데이터로
가져오면 됩니다.

### 5.4 FRED — 주말/공휴일 NaN은 정상

FRED 시리즈는 미국 영업일만 발표됩니다. 토/일/미국 공휴일엔 새 값 없음.
`_align_to_daily()`가 `merge_asof(direction="backward")`로 직전 영업일 값을
forward-fill하므로 NaN으로 빠지지 않습니다. 단, **fetcher가 빈 응답을
받았을 때 기존 parquet의 마지막 row를 보존**해야 합니다 (덮어쓰기 ❌).

### 5.5 CFTC — 주간 publication

CFTC TFF는 **매주 금요일 약 15:30 ET (≈ 20:30 UTC) publish**합니다. 운영
서비스의 cron은 UTC 00:10이므로:

- 월~목: 직전 화요일 position이 그대로 사용 (이미 4일 이상 지남, OK)
- 금: 어제까지의 position. publish 직후라 새 row가 있을 수 있고 없을
  수도 있음 — fetcher가 *적어도 직전 주 row*는 가지고 있어야 함
- 토/일: 가장 최근 금요일 publish 값 + 4일 lag 적용 후 사용

### 5.6 GDELT — TimelineTone API rate-limit 시 NaN

`gdelt.parquet`의 `avg_tone` 컬럼은 GDELT TimelineTone API가 rate-limit
당하면 NaN이 됩니다. `_gdelt_features()`가 자동으로 `gdelt_tone` /
`gdelt_tone_chg_3d`를 생략하므로 X.shape이 달라질 수 있습니다 (7개 vs 9개).
**운영 모델의 학습 시 X.shape[1]과 항상 일치하는지 확인**(아래 §7 참조).

### 5.7 Timezone — 절대 UTC

모든 timestamp, fetcher, build_features 호출은 **UTC**. 다음은 금지:
- `tz_localize("Asia/Seoul")` / `tz_convert("EST")` 등
- `datetime.now()` (naive) — 항상 `datetime.now(timezone.utc)`
- 파일 mtime 기준 시각 — naive로 들어오면 잘못된 비교

### 5.8 마지막 row의 NaN 처리

`build_features()` 끝부분의 `valid_idx = X["d_w180_cum_ret"].notna()`는
처음 180일을 잘라냅니다. 추론 시점은 *데이터 시작점에서 충분히 떨어진
시점*이므로 일반적으로 영향 없음. 단, 새 feature를 추가할 때 lookback이
180일을 초과하면 *마지막 row*가 누락될 수 있으므로 features.py 수정 시
주의.

---

## 6. Fetcher 매일 호출 순서

순서가 중요합니다 — on-chain 산출이 가장 느리므로 가장 먼저, market
데이터는 외부 API 응답이 늦으면 retry하므로 마지막에.

```
1. Bitcoin Core IBD 상태 → ibd_done() True 확인
2. on-chain (W:\onchain) — _refresh_onchain():
   - rpc_snapshot()        ~10s
   - chain_scan_run()      수십 분 (multi-process, incremental: 마지막 처리 블록부터)
   - join → data/coinmetrics.parquet 덮어쓰기
3. binance_fetcher.py — BTCUSDT/ETHUSDT 1d/1w/1M/15m
4. fred_fetcher.py — 21 macro series + M2
5. cftc_fetcher.py — TFF positioning
6. fng_fetcher.py — F&G index
7. gdelt_fetcher.py — news attention + tone
8. coingecko_fetcher.py — 보조 가격 (선택)
9. build_features() + predict_for_today()
```

각 fetcher가 *append-only* 동작인지 *full re-fetch*인지 확인:
- Binance kline: full re-fetch 가능 (immutable), incremental도 OK
- FRED: full re-fetch (시리즈 길이 짧음, 빠름)
- CFTC: incremental (주 1회 새 row)
- GDELT: 매일 새 row 추가
- on-chain: incremental (마지막 처리 블록 height부터)

**기존 row의 값은 절대 변경되지 말 것** (immutable 데이터). 외부 데이터
provider가 historical revision을 publish하면 (예: CFTC가 과거 수정) 그건
별도 검토 후 수동 반영. 자동 덮어쓰기는 ❌.

---

## 7. 모델 호출 — X 매트릭스 alignment

가장 흔한 silent failure: 학습 시 X 컬럼과 추론 시 X 컬럼이 *조금 다름*.

### 7.1 학습 시 컬럼 fingerprint 저장

학습 sweep 끝나면 모델 pkl과 함께 X 컬럼 리스트도 저장하는 게 안전:
```python
import json
with open(f"data/live_models_ebm/{variant}.cols.json", "w") as f:
    json.dump(list(X.columns), f)
```

### 7.2 추론 시 검증

```python
X, _, _ = build_features(DATA_DIR, daily_df=bar, **flags)
with open(f"data/live_models_ebm/{variant}.cols.json") as f:
    train_cols = json.load(f)

infer_cols = list(X.columns)
missing = [c for c in train_cols if c not in infer_cols]
extra   = [c for c in infer_cols if c not in train_cols]

if missing or extra:
    # 가장 안전한 보정: train_cols 순서로 reindex, missing은 NaN
    X = X.reindex(columns=train_cols)
    log.warning(f"column mismatch  missing={missing}  extra={extra}")
```

특히 `gdelt_tone` 누락(§5.6)이 가장 흔한 mismatch 원인.

### 7.3 dtype 정합

EBM/XGB는 float64 가정. integer/object 컬럼이 섞이면 fit과 다른 binning이
일어날 수 있음. `X = X.astype(float)`로 명시 변환 권장.

---

## 8. Calibration 보존 — predict_proba 의미

학습 모델은 `cal_method='isotonic'` + `cal_cv=3`으로 fit됨. 즉 모델 객체는
`CalibratedClassifierCV(estimator=EBM/XGB, method='isotonic', cv=3)`.

→ **`predict_proba(X)[:, 1]`이 직접 calibrated probability**입니다. 예:
- `prob_up = 0.65` → "이 row에서 다음날 up일 실제 확률은 약 65%"
- threshold 0.6은 "60% 신뢰 이상이면 매수" — calibrated 의미 그대로

threshold-based decision (buy/sell/hold)이 의미를 가지려면 **이 calibration이
유지**돼야 합니다. 다음 행위가 calibration을 깹니다:

- ❌ `predict_proba` 결과에 추가 sigmoid/softmax 적용
- ❌ rank-based 비교 (e.g. quantile transform) 후 임계
- ❌ 자체 정규화 (e.g. `(p - p.min()) / (p.max() - p.min())`)
- ❌ X에 학습 때 없던 feature 추가 (calibration 분포가 깨짐)
- ✅ `prob_up >= 0.60` 같은 직접 비교만

---

## 9. Decision logic

학습 시 sweep에서 선정된 *prob_th* 한 값을 모델별로 보존하고, 그 임계와
비교합니다.

### static mode
```python
prob_up = model.predict_proba(today_row)[0, 1]
if   prob_up >= prob_th_up:   decision = "BUY"
elif prob_up <= prob_th_down: decision = "SELL"   # 통상 0.50 그대로
else:                          decision = "HOLD"
```

### dynamic mode
```python
# dynamic_k가 학습 시 best로 선정된 값 (예: dynk12 → k=1.2)
sigma_30 = close_history.pct_change().rolling(30, min_periods=30).std().iloc[-1]
adjusted_th_up = 0.50 + k * sigma_30   # 학습과 동일 공식 (features.py::compute_dynamic_threshold)
```

(정확한 공식은 학습 sweep에서 사용한 `compute_dynamic_threshold` /
`build_labels`를 따르면 됩니다 — features.py 참조.)

### Ensemble (선택)

여러 모델을 운영하려면 단순 평균 또는 가중 평균:
```python
probs = [m.predict_proba(today_row)[0, 1] for m in models]
prob_up = sum(probs) / len(probs)         # 또는 PSR-weighted
```

ensemble은 각 모델이 calibrated일 때만 의미가 있습니다 (calibrated의
평균도 잘 calibrated).

---

## 10. ✅ Daily checklist (운영자가 매일 확인)

```
[ ] Bitcoin Core ibd_done() == True
[ ] btc_1d.parquet 마지막 close_time = 어제 23:59:59.999 UTC
[ ] btc_15m.parquet 마지막 close_time이 현재 시각보다 > 15분 전
[ ] macro_fred.parquet 마지막 row가 어제 또는 직전 영업일
[ ] cot.parquet 가장 최근 row가 *4일 이상* 과거 (lag 보정 후 사용 가능)
[ ] coinmetrics.parquet이 _refresh_onchain()에 의해 오늘 갱신됨
[ ] X.shape[1] == 학습 시 X.shape[1] (cols fingerprint 일치)
[ ] X.iloc[-1] dtype 모두 float, 추가 inf 없음
[ ] predict_proba 결과 ∈ [0, 1]
[ ] decision 출력 + DB/parquet 저장 + 알림 발송
[ ] 다음날 아침 outcome (전날 결정 + 실제 return) 비교 로그 작성
```

`predict_for_today()`의 JSON 출력 + 위 checklist 통과 시그널을 모두
모니터링 시스템 (e.g. Slack/email)에 발송.

---

## 11. 자주 발생하는 함정 / 디버깅 가이드

| 증상 | 흔한 원인 | 대응 |
|---|---|---|
| `prob_up == NaN` | X.iloc[-1]에 모든 컬럼이 NaN 또는 핵심 feature NaN | 어떤 컬럼이 NaN인지 검사: `X.iloc[-1].isna().sum()` |
| `prob_up`이 매일 거의 동일 (e.g. 0.5xx 고정) | X.iloc[-1]이 미완성 데이터 (전부 마지막 fill값) | fetcher가 새 row 추가 못 했는지, last close_time 검사 |
| `X.shape[1] != train cols` | gdelt_tone NaN 또는 새 feature 추가됨 | reindex(train_cols) + 사람 검토 |
| on-chain feature가 매일 같은 값 | `_refresh_onchain()`이 silently fallback | IBD 상태 + RPC 응답 확인 |
| `prob_up` 평소와 매우 다름 (jump) | distribution shift (특히 on-chain swap 직후) | §4.4 검증 다시 수행 |
| Sunday/공휴일 cron 출력 없음 | FRED/CFTC publish 없음 → 일부 fetcher 실패 → script abort | fetcher별 실패를 fatal로 두지 말고 NaN으로 처리 |
| `predict_proba` 분포가 0 또는 1로 몰림 | X에 leakage 있음 (e.g. 미래 정보가 marginal하게 새어듦) | features 추가 후 *반드시* walk-forward로 재검증 |
| 학습은 calib인데 추론이 nocalib처럼 동작 | saved pkl이 wrapping 안 된 raw EBM/XGB | pkl을 다시 학습 끝에 `CalibratedClassifierCV` wrapped로 저장 |

---

## 12. Failure modes & fallback 정책

| 실패 | 정책 |
|---|---|
| Bitcoin Core RPC 무응답 | `_refresh_onchain()` skip + 직전 `coinmetrics.parquet` 그대로 사용. on-chain 4 feature가 stale해짐. 24시간 내 복구 안 되면 alert |
| Binance API rate-limit | retry with backoff (`binance_fetcher.py` 내부에 이미 있음). 끝까지 실패 시 추론 abort + alert |
| FRED 한 시리즈 누락 | 해당 컬럼 NaN. 모델은 NaN 처리 가능. log warning |
| GDELT TimelineTone 누락 | `gdelt_tone` / `gdelt_tone_chg_3d` 생략. reindex(train_cols)으로 NaN 채움 |
| pickle load 실패 | fallback 모델 (예: 가장 보수적인 single best EBM) 시도. 둘 다 실패 → HOLD + alert |
| 추론 결과 |`prob_up - 0.5| < 0.01 (signal 거의 없음) | 자동으로 HOLD |

---

## 13. Reference (코드)

- `src/features.py` — feature pipeline (학습/추론 공통)
- `src/model.py` — WFConfig, `_fit_calibrated`, `walk_forward_predict`
- `src/live_inference_btc_core.py` — 일일 추론 entry point
- `src/btc_core_rpc.py` — Bitcoin Core RPC snapshot
- `src/btc_core_chain_scan.py` — 블록 스캔 (TxCnt/AdrActCnt)
- `src/utc2130.py` — UTC 21:30 daily bar builder (15m → daily)
- `src/coingecko_fetcher.py`, `binance_fetcher.py`, `fred_fetcher.py`,
  `cftc_fetcher.py`, `fng_fetcher.py`, `gdelt_fetcher.py`,
  `coinmetrics_fetcher.py`
- `FEATURE_SOURCES.md` — feature ↔ source 상세 매핑
- `FEATURE_VERSIONS.md` — v0/v3/v4/v5/v5_2/v6/v7/v5_3 분기 정의

---

## 14. 학습 → 운영 전환 체크리스트 (1회)

sweep 끝난 후 운영 시작 *직전* 한 번:

```
[ ] 선정 모델의 pickle을 data/live_models_{ebm,xgb}/ 에 저장
    (학습 sweep의 final-refit step 모델 + cal wrapping 포함)
[ ] X 컬럼 fingerprint(.cols.json)도 같이 저장
[ ] data/coinmetrics.parquet 학습 시 버전을 _archive_coinmetrics_train.parquet로 백업
[ ] §4.4 분포 일치 검증 통과
[ ] live_inference_btc_core.py를 v5_2/v7/v5_3 flag도 처리하도록 확장
    (현재 코드는 v3/v4/v5만 인식 — line 87-93)
[ ] 빈 cron 시뮬레이션: 수동으로 한 번 돌려 prob_up 출력 확인
[ ] Daily checklist의 모든 항목이 자동화돼 알림 발송
```

이 체크리스트를 통과해야 production 시작.

---

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-05-13 | 초기 작성. 학습=Coin Metrics → 추론=자체 노드 swap 명시. calibration 보존, fetcher 호출 순서, daily checklist 포함. |
