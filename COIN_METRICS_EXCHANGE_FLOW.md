# Coin Metrics 거래소 입출금 (FlowInExUSD / FlowOutExUSD) 방법론 + 자체 구축 계획

## 배경 (정정: 2026-05-12)

**현재 프로젝트에서는 거래소 flow 지표(`FlowInExUSD` / `FlowOutExUSD`)를 사용하지 않는다.**
초기 검토에서 사용 중이라고 잘못 단정했던 부분을 정정한다.

- `coinmetrics_fetcher.py:DEFAULT_METRICS` 는 PriceUSD / CapMrktCurUSD / SplyCur /
  AdrActCnt / TxCnt / HashRate 6개만 받는다.
- `coinmetrics.parquet` 의 실제 컬럼도 위 6개뿐이다.
- `features.py` 어디에서도 `FlowInEx` / `FlowOutEx` 컬럼을 읽지 않는다.

이 문서는 **만약 향후 거래소 flow를 추가하고 싶을 때** 어떻게 자체 구축할 수 있는지에 대한
참고 자료로 남긴다 — 현재 모델에는 영향 없다. 자체 on-chain 파이프라인
(`btc_core_chain_scan.py`)으로의 전환 가능성을 평가하기 위해 Coin Metrics 가 어떻게 이
지표를 산출하는지 정리해 둔다.

## Coin Metrics 방법론

### Step 1 — Seed 주소 큐레이션 (수동)

거래소(Binance, Coinbase, Kraken, Bitfinex 등)마다 최소 1개의 확인된 wallet 주소를
seed로 보유한다. 시드 확보 경로:

- 거래소 공식 disclosure (proof-of-reserves 공개 주소)
- 공개된 cold wallet 주소
- 거래소 발표/announcement
- 사용자 tagged transactions (입금/출금 영수증)
- 수동 OSINT 리서치

### Step 2 — Common-Input-Ownership Heuristic (CIOH)

Coin Metrics 공식 인용:

> "Exchange flows are estimated using the common-input-ownership heuristic,
> which assumes that addresses that are inputs to the same transaction share
> an owner. While this technique is precise, it requires at least one seed
> address for every exchange, limiting coverage to a predetermined universe
> of exchanges."

알고리즘:

```
1. seed 주소 S 를 클러스터 C 에 추가
2. 체인 전체 트랜잭션 T 순회:
   - T 의 어떤 input 주소가 C 에 있으면 → T 의 다른 input 주소도 모두 C 에 추가
3. 새로 추가된 주소가 있다면 2 반복
4. 수렴하면 C = 그 거래소가 소유한 모든 주소 집합
```

CIOH는 *"한 트랜잭션의 input들은 모두 같은 owner가 sign 한다"* 라는 단순한 가정에
기반한다 (Meiklejohn et al. 2013 "A Fistful of Bitcoins" 페이퍼에서 정립).

### Step 3 — Flow 집계

매일 블록 스캔:

- 트랜잭션 T 의 **input이 클러스터에 속함** → BTC가 거래소에서 나감 → `FlowOut`
- 트랜잭션 T 의 **output이 클러스터에 속함** → BTC가 거래소로 들어옴 → `FlowIn`
- 양쪽 모두 라벨링된 거래소 주소이면 (inter-exchange transfer) → 두 flow에서 모두 제외
- BTC 양에 그 날 BTC USD 가격을 곱해 USD 환산

### 한계 (Coin Metrics 공식 명시)

- **UTXO 체인 전용**: Bitcoin 류 OK, Ethereum 같은 account-based 체인은 다른 방법 필요
- **CoinJoin이 휴리스틱 깨뜨림**: Wasabi, Whirlpool, JoinMarket 같은 mixing service의 트랜잭션은
  input 공동 소유 가정이 무효 → false cluster expansion 위험
- **Peeling chain**: 잔돈 패턴이 휴리스틱에 noise 추가
- **Seed 누락 거래소**: seed가 없는 거래소는 클러스터링이 시작도 안 됨 (coverage gap)
- **Address rotation**: 거래소가 새 cold wallet 만들면 별도로 발견해서 seed 추가해야 함

## 자체 구축 (`btc_core_chain_scan.py` 확장) 가능성

| 단계 | 난이도 | 비고 |
|---|---|---|
| bitcoind RPC로 모든 tx input/output 스캔 | 쉬움 | `getblock verbosity=2` 사용. 이미 `btc_core_chain_scan.py`가 비슷한 일 수행 |
| CIOH 알고리즘 구현 | 쉬움 | union-find로 며칠 작업. 오픈소스 참고 가능 (BlockSci) |
| 거래소 seed 주소 큐레이션 | 중간 | 초기 dataset은 공개된 cold wallet 모음 + 거래소 공식 disclosure로 가능. 신규 wallet 추적이 ongoing 작업 |
| CoinJoin / peeling 처리 | 중간~어려움 | Mixer tx 식별 휴리스틱 적용 — 정확도 trade-off |
| 품질 검증/유지보수 | 어려움 | Coin Metrics 가 가장 차별화하는 부분 — 전담팀 ongoing 큐레이션 |

### 비교 표

| 항목 | Coin Metrics Network Data Pro | 자체 구축 |
|---|---|---|
| 정확도 | 높음 (수년 큐레이션 누적) | 중~높음 (seed 품질에 비례) |
| 유지보수 | 그쪽 책임 | 월별 신규 cold wallet 추적 필요 |
| 비용 | 구독료 (수천~수만 USD/년 추정) | 1회 구현 + 운영 인력 비용 |
| 라이선스 | 상업 가능 (paid plan) | 자체 산출이라 라이선스 제약 없음 |

## Production 전환 계획

### 사전 조건

1. **bitcoind IBD 완료** — 현재 33% 진행 (2026-05-12 기준), 약 2-3일 후 완료 예상
2. **새 EBM sweep 완료** — 거래소 flow feature의 실제 importance 측정. Importance 결과에 따라
   다음 두 시나리오로 분기:

### 시나리오 A: 거래소 flow의 feature importance가 미미한 경우

`_mvrv_v2_features` 내 FlowInExUSD/FlowOutExUSD 의존 항목을 **그냥 제거**.
모델이 거래소 flow 없이도 충분히 학습됐다는 의미. Coin Metrics 의존 최소화 가능.

남는 feature: `d_mvrv_z` (가격만으로 계산 가능한 MVRV proxy) 유지 — Coin Metrics 불필요.

### 시나리오 B: 거래소 flow가 강한 predictor인 경우

세 가지 옵션:

**Option B1 — 자체 구축 (장기적 정공법)**

```
btc_core_chain_scan.py  (이미 존재)
  + cluster_builder.py    (신규 — CIOH 알고리즘 + seed 큐레이션)
  + exchange_flow.py      (신규 — 클러스터에서 일별 in/out 집계)
  + 시드 주소 DB           (data/exchange_seeds.json — 큐레이션 진행)
```

- 구현 기간: 2-4주
- 유지보수: 분기별 신규 cold wallet 추가
- 라이선스: 자체 산출이라 무제한 상업 가능

**Option B2 — 상업 벤더 (예: Glassnode, CryptoQuant)**

- 비용: 월 ~$수백~수천 (tier에 따라)
- 즉시 사용 가능
- 일부 벤더는 commercial license 명시

**Option B3 — Coin Metrics Network Data Pro 구독**

- 가장 직접적인 마이그레이션 (feature 컬럼 그대로 유지)
- 가격 확인 필요 (영업 문의)

## 참고 자료

- [Asset Metrics FAQs - Coin Metrics docs](https://github.com/coinmetrics/docs-website/blob/master/asset-metrics/asset-metrics-faqs.md)
- [Exchange Flows | Data Encyclopedia - Coin Metrics](https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/mining/exchange-flows)
- [FlowInExUSD - Coin Metrics docs](https://docs.coinmetrics.io/info/metrics/FlowInExUSD)
- [Bitcoin On-Chain Exchange Metrics: The Good, The Bad, The Ugly - Glassnode](https://insights.glassnode.com/exchange-metrics/)
- Meiklejohn, S. et al. (2013). *A Fistful of Bitcoins: Characterizing Payments Among Men with No Names.* — CIOH 최초 정립 논문
- BlockSci (오픈소스 Bitcoin cluster analyzer) — https://github.com/citp/BlockSci
