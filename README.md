# ml_test_v2 — BTC direction sweep (2019-12-01 cutoff, EBM + XGB, isotonic-calibrated)

Companion experiment to [`ml_test`](https://github.com/tgj1994/ml_test).
The only experimental difference is the **training data start date**:

|                    | ml_test (v1)              | **ml_test_v2** (this repo) |
|--------------------|---------------------------|----------------------------|
| Data start         | 2017-08 (Binance inception) | **2019-12-01 UTC**         |
| Data rows          | ~3,012 daily              | ~2,176 daily (72%)         |
| Code / fetchers    | identical                 | identical                  |
| Calibration        | isotonic cv=3             | identical                  |
| Hyperparameters    | WFConfig default          | identical                  |
| Feature versions   | v0/v3/v4/v5/v5_2/v6/v7/v5_3 | identical                |
| Phase chain        | 19 phases                 | identical                  |

Purpose: ablation on the value of the 2018-2019 pre-COVID / post-2017-bull
training tail. v2 trains only on the post-2019 regime (covers two halvings:
2020-05 and 2024-04) and lets you compare the same model fit on a more
recent / shorter window.

## Quick start (server-side)

```bash
git clone <this-repo-url>.git ml_test_v2
cd ml_test_v2
chmod +x start.sh
nohup bash start.sh > sweep.log 2>&1 &
echo $! > sweep.pid
tail -f sweep.log
```

`start.sh` will:

1. Install `uv` to `~/.local/bin` if missing.
2. `uv sync` from `uv.lock`.
3. Verify the 2019-12-01 cutoff is intact in `data/` (sanity check).
4. Run EBM chain (v0→v3→v4→v6 → v5_sel→v5 → v5_2_sel→v5_2 → v7 → v5_3_sel→v5_3).
5. Run XGB chain (same 8 versions).
6. Write per-variant logs under `logs/` and report artifacts under
   `reports/utc2130/` (EBM) and `reports/xgb/utc2130/` (XGB).

To resume after a crash, just run `start.sh` again — `--skip-done` skips
variants whose log already ends with a `Markdown:` line.

## Hardware tuning

Defaults assume a 64-vCPU CPU-only Linux host (AMD EPYC class) at ~62.5%
load:

| Variable   | Default | Meaning                                    |
|------------|--------:|--------------------------------------------|
| `THROTTLE` | 20      | parallel sweep children (each = THREADS cores) |
| `THREADS`  | 2       | EBM `n_jobs` / XGB `n_jobs` per child      |

Override at invocation time:

```bash
THROTTLE=24 THREADS=2 nohup bash start.sh > sweep.log 2>&1 &   # ~75%
THROTTLE=16 THREADS=2 nohup bash start.sh > sweep.log 2>&1 &   # ~50%
```

## Time estimate

v2 has ~22% fewer training rows than v1, so each phase is roughly 78% of
v1's wall-clock. On a 64-vCPU host at `throttle=20`, expect each version
sweep to land in **~5–6 hours**, and the full chain (19 phases) in
**3.5–4 days**.

## Layout

```
ml_test_v2/
├── start.sh                 # full sweep entrypoint (v2 — 2019-12-01 cutoff)
├── pyproject.toml uv.lock   # Python deps (pinned)
├── data/                    # raw input parquets, pre-cut to 2019-12-01+ (~12 MB)
├── src/                     # features, model, fetchers, runners
├── runners/run_all_M_sm.py  # parallel sweep launcher
├── analysis/                # v5 selector, robust_full_ebm.py, robust_full_xgb.py
├── main_th_sweep/v{0..7,5_2,5_3}/   # per-variant sweep entry points
├── FEATURE_SOURCES.md       # raw → derived feature mapping
├── FEATURE_VERSIONS.md      # what each version flag adds
├── INFERENCE_RUNBOOK.md     # live-service inference operations
└── COIN_METRICS_EXCHANGE_FLOW.md  # historical note on exchange flow
```

## Calibration is mandatory

Both EBM and XGB train through `CalibratedClassifierCV(method='isotonic',
cv=3)` so `predict_proba` returns calibrated probabilities suitable for
threshold-based buy/sell/hold decisions. Do **not** set
`XGB_NO_CALIB=1` or remove `EBM_CAL_METHOD=isotonic` from `start.sh`.

## Comparing to ml_test (v1)

After both repos finish their full chain:

```bash
# v1 results (ml_test)
ls ml_test/reports/utc2130/robustness_all_ebm.csv
ls ml_test/reports/xgb/utc2130/robustness_all_xgb.csv

# v2 results (ml_test_v2 — this repo)
ls ml_test_v2/reports/utc2130/robustness_all_ebm.csv
ls ml_test_v2/reports/xgb/utc2130/robustness_all_xgb.csv
```

Same schema for both. Join on `(variant, mode, label_or_k, best_prob_th)`
to see how each metric (PSR / DSR / MaxDD / Sharpe / 730d return / PBO)
shifts between full-history and 2019-12+ training.
