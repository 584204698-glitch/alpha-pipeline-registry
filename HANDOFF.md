# Alpha Pipeline Handoff

## 1. What this project is
This repository implements a local automated crypto perpetual alpha-mining pipeline under `/mnt/e/alpha_pipeline`.

Goal:
- generate factor ideas mainly through DeepSeek
- convert ideas into executable factor Python files
- run local backtests on parquet market data
- hard-reject weak/overfit factors through deterministic validator rules
- deploy only PASS factors into production config with default-disabled safety switch
- run health/audit reporting and scheduled autodiscover loops

This is an engineering-complete pipeline, but not yet a business-complete pipeline:
- code architecture is in place
- tests pass
- scheduled jobs run
- data download and audit loops run
- **no factor has passed validator yet**

---

## 2. Current project status at handoff
Live checked at handoff time:
- pytest: `20 passed, 4 warnings`
- logged discover rounds: `58`
- logged deploy events: `0`
- validator rejections logged: `99`
- generated factor files under `factors/alpha_*.py`: `25`
- production factors deployed: `0`
- latest autodiscover summary provider: `deepseek`
- latest autodiscover summary result often remains low-yield and may produce `ideas_returned=0..1`

### Data status
Current download progress files show (completed 200-market expansion):
- `15m`: 200 markets success, 0 failed
- `1h`: 200 markets success, 0 failed
- `2h`: 200 markets success, 0 failed
- `4h`: 200-market run completed (198 success, 0 failed; 2 non-fetched within the batch)

Important nuance:
- main backtest dataset is still the single file `data/data_storage.parquet`
- interval-specific raw market parquet shards are also stored under `data/markets/<interval>/...`
- all four intervals are now at production scale (200 markets)

### Registry status
- `logs/tested_idea_families.json` has been backfilled from existing `factors/alpha_*.py` files
- 16 unique family fingerprints now registered
- This reduces re-testing of known historical families

### Key practical conclusion
The pipeline is no longer mainly blocked by obvious runtime bugs.
The main bottleneck is now **idea quality**, **repeat-family suppression**, and **DeepSeek output quality**, not missing architecture.

---

## 3. Directory map
```text
/mnt/e/alpha_pipeline/
├── HANDOFF.md
├── config/
│   ├── main_config.json
│   └── production_factors.json
├── data/
│   ├── data_storage.parquet
│   ├── download_progress_15m.json
│   ├── download_progress_1h.json
│   ├── download_progress_2h.json
│   ├── download_progress_4h.json
│   └── markets/
├── factors/
│   ├── __init__.py
│   ├── base.py
│   └── alpha_*.py
├── logs/
│   ├── pipeline.log
│   ├── autodiscover_last_summary.json
│   ├── tested_idea_families.json
│   └── audit_report_*.md
├── idea_generator.py
├── idea_registry.py
├── code_generator.py
├── backtest_engine.py
├── validator.py
├── deployer.py
├── data_ingestion.py
├── download_data.py
├── progress_report.py
├── health_check.py
├── runtime_env.py
├── pipeline_utils.py
├── main.py
├── requirements.txt
└── tests/test_pipeline.py
```

---

## 4. High-level workflow for the next model
If you are taking over as the next model, think of the project as 4 connected loops:

### Loop A: Idea generation
- `idea_generator.py` queries DeepSeek
- prompt includes historical failures and duplicate-family warnings
- output must conform to factor schema
- malformed output is recovered if possible

### Loop B: Duplicate-family control
- `idea_registry.py` computes a structural family fingerprint from formula AST + inputs + timeframes
- near-clones should be rejected before expensive backtests
- user explicitly asked for **upstream duplicate suppression**, not only post-hoc counting

### Loop C: Backtest and hard rejection
- `code_generator.py` emits a factor file
- `backtest_engine.py` computes IC/IR/quintiles/regime/correlations
- `validator.py` applies 6 hard deterministic reject rules
- only PASS factors go forward

### Loop D: Deployment and operations
- `deployer.py` writes PASS factors into `production_factors.json`
- deployed factors are always disabled by default
- `health_check.py` writes audit reports
- Hermes cron scripts keep autodiscover and progress reporting alive

In short:
```text
DeepSeek ideas -> family dedupe -> factor code -> backtest -> validator -> deploy(if PASS) -> health/audit
```

---

## 5. Important files and what each one does

### `main.py`
Main orchestration entrypoint.

Responsibilities:
- bootstrap directory/config/data skeleton
- load `.env.runtime`
- collect failed-factor feedback from previous runs
- call idea generation
- append queued/failed/deployed family entries into registry
- process each idea through generate -> backtest -> validate -> deploy
- never crash whole run because one factor crashes

Important functions/classes:
- `bootstrap_environment(project_root)`
- `build_failure_feedback_record(processed_item)`
- `PipelineRunner.process_idea()`
- `PipelineRunner.discover()`

Important current behavior:
- failures now feed back as either `runtime_error` or `validator_fail`
- duplicate filtering is expected to happen inside `idea_generator.fetch_factor_ideas()` via `idea_registry`
- summary is printed as a Python dict to stdout

### `idea_generator.py`
DeepSeek-facing idea generator and fallback/recovery layer.

Important behavior:
- default DeepSeek model constant is `deepseek-v4-pro`
- prompt includes:
  - failed-factor summary
  - historical family registry summary
  - originality guardrails
  - quality guardrails
- can recover semi-structured or malformed JSON/text from DeepSeek responses
- local fallback templates still exist and are conservative

Important risk:
- historical repeated family patterns are still strongly represented by fallback/recovery behavior, especially `RecoveredDeepSeekIdea1`
- DeepSeek often returns only 1 recoverable idea instead of a full batch
- in some runs DeepSeek output is fully filtered away by dedupe, producing `ideas_after_dedupe=0`

Important functions:
- `_quality_guardrails_text()`
- `_deepseek_prompt()`
- `_fetch_from_deepseek()`
- `_recover_ideas_from_text()`
- `fetch_factor_ideas()`

### `idea_registry.py`
Persistent duplicate-family registry and structural fingerprinting.

What it does:
- canonicalizes formulas via AST
- computes `family_fingerprint`
- treats additive/multiplicative/commutative structure carefully
- blocks same-principle ideas even if only renamed / reordered / retuned
- writes registry to `logs/tested_idea_families.json`

Important limitation now:
- registry persistence exists, but current registry file only contains a very small number of entries
- most historical factor files have **not** yet been backfilled into the registry
- so historical duplicates in old files may still not all be represented in registry memory

### `code_generator.py`
Turns an idea dict into executable factor code.

What it guarantees:
- generates `alpha_<snake_case>.py`
- factor class inherits `FactorRegistry`
- factor `compute()` uses `evaluate_formula()` from `factors/base.py`
- rejects placeholder name `FactorName`

### `factors/base.py`
Base factor runtime and formula evaluator.

Important pieces:
- `FactorRegistry`
- vectorized transform library:
  - `delta`
  - `pct_change`
  - `rolling_mean/std/max/min/sum/quantile`
  - `zscore`
  - `rank_pct`
  - `indicator`
  - `clip`
  - `abs`
  - `sign`
- `evaluate_formula()` uses restricted `eval()` context
- `load_factor_from_path()` dynamically imports factor files

This file is the heart of formula execution.
If new operators are needed, they must be added here and also whitelisted in safety logic.

### `backtest_engine.py`
Runs local factor backtests.

What it computes:
- fee-adjusted forward returns
- overall IC
- overall IR / ICIR-style measure
- in-sample vs out-of-sample ICIR
- quintile returns
- regime ICs
- correlations vs active production factors
- signal summary including `long_ratio`

Important notes:
- default dataset path is `data/data_storage.parquet`
- supports override via `ALPHA_PIPELINE_DATA_PATH`
- OOS split logic:
  - if >180 timestamps: first 120 IS, next 60 OOS
  - else falls back to ~66/34 split

### `validator.py`
Hard deterministic overfit gate.

Current implemented checks in order:
1. look-ahead bias
2. OOS consistency
3. parameter stability
4. market regime invariance
5. multicollinearity limit
6. signal symmetry

Single-failure veto behavior:
- first failed check stops later checks
- returns `PASS` or `FAIL`
- logs exact failed check

Important caveat:
- parameter stability recomputes IC against raw forward returns from `BacktestEngine._forward_returns()`, not fee-adjusted returns
- multicollinearity only checks active deployed factors in `production_factors.json`, and because deployed count is 0, this check is currently effectively weak in practice

### `deployer.py`
Production registry writer.

What it does:
- appends factor metadata into `config/production_factors.json`
- writes `FACTOR_<NAME>_ENABLED = false`
- never deploys enabled-by-default

Current state:
- file is empty because no factor has passed

### `health_check.py`
Writes audit markdown reports.

Checks:
- data freshness from `data_storage.parquet`
- memory estimate from `/proc/meminfo`
- deployed factor count in last 24h
- recent factor signal pnl for deployed factors
- writes `logs/audit_report_<timestamp>.md`

Current state:
- audit report generation is working
- there are many audit reports already in `logs/`

### `data_ingestion.py`
Live Coinglass data download/build logic.

Important behavior:
- filters out obvious non-crypto symbols
- restricts to Binance/OKX by default
- downloads four endpoint families per market:
  - price
  - open interest
  - funding rate
  - taker volume
- writes per-market parquet shards
- writes combined dataset to `data/data_storage.parquet`
- writes interval progress JSONs

Important function:
- `build_live_dataset(...)`

### `download_data.py`
CLI wrapper around `build_live_dataset()`.
Used for staged universe expansion runs like 20 -> 50 -> 100 -> 200.

### `progress_report.py`
Builds a human-readable progress report from health-check and current config/data status.
Used by the hourly cron script.

### `pipeline_utils.py`
Shared utilities:
- structured file logging
- JSON read/write
- UTC timestamp helper
- safe float coercion

Logging format matches the requested structure:
`[timestamp] [MODULE] [LEVEL] - message`

### `runtime_env.py`
Loads environment variables from `.env.runtime` without overwriting already-set env vars.
Used by `main.py` and `download_data.py`.

### `tests/test_pipeline.py`
Main regression suite.

Current verified state:
- all tests pass
- contains tests for:
  - idea generation schema/recovery
  - duplicate-family detection
  - structural fingerprint commutativity
  - generated factor/backtest behavior
  - validator schema
  - deployment disabled switch
  - failure feedback recording

---

## 6. Scheduled automation and Hermes-side scripts
Hermes-side scripts live under:
- `/home/hermes/.hermes/scripts/`

Important scripts:

### `alpha_pipeline_autodiscover.py`
- runs `main.py --mode discover --batch-size 5`
- parses stdout dict summary
- writes `logs/autodiscover_last_summary.json`
- currently scheduled every 30 minutes via cron job

### `alpha_pipeline_supervised_discover.py`
- runs `main.py --mode discover --batch-size 3`
- lighter supervised discover helper

### `alpha_pipeline_hourly_progress.py`
- runs `progress_report.py`
- emits progress text for hourly updates

Current Hermes cron jobs:
1. `alpha-pipeline-autodiscover`
   - schedule: `every 30m`
   - script: `alpha_pipeline_autodiscover.py`
   - delivery: `local`
   - no_agent: `true`

2. `alpha-pipeline-hourly-progress`
   - schedule: `0 * * * *`
   - script: `alpha_pipeline_hourly_progress.py`
   - delivery: `origin`
   - no_agent: `true`

---

## 7. Configuration files and live runtime config

### `config/main_config.json`
Current important values:
- `deepseek_api_key_env`: `DEEPSEEK_API_KEY`
- `coinglass_api_key_env`: `COINGLASS_API_KEY`
- `default_mode`: `discover`
- `default_batch_size`: `10`
- `cpu_usage_cap`: `0.7`
- `memory_cap_gb`: `100`
- `transaction_fee_bps`: `5.0`
- `live_symbol_limit`: `20` in file right now
- `live_history_limit`: `120`
- `live_intervals`: `15m,1h,2h,4h`
- `live_pause_seconds`: `4.5`
- `live_exchanges`: `Binance, OKX`

Note:
- operationally the project has already downloaded up to 200 for several intervals even if config still shows smaller defaults

### `config/production_factors.json`
Current contents:
```json
{"factors": []}
```

### `logs/tested_idea_families.json`
Purpose:
- persistent family dedupe memory

Current limitation:
- contains too few entries compared with full historical file set

### Project runtime env
Project-level API keys are loaded from:
```text
/mnt/e/alpha_pipeline/.env.runtime
```

`main.py` and `download_data.py` call `load_runtime_env()` at startup.
If DeepSeek mysteriously falls back to local bootstrap, check this file first.

### Hermes brain config
Hermes itself now uses official DeepSeek by default.
Live verified target config:
- provider: `deepseek`
- model: `deepseek-v4-pro`
- base_url: `https://api.deepseek.com/v1`

This was validated by a real command:
```bash
hermes chat -Q -q "Reply with exactly OK and nothing else."
```
which returned:
```text
OK
```

---

## 8. Repetition / quality-control state
The user explicitly wanted quality and duplicate-rate supervision to be **upstream in the prompt**, not only audited afterward.
This has been partly implemented.

Implemented now:
- failed-factor summary enters prompt
- duplicate-family summary enters prompt
- originality/quality hard rules are included in DeepSeek prompt
- structural family dedupe exists before backtest
- validator remains strict

Still weak / incomplete:
- historical factor files not fully backfilled into registry
- fallback/recovery paths still bias toward repeated family shapes
- latest registry file is tiny relative to historical factor population
- latest summaries still commonly resolve to `RecoveredDeepSeekIdea1` or get deduped to zero

Most repeated historical families observed:
- `RecoveredDeepSeekIdea1`
- `TestMomentumBalance`
- `TestFundingDivergence`
- repeated variants of:
  - `OI_Funding_Divergence_*`
  - `Taker_Imbalance_Reversion_*`
  - `VolAdj_Momentum_Pressure_*`

---

## 9. Known blockers and important caveats

### A. No PASS factor yet
This is the biggest business blocker.
Current project is operational but not yet producing approved factors.

### B. DeepSeek quality remains weak in practice
Observed behaviors:
- often only 1 idea is recoverable out of requested 5
- malformed or semi-structured output is common
- fallback/recovery often degenerates to repetitive conservative templates
- some runs now become `ideas_returned=0` after dedupe removes repeated family structures

### C. DeepSeek brain is now switched successfully
Earlier this was blocked. It is no longer blocked.
The switch was completed using the provided official DeepSeek API key.

Live verification completed:
- official DeepSeek `/models` returned:
  - `deepseek-v4-flash`
  - `deepseek-v4-pro`
- direct official completion test to `deepseek-v4-pro` returned HTTP 200
- Hermes default `hermes chat` returned `OK`

### D. Pipeline discover can now use official DeepSeek key from `.env.runtime`
A real discover run completed after the switch.
Current observed output can still be zero-yield after dedupe, so "DeepSeek works" does not mean "factor quality is solved".

### E. Market data download is now complete
- 15m/1h/2h: 200 success each
- 4h: 198 success (200-market batch run completed)
- All intervals at production scale

### F. Registry backfill is complete
- 16 unique family fingerprints from 25 historical alpha files registered
- Future discover runs will skip known repeated families
- Falling back to `RecoveredDeepSeekIdea1` should now be caught by dedupe

---

## 10. Exact usage guide for the next model
This section is intentionally detailed so a DeepSeek-based successor can operate without guessing.

### A. First 5 minutes after takeover
Run these checks in order:

```bash
cd /mnt/e/alpha_pipeline
.venv/bin/python -m pytest tests/test_pipeline.py -q
.venv/bin/python main.py --mode discover --batch-size 3
cat logs/autodiscover_last_summary.json
```

Then inspect:
```bash
cat config/production_factors.json
cat data/download_progress_15m.json
cat data/download_progress_1h.json
cat data/download_progress_2h.json
cat data/download_progress_4h.json
```

And if needed, inspect Hermes-side scheduler:
```bash
hermes cron list
```

### B. Normal working loop
Use this loop repeatedly:
1. inspect latest autodiscover summary
2. inspect whether ideas were empty, duplicated, crashed, or failed validator
3. identify the dominant failure class:
   - no ideas
   - duplicate families
   - malformed output
   - validator failure
   - runtime crash
4. change the smallest relevant part of the system
5. rerun tests
6. rerun discover
7. only then decide next change

### C. If no ideas are produced
Check:
- `idea_generator.py`
- `logs/autodiscover_last_summary.json`
- whether DeepSeek returned malformed content
- whether dedupe removed everything

Typical fixes:
- improve prompt constraints while preserving structure diversity
- reduce fallback collapse into repeated templates
- inspect `_recover_ideas_from_text()`

### D. If everything is deduped away
Check:
- `idea_registry.py`
- `logs/tested_idea_families.json`
- whether DeepSeek is just generating old families again

Typical fixes:
- backfill historical alpha files into the registry
- expand the prompt’s negative examples with recent repeated families
- push for materially different interaction structures, not parameter tweaks

### E. If ideas reach validator but fail quality
Read the exact failed check from:
- `logs/autodiscover_last_summary.json`
- `logs/pipeline.log`

Typical fixes by failure type:
- `oos_consistency_pass`: push for simpler, more robust cross-horizon structure
- `parameter_stability_pass`: avoid brittle threshold-heavy formulas
- `market_regime_invariance_pass`: force multi-regime plausibility in prompt
- `signal_symmetry_pass`: reduce one-sided bias in formula design
- `correlation_limit_pass`: if deployed factors ever appear, actively diversify interactions

### F. If a factor crashes at runtime
Check:
- generated factor file in `factors/`
- `factors/base.py`
- `pipeline.log`
- safe formula vocabulary in `idea_generator.py`

Do not let runtime crashes get confused with “bad research”.
The user specifically asked to avoid bug-caused mining failures.

### G. If you need to refresh or expand market data
Manual command:
```bash
cd /mnt/e/alpha_pipeline
.venv/bin/python download_data.py --symbol-limit 200 --history-limit 120 --intervals 15m 1h 2h 4h --pause-seconds 4.5
```

But note the user preference:
- once local market data is sufficiently downloaded, do **not** waste time repeatedly re-downloading unless a patch requires it

### H. If you need to inspect progress quickly
```bash
cd /mnt/e/alpha_pipeline
.venv/bin/python progress_report.py --project-root /mnt/e/alpha_pipeline --batch-size 10
```

### I. If you need to verify the Hermes brain itself
```bash
hermes chat -Q -q "Reply with exactly OK and nothing else."
```
Expected result:
```text
OK
```

---

## 11. Day-to-day command cheat sheet

### Run tests
```bash
cd /mnt/e/alpha_pipeline
.venv/bin/python -m pytest tests/test_pipeline.py -q
```

### Run one discover pass
```bash
cd /mnt/e/alpha_pipeline
.venv/bin/python main.py --mode discover --batch-size 5
```

### Run health check once
```bash
cd /mnt/e/alpha_pipeline
.venv/bin/python health_check.py
```

### Run data download manually
```bash
cd /mnt/e/alpha_pipeline
.venv/bin/python download_data.py --symbol-limit 200 --history-limit 120 --intervals 15m 1h 2h 4h --pause-seconds 4.5
```

### Run progress report
```bash
cd /mnt/e/alpha_pipeline
.venv/bin/python progress_report.py --project-root /mnt/e/alpha_pipeline --batch-size 10
```

### Inspect latest autodiscover summary
```bash
cat /mnt/e/alpha_pipeline/logs/autodiscover_last_summary.json
```

### Inspect production factors
```bash
cat /mnt/e/alpha_pipeline/config/production_factors.json
```

### Inspect duplicate registry
```bash
cat /mnt/e/alpha_pipeline/logs/tested_idea_families.json
```

### Inspect cron jobs
```bash
hermes cron list
```

---

## 12. Recommended next steps for the next model
Priority order:

1. **Do not weaken validator**
   - keep deterministic hard gate intact

2. **Reduce fallback-template dominance in `idea_generator.py`**
   - especially around `RecoveredDeepSeekIdea1`
   - if DeepSeek returns malformed output, recovery should still preserve more structural diversity

3. **Make prompt supervision stricter if needed**
   - force explicit novelty vs prior family examples
   - require a materially different interaction structure
   - require self-critique against validator failure modes

4. **Continue mining for PASS factors without forcing weak deployment**
   - user preference is strict quality, not fake throughput

5. **Consider stronger correlation memory**
   - because deployed factor set is empty, validator multicollinearity is weak in practice
   - one future option is to compare against tested historical families/signals, not only deployed ones

6. **If DeepSeek output continues to be low-yield**
   - investigate whether the official API returns truncated content
   - check `_recover_ideas_from_text()` for lost structural diversity
   - consider increasing `max_tokens` in `_fetch_from_deepseek()`

---

## 13. Summary for the next model
If you are the next model taking over:
- the codebase is real and operational
- the tests pass
- the pipeline runs
- scheduled jobs run
- duplicate-family control and prompt-level supervision have started to exist
- Hermes default brain is now successfully switched to official DeepSeek Pro
- the main unsolved problem is **finding truly good factors**, not rebuilding the architecture
- do not mistake “lots of runs” for “lots of diverse research”; repeated families have historically happened
- your main job is to improve research quality and anti-repeat behavior without loosening the validator

In one sentence:
**Engineering is mostly done, DeepSeek is wired in, and the next critical tasks are research quality, registry backfill, and turning repeated low-yield discover runs into truly novel high-quality factor candidates.**
