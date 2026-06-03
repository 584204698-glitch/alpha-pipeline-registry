# Alpha Pipeline Implementation Plan

> **For Hermes:** implement directly in this session with strict TDD where practical.

**Goal:** Build a fully local, directly executable crypto perpetual alpha-factor discovery, codegen, backtest, validation, deployment, and health-audit pipeline.

**Architecture:** Create a self-contained Python project under `alpha_pipeline/` with JSON config, generated factor modules, a vectorized backtest engine, deterministic overfit validator, deployment gate, and a background audit script. Use a synthetic bootstrap parquet dataset when no real parquet exists so the pipeline can be executed and verified immediately.

**Tech Stack:** Python 3.12, pandas, numpy, joblib, pyarrow, requests, unittest.

---

### Task 1: Scaffold tests and dependencies
- Create `alpha_pipeline/requirements.txt`
- Create `alpha_pipeline/tests/test_pipeline.py`
- Run the tests first and observe failure before implementation

### Task 2: Implement shared factor foundation
- Create `alpha_pipeline/factors/base.py`
- Create `alpha_pipeline/factors/__init__.py`
- Implement formula evaluation, factor registry base class, and factor discovery helpers

### Task 3: Implement pipeline modules
- Create `idea_generator.py`, `code_generator.py`, `backtest_engine.py`, `validator.py`, `deployer.py`, `health_check.py`, `main.py`
- Create configs and example factor file

### Task 4: Create executable sample environment
- Install dependencies in a local virtualenv
- Bootstrap sample parquet data
- Run unit tests and end-to-end pipeline execution

### Task 5: Verify artifacts and summarize limitations
- Confirm logs, configs, factor files, and reports are generated
- Summarize real execution results and known assumptions
