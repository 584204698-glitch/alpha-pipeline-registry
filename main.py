from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_engine import BacktestEngine
from code_generator import FactorCodeGenerator
from data_ingestion import build_live_dataset
from deployer import Deployer
from health_check import run_health_check_once
from idea_generator import fetch_factor_ideas
from idea_registry import append_registry_entries
from pipeline_utils import get_logger, read_json, write_json
from runtime_env import load_runtime_env
from validator import OverfitValidator


def bootstrap_environment(project_root: Path) -> Path:
    project_root = Path(project_root)
    (project_root / "config").mkdir(parents=True, exist_ok=True)
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "factors").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    main_config = project_root / "config" / "main_config.json"
    production = project_root / "config" / "production_factors.json"
    init_file = project_root / "factors" / "__init__.py"
    data_path = project_root / "data" / "data_storage.parquet"

    if not main_config.exists():
        write_json(
            main_config,
            {
                "deepseek_api_key_env": "DEEPSEEK_API_KEY",
                "default_mode": "discover",
                "default_batch_size": 10,
                "cpu_usage_cap": 0.7,
                "memory_cap_gb": 100,
                "coinglass_api_key_env": "COINGLASS_API_KEY",
                "live_symbol_limit": 200,
                "live_history_limit": 200,
                "live_interval": "1h",
                "live_pause_seconds": 0.0,
                "live_exchanges": ["Binance", "OKX"],
                "data_path": str(data_path),
            },
        )
    if not production.exists():
        write_json(production, {"factors": []})
    if not init_file.exists():
        init_file.write_text("\n", encoding="utf-8")
    if not data_path.exists():
        _bootstrap_sample_data(data_path)
    return data_path


def _bootstrap_sample_data(data_path: Path) -> None:
    rng = np.random.default_rng(42)
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    timestamps = pd.date_range("2025-01-01", periods=220, freq="4h", tz="UTC")
    frames = []
    for idx, symbol in enumerate(symbols):
        base = 100 + idx * 20
        trend = np.linspace(0, 15, len(timestamps))
        noise = rng.normal(0, 1.2, len(timestamps)).cumsum()
        close = base + trend + noise
        open_ = close + rng.normal(0, 0.3, len(timestamps))
        high = np.maximum(open_, close) + rng.uniform(0.1, 1.5, len(timestamps))
        low = np.minimum(open_, close) - rng.uniform(0.1, 1.5, len(timestamps))
        volume = 1000 + rng.normal(0, 120, len(timestamps)) + idx * 50
        taker_volume = volume * (0.45 + 0.1 * np.sin(np.linspace(0, 8, len(timestamps))))
        funding_rate = rng.normal(0, 0.0008, len(timestamps)) + np.sin(np.linspace(0, 10, len(timestamps))) * 0.0004
        open_interest = 5000 + trend * 40 + rng.normal(0, 70, len(timestamps)).cumsum()
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": symbol,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.clip(volume, 50, None),
                "taker_volume": np.clip(taker_volume, 20, None),
                "funding_rate": funding_rate,
                "open_interest": np.clip(open_interest, 1000, None),
            }
        )
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True).set_index(["timestamp", "symbol"]).sort_index()
    data.to_parquet(data_path)


def build_failure_feedback_record(processed_item: dict[str, Any]) -> dict[str, Any] | None:
    factor_name = (
        processed_item.get("idea", {}).get("factor_name")
        or processed_item.get("backtest_results", {}).get("factor_name")
        or "unknown"
    )
    error_text = processed_item.get("error")
    if error_text:
        return {
            "factor_name": factor_name,
            "failure_reason": "runtime_error",
            "failed_checks": ["runtime_error"],
            "metrics_snapshot": {},
            "error_excerpt": str(error_text)[:240],
        }

    validation = processed_item.get("validation") or {}
    if validation.get("verdict") != "FAIL":
        return None

    checks = validation.get("checks") or {}
    failed_checks = [name for name, passed in checks.items() if passed is False]
    backtest = processed_item.get("backtest_results") or {}
    metrics = backtest.get("metrics") or {}
    signal_summary = backtest.get("signal_summary") or {}
    return {
        "factor_name": factor_name,
        "failure_reason": "validator_fail",
        "failed_checks": failed_checks,
        "metrics_snapshot": {
            "overall_ic": metrics.get("overall_ic"),
            "out_of_sample_icir": metrics.get("out_of_sample_icir"),
            "long_ratio": signal_summary.get("long_ratio"),
        },
    }


class PipelineRunner:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.logger = get_logger(self.project_root, "MAIN")
        bootstrap_environment(self.project_root)
        self.main_config = read_json(self.project_root / "config" / "main_config.json", default={}) or {}
        self.production_path = self.project_root / "config" / "production_factors.json"

    def process_idea(self, idea: dict[str, Any]) -> dict[str, Any]:
        generator = FactorCodeGenerator(self.project_root)
        engine = BacktestEngine(self.project_root)
        deployer = Deployer(self.production_path)

        factor_path = generator.generate_factor_file(idea)
        results = engine.run_backtest(factor_path)
        validator = OverfitValidator(
            current_factor_code=factor_path.read_text(encoding="utf-8"),
            backtest_results=results,
            factor_registry_path=str(self.production_path),
            project_root=self.project_root,
            factor_path=factor_path,
        )
        verdict = validator.validate_all()
        deployed = False
        if verdict["verdict"] == "PASS":
            deployer.deploy(results)
            deployed = True
        return {"idea": idea, "backtest_results": results, "validation": verdict, "deployed": deployed}

    def discover(self, batch_size: int) -> dict[str, Any]:
        production = read_json(self.production_path, default={"factors": []}) or {"factors": []}
        failed_summary = [
            {"factor_name": item.get("factor_name"), "failure_reason": item.get("last_failure_reason", "historical rejection")}
            for item in production.get("factors", [])
        ]
        history_path = self.project_root / "logs" / "autodiscover_last_summary.json"
        historical = read_json(history_path, default={}) or {}
        for processed_item in historical.get("processed", []):
            record = build_failure_feedback_record(processed_item)
            if record:
                failed_summary.append(record)

        deepseek_key = os.environ.get(self.main_config.get("deepseek_api_key_env", "DEEPSEEK_API_KEY"), "")
        generated = fetch_factor_ideas(api_key=deepseek_key, failed_factors_summary=failed_summary, batch_size=batch_size)
        raw_ideas = list(generated.get("ideas", []))
        skipped_duplicates = list(generated.get("skipped_duplicates", []))

        fresh_ideas = raw_ideas
        if fresh_ideas:
            append_registry_entries(self.project_root, fresh_ideas, status="queued", reason="awaiting_backtest")
        if skipped_duplicates:
            self.logger.info(
                f"Skipped {len(skipped_duplicates)} duplicate idea families: "
                + ", ".join(item.get("factor_name", "unknown") for item in skipped_duplicates)
            )

        processed = []
        for idea in fresh_ideas:
            try:
                processed_item = self.process_idea(idea)
            except Exception as exc:  # noqa: BLE001
                self.logger.error(
                    f"Factor [{idea.get('factor_name', 'unknown')}] crashed: {exc}\n{traceback.format_exc()}"
                )
                processed_item = {"idea": idea, "error": str(exc), "deployed": False}
            processed.append(processed_item)

            if processed_item.get("deployed"):
                append_registry_entries(self.project_root, [idea], status="deployed", reason="validator_pass")
            else:
                failure_record = build_failure_feedback_record(processed_item)
                append_registry_entries(
                    self.project_root,
                    [idea],
                    status="failed",
                    reason=(failure_record or {}).get("failure_reason", "not_deployed"),
                )

        summary = {
            "mode": "discover",
            "provider": generated.get("provider"),
            "ideas_requested": batch_size,
            "ideas_returned": len(raw_ideas),
            "ideas_after_dedupe": len(fresh_ideas),
            "skipped_duplicates": skipped_duplicates,
            "processed": processed,
            "deployed_count": sum(1 for item in processed if item.get("deployed")),
        }
        self.logger.info(
            "Discovery run complete. "
            f"ideas_returned={summary['ideas_returned']} ideas_after_dedupe={summary['ideas_after_dedupe']} "
            f"deployed_count={summary['deployed_count']}"
        )
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated alpha factor discovery pipeline")
    parser.add_argument("--mode", default="discover", choices=["discover", "health-check"])
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--live-data", action="store_true", help="Refresh data/data_storage.parquet from Coinglass before discover")
    parser.add_argument("--symbol-limit", type=int, default=None)
    parser.add_argument("--history-limit", type=int, default=None)
    parser.add_argument("--interval", type=str, default=None)
    parser.add_argument("--live-pause-seconds", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    load_runtime_env(root / ".env.runtime")
    bootstrap_environment(root)
    if args.mode == "health-check":
        report = run_health_check_once(root)
        print(report)
        return

    runner = PipelineRunner(root)
    live_summary: dict[str, Any] | None = None
    if args.live_data:
        config = runner.main_config
        coinglass_env = config.get("coinglass_api_key_env", "COINGLASS_API_KEY")
        coinglass_key = os.environ.get(coinglass_env, "")
        if not coinglass_key:
            raise RuntimeError(f"Missing required environment variable [{coinglass_env}] for live data refresh")
        exchanges = tuple(config.get("live_exchanges", ["Binance", "OKX"]))
        _, live_summary = build_live_dataset(
            project_root=root,
            api_key=coinglass_key,
            exchanges=exchanges,
            symbol_limit=args.symbol_limit or int(config.get("live_symbol_limit", 200)),
            interval=args.interval or str(config.get("live_interval", "1h")),
            history_limit=args.history_limit or int(config.get("live_history_limit", 200)),
            pause_seconds=args.live_pause_seconds if args.live_pause_seconds is not None else float(config.get("live_pause_seconds", 0.0)),
        )
    result = runner.discover(batch_size=args.batch_size)
    if live_summary is not None:
        result["live_data"] = live_summary
    print(result)


if __name__ == "__main__":
    main()
