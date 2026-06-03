import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from idea_generator import fetch_factor_ideas
from code_generator import FactorCodeGenerator
from backtest_engine import BacktestEngine
from deployer import Deployer
from health_check import run_health_check_once
from main import bootstrap_environment, build_failure_feedback_record
from validator import OverfitValidator


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "logs" / "tested_idea_families.json"
registry_path = REGISTRY_PATH


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        bootstrap_environment(cls.project_root)
        cls.data_path = cls.project_root / "data" / "data_storage.parquet"
        cls.production_path = cls.project_root / "config" / "production_factors.json"
        cls.main_config = cls.project_root / "config" / "main_config.json"

    def setUp(self):
        REGISTRY_PATH.write_text(json.dumps({"ideas": []}, indent=2), encoding="utf-8")

    def test_fetch_factor_ideas_schema(self):
        ideas = fetch_factor_ideas("demo", [{"factor_name": "bad", "failure_reason": "unstable"}], batch_size=2)
        self.assertEqual(len(ideas["ideas"]), 2)
        first = ideas["ideas"][0]
        self.assertIn("factor_name", first)
        self.assertIn("parameters", first)

    def test_fetch_factor_ideas_uses_deepseek_payload_when_api_key_present(self):
        api_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "ideas": [
                                    {
                                        "factor_name": "ApiFactorOne",
                                        "rationale": "api generated",
                                        "timeframes": ["15m", "2h"],
                                        "inputs": ["close", "open_interest", "funding_rate"],
                                        "mathematical_formula": "zscore(delta(open_interest, 4), 12)",
                                        "parameters": {"oi_lookback": 4, "window": 12},
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        with patch("requests.post", return_value=Mock(status_code=200, json=lambda: api_payload, text=json.dumps(api_payload))) as mock_post:
            ideas = fetch_factor_ideas("live-key", [{"factor_name": "bad", "failure_reason": "unstable"}], batch_size=1)
        self.assertEqual(ideas["provider"], "deepseek")
        self.assertEqual(ideas["ideas"][0]["factor_name"], "ApiFactorOne")
        self.assertTrue(mock_post.called)

    def test_fetch_factor_ideas_falls_back_when_deepseek_formula_is_not_expression(self):
        api_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "ideas": [
                                    {
                                        "factor_name": "BadApiFactor",
                                        "rationale": "bad formula",
                                        "timeframes": ["15m", "2h"],
                                        "inputs": ["close", "volume"],
                                        "mathematical_formula": "tmp = zscore(close, 20); factor = tmp * -1",
                                        "parameters": {"window": 20},
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        registry_path.write_text(
            json.dumps({"ideas": []}, indent=2), encoding="utf-8"
        )
        with patch("requests.post", return_value=Mock(status_code=200, json=lambda: api_payload, text=json.dumps(api_payload))):
            ideas = fetch_factor_ideas("live-key", [{"factor_name": "bad", "failure_reason": "unstable"}], batch_size=1)
        self.assertEqual(ideas["provider"], "deepseek")
        self.assertNotIn(";", ideas["ideas"][0]["mathematical_formula"])
        self.assertEqual(ideas["skipped_duplicates"], [])
        self.assertNotEqual(ideas["ideas"][0]["factor_name"], "BadApiFactor")

    def test_fetch_factor_ideas_falls_back_when_deepseek_uses_unknown_functions(self):
        api_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "ideas": [
                                    {
                                        "factor_name": "UnknownFunctionFactor",
                                        "rationale": "bad formula",
                                        "timeframes": ["15m", "2h"],
                                        "inputs": ["close", "funding_rate", "volume"],
                                        "mathematical_formula": "Zscore(ema(funding_rate, fast_period) - ema(funding_rate, slow_period)) * sign(roc(close, roc_period)) * zscore(volume, vol_period)",
                                        "parameters": {"fast_period": 3, "slow_period": 10, "roc_period": 5, "vol_period": 20},
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        registry_path.write_text(
            json.dumps({"ideas": []}, indent=2), encoding="utf-8"
        )
        with patch("requests.post", return_value=Mock(status_code=200, json=lambda: api_payload, text=json.dumps(api_payload))):
            ideas = fetch_factor_ideas("live-key", [{"factor_name": "bad", "failure_reason": "unstable"}], batch_size=1)
        self.assertEqual(ideas["provider"], "deepseek")
        self.assertNotEqual(ideas["ideas"][0]["factor_name"], "UnknownFunctionFactor")

    def test_fetch_factor_ideas_skips_duplicate_families_from_registry(self):
        api_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "ideas": [
                                    {
                                        "factor_name": "MomentumPressureA",
                                        "rationale": "dup",
                                        "timeframes": ["15m", "1h", "4h"],
                                        "inputs": ["close", "volume", "open_interest"],
                                        "mathematical_formula": "zscore(pct_change(close, 4), 12) * zscore(delta(open_interest, 4), 12)",
                                        "parameters": {"lag": 4, "window": 12},
                                    },
                                    {
                                        "factor_name": "FundingFlowDivergence",
                                        "rationale": "unique",
                                        "timeframes": ["15m", "1h", "4h"],
                                        "inputs": ["taker_volume", "funding_rate", "close"],
                                        "mathematical_formula": "zscore(delta(taker_volume, 3), 12) * indicator(funding_rate > rolling_quantile(funding_rate, 24, 0.8)) * -1",
                                        "parameters": {"lag": 3, "window": 12, "quantile_window": 24, "funding_quantile": 0.8},
                                    },
                                ]
                            }
                        )
                    }
                }
            ]
        }
        registry_path = self.project_root / "logs" / "tested_idea_families.json"
        registry_path.write_text(
            json.dumps(
                {
                    "ideas": [
                        {
                            "family_fingerprint": importlib.import_module("idea_registry").idea_family_fingerprint(
                                {
                                    "factor_name": "OldMomentumPressure",
                                    "timeframes": ["15m", "1h", "4h"],
                                    "inputs": ["close", "volume", "open_interest"],
                                    "mathematical_formula": "zscore(pct_change(close, 9), 20) * zscore(delta(open_interest, 9), 20)",
                                    "parameters": {"lag": 9, "window": 20},
                                }
                            ),
                            "status": "failed",
                        }
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with patch("requests.post", return_value=Mock(status_code=200, json=lambda: api_payload, text=json.dumps(api_payload))):
            ideas = fetch_factor_ideas("live-key", [{"factor_name": "bad", "failure_reason": "unstable"}], batch_size=2)
        self.assertEqual([item["factor_name"] for item in ideas["ideas"]], ["FundingFlowDivergence"])
        self.assertEqual(len(ideas["skipped_duplicates"]), 1)
        self.assertEqual(ideas["skipped_duplicates"][0]["reason"], "duplicate_family")

    def test_fetch_factor_ideas_recovers_truncated_json_using_reasoning_content(self):
        api_payload = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": '{"ideas":[{"factor_name":"RecoveredFactor","rationale":"ok","timeframes":"15m/2h/4h","inputs":"close,volume,open_interest","mathematical_formula":"zscore(delta(open_interest, 4), 12)","parameters":{"lag":4,"window":12}}]'
                    }
                }
            ]
        }
        registry_path.write_text(
            json.dumps({"ideas": []}, indent=2), encoding="utf-8"
        )
        with patch("requests.post", return_value=Mock(status_code=200, json=lambda: api_payload, text=json.dumps(api_payload))):
            ideas = fetch_factor_ideas("live-key", [{"factor_name": "bad", "failure_reason": "unstable"}], batch_size=1)
        self.assertEqual(ideas["provider"], "deepseek")
        self.assertEqual(ideas["ideas"][0]["factor_name"], "RecoveredFactor")
        self.assertEqual(ideas["ideas"][0]["timeframes"], ["15m", "2h", "4h"])
        self.assertEqual(ideas["ideas"][0]["inputs"], ["close", "volume", "open_interest"])

    def test_fetch_factor_ideas_recovers_from_semistructured_text(self):
        api_payload = {
            "choices": [
                {
                    "message": {
                        "content": "factor_name: FundingPressureRecovery\nrationale: combine funding and oi\ntimeframes: 15m/2h/4h\ninputs: close, volume, open_interest\nmathematical_formula: zscore(delta(open_interest, 4), 12)\nparameters: {\"lag\": 4, \"window\": 12}",
                        "reasoning_content": ""
                    }
                }
            ]
        }
        registry_path.write_text(
            json.dumps({"ideas": []}, indent=2), encoding="utf-8"
        )
        with patch("requests.post", return_value=Mock(status_code=200, json=lambda: api_payload, text=json.dumps(api_payload))):
            ideas = fetch_factor_ideas("live-key", [{"factor_name": "bad", "failure_reason": "unstable"}], batch_size=1)
        self.assertEqual(ideas["provider"], "deepseek")
        self.assertEqual(ideas["ideas"][0]["factor_name"], "FundingPressureRecovery")
        self.assertEqual(ideas["ideas"][0]["timeframes"], ["15m", "2h", "4h"])
        self.assertEqual(ideas["ideas"][0]["inputs"], ["close", "volume", "open_interest"])

    def test_fetch_factor_ideas_sends_failure_evidence_and_preserves_hypothesis_fields(self):
        api_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "ideas": [
                                    {
                                        "factor_name": "EvidenceAwareFactor",
                                        "rationale": "Built to avoid prior OOS/regime failures using more balanced signal geometry.",
                                        "hypothesis": "When taker pressure spikes against macro trend and funding remains crowded, reversal odds improve.",
                                        "observation": "Past failures were too one-sided and fragile in low-volatility regimes.",
                                        "novelty_claim": "Combines taker imbalance with funding crowding and multi-horizon contradiction.",
                                        "risk_notes": "Could fail during strong trend continuation.",
                                        "timeframes": ["15m", "1h", "4h"],
                                        "inputs": ["close", "taker_volume", "funding_rate", "open_interest"],
                                        "mathematical_formula": "zscore(delta(taker_volume, 3), 12) * indicator(funding_rate > rolling_quantile(funding_rate, 24, 0.8)) * -1",
                                        "parameters": {"lag": 3, "window": 12, "quantile_window": 24, "funding_quantile": 0.8}
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        failed = [{
            "factor_name": "OldBadFactor",
            "failure_reason": "validator_fail",
            "failed_checks": ["oos_consistency_pass", "market_regime_invariance_pass"],
            "metrics_snapshot": {"overall_ic": 0.01, "out_of_sample_icir": 0.02, "long_ratio": 0.91},
        }]
        registry_path = self.project_root / "logs" / "tested_idea_families.json"
        registry_path.write_text(
            json.dumps(
                {
                    "ideas": [
                        {
                            "factor_name": "DuplicateFamilyA",
                            "family_fingerprint": "abc",
                            "status": "failed",
                            "reason": "duplicate_family",
                            "formula": "zscore(delta(open_interest, 4), 12) * indicator(funding_rate < rolling_quantile(funding_rate, 24, 0.2))",
                        }
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with patch("requests.post", return_value=Mock(status_code=200, json=lambda: api_payload, text=json.dumps(api_payload))) as mock_post:
            ideas = fetch_factor_ideas("live-key", failed, batch_size=1)
        prompt = mock_post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertIn("oos_consistency_pass", prompt)
        self.assertIn("long_ratio=0.91", prompt)
        self.assertIn("duplicate_family", prompt)
        self.assertIn("Hard originality rule", prompt)
        self.assertIn("DuplicateFamilyA", prompt)
        first = ideas["ideas"][0]
        self.assertEqual(first["factor_name"], "EvidenceAwareFactor")
        self.assertIn("hypothesis", first)
        self.assertIn("novelty_claim", first)
        self.assertIn("risk_notes", first)

    def test_build_failure_feedback_record_captures_failed_checks_and_metrics(self):
        processed = {
            "idea": {"factor_name": "FragileFactor"},
            "backtest_results": {
                "metrics": {
                    "overall_ic": 0.01,
                    "out_of_sample_icir": 0.04,
                },
                "signal_summary": {"long_ratio": 0.88},
            },
            "validation": {
                "verdict": "FAIL",
                "checks": {
                    "look_ahead_bias_free": True,
                    "oos_consistency_pass": False,
                    "parameter_stability_pass": False,
                    "market_regime_invariance_pass": True,
                    "correlation_limit_pass": True,
                    "signal_symmetry_pass": False,
                },
            },
            "deployed": False,
        }
        record = build_failure_feedback_record(processed)
        self.assertEqual(record["factor_name"], "FragileFactor")
        self.assertEqual(record["failure_reason"], "validator_fail")
        self.assertIn("oos_consistency_pass", record["failed_checks"])
        self.assertIn("parameter_stability_pass", record["failed_checks"])
        self.assertEqual(record["metrics_snapshot"]["long_ratio"], 0.88)

    def test_build_failure_feedback_record_captures_runtime_errors(self):
        processed = {
            "idea": {"factor_name": "CrashyFactor"},
            "error": "ZeroDivisionError: division by zero",
            "deployed": False,
        }
        record = build_failure_feedback_record(processed)
        self.assertEqual(record["factor_name"], "CrashyFactor")
        self.assertEqual(record["failure_reason"], "runtime_error")
        self.assertIn("runtime_error", record["failed_checks"])
        self.assertIn("division by zero", record["error_excerpt"])

    def test_idea_registry_family_fingerprint_blocks_renamed_repeats(self):
        idea_registry = importlib.import_module("idea_registry")
        idea_a = {
            "factor_name": "MomentumPressureA",
            "timeframes": ["15m", "1h", "4h"],
            "inputs": ["close", "volume", "open_interest"],
            "mathematical_formula": "zscore(pct_change(close, 4), 12) * zscore(delta(open_interest, 4), 12)",
            "parameters": {"lag": 4, "window": 12},
        }
        idea_b = {
            "factor_name": "MacroMicroConflictB",
            "timeframes": ["4h", "1h", "15m"],
            "inputs": ["open_interest", "volume", "close"],
            "mathematical_formula": "zscore(pct_change(close, 9), 20) * zscore(delta(open_interest, 9), 20)",
            "parameters": {"lag": 9, "window": 20},
        }
        self.assertEqual(
            idea_registry.idea_family_fingerprint(idea_a),
            idea_registry.idea_family_fingerprint(idea_b),
        )

    def test_idea_registry_family_fingerprint_is_commutative_for_same_interaction(self):
        idea_registry = importlib.import_module("idea_registry")
        idea_a = {
            "factor_name": "FundingCrowdingA",
            "timeframes": ["15m", "1h", "4h"],
            "inputs": ["funding_rate", "open_interest"],
            "mathematical_formula": "zscore(delta(open_interest, lag), window) * indicator(funding_rate < rolling_quantile(funding_rate, quantile_window, funding_quantile))",
            "parameters": {"lag": 4, "window": 12, "quantile_window": 24, "funding_quantile": 0.2},
        }
        idea_b = {
            "factor_name": "FundingCrowdingB",
            "timeframes": ["4h", "1h", "15m"],
            "inputs": ["open_interest", "funding_rate"],
            "mathematical_formula": "indicator(funding_rate < rolling_quantile(funding_rate, q_window, q_level)) * zscore(delta(open_interest, lookback), look_window)",
            "parameters": {"lookback": 6, "look_window": 18, "q_window": 36, "q_level": 0.25},
        }
        self.assertEqual(
            idea_registry.idea_family_fingerprint(idea_a),
            idea_registry.idea_family_fingerprint(idea_b),
        )

    def test_idea_registry_distinguishes_different_interaction_structures(self):
        idea_registry = importlib.import_module("idea_registry")
        additive = {
            "factor_name": "AdditiveCombo",
            "timeframes": ["15m", "1h", "4h"],
            "inputs": ["close", "open_interest"],
            "mathematical_formula": "zscore(pct_change(close, 4), 12) + zscore(delta(open_interest, 4), 12)",
            "parameters": {"lag": 4, "window": 12},
        }
        multiplicative = {
            "factor_name": "MultiplicativeCombo",
            "timeframes": ["15m", "1h", "4h"],
            "inputs": ["close", "open_interest"],
            "mathematical_formula": "zscore(pct_change(close, 4), 12) * zscore(delta(open_interest, 4), 12)",
            "parameters": {"lag": 4, "window": 12},
        }
        self.assertNotEqual(
            idea_registry.idea_family_fingerprint(additive),
            idea_registry.idea_family_fingerprint(multiplicative),
        )

    def test_idea_registry_filters_existing_and_in_batch_duplicates(self):
        idea_registry = importlib.import_module("idea_registry")
        duplicate_a = {
            "factor_name": "MomentumPressureA",
            "timeframes": ["15m", "1h", "4h"],
            "inputs": ["close", "volume", "open_interest"],
            "mathematical_formula": "zscore(pct_change(close, 4), 12) * zscore(delta(open_interest, 4), 12)",
            "parameters": {"lag": 4, "window": 12},
        }
        duplicate_b = {
            "factor_name": "SamePrincipleDifferentName",
            "timeframes": ["15m", "1h", "4h"],
            "inputs": ["close", "volume", "open_interest"],
            "mathematical_formula": "zscore(pct_change(close, 6), 18) * zscore(delta(open_interest, 6), 18)",
            "parameters": {"lag": 6, "window": 18},
        }
        unique = {
            "factor_name": "FundingFlowDivergence",
            "timeframes": ["15m", "1h", "4h"],
            "inputs": ["taker_volume", "funding_rate", "close"],
            "mathematical_formula": "zscore(delta(taker_volume, 3), 12) * indicator(funding_rate > rolling_quantile(funding_rate, 24, 0.8)) * -1",
            "parameters": {"lag": 3, "window": 12, "quantile_window": 24, "funding_quantile": 0.8},
        }
        registry = {
            "ideas": [
                {
                    "family_fingerprint": idea_registry.idea_family_fingerprint(duplicate_a),
                    "status": "failed",
                }
            ]
        }
        fresh, skipped = idea_registry.filter_duplicate_ideas([duplicate_a, duplicate_b, unique], registry)
        self.assertEqual([item["factor_name"] for item in fresh], ["FundingFlowDivergence"])
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all(item["reason"] == "duplicate_family" for item in skipped))

    def test_live_symbol_filter_excludes_stock_tokens_and_limits_count(self):
        data_ingestion = importlib.import_module("data_ingestion")
        symbols = [
            "BTC",
            "ETH",
            "SOL",
            "AAPL",
            "TSLA",
            "AMZN",
            "COIN",
            "NVDA",
            "DOGE",
        ] + [f"ALT{i}" for i in range(300)]
        filtered = data_ingestion.filter_supported_symbols(symbols, limit=200)
        self.assertLessEqual(len(filtered), 200)
        self.assertIn("BTC", filtered)
        self.assertNotIn("AAPL", filtered)
        self.assertNotIn("TSLA", filtered)
        self.assertNotIn("NVDA", filtered)

    def test_generated_factor_and_backtest(self):
        generator = FactorCodeGenerator(self.project_root)
        idea = {
            "factor_name": "TestMomentumBalance",
            "rationale": "test",
            "timeframes": ["15m", "2h"],
            "inputs": ["close", "volume", "taker_volume"],
            "mathematical_formula": "(pct_change(close, 6) * zscore(volume, 8)) - zscore(taker_volume, 8)",
            "parameters": {"window": 8, "lag": 6},
        }
        factor_path = generator.generate_factor_file(idea)
        self.assertTrue(factor_path.exists())

        engine = BacktestEngine(self.project_root)
        results = engine.run_backtest(factor_path)
        self.assertEqual(results["factor_name"], "TestMomentumBalance")
        self.assertIn("overall_ic", results["metrics"])
        self.assertEqual(len(results["metrics"]["quantile_returns"]), 5)

    def test_cost_adjusted_forward_returns_applies_transaction_costs(self):
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2026-01-01", tz="UTC"), "BTCUSDT"),
                (pd.Timestamp("2026-01-02", tz="UTC"), "BTCUSDT"),
                (pd.Timestamp("2026-01-03", tz="UTC"), "BTCUSDT"),
            ],
            names=["timestamp", "symbol"],
        )
        signal = pd.Series([1.0, -1.0, 0.0], index=idx)
        raw_returns = pd.Series([0.01, 0.02, 0.0], index=idx)
        # total_cost_bps = fee(5) + slippage(3) + spread(1) = 9 bps = 0.0009 per side
        adjusted = BacktestEngine._cost_adjusted_forward_returns(signal, raw_returns, total_cost_bps=9.0)
        expected = pd.Series([0.0091, 0.0182, -0.0009], index=idx)
        self.assertTrue(np.allclose(adjusted.values, expected.values))

    def test_validator_output_schema(self):
        generator = FactorCodeGenerator(self.project_root)
        idea = {
            "factor_name": "TestFundingDivergence",
            "rationale": "test",
            "timeframes": ["15m", "4h"],
            "inputs": ["funding_rate", "open_interest", "close"],
            "mathematical_formula": "zscore(delta(open_interest, 4), 12) * indicator(funding_rate < rolling_quantile(funding_rate, 24, 0.2))",
            "parameters": {"oi_lookback": 4, "window": 12, "quantile_window": 24, "threshold": 0.2},
        }
        factor_path = generator.generate_factor_file(idea)
        engine = BacktestEngine(self.project_root)
        results = engine.run_backtest(factor_path)
        validator = OverfitValidator(
            current_factor_code=factor_path.read_text(),
            backtest_results=results,
            factor_registry_path=str(self.production_path),
            project_root=self.project_root,
            factor_path=factor_path,
        )
        verdict = validator.validate_all()
        self.assertIn(verdict["verdict"], {"PASS", "FAIL"})
        self.assertIn("checks", verdict)

    def test_deployer_appends_disabled_factor(self):
        temp_dir = Path(tempfile.mkdtemp())
        prod = temp_dir / "production_factors.json"
        prod.write_text(json.dumps({"factors": []}, indent=2), encoding="utf-8")
        deployer = Deployer(prod)
        payload = {
            "factor_name": "DeployMe",
            "factor_file": "factors/deploy_me.py",
            "parameters": {"x": 1},
            "metrics": {"overall_ic": 0.4, "overall_ir": 0.5},
        }
        deployer.deploy(payload)
        stored = json.loads(prod.read_text(encoding="utf-8"))
        self.assertEqual(stored["factors"][0]["enabled_key"], "FACTOR_DEPLOYME_ENABLED")
        self.assertFalse(stored["factors"][0]["FACTOR_DEPLOYME_ENABLED"])

    def test_health_check_writes_report(self):
        report = run_health_check_once(self.project_root)
        self.assertTrue(report.exists())
        self.assertIn("# Alpha Pipeline Audit Report", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
