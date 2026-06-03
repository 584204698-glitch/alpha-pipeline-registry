from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_engine import BacktestEngine
from factors import load_factor_from_path
from pipeline_utils import get_logger, read_json, safe_float, utc_now_z

# Import new statistical inference modules
from research.statistics import factor_statistical_report
from research.multiple_testing import factor_pvalue_from_ic


class OverfitValidator:
    def __init__(
        self,
        current_factor_code: str,
        backtest_results: dict,
        factor_registry_path: str,
        project_root: Path | None = None,
        factor_path: Path | None = None,
    ):
        self.code = current_factor_code
        self.results = backtest_results
        self.registry_path = factor_registry_path
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parent
        self.factor_path = Path(factor_path) if factor_path else self.project_root / self.results.get("factor_file", "")
        self.logger = get_logger(self.project_root, "VALIDATOR")

    # ── 6 hard checks (unchanged) ──────────────────────────────

    def _look_ahead_bias_check(self) -> bool:
        forbidden_tokens = [".shift(-", "shift(periods=-", "iloc[i+", "timestamp +"]
        if any(token in self.code for token in forbidden_tokens):
            return False
        tree = ast.parse(self.code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
                if node.args and isinstance(node.args[0], ast.UnaryOp) and isinstance(node.args[0].op, ast.USub):
                    return False
                for kw in node.keywords:
                    if kw.arg == "periods" and isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.op, ast.USub):
                        return False
        return True

    def _oos_consistency_check(self) -> bool:
        metrics = self.results.get("metrics", {})
        is_icir = safe_float(metrics.get("in_sample_icir"))
        oos_icir = safe_float(metrics.get("out_of_sample_icir"))
        return oos_icir > 0.15 and (is_icir - oos_icir) < 0.3

    def _extract_numeric_parameters(self) -> dict[str, float]:
        params = self.results.get("parameters", {}) or {}
        return {
            key: float(value)
            for key, value in params.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    def _parameter_stability_check(self) -> bool:
        if not self.factor_path.exists():
            return False
        base_ic = safe_float(self.results.get("metrics", {}).get("overall_ic"))
        numeric_params = self._extract_numeric_parameters()
        if not numeric_params:
            return True
        data = pd.read_parquet(self.project_root / "data" / "data_storage.parquet")
        if not isinstance(data.index, pd.MultiIndex):
            data = data.set_index(["timestamp", "symbol"]).sort_index()
        loaded = load_factor_from_path(self.factor_path)
        for multiplier in (0.7, 1.3):
            mutated = dict(loaded.factor.parameters)
            for key, value in numeric_params.items():
                candidate = value * multiplier
                mutated[key] = max(1, int(round(candidate))) if float(value).is_integer() else candidate
            test_factor = loaded.factor.__class__(parameters=mutated)
            signal = test_factor.compute(data).reindex(data.index).fillna(0.0)
            forward_returns = BacktestEngine._forward_returns(data)
            trial_ic = safe_float(signal.corr(forward_returns, method="spearman"))
            if base_ic == 0:
                continue
            if trial_ic == 0 or (trial_ic > 0) != (base_ic > 0):
                return False
        return True

    def _market_regime_invariance_check(self) -> bool:
        regime_ic = self.results.get("metrics", {}).get("regime_ic", {}) or {}
        positive = sum(1 for value in regime_ic.values() if safe_float(value) > 0)
        return positive >= 2

    def _multicollinearity_check(self) -> bool:
        correlations = self.results.get("correlations", {}) or {}
        for _, pair in correlations.items():
            pearson = abs(safe_float(pair.get("pearson")))
            spearman = abs(safe_float(pair.get("spearman")))
            if pearson > 0.5 or spearman > 0.5:
                return False
        return True

    def _signal_symmetry_check(self) -> bool:
        long_ratio = safe_float(self.results.get("signal_summary", {}).get("long_ratio"))
        return 0.30 <= long_ratio <= 0.70

    # ── Statistical inference (NEW) ──────────────────────────

    def _generate_statistical_report(self) -> dict:
        """Generate Newey-West, bootstrap CI, positive IC ratio report."""
        ic_series_data = self.results.get("ic_series", {})
        daily_ic = ic_series_data.get("daily_ic", [])
        oos_ic = ic_series_data.get("oos_daily_ic", [])

        name = self.results.get("factor_name", "unknown")
        report = factor_statistical_report(
            np.array(daily_ic, dtype=float),
            factor_name=name,
        )
        # Add OOS-specific stats
        if oos_ic:
            oos_arr = np.array(oos_ic, dtype=float)
            oos_arr = oos_arr[~np.isnan(oos_arr)]
            if len(oos_arr) > 1:
                report["oos_mean_ic"] = float(oos_arr.mean())
                report["oos_positive_ratio"] = float((oos_arr > 0).mean())
        return report

    @staticmethod
    def _stat_pass(stat_report: dict) -> bool:
        """Check if statistical report meets RESEARCH_PASS threshold."""
        return stat_report.get("decision", "FAIL") in ("RESEARCH_PASS", "PAPER_CANDIDATE")

    # ── Main validation pipeline ──────────────────────────────

    def validate_all(self) -> dict:
        checks = {
            "look_ahead_bias_free": False,
            "oos_consistency_pass": False,
            "parameter_stability_pass": False,
            "market_regime_invariance_pass": False,
            "correlation_limit_pass": False,
            "signal_symmetry_pass": False,
        }

        ordered_checks = [
            ("look_ahead_bias_free", self._look_ahead_bias_check),
            ("oos_consistency_pass", self._oos_consistency_check),
            ("parameter_stability_pass", self._parameter_stability_check),
            ("market_regime_invariance_pass", self._market_regime_invariance_check),
            ("correlation_limit_pass", self._multicollinearity_check),
            ("signal_symmetry_pass", self._signal_symmetry_check),
        ]

        # Phase 1: Hard gate checks (single-veto)
        verdict = "PASS"
        for check_name, func in ordered_checks:
            passed = bool(func())
            checks[check_name] = passed
            if not passed:
                verdict = "FAIL"
                self.logger.info(
                    f"Factor [{self.results.get('factor_name', 'unknown')}] failed [{check_name}]. Rejecting deployment."
                )
                break

        # Phase 2: Statistical inference (NEW — always run, even on FAIL)
        stat_report = self._generate_statistical_report()
        stat_pass = self._stat_pass(stat_report)

        # Phase 3: Final decision
        factor_name = self.results.get("factor_name", "unknown")
        decision = "FAIL"
        if verdict == "PASS":
            if stat_report.get("decision") == "PAPER_CANDIDATE":
                decision = "PAPER_CANDIDATE"
            elif stat_pass:
                decision = "RESEARCH_PASS"
            else:
                decision = "PASS"  # Hard gates pass, but stats borderline

        # Write factor_report.json
        report_path = self.project_root / "logs" / f"factor_report_{factor_name}.json"
        full_report = {
            "factor_name": factor_name,
            "timestamp": utc_now_z(),
            "verdict": verdict,
            "decision": decision,
            "hard_checks": checks,
            "statistics": stat_report,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(full_report, indent=2, default=str), encoding="utf-8")

        payload = {
            "factor_name": factor_name,
            "timestamp": utc_now_z(),
            "verdict": verdict,
            "decision": decision,
            "checks": checks,
            "statistics": stat_report,
            "report_path": str(report_path),
        }
        return payload
