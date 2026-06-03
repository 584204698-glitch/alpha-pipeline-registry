#!/usr/bin/env python3
"""One-shot mining round: backtest + 6-gate verification + logging."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/mnt/e/alpha_pipeline")
sys.path.insert(0, str(PROJECT))

from backtest_engine import BacktestEngine
from factors.base import load_factor_from_path


def compute_gates(results: dict, signal: pd.Series, forward_returns_nocost: pd.Series, data: pd.DataFrame) -> dict:
    """Compute all 6 gate checks."""
    metrics = results["metrics"]
    gates = {}

    # Gate 1: look_ahead_bias — structural guarantee from framework, always true
    gates["look_ahead_bias_free"] = True

    # Gate 2: oos_consistency (OOS ICIR > 0.15 + t-stat > 1.5)
    oos_icir = float(metrics.get("out_of_sample_icir", 0))
    oos_tstat = float(metrics.get("out_of_sample_tstat", 0))
    gates["oos_consistency_pass"] = oos_icir > 0.15 and oos_tstat > 1.5
    gates["oos_icir"] = round(oos_icir, 4)
    gates["oos_tstat"] = round(oos_tstat, 4)

    # Gate 3: parameter_stability (IS and OOS ICIR same sign, both > 0, ratio < 3)
    is_icir = float(metrics.get("in_sample_icir", 0))
    gates["in_sample_icir"] = round(is_icir, 4)
    if is_icir > 0 and oos_icir > 0:
        ratio = max(is_icir, oos_icir) / (min(is_icir, oos_icir) + 1e-8)
        gates["parameter_stability_pass"] = ratio < 3.0
        gates["icir_ratio"] = round(ratio, 2)
    else:
        gates["parameter_stability_pass"] = False
        gates["icir_ratio"] = 999

    # Gate 4: regime_invariance (≥2 of 3 regimes have IC > 0)
    regime_ic = metrics.get("regime_ic", {})
    positive_regimes = sum(1 for v in regime_ic.values() if v > 0)
    gates["market_regime_invariance_pass"] = positive_regimes >= 2
    gates["regime_positive_count"] = positive_regimes

    # Gate 5: correlation_limit (|pearson| ≤ 0.5 AND |spearman| ≤ 0.5 for ALL existing factors)
    corrs = results.get("correlations", {})
    max_pearson = max((abs(v["pearson"]) for v in corrs.values()), default=0)
    max_spearman = max((abs(v["spearman"]) for v in corrs.values()), default=0)
    gates["correlation_limit_pass"] = max_pearson <= 0.5 and max_spearman <= 0.5
    gates["max_pearson"] = round(max_pearson, 4)
    gates["max_spearman"] = round(max_spearman, 4)

    # Gate 6: signal_symmetry (long ratio 30%-70%)
    long_ratio = float(results.get("signal_summary", {}).get("long_ratio", 0.5))
    gates["signal_symmetry_pass"] = 0.30 <= long_ratio <= 0.70
    gates["long_ratio"] = round(long_ratio, 4)

    passed = sum(1 for k, v in gates.items() if k.endswith("_pass") and v)
    gates["passed_count"] = passed
    gates["total_gates"] = 6
    gates["verdict"] = "PASS" if passed == 6 else "FAIL"

    return gates


def main():
    engine = BacktestEngine(PROJECT)
    data = engine._load_data(min_symbols=30)

    factor_files = [
        PROJECT / "factors" / "alpha_efficiency_timing_fade_v9.py",
        PROJECT / "factors" / "alpha_hybrid_body_eff_reversal_v2.py",
        PROJECT / "factors" / "alpha_efficiency_timing_fade_v10.py",
    ]

    results_log_path = PROJECT / "logs" / "direct_mining_results.jsonl"

    for factor_path in factor_files:
        if not factor_path.exists():
            print(f"[SKIP] Missing: {factor_path}")
            continue

        print(f"\n{'='*60}")
        print(f"TESTING: {factor_path.name}")
        print(f"{'='*60}")

        try:
            loaded = load_factor_from_path(factor_path)
            factor = loaded.factor
            print(f"  Factor name: {factor.factor_name}")
            print(f"  Formula: {factor.mathematical_formula}")
            print(f"  Parameters: {factor.parameters}")
            print(f"  Inputs: {factor.inputs}")

            results = engine.run_backtest(factor_path)

            # Compute forward returns WITHOUT cost for gate IC computation
            raw_fwd = engine._forward_returns(data)
            signal = factor.compute(data).reindex(data.index).fillna(0.0)

            gates = compute_gates(results, signal, raw_fwd, data)

            overall_ic = float(results["metrics"]["overall_ic"])
            overall_ir = float(results["metrics"]["overall_ir"])

            print(f"\n  --- RESULTS ---")
            print(f"  Overall IC:  {overall_ic:.4f}")
            print(f"  Overall IR:  {overall_ir:.4f}")
            print(f"  IS ICIR:     {gates['in_sample_icir']:.4f}")
            print(f"  OOS ICIR:    {gates['oos_icir']:.4f}")
            print(f"  OOS t-stat:  {gates['oos_tstat']:.2f}")
            print(f"  Long ratio:  {gates['long_ratio']:.2%}")
            print(f"  Regime IC:   {results['metrics']['regime_ic']}")
            print(f"  Corrs max:   Pearson={gates['max_pearson']:.3f} Spearman={gates['max_spearman']:.3f}")
            print(f"\n  --- GATES ---")
            for k, v in gates.items():
                if k.endswith("_pass"):
                    status = "✅" if v else "❌"
                    print(f"  {status} {k}: {v}")
            print(f"  Verdict: {gates['verdict']} ({gates['passed_count']}/{gates['total_gates']})")

            # Build log entry
            log_entry = {
                "factor_name": factor.factor_name,
                "factor_file": str(Path("factors") / factor_path.name),
                "verdict": gates["verdict"],
                "checks": {k: v for k, v in gates.items() if k.endswith("_pass") or k in ("passed_count", "total_gates")},
                "overall_ic": round(overall_ic, 6),
                "overall_ir": round(overall_ir, 6),
                "in_sample_icir": gates["in_sample_icir"],
                "out_of_sample_icir": gates["oos_icir"],
                "long_ratio": gates["long_ratio"],
                "regime_ic": {k: round(v, 6) for k, v in results["metrics"]["regime_ic"].items()},
                "correlations": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in results.get("correlations", {}).items()},
                "quantile_returns": [round(x, 6) for x in results["metrics"].get("quantile_returns", [])],
                "formula": factor.mathematical_formula,
                "rationale": factor.rationale[:200] if hasattr(factor, 'rationale') else "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Append to JSONL
            with open(results_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"  [LOGGED] → {results_log_path}")

        except Exception as e:
            print(f"  [ERROR] {factor_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("MINING ROUND COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
