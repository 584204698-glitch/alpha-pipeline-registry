#!/usr/bin/env python3
"""Paper Trading Death Filter — Full Battery.

Runs paper trading on all 5 production factors across:
- 9 configs (5/10/20% × 1/2/3 bar holds)
- 3 cost levels (9/12/15 bps per side)
- Regime attribution
- Classification: trade_signal / filter_only / kill

Output: logs/paper_trading_battery.json + factor-level reports
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research.paper_trading import PaperConfig, PaperTradingSimulator, run_paper_battery
from research.regime_detector import detect_regimes, regime_summary
from research.factor_ensemble import FactorEnsemble, compute_factor_correlations
from research.registry import FactorRegistry as SharedRegistry
from backtest_engine import BacktestEngine


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "logs" / "paper_trading_battery.json"
REGISTRY_ROOT = PROJECT / "registry"


def format_result_table(results: dict[str, list], regime_dist: dict) -> str:
    """Format paper trading results as a readable table."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("PAPER TRADING DEATH FILTER RESULTS")
    lines.append(f"{'='*80}")
    lines.append(f"\nRegime distribution: {regime_dist.get('distribution', {})}")
    lines.append("")

    # Summary table
    lines.append(f"{'Factor':<25} {'Best Config':<18} {'Net PnL':>10} {'PF':>7} {'Cost/α':>8} {'Class':>14}")
    lines.append("-" * 85)

    classification: dict[str, list[str]] = {"trade_signal": [], "filter_only": [], "kill": []}

    for fname, config_results in results.items():
        if not config_results:
            continue
        # Find best config by net PnL
        best = max(config_results, key=lambda r: r.net_pnl)
        classification[best.classification].append(fname)

        cfg_str = f"{best.config.top_frac:.0%}/{best.config.hold_bars}b"
        cost_ratio = f"{best.cost_gross_ratio:.0%}"
        lines.append(
            f"{fname:<25} {cfg_str:<18} {best.net_pnl:>10.4f} {best.pf:>7.2f} {cost_ratio:>8} {best.classification:>14}"
        )

    lines.append("-" * 85)
    lines.append(f"\ntrade_signal: {len(classification['trade_signal'])} factors")
    for f in classification["trade_signal"]:
        best = max(results[f], key=lambda r: r.net_pnl)
        lines.append(f"  ✓ {f}: PF={best.pf:.2f} Net={best.net_pnl:.4f} Turnover={best.turnover:.1f}/bar")
        lines.append(f"    Long={best.long_pnl:.4f} Short={best.short_pnl:.4f} TailContrib={best.tail_contribution:.1%}")

    lines.append(f"\nfilter_only: {len(classification['filter_only'])} factors")
    for f in classification["filter_only"]:
        best = max(results[f], key=lambda r: r.net_pnl)
        lines.append(f"  ~ {f}: Cost/Gross={best.cost_gross_ratio:.0%} — use as filter, not standalone")

    lines.append(f"\nkill: {len(classification['kill'])} factors")
    for f in classification["kill"]:
        best = max(results[f], key=lambda r: r.net_pnl)
        lines.append(f"  ✗ {f}: Net={best.net_pnl:.4f} PF={best.pf:.2f} — remove from production")
        # Show detail
        for r in results[f]:
            if r.config.top_frac == 0.10 and r.config.hold_bars == 1:
                lines.append(f"    (10%/1b): Gross={r.gross_pnl:.4f} Net={r.net_pnl:.4f} PF={r.pf:.2f}")
                break

    lines.append(f"\n{'='*80}")
    return "\n".join(lines)


def main():
    print("Loading data and detecting regimes...")
    engine = BacktestEngine(PROJECT)
    data = engine._load_data()

    # Detect regimes
    regimes = detect_regimes(data)
    r_summary = regime_summary(regimes)
    print(f"Regimes: {r_summary['distribution']}")

    # Identify production factors
    prod = json.loads((PROJECT / "config" / "production_factors.json").read_text())
    factor_names = [f["factor_name"] for f in prod.get("factors", [])]
    print(f"Production factors: {factor_names}")

    # Run paper trading battery
    print("\nRunning paper trading battery (9 configs × 5 factors)...")
    results = run_paper_battery(PROJECT, factor_names=factor_names, regime_labels=regimes)

    # Format and print
    report = format_result_table(results, r_summary)
    print(report)

    # Save JSON
    output: dict[str, Any] = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "regime_summary": {str(k): v for k, v in r_summary.items()},
        "factors": {},
    }
    for fname, config_results in results.items():
        output["factors"][fname] = [
            {
                "config": {
                    "top_frac": r.config.top_frac,
                    "bottom_frac": r.config.bottom_frac,
                    "hold_bars": r.config.hold_bars,
                    "round_trip_cost_bps": r.config.round_trip_cost_bps * 10000,
                },
                "gross_pnl": r.gross_pnl,
                "net_pnl": r.net_pnl,
                "cost_total": r.cost_total,
                "cost_gross_ratio": r.cost_gross_ratio,
                "turnover": r.turnover,
                "pf": r.pf,
                "long_pnl": r.long_pnl,
                "short_pnl": r.short_pnl,
                "n_trades": r.n_trades,
                "n_bars_traded": r.n_bars_traded,
                "regime_pnl": {str(k): v for k, v in r.regime_pnl.items()},
                "tail_contribution": r.tail_contribution,
                "classification": r.classification,
            }
            for r in config_results
        ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved to {OUTPUT}")

    # Factor ensemble weights
    print("\nComputing factor ensemble weights...")
    ensemble = FactorEnsemble(PROJECT)
    factors_dir = PROJECT / "factors"
    for fname in factor_names:
        fname_clean = fname.lower().replace("_", "")
        for py_file in factors_dir.glob("alpha_*.py"):
            if fname_clean in py_file.stem.lower().replace("_", ""):
                ensemble.add_factor(py_file)
                break

    weights = ensemble.compute_weights()
    corr_matrix = compute_factor_correlations(PROJECT, factor_names)
    print(f"Weights: {weights}")
    print(f"\nCorrelation matrix:\n{corr_matrix.to_string()}")

    # Save ensemble config
    ensemble_out = {
        "weights": {str(k): float(v) for k, v in weights.items()},
        "correlation_matrix": corr_matrix.to_dict(),
    }
    (PROJECT / "logs" / "ensemble_weights.json").write_text(
        json.dumps(ensemble_out, indent=2, default=str)
    )
    print("Saved ensemble weights to logs/ensemble_weights.json")

    # ── Registry integration ──────────────────────────────────
    registry = SharedRegistry(REGISTRY_ROOT)

    # Register all factors and submit proposals for paper_pass / trade_signal
    for fname in factor_names:
        # Find factor file and parameters
        factor_file = ""
        params = {}
        inputs_list = []
        fname_clean = fname.lower().replace("_", "")
        for py_file in factors_dir.glob("alpha_*.py"):
            if fname_clean in py_file.stem.lower().replace("_", ""):
                factor_file = str(py_file.relative_to(PROJECT))
                from factors import load_factor_from_path
                loaded = load_factor_from_path(py_file)
                params = dict(loaded.factor.parameters)
                inputs_list = loaded.factor.get_required_data()
                break

        if not factor_file:
            continue

        # Register
        reg_result = registry.register_factor(
            factor_id=fname,
            factor_file=factor_file,
            parameters=params,
            inputs=inputs_list,
            who="research_hermes",
        )

        if reg_result.get("error") and "already exists" not in reg_result["error"]:
            print(f"  Register {fname}: {reg_result['error']}")
            continue

        # Update status based on classification
        if fname in results:
            best = max(results[fname], key=lambda r: r.net_pnl)
            cls = best.classification

            if cls == "trade_signal":
                registry.update_status(fname, "paper_candidate", who="research_hermes")
                registry.update_status(fname, "paper_pass", who="research_hermes", metadata={
                    "paper_pf": best.pf,
                    "paper_net_pnl": best.net_pnl,
                    "paper_cost_gross_ratio": best.cost_gross_ratio,
                    "best_config": f"{best.config.top_frac:.0%}/{best.config.hold_bars}b",
                })
                registry.propose_for_review(
                    factor_id=fname,
                    paper_metrics={
                        "pf": best.pf,
                        "net_pnl": best.net_pnl,
                        "cost_gross_ratio": best.cost_gross_ratio,
                        "classification": cls,
                    },
                    recommendation="promote_to_shadow",
                )
                print(f"  {fname}: → paper_pass, proposed for shadow review")

            elif cls == "filter_only":
                registry.update_status(fname, "paper_candidate", who="research_hermes")
                registry.update_status(fname, "paper_pass", who="research_hermes", metadata={
                    "paper_pf": best.pf,
                    "paper_cost_gross_ratio": best.cost_gross_ratio,
                    "role": "filter_only",
                })
                print(f"  {fname}: → paper_pass (filter_only, no trade proposal)")

            else:  # kill
                registry.update_status(fname, "paper_candidate", who="research_hermes")
                registry.update_status(fname, "paper_fail", who="research_hermes", metadata={
                    "paper_pf": best.pf,
                    "paper_net_pnl": best.net_pnl,
                    "kill_reason": f"classification={cls}, PF={best.pf:.2f}, Net={best.net_pnl:.4f}",
                })
                print(f"  {fname}: → paper_fail (killed)")

    print(f"\nRegistry saved to {REGISTRY_ROOT}")
    print(f"Pending reviews: {len(registry.get_pending_reviews())}")


if __name__ == "__main__":
    main()
