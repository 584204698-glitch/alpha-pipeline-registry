"""Batch backtest all 6 new factors from external AI methodology review."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_engine import BacktestEngine
from validator import OverfitValidator

PROJECT = Path(__file__).resolve().parent
FACTORS = [
    "alpha_fpr_cs_neutral.py",
    "alpha_fpr_statewise_norm.py",
    "alpha_fpr_dir_strength_split.py",
    "alpha_wick_absorption.py",
    "alpha_funding_gated_absorption.py",
    "alpha_trend_trap_absorption.py",
]

engine = BacktestEngine(PROJECT)
results_all = []

for fname in FACTORS:
    fpath = PROJECT / "factors" / fname
    if not fpath.exists():
        print(f"SKIP {fname} — file not found")
        continue

    print(f"\n{'='*60}")
    print(f"BACKTESTING: {fname}")
    print(f"{'='*60}")

    try:
        bt = engine.run_backtest(fpath)
    except Exception as e:
        print(f"  BACKTEST ERROR: {e}")
        results_all.append({"file": fname, "error": str(e)})
        continue

    m = bt["metrics"]
    ss = bt.get("signal_summary", {})
    print(f"  IC={m['overall_ic']:.4f}  IR={m['overall_ir']:.3f}")
    print(f"  IS_ICIR={m['in_sample_icir']:.3f}  OOS_ICIR={m['out_of_sample_icir']:.3f}")
    print(f"  long_ratio={ss.get('long_ratio', 0):.3f}  nonzero={ss.get('nonzero_ratio', 0):.3f}")
    print(f"  regime_ic: {m['regime_ic']}")
    print(f"  quintiles: {[f'{v:.4f}' for v in m['quantile_returns']]}")

    try:
        v = OverfitValidator(
            current_factor_code=fpath.read_text(encoding="utf-8"),
            backtest_results=bt,
            factor_registry_path=str(PROJECT / "config" / "production_factors.json"),
            project_root=PROJECT,
            factor_path=fpath,
        )
        verdict = v.validate_all()
    except Exception as e:
        print(f"  VALIDATOR ERROR: {e}")
        verdict = {"verdict": "ERROR", "error": str(e)}

    verdict_str = verdict.get("verdict", "???")
    checks = verdict.get("checks", {})
    print(f"  VERDICT: {verdict_str}")
    for ck, cv in checks.items():
        print(f"    {ck}: {'PASS' if cv else 'FAIL'}")

    results_all.append({
        "file": fname,
        "factor_name": bt.get("factor_name"),
        "ic": m["overall_ic"],
        "ir": m["overall_ir"],
        "is_icir": m["in_sample_icir"],
        "oos_icir": m["out_of_sample_icir"],
        "long_ratio": ss.get("long_ratio"),
        "regime_ic": m["regime_ic"],
        "verdict": verdict_str,
        "checks": {k: v for k, v in checks.items()},
    })

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for r in results_all:
    v = r.get("verdict", "ERR")
    flags = " ✅" if v == "PASS" else " ❌"
    print(f"  {r.get('factor_name', r['file'])}: {v} IC={r.get('ic', 0):.4f} OOS={r.get('oos_icir', 0):.3f} long={r.get('long_ratio', 0):.3f}{flags}")

# Save results
out_path = PROJECT / "logs" / "external_review_results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results_all, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
