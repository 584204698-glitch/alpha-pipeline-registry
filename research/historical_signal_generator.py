"""
Generate full historical shadow signals from all 4 scanners.
Feeds into Paper Order Simulator for comprehensive execution audit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from execution_simulator import (
    simulate_orders, build_report, load_symbol_cost, build_spread_fallback,
    TOP_N_MODES, COST_TIERS,
)


def generate_historical_signals(data: pd.DataFrame) -> list[dict]:
    """Run all 4 scanners across full history, output signal list."""
    signals = []

    # Import scanners
    sys.path.insert(0, str(ROOT))
    from event_scanner import DeleveragingEventScanner
    from relative_strength_shock import detect_relative_strength_shock
    from oi_shock_absorption import detect_oi_shock_absorption
    from funding_carry_scanner import FundingCarryScanner

    from research.regime_detector import detect_regime_fast
    regime_map = detect_regime_fast(data)

    ts_unique = sorted(data.index.get_level_values("timestamp").unique())
    # Only scan bars with enough symbols
    min_syms = 30

    print(f"  Scanning {len(ts_unique)} bars...")

    # Deleveraging
    print("    Deleveraging...")
    scanner = DeleveragingEventScanner(data, regime_map, hold_bars=2)
    scanner.scan()
    for s in scanner.signals:
        d = s.to_dict() if hasattr(s, 'to_dict') else s
        signals.append({
            "timestamp": str(d.get("timestamp", d.get("ts", ""))),
            "symbol": d.get("symbol", ""),
            "event_type": "DeleveragingReversal",
            "status": "SHADOW_SIGNAL",
            "regime": d.get("regime", ""),
            "direction": d.get("direction", "long"),
            "event_score": d.get("event_score", d.get("score", 0)),
            "hold_bars": 2,
        })

    # RS Shock
    print("    RS Shock...")
    rs_events = detect_relative_strength_shock(data, regime_map=regime_map)
    for ev in rs_events:
        signals.append({
            "timestamp": str(ev["timestamp"]),
            "symbol": ev["symbol"],
            "event_type": "RelativeStrengthShock",
            "status": "SHADOW_SIGNAL",
            "regime": ev.get("regime", ""),
            "direction": ev.get("direction", "long"),
            "event_score": ev.get("event_score", 0),
            "hold_bars": 2,
        })

    # OI Shock
    print("    OI Shock...")
    oi_events = detect_oi_shock_absorption(data)
    for ev in oi_events:
        signals.append({
            "timestamp": str(ev["timestamp"]),
            "symbol": ev["symbol"],
            "event_type": "OIShockAbsorption",
            "status": "SHADOW_SIGNAL",
            "regime": ev.get("regime", ""),
            "direction": ev.get("direction", "long"),
            "event_score": ev.get("event_score", 0),
            "hold_bars": 2,
        })

    # Funding Carry
    print("    Funding Carry...")
    with open(ROOT / "registry" / "live" / "event_universe.json") as f:
        universe_data = json.load(f)
        universe = set()
        for k, v in universe_data.items():
            if isinstance(v, dict) and "symbols" in v:
                universe.update(v["symbols"])

    fc_scanner = FundingCarryScanner(data, regime_map, universe)
    fc_scanner.scan_all()
    for s in fc_scanner.signals:
        d = s if isinstance(s, dict) else s.__dict__
        signals.append({
            "timestamp": str(d.get("timestamp", "")),
            "symbol": d.get("symbol", ""),
            "event_type": "FundingCarryEU",
            "status": "SHADOW_SIGNAL",
            "regime": d.get("regime", ""),
            "direction": "long",
            "event_score": d.get("event_score", d.get("score", 0)),
            "hold_bars": 12,
        })

    # Deduplicate
    seen = set()
    dedup = []
    for s in signals:
        k = (s["timestamp"], s["symbol"], s["event_type"])
        if k not in seen:
            seen.add(k)
            dedup.append(s)

    return dedup


def main():
    print("=" * 60)
    print("Full Historical Paper Order Simulation")
    print("=" * 60)

    print("\n[1/3] Loading data...")
    data = pd.read_parquet(ROOT / "data" / "data_storage_1h.parquet")
    print(f"  {len(data):,} rows, {data.index.get_level_values('symbol').nunique()} symbols")

    print("\n[2/3] Generating historical signals...")
    signals = generate_historical_signals(data)
    print(f"  Total signals: {len(signals)}")
    by_alpha = {}
    for s in signals:
        a = s["event_type"]
        by_alpha[a] = by_alpha.get(a, 0) + 1
    for a, n in sorted(by_alpha.items()):
        print(f"    {a}: {n}")

    if not signals:
        print("  No signals generated!")
        return

    print("\n[3/3] Running Paper Order Simulator...")
    # Load cost profile
    cost_path = ROOT / "registry" / "live" / "symbol_cost_profile.json"
    symbol_cost = load_symbol_cost(cost_path)
    if not symbol_cost:
        spread = build_spread_fallback(data)
        for sym in spread:
            half = spread[sym] / 2
            total = max(4.0, min(3.0 + half, 50.0))
            symbol_cost[sym] = {
                "spread_bps_mean": round(spread[sym], 1),
                "half_spread_bps": round(half, 1),
                "total_cost_bps": round(total, 1),
                "cost_bucket": "medium",
            }
        # Save it too
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        cost_path.write_text(json.dumps(symbol_cost, indent=2))

    all_orders = []
    order_path = ROOT / "logs" / "shadow" / "simulated_orders.jsonl"
    report_path = ROOT / "logs" / "shadow" / "paper_execution_report.json"

    for top_n in TOP_N_MODES:
        for entry_name, delay_bars in [("scan_HH_15", 1)]:
            print(f"  {entry_name}_top{top_n} ...")
            orders, _ = simulate_orders(
                signals, data, symbol_cost, entry_name, delay_bars, top_n
            )
            all_orders.extend(orders)

    # Close entry reference
    for top_n in TOP_N_MODES:
        orders, _ = simulate_orders(
            signals, data, symbol_cost, "close_entry", 0, top_n
        )
        all_orders.extend(orders)

    # Write orders
    order_path.parent.mkdir(parents=True, exist_ok=True)
    with open(order_path, "w") as f:
        for o in all_orders:
            f.write(json.dumps(o.dict, default=str) + "\n")
    print(f"  Orders: {order_path} ({len(all_orders)} entries)")

    # Build report
    ts = data.index.get_level_values("timestamp")
    report = build_report(all_orders, {}, {}, f"{ts.min()} → {ts.max()}")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"  Report: {report_path}")

    decision = report.get("decision", {})
    print(f"\n{'='*60}")
    print(f"Decision: {decision.get('verdict', '?')}")
    for r in decision.get("reasons", []):
        print(f"  • {r}")

    # Quick summary
    primary = report.get("by_top_mode", {}).get("scan_HH_15_1bar_top5", {})
    if primary:
        print(f"\n  Primary (scan_HH_15_top5):")
        print(f"    n={primary['n_trades']}, Net={primary['net_pnl_bps']:.0f}bps, PF={primary['pf']:.2f}")
        print(f"    Hit={primary['hit_rate']:.1%}, net_wo_top3={primary['net_without_top3']:.0f}bps")


if __name__ == "__main__":
    main()
