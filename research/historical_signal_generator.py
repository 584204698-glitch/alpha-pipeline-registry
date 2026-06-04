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
sys.path.insert(0, str(ROOT))

from execution_simulator import (
    simulate_orders, build_report, load_symbol_cost, build_spread_fallback,
    TOP_N_MODES, COST_TIERS,
)


def load_universe() -> set[str]:
    with open(ROOT / "registry" / "live" / "event_universe.json") as f:
        d = json.load(f)
    syms = set()
    def collect(v):
        if isinstance(v, list): syms.update(v)
        elif isinstance(v, dict):
            for vv in v.values(): collect(vv)
    collect(d)
    return syms


def generate_historical_signals(data: pd.DataFrame) -> list[dict]:
    from event_scanner import DeleveragingEventScanner
    from relative_strength_shock import detect_relative_strength_shock
    from oi_shock_absorption import detect_oi_shock_absorption
    from funding_carry_scanner import FundingCarryScanner
    from research.regime_detector import detect_regime_fast

    signals: list[dict] = []
    ts_unique = sorted(data.index.get_level_values("timestamp").unique())
    universe = load_universe()
    print(f"  Universe: {len(universe)} symbols, {len(ts_unique)} bars")

    # Regime map
    print("    Computing regime map...")
    regime_map = detect_regime_fast(data).to_dict()
    btc_present = "BTCUSDT" in data.index.get_level_values("symbol")

    # 1. Deleveraging — per-bar scan
    print("    Deleveraging...")
    scanner = DeleveragingEventScanner(data, regime_map, hold_bars=2)
    for i, ts in enumerate(ts_unique):
        if i % 200 == 0:
            print(f"      {i}/{len(ts_unique)}")
        for s in scanner.scan(ts):
            d = s.__dict__ if hasattr(s, '__dict__') else (s if isinstance(s, dict) else {})
            sym = d.get("symbol", "")
            state = d.get("state", "")
            if sym not in universe:
                continue
            # Only take CONFIRMED or QUALIFIED events (not REJECTED/watched)
            from event_scanner import EventState
            if isinstance(state, EventState):
                if state in (EventState.REJECTED, EventState.WATCH):
                    continue
            signals.append({
                "timestamp": str(ts),
                "symbol": sym,
                "event_type": "DeleveragingReversal",
                "status": "SHADOW_SIGNAL",
                "regime": regime_map.get(ts, "unknown"),
                "direction": "long",
                "event_score": d.get("event_score", 0),
                "hold_bars": 2,
            })

    # 2. OI Shock — scans all at once
    print("    OI Shock...")
    oi_events = detect_oi_shock_absorption(data, universe, btc_regime_map=regime_map)
    for ev in oi_events:
        sym = ev.get("symbol", "")
        if sym not in universe:
            continue
        signals.append({
            "timestamp": str(ev["timestamp"]),
            "symbol": sym,
            "event_type": "OIShockAbsorption",
            "status": "SHADOW_SIGNAL",
            "regime": ev.get("regime", "unknown"),
            "direction": ev.get("direction", "long"),
            "event_score": ev.get("event_score", 0),
            "hold_bars": 2,
        })

    # 3. RS Shock — scans all at once
    print("    RS Shock...")
    rs_events = detect_relative_strength_shock(data, universe, btc_regime_map=regime_map)
    for ev in rs_events:
        sym = ev.get("symbol", "")
        if sym not in universe:
            continue
        signals.append({
            "timestamp": str(ev["timestamp"]),
            "symbol": sym,
            "event_type": "RelativeStrengthShock",
            "status": "SHADOW_SIGNAL",
            "regime": ev.get("regime", "unknown"),
            "direction": ev.get("direction", "long"),
            "event_score": ev.get("event_score", 0),
            "hold_bars": 2,
        })

    # 4. Funding Carry — per-bar scan
    print("    Funding Carry...")
    fc_scanner = FundingCarryScanner(data, regime_map, universe)
    for i, ts in enumerate(ts_unique):
        if i % 200 == 0:
            print(f"      {i}/{len(ts_unique)}")
        for s in fc_scanner.scan(ts):
            d = s.__dict__ if hasattr(s, '__dict__') else (s if isinstance(s, dict) else {})
            sym = d.get("symbol", "")
            state = d.get("state", "")
            if sym not in universe:
                continue
            # Only SHADOW_SIGNAL events (not REJECTED)
            from funding_carry_scanner import EventState as FCEventState
            if isinstance(state, FCEventState):
                if state != FCEventState.SHADOW_SIGNAL:
                    continue
            signals.append({
                "timestamp": str(ts),
                "symbol": sym,
                "event_type": "FundingCarryEU",
                "status": "SHADOW_SIGNAL",
                "regime": regime_map.get(ts, "unknown"),
                "direction": "long",
                "event_score": d.get("event_score", 0),
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

    # Save raw historical signals for statistical analysis
    signal_path = ROOT / "logs" / "shadow" / "historical_signals.jsonl"
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(signal_path, "w") as f:
        for s in dedup:
            f.write(json.dumps(s, default=str) + "\n")

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
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        cost_path.write_text(json.dumps(symbol_cost, indent=2))

    all_orders = []
    order_path = ROOT / "logs" / "shadow" / "simulated_orders.jsonl"
    report_path = ROOT / "logs" / "shadow" / "paper_execution_report.json"

    for top_n in TOP_N_MODES:
        print(f"  scan_HH_15_top{top_n} ...")
        orders, _ = simulate_orders(signals, data, symbol_cost, "scan_HH_15", 1, top_n)
        all_orders.extend(orders)

    for top_n in TOP_N_MODES:
        print(f"  close_entry_top{top_n} ...")
        orders, _ = simulate_orders(signals, data, symbol_cost, "close_entry", 0, top_n)
        all_orders.extend(orders)

    with open(order_path, "w") as f:
        for o in all_orders:
            f.write(json.dumps(o.dict, default=str) + "\n")
    print(f"  Orders: {order_path} ({len(all_orders)} entries)")

    ts = data.index.get_level_values("timestamp")
    report = build_report(all_orders, {}, {}, f"{ts.min()} → {ts.max()}")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"  Report: {report_path}")

    decision = report.get("decision", {})
    print(f"\n{'='*60}")
    print(f"Decision: {decision.get('verdict', '?')}")
    for r in decision.get("reasons", []):
        print(f"  • {r}")
    primary = report.get("by_top_mode", {}).get("scan_HH_15_1bar_top5", {})
    if primary:
        print(f"\n  n={primary['n_trades']} Net={primary['net_pnl_bps']:.0f}bps PF={primary['pf']:.2f} net_wo_top3={primary['net_without_top3']:.0f}bps")


if __name__ == "__main__":
    main()
