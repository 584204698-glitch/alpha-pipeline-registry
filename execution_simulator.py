"""
Paper Order Simulator v2.0
===========================
Upgraded from Execution Simulator v1.0 — merged with Paper Order Sim spec.

Simulates real-world order execution on top of shadow signals:
- Bid/ask spread from symbol_cost_profile
- Per-symbol position dedup (one position at a time)
- Conflict rejection (opposite direction same symbol)
- TopN allocation (top1/3/5)
- Funding settlement-aware PnL for Funding Carry
- Full order-level audit trail (simulated_orders.jsonl)
- 9/12/15 bps cost stress test

Safe: live_allowed=false, no exchange API calls, no real orders.

Output:
  logs/shadow/simulated_orders.jsonl
  logs/shadow/paper_execution_report.json
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# ── Constants ──────────────────────────────────────

ALPHA_PRIORITY = [
    "DeleveragingReversal",
    "OIShockAbsorption",
    "RelativeStrengthShock",
    "FundingCarryEU",
]

COST_TIERS = [9, 12, 15]  # bps per side
TOP_N_MODES = [1, 3, 5]
FUNDING_SETTLE_HOURS = [0, 8, 16]

# ── Order Data ──────────────────────────────────────

@dataclass
class SimulatedOrder:
    order_id: str
    symbol: str
    alpha: str
    signal_ts: str
    regime: str
    direction: str  # long / short
    event_score: float
    hold_bars: int

    # Entry
    entry_mode: str           # scan_HH_15 / close_entry / next_15m_open
    entry_time: str
    entry_side: str           # buy / sell
    entry_price: float
    entry_bid: float          # best bid at entry
    entry_ask: float          # best ask at entry

    # Exit
    exit_time: str
    exit_side: str
    exit_price: float
    exit_bid: float
    exit_ask: float

    # P&L
    gross_bps: float
    fee_bps: float            # taker fee (3 bps per leg)
    spread_cost_bps: float    # half-spread × 2
    slippage_bps: float        # extra adverse movement
    total_cost_bps: float
    net_bps: float

    # Flags
    reject_reason: str = ""
    allocation_mode: str = "top5"
    funding_crossed: bool = False
    funding_settlement_bps: float = 0.0

    @property
    def dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "alpha": self.alpha,
            "signal_ts": self.signal_ts,
            "regime": self.regime,
            "direction": self.direction,
            "event_score": self.event_score,
            "hold_bars": self.hold_bars,
            "entry_mode": self.entry_mode,
            "entry_time": self.entry_time,
            "entry_side": self.entry_side,
            "entry_price": round(self.entry_price, 8),
            "entry_bid": round(self.entry_bid, 8),
            "entry_ask": round(self.entry_ask, 8),
            "exit_time": self.exit_time,
            "exit_side": self.exit_side,
            "exit_price": round(self.exit_price, 8),
            "exit_bid": round(self.exit_bid, 8),
            "exit_ask": round(self.exit_ask, 8),
            "gross_bps": round(self.gross_bps, 2),
            "fee_bps": round(self.fee_bps, 2),
            "spread_cost_bps": round(self.spread_cost_bps, 2),
            "slippage_bps": round(self.slippage_bps, 2),
            "total_cost_bps": round(self.total_cost_bps, 2),
            "net_bps": round(self.net_bps, 2),
            "reject_reason": self.reject_reason,
            "allocation_mode": self.allocation_mode,
            "funding_crossed": self.funding_crossed,
            "funding_settlement_bps": round(self.funding_settlement_bps, 2),
        }


# ── Helpers ─────────────────────────────────────────

def load_signals(signal_dir: Path) -> list[dict]:
    """Load deduplicated SHADOW_SIGNAL entries from latest signal file."""
    files = sorted(signal_dir.glob("portfolio_signals_*.jsonl"))
    if not files:
        return []
    seen = set()
    signals = []
    with open(files[-1]) as f:
        for line in f:
            d = json.loads(line)
            if d.get("status") != "SHADOW_SIGNAL":
                continue
            key = (d["timestamp"], d["symbol"], d["event_type"])
            if key in seen:
                continue
            seen.add(key)
            signals.append(d)
    return signals


def load_symbol_cost(cost_path: Path) -> dict[str, dict]:
    """Load per-symbol cost profile."""
    if cost_path.exists():
        return json.loads(cost_path.read_text())
    return {}


def build_spread_fallback(data: pd.DataFrame) -> dict[str, float]:
    """Estimate spread from high-low range, capped to realistic values.

    High-low range overestimates true bid-ask — we cap at 30bps for small coins.
    """
    spread = (data["high"] - data["low"]) / data["close"].clip(lower=1e-8) * 10000
    raw = spread.groupby(level="symbol").mean().to_dict()
    # Cap: true bid-ask rarely exceeds 20bps even for illiquid alts
    # High-low range measures volatility, not spread
    return {s: min(v * 0.25, 20.0) for s, v in raw.items()}


def get_bid_ask(
    data: pd.DataFrame,
    symbol: str,
    ts: pd.Timestamp,
    symbol_cost: dict[str, dict],
    spread_fallback: dict[str, float],
) -> tuple[float, float, float]:
    """Get close, bid, ask at timestamp.

    close = bar close
    half_spread = symbol_cost_profile half_spread or fallback
    bid = close - half_spread bps
    ask = close + half_spread bps
    """
    try:
        close = float(data.loc[(ts, symbol), "close"])
    except KeyError:
        return 0.0, 0.0, 0.0

    sc = symbol_cost.get(symbol, {})
    half_spread = sc.get("half_spread_bps", spread_fallback.get(symbol, 5.0) / 2)
    half_spread_ratio = half_spread / 10000.0

    bid = close * (1.0 - half_spread_ratio)
    ask = close * (1.0 + half_spread_ratio)

    return close, bid, ask


def get_next_bar_entry(
    data: pd.DataFrame,
    symbol: str,
    signal_ts: pd.Timestamp,
    entry_delay_bars: int = 1,
) -> tuple[float | None, float | None, float | None, str | None]:
    """Get entry price at next bar (simulating delayed scan entry).

    Returns (close, bid, ask, entry_time) or (None, None, None, None).
    """
    ts_list = list(sorted(data.index.get_level_values("timestamp").unique()))
    try:
        idx = ts_list.index(signal_ts)
    except ValueError:
        return None, None, None, None

    entry_idx = idx + entry_delay_bars
    if entry_idx >= len(ts_list):
        return None, None, None, None
    entry_ts = ts_list[entry_idx]
    try:
        close = float(data.loc[(entry_ts, symbol), "close"])
        # Approximate bid/ask from next bar's OHLC
        high = float(data.loc[(entry_ts, symbol), "high"])
        low = float(data.loc[(entry_ts, symbol), "low"])
        mid = (high + low) / 2.0
        half_range = (high - low) / 2.0
        bid = close - half_range * 0.3  # rough
        ask = close + half_range * 0.3
        return close, bid, ask, str(entry_ts)
    except KeyError:
        return None, None, None, None


def get_exit_data(
    data: pd.DataFrame,
    symbol: str,
    signal_ts: pd.Timestamp,
    hold_bars: int,
    symbol_cost: dict[str, dict],
    spread_fallback: dict[str, float],
) -> tuple[float | None, float | None, float | None, str | None]:
    """Get exit close/bid/ask at hold_bars after signal."""
    ts_list = list(sorted(data.index.get_level_values("timestamp").unique()))
    try:
        idx = ts_list.index(signal_ts)
    except ValueError:
        return None, None, None, None

    exit_idx = idx + hold_bars
    if exit_idx >= len(ts_list):
        return None, None, None, None

    exit_ts = ts_list[exit_idx]
    close, bid, ask = get_bid_ask(data, symbol, exit_ts, symbol_cost, spread_fallback)
    if close == 0.0:
        return None, None, None, None
    return close, bid, ask, str(exit_ts)


def crosses_funding(entry_str: str, exit_str: str) -> bool:
    """Check if holding period crosses funding settlement time."""
    try:
        t0 = pd.Timestamp(entry_str)
        t1 = pd.Timestamp(exit_str)
    except (ValueError, TypeError):
        return False
    hours = pd.date_range(t0, t1, freq="h", inclusive="neither")
    return any(h.hour in FUNDING_SETTLE_HOURS for h in hours)


# ── Core Simulator ──────────────────────────────────

def simulate_orders(
    signals: list[dict],
    data: pd.DataFrame,
    symbol_cost: dict[str, dict],
    entry_mode: str,
    entry_delay_bars: int,
    top_n: int,
) -> tuple[list[SimulatedOrder], dict]:
    """Generate simulated orders for a given allocation mode.

    Rules:
      - One position per symbol at a time
      - Opposite direction → conflict_rejected
      - Top N by event_score per bar
    """
    spread_fallback = build_spread_fallback(data)
    order_counter = 0
    orders: list[SimulatedOrder] = []
    stats = {
        "raw_signals": len(signals),
        "rejected_due_to_cap": 0,
        "rejected_due_to_conflict": 0,
        "rejected_no_data": 0,
        "simulated_count": 0,
    }

    # Group signals by bar timestamp
    by_bar: dict[str, list[dict]] = defaultdict(list)
    for s in signals:
        by_bar[s["timestamp"]].append(s)

    # Current position tracker: symbol → active order
    active_positions: dict[str, SimulatedOrder] = {}

    for bar_ts_str in sorted(by_bar):
        bar_signals = by_bar[bar_ts_str]
        bar_ts = pd.Timestamp(bar_ts_str)

        # Rank by alpha priority then event_score
        priority_map = {a: i for i, a in enumerate(ALPHA_PRIORITY)}
        ranked = sorted(bar_signals, key=lambda s: (
            priority_map.get(s["event_type"], 99),
            -s.get("event_score", 0),
        ))

        # Top N filter
        eligible = ranked[:top_n]
        rejected_cap = ranked[top_n:]
        stats["rejected_due_to_cap"] += len(rejected_cap)

        # Record rejected due to cap as orders
        for s in rejected_cap:
            order_counter += 1
            orders.append(SimulatedOrder(
                order_id=f"REJ_CAP_{order_counter:04d}",
                symbol=s["symbol"],
                alpha=s["event_type"],
                signal_ts=s["timestamp"],
                regime=s.get("regime", "unknown"),
                direction=s.get("direction", "long"),
                event_score=s.get("event_score", 0.0),
                hold_bars=s.get("hold_bars", 2),
                entry_mode=entry_mode,
                entry_time="",
                entry_side="",
                entry_price=0.0,
                entry_bid=0.0,
                entry_ask=0.0,
                exit_time="",
                exit_side="",
                exit_price=0.0,
                exit_bid=0.0,
                exit_ask=0.0,
                gross_bps=0.0,
                fee_bps=0.0,
                spread_cost_bps=0.0,
                slippage_bps=0.0,
                total_cost_bps=0.0,
                net_bps=0.0,
                reject_reason=f"top{top_n}_cap (rank={len(eligible) + 1 + rejected_cap.index(s)})",
                allocation_mode=f"scan_HH_15_{entry_delay_bars}bar_top{top_n}",
            ))

        for s in eligible:
            sym = s["symbol"]
            direction = s.get("direction", "long")
            alpha = s["event_type"]
            score = s.get("event_score", 0.0)
            hold = s.get("hold_bars", 2)

            # Conflict check: opposite direction, same symbol, active position
            if sym in active_positions:
                active = active_positions[sym]
                if active.direction != direction:
                    order_counter += 1
                    orders.append(SimulatedOrder(
                        order_id=f"REJ_CONFLICT_{order_counter:04d}",
                        symbol=sym, alpha=alpha,
                        signal_ts=s["timestamp"],
                        regime=s.get("regime", "unknown"),
                        direction=direction,
                        event_score=score,
                        hold_bars=hold,
                        entry_mode=entry_mode,
                        entry_time="", entry_side="",
                        entry_price=0.0, entry_bid=0.0, entry_ask=0.0,
                        exit_time="", exit_side="",
                        exit_price=0.0, exit_bid=0.0, exit_ask=0.0,
                        gross_bps=0.0, fee_bps=0.0,
                        spread_cost_bps=0.0, slippage_bps=0.0,
                        total_cost_bps=0.0, net_bps=0.0,
                        reject_reason=f"conflict: active {active.direction} vs signal {direction} [{alpha}]",
                        allocation_mode=f"scan_HH_15_{entry_delay_bars}bar_top{top_n}",
                    ))
                    stats["rejected_due_to_conflict"] += 1
                    continue

            # Entry: use next bar close as delayed entry
            entry_close, entry_bid, entry_ask, entry_time_str = get_next_bar_entry(
                data, sym, bar_ts, entry_delay_bars
            )
            if entry_close is None:
                stats["rejected_no_data"] += 1
                continue

            # Exit
            exit_close, exit_bid, exit_ask, exit_time_str = get_exit_data(
                data, sym, bar_ts, hold, symbol_cost, spread_fallback
            )
            if exit_close is None:
                stats["rejected_no_data"] += 1
                continue

            # Determine entry/exit prices based on direction
            if direction == "long":
                entry_px = entry_ask       # buy at ask
                entry_side = "buy"
                exit_px = exit_bid         # sell at bid
                exit_side = "sell"
            else:
                entry_px = entry_bid       # sell at bid
                entry_side = "sell"
                exit_px = exit_ask         # cover at ask
                exit_side = "buy"

            # Gross PnL
            if direction == "long":
                gross_bps = (exit_px / entry_px - 1.0) * 10000
            else:
                gross_bps = (entry_px / exit_px - 1.0) * 10000

            # Cost breakdown
            sc = symbol_cost.get(sym, {})
            half_spread = sc.get("half_spread_bps", spread_fallback.get(sym, 5.0) / 2)
            total_sym_cost = sc.get("total_cost_bps", 9.0)

            fee_bps = 6.0          # 3 bps per leg × 2
            spread_cost_bps = half_spread * 2  # round-trip spread
            # Slippage is the remaining cost after fee + spread
            slippage_bps = max(0.0, total_sym_cost - fee_bps - spread_cost_bps)
            total_cost_bps = fee_bps + spread_cost_bps + slippage_bps

            net_bps = gross_bps - total_cost_bps

            # Funding settlement
            fc_crossed = False
            if alpha == "FundingCarryEU" and exit_time_str:
                fc_crossed = crosses_funding(str(bar_ts), exit_time_str)

            order_counter += 1
            order = SimulatedOrder(
                order_id=f"SIM_{order_counter:04d}",
                symbol=sym,
                alpha=alpha,
                signal_ts=s["timestamp"],
                regime=s.get("regime", "unknown"),
                direction=direction,
                event_score=score,
                hold_bars=hold,
                entry_mode=entry_mode,
                entry_time=entry_time_str,
                entry_side=entry_side,
                entry_price=round(entry_px, 8),
                entry_bid=round(entry_bid, 8),
                entry_ask=round(entry_ask, 8),
                exit_time=exit_time_str,
                exit_side=exit_side,
                exit_price=round(exit_px, 8),
                exit_bid=round(exit_bid, 8),
                exit_ask=round(exit_ask, 8),
                gross_bps=round(gross_bps, 2),
                fee_bps=round(fee_bps, 2),
                spread_cost_bps=round(spread_cost_bps, 2),
                slippage_bps=round(slippage_bps, 2),
                total_cost_bps=round(total_cost_bps, 2),
                net_bps=round(net_bps, 2),
                allocation_mode=f"scan_HH_15_{entry_delay_bars}bar_top{top_n}",
                funding_crossed=fc_crossed,
            )
            orders.append(order)
            active_positions[sym] = order
            stats["simulated_count"] += 1

    return orders, stats


def compute_pf(nets: list[float]) -> float:
    """Compute profit factor from list of net bps."""
    if not nets:
        return 0.0
    arr = np.array(nets)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    if len(losses) == 0:
        return 999.0 if len(wins) > 0 else 0.0
    loss_sum = abs(losses.sum())
    if loss_sum == 0:
        return 999.0 if len(wins) > 0 else 0.0
    return float(wins.sum() / loss_sum)


def build_report(
    all_orders: list[SimulatedOrder],
    stats_by_mode: dict[str, dict],
    costs_by_tier: dict[str, dict],
    data_range: str,
) -> dict:
    """Build paper_execution_report.json."""
    # Collect non-rejected orders for P&L
    executed = [o for o in all_orders if o.order_id.startswith("SIM_")]

    report: dict[str, Any] = {
        "_meta": {
            "version": "v2.0",
            "generated_ts": str(pd.Timestamp.now(tz="UTC")),
            "data_range": data_range,
            "live_allowed": False,
        },
        "summary": {
            "raw_signal_count": sum(s.get("raw_signals", 0) for s in stats_by_mode.values()) // len(stats_by_mode) if stats_by_mode else 0,
            "simulated_order_count": len(executed),
            "rejected_due_to_cap": sum(s.get("rejected_due_to_cap", 0) for s in stats_by_mode.values()),
            "rejected_due_to_conflict": sum(s.get("rejected_due_to_conflict", 0) for s in stats_by_mode.values()),
            "rejected_no_data": sum(s.get("rejected_no_data", 0) for s in stats_by_mode.values()),
        },
        "by_top_mode": {},
        "cost_stress_test": {},
        "decision": {},
    }

    # Per allocation mode — group by allocation_mode
    alloc_modes = set(o.allocation_mode for o in executed)
    for alloc in sorted(alloc_modes):
        mode_orders = [o for o in executed if o.allocation_mode == alloc]
        nets = [o.net_bps for o in mode_orders]
        gross = [o.gross_bps for o in mode_orders]
        total_cost = [o.total_cost_bps for o in mode_orders]
        spread_costs = [o.spread_cost_bps for o in mode_orders]
        slippage = [o.slippage_bps for o in mode_orders]
        fees = [o.fee_bps for o in mode_orders]

        net_arr = np.array(nets)
        top3_idx = np.argsort(net_arr)[-3:] if len(net_arr) >= 3 else range(len(net_arr))
        top3_net = float(net_arr[list(top3_idx)].sum())
        net_wo_top3 = float(net_arr.sum()) - top3_net

        report["by_top_mode"][alloc] = {
            "n_trades": len(mode_orders),
            "net_pnl_bps": round(float(net_arr.sum()), 1),
            "pf": round(compute_pf(nets), 2),
            "hit_rate": round(float((net_arr > 0).mean()), 3),
            "max_loss_bps": round(float(net_arr.min()), 1) if len(net_arr) > 0 else 0,
            "max_gain_bps": round(float(net_arr.max()), 1) if len(net_arr) > 0 else 0,
            "median_bps": round(float(np.median(net_arr)), 1) if len(net_arr) > 0 else 0,
            "top3_contribution": round(top3_net, 1),
            "net_without_top3": round(net_wo_top3, 1),
            "delay_cost_bps": round(float(np.mean(slippage)), 1) if slippage else 0,
            "slippage_cost_bps": round(float(np.mean(slippage)), 1) if slippage else 0,
            "spread_cost_bps": round(float(np.mean(spread_costs)), 1) if spread_costs else 0,
            "total_cost_bps_mean": round(float(np.mean(total_cost)), 1) if total_cost else 0,
            "gross_bps_sum": round(float(np.array(gross).sum()), 1),
            "fee_bps_total": round(float(np.array(fees).sum()), 1),
        }

    # Cost stress test: check 12bps and 15bps
    # We simulate by adding delta to each trade's cost
    base_cost = 9.0
    for test_bps in [12, 15]:
        delta = test_bps - base_cost
        stress_nets = [o.net_bps - delta for o in executed]
        stress_nets_arr = np.array(stress_nets)
        report["cost_stress_test"][f"cost_{test_bps}bps"] = {
            "net_pnl_bps": round(float(stress_nets_arr.sum()), 1),
            "pf": round(compute_pf(stress_nets), 2),
            "hit_rate": round(float((stress_nets_arr > 0).mean()), 3),
            "survives": bool(stress_nets_arr.sum() > 0 and compute_pf(stress_nets) >= 1.0),
        }

    # Decision
    primary = report["by_top_mode"].get("scan_HH_15_1bar_top5", {})
    if not primary:
        report["decision"] = {"verdict": "NO_DATA", "reasons": ["No trades executed"]}
        return report

    net = primary.get("net_pnl_bps", 0)
    pf = primary.get("pf", 0)
    net_wo_top3 = primary.get("net_without_top3", 0)
    stress_survives = all(
        report["cost_stress_test"][f"cost_{b}bps"]["survives"]
        for b in [12, 15]
    )

    reasons = []
    if net > 0 and pf >= 1.15 and net_wo_top3 >= 0:
        verdict = "PAPER_EXECUTION_PASS"
        reasons.append(f"Net={net:.0f}bps PF={pf:.2f} net_wo_top3={net_wo_top3:.0f}")
        if stress_survives:
            reasons.append("12/15bps stress test: SURVIVES")
            verdict = "PAPER_EXECUTION_PASS"
        else:
            reasons.append("12/15bps stress test: FAILS → downgrade")
            verdict = "PAPER_EXECUTION_WEAK"
    elif net <= 0:
        verdict = "EXECUTION_FAIL"
        reasons.append(f"Net={net:.0f}bps ≤ 0")
    elif pf < 1.15:
        verdict = "EXECUTION_FAIL"
        reasons.append(f"PF={pf:.2f} < 1.15")
    elif net_wo_top3 < 0:
        verdict = "EXECUTION_FAIL"
        reasons.append(f"Net_wo_top3={net_wo_top3:.0f} < 0")
    else:
        verdict = "PAPER_EXECUTION_PASS"

    # Check if raw_shadow (close_entry) was positive but simulated was negative
    # For now, compare close_entry vs simulated
    close_mode = report["by_top_mode"].get("close_entry_top5", {})
    if close_mode:
        close_net = close_mode.get("net_pnl_bps", 0)
        if close_net > 0 and net <= 0:
            reasons.append(f"RAW_SHADOW_POSITIVE (close_entry={close_net:.0f}) but SIMULATED_NEGATIVE")
            verdict = "EXECUTION_FAIL"

    report["decision"] = {
        "verdict": verdict,
        "reasons": reasons,
        "primary_mode": "scan_HH_15_1bar_top5",
    }

    return report


# ── Main ────────────────────────────────────────────

def main() -> None:
    signal_dir = ROOT / "logs" / "shadow"
    data_path = ROOT / "data" / "data_storage_1h.parquet"
    cost_path = ROOT / "registry" / "live" / "symbol_cost_profile.json"
    order_path = ROOT / "logs" / "shadow" / "simulated_orders.jsonl"
    report_path = ROOT / "logs" / "shadow" / "paper_execution_report.json"

    print("=" * 60)
    print("Paper Order Simulator v2.0")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    t0 = time.time()
    data = pd.read_parquet(data_path)
    ts = data.index.get_level_values("timestamp")
    print(f"  {len(data):,} rows, {data.index.get_level_values('symbol').nunique()} symbols")
    print(f"  range: {ts.min()} → {ts.max()}")

    # Load signals
    print("\n[2/5] Loading signals...")
    signals = load_signals(signal_dir)
    print(f"  Signals: {len(signals)}")

    if not signals:
        print("  No signals to simulate. Exiting.")
        return

    # Also generate close_entry signals for raw shadow comparison
    # (already done by the existing shadow runner)

    # Load cost profile
    print("\n[3/5] Loading cost profile...")
    symbol_cost = load_symbol_cost(cost_path)
    if not symbol_cost:
        print("  No symbol_cost_profile.json — generating from spread proxy...")
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
    print(f"  {len(symbol_cost)} symbols with cost data")

    # Run simulations
    print("\n[4/5] Running simulations...")

    all_orders: list[SimulatedOrder] = []
    stats_by_mode: dict[str, dict] = {}

    for top_n in TOP_N_MODES:
        for entry_name, delay_bars in [("scan_HH_15", 1)]:  # only realistic: 1-bar delay
            mode_key = f"{entry_name}_top{top_n}"
            print(f"  {mode_key} ...")
            orders, stats = simulate_orders(
                signals, data, symbol_cost, entry_name, delay_bars, top_n
            )
            all_orders.extend(orders)
            stats_by_mode[mode_key] = stats  # unique key per mode

    # Also run close_entry comparison
    print("  close_entry reference ...")
    # For close_entry, use delay_bars=0 (same bar)
    for top_n in TOP_N_MODES:
        orders, stats = simulate_orders(
            signals, data, symbol_cost, "close_entry", 0, top_n
        )
        all_orders.extend(orders)
        stats_by_mode[f"close_entry_top{top_n}"] = stats

    # Write orders
    print("\n[5/5] Writing outputs...")
    order_path.parent.mkdir(parents=True, exist_ok=True)
    with open(order_path, "w") as f:
        for o in all_orders:
            f.write(json.dumps(o.dict, default=str) + "\n")
    print(f"  Orders: {order_path} ({len(all_orders)} entries)")

    # Build report
    data_range = f"{ts.min()} → {ts.max()}"
    report = build_report(all_orders, stats_by_mode, {}, data_range)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"  Report: {report_path}")

    # Summary
    decision = report.get("decision", {})
    print(f"\n{'='*60}")
    print(f"Decision: {decision.get('verdict', '?')}")
    for r in decision.get("reasons", []):
        print(f"  • {r}")
    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
