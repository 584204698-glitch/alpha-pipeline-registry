"""Shadow Readiness Audit — 5 checks for all 3 event types.

Checks:
A. Extreme contribution (remove top 1/3/5% trades)
B. Time slice (early/middle/late)
C. Cost stress (9/12/15 bps + delay + slippage)
D. Event conflict handling
E. Shadow mode field completeness

Zero new mining — pure audit.
"""

import sys, json, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
sys.path.insert(0, str(ROOT))

from backtest_engine import BacktestEngine
from research.regime_detector import detect_regime_fast
from event_scanner import backtest_scanner
from relative_strength_shock import detect_relative_strength_shock
from oi_shock_absorption import detect_oi_shock_absorption


def run_all_trades() -> dict[str, pd.DataFrame]:
    """Generate all 3 event trades, return unified DataFrames."""
    engine = BacktestEngine(str(ROOT))
    data = engine._load_data()
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))

    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    small_syms = set(vol_rank[(vol_rank >= 20) & (vol_rank <= 100)].index)

    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}

    all_dfs = {}

    # --- 1. Deleveraging Reversal ---
    r_delev = backtest_scanner(data, regime_map, hold_bars=2, entry_mode="current_close", use_regime_rules=True)
    # Reconstruct trades from scanner results
    delev_trades = _reconstruct_scanner_trades(data, regime_map, ts_list, ts_to_idx, small_syms)
    all_dfs["DeleveragingReversal"] = delev_trades if not delev_trades.empty else _empty_df()

    # --- 2. RS Shock ---
    rs_events = detect_relative_strength_shock(
        data, small_syms, regime_map,
        rs_threshold=0.05, vol_z_min=2.0, oi_delta_z_min=0.5,
        close_loc_min=0.50, event_score_min=0.60, cooldown_bars=8,
    )
    rs_trades = _event_to_trades(data, rs_events, ts_list, ts_to_idx, hold_bars=2)
    all_dfs["RelativeStrengthShock"] = rs_trades if not rs_trades.empty else _empty_df()

    # --- 3. OI Shock Absorption ---
    oi_events = detect_oi_shock_absorption(
        data, small_syms, regime_map,
        oi_delta_z_min=2.5, vol_z_min=1.5, cooldown_bars=8,
    )
    oi_trades = _event_to_trades(data, oi_events, ts_list, ts_to_idx, hold_bars=2)
    all_dfs["OIShockAbsorption"] = oi_trades if not oi_trades.empty else _empty_df()

    return all_dfs


def _reconstruct_scanner_trades(data, regime_map, ts_list, ts_to_idx, small_syms):
    """Reconstruct actual scanner trades for audit."""
    from event_scanner import DeleveragingEventScanner, EventState
    scanner = DeleveragingEventScanner(data, regime_map, hold_bars=2)
    trades = []
    for i, ts in enumerate(ts_list):
        results = scanner.scan(ts)
        for r in results:
            if r.state == EventState.CONFIRMED:
                entry_ts = ts
                try:
                    entry_px = float(data.loc[(entry_ts, r.symbol), "close"])
                except KeyError:
                    continue
                exit_idx = i + 2
                if exit_idx >= len(ts_list):
                    continue
                exit_ts = ts_list[exit_idx]
                try:
                    exit_px = float(data.loc[(exit_ts, r.symbol), "close"])
                except KeyError:
                    continue
                ret = (exit_px / entry_px) - 1.0
                gbp = ret * 10000
                regime = regime_map.get(ts, "unknown")
                trades.append({
                    "symbol": r.symbol,
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "event_type": "DeleveragingReversal",
                    "direction": "long",
                    "regime": regime,
                    "event_score": r.event_score,
                    "gross_bps": gbp,
                    "net_9bps": gbp - 9,
                    "net_12bps": gbp - 12,
                    "net_15bps": gbp - 15,
                    "oi_z": r.metrics.get("oi_z", 0),
                    "ret_1h": r.metrics.get("ret_1h", 0),
                    "close_loc": r.metrics.get("close_loc", 0),
                })
    return pd.DataFrame(trades)


def _event_to_trades(data, events, ts_list, ts_to_idx, hold_bars=2):
    """Convert event dicts to trade DataFrames with cost levels."""
    trades = []
    for ev in events:
        ts = ev["timestamp"]
        sym = ev["symbol"]
        direction = ev["direction"]
        ts_idx = ts_to_idx.get(ts)
        if ts_idx is None or ts_idx >= len(ts_list) - hold_bars:
            continue
        try:
            ep = float(data.loc[(ts, sym), "close"])
            xp = float(data.loc[(ts_list[ts_idx + hold_bars], sym), "close"])
        except KeyError:
            continue
        ret = (xp / ep) - 1.0
        if direction == "short":
            ret = -ret
        gbp = ret * 10000
        trades.append({
            "symbol": sym,
            "entry_ts": ev["timestamp"],
            "exit_ts": ts_list[ts_idx + hold_bars],
            "event_type": ev.get("event_type", "Unknown"),
            "direction": direction,
            "regime": ev.get("regime", "unknown"),
            "event_score": ev.get("event_score", 0),
            "gross_bps": gbp,
            "net_9bps": gbp - 9,
            "net_12bps": gbp - 12,
            "net_15bps": gbp - 15,
            "rs_pct": ev.get("rs_pct", 0),
            "vol_z": ev.get("vol_z", 0),
            "oi_z": ev.get("oi_z", 0),
            "close_loc": ev.get("close_loc", 0),
        })
    return pd.DataFrame(trades)


def _empty_df():
    return pd.DataFrame(columns=["symbol", "entry_ts", "event_type", "direction",
                                  "regime", "event_score", "gross_bps",
                                  "net_9bps", "net_12bps", "net_15bps"])


# ── A. Extreme Contribution ─────────────────────────

def audit_extreme(df: pd.DataFrame, name: str) -> dict:
    """Check if removing top trades destroys net PnL."""
    if df.empty:
        return {"status": "NO_TRADES"}

    net_col = "net_9bps"
    arr = df[net_col].sort_values(ascending=False).values
    total_net = arr.sum()
    n = len(arr)

    # Remove top 1
    net_no_top1 = arr[1:].sum()
    pct_remain_1 = net_no_top1 / total_net * 100 if total_net != 0 else 0

    # Remove top 3
    remove_n = min(3, n)
    net_no_top3 = arr[remove_n:].sum()
    pct_remain_3 = net_no_top3 / total_net * 100 if total_net != 0 else 0

    # Remove top 5%
    remove_5pct = max(1, int(n * 0.05))
    net_no_top5pct = arr[remove_5pct:].sum()
    pct_remain_5pct = net_no_top5pct / total_net * 100 if total_net != 0 else 0

    # Top 3 contribution %
    top3_sum = arr[:3].sum()
    top3_pct = top3_sum / total_net * 100 if total_net > 0 else 0

    # Max single trade
    max_trade = arr[0]

    verdict = "PASS"
    warnings = []
    if pct_remain_1 < 70:
        verdict = "WARN"
        warnings.append(f"remove_top_1 leaves only {pct_remain_1:.0f}% of net")
    if pct_remain_3 < 50:
        verdict = "FAIL"
        warnings.append(f"remove_top_3 leaves only {pct_remain_3:.0f}% of net")
    if top3_pct > 40 and total_net > 0:
        warnings.append(f"top 3 trades = {top3_pct:.0f}% of net — concentration risk")

    return {
        "n_trades": n,
        "total_net_9bps": round(total_net, 0),
        "max_single_trade": round(max_trade, 0),
        "top3_pct_of_net": round(top3_pct, 0),
        "net_no_top1": round(net_no_top1, 0),
        "pct_remain_no_top1": round(pct_remain_1, 0),
        "net_no_top3": round(net_no_top3, 0),
        "pct_remain_no_top3": round(pct_remain_3, 0),
        "net_no_top5pct": round(net_no_top5pct, 0),
        "pct_remain_no_top5pct": round(pct_remain_5pct, 0),
        "verdict": verdict,
        "warnings": warnings,
    }


# ── B. Time Slice ──────────────────────────────────

def audit_time_slice(df: pd.DataFrame, name: str) -> dict:
    """Check performance across early/middle/late periods."""
    if df.empty:
        return {"status": "NO_TRADES"}

    df = df.copy()
    df["entry_ts_dt"] = pd.to_datetime(df["entry_ts"])
    df = df.sort_values("entry_ts_dt")

    n = len(df)
    slice_size = max(1, n // 3)
    slices = {
        "early": df.iloc[:slice_size],
        "middle": df.iloc[slice_size:2*slice_size],
        "late": df.iloc[2*slice_size:],
    }

    net_col = "net_9bps"
    result = {}
    any_neg = False
    for label, sub in slices.items():
        if sub.empty:
            result[label] = {"n": 0, "net": 0, "pf": 0, "hit": 0}
            continue
        arr = sub[net_col].values
        pos = arr[arr > 0].sum()
        neg = abs(arr[arr < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        result[label] = {
            "n": len(sub),
            "net": round(arr.sum(), 0),
            "pf": round(pf, 2),
            "hit": round((arr > 0).mean() * 100, 0),
            "date_range": f"{sub['entry_ts_dt'].min().date()} → {sub['entry_ts_dt'].max().date()}",
        }
        if arr.sum() < 0:
            any_neg = True

    result["verdict"] = "WARN" if any_neg else "PASS"
    result["any_slice_negative"] = any_neg

    return result


# ── C. Cost Stress ─────────────────────────────────

def audit_cost_stress(df: pd.DataFrame, name: str) -> dict:
    """Test cost levels + entry delay + extra slippage."""
    if df.empty:
        return {"status": "NO_TRADES"}

    result = {}

    # Cost levels
    for cost in [9, 12, 15]:
        col = f"net_{cost}bps"
        arr = df[col].values
        pos = arr[arr > 0].sum()
        neg = abs(arr[arr < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        alive = arr.sum() > 0
        result[f"cost_{cost}bps"] = {
            "net": round(arr.sum(), 0),
            "pf": round(pf, 2),
            "hit": round((arr > 0).mean() * 100, 1),
            "alive": alive,
        }

    # Entry delay: simulate entry 1 bar later (current entry price vs next bar close)
    if "entry_ts" in df.columns and len(df) > 0:
        # Use exit price from the next bar as simulation
        # For simplicity: add 5 bps extra cost to simulate delay+slippage
        col = "net_9bps"
        delay_arr = df[col].values - 5  # 5bps extra = 1-bar delay penalty
        pos = delay_arr[delay_arr > 0].sum()
        neg = abs(delay_arr[delay_arr < 0].sum())
        pf_d = pos / neg if neg > 0 else float("inf")
        result["delay_1bar_plus5bps"] = {
            "net": round(delay_arr.sum(), 0),
            "pf": round(pf_d, 2),
            "hit": round((delay_arr > 0).mean() * 100, 1),
            "alive": delay_arr.sum() > 0,
        }

    # Worst case: 15bps + delay
    worst = df["net_15bps"].values - 5
    pos = worst[worst > 0].sum()
    neg = abs(worst[worst < 0].sum())
    pf_w = pos / neg if neg > 0 else float("inf")
    result["worst_15bps_delay"] = {
        "net": round(worst.sum(), 0),
        "pf": round(pf_w, 2),
        "hit": round((worst > 0).mean() * 100, 1),
        "alive": worst.sum() > 0,
    }

    # Verdict
    base_alive = result.get("cost_9bps", {}).get("alive", False)
    stress_alive = result.get("cost_12bps", {}).get("alive", False)
    worst_alive = result.get("worst_15bps_delay", {}).get("alive", False)

    if base_alive and stress_alive and worst_alive:
        result["verdict"] = "PASS_STRONG"
    elif base_alive and stress_alive:
        result["verdict"] = "PASS"
    elif base_alive:
        result["verdict"] = "PASS_BASELINE"
    else:
        result["verdict"] = "FAIL"

    return result


# ── D. Event Conflict ──────────────────────────────

def audit_conflicts(all_dfs: dict[str, pd.DataFrame]) -> dict:
    """Check for overlapping events on same symbol+timestamp."""
    # Combine all trades
    all_trades = []
    for evt_name, df in all_dfs.items():
        if df.empty:
            continue
        sub = df[["symbol", "entry_ts", "event_type", "direction"]].copy()
        all_trades.append(sub)

    if len(all_trades) <= 1:
        return {"conflict_count": 0, "verdict": "PASS (single event type)"}

    combined = pd.concat(all_trades)
    combined["ts_str"] = combined["entry_ts"].astype(str)

    # Find duplicates on (symbol, timestamp)
    dupes = combined.groupby(["symbol", "ts_str"]).agg(
        event_types=("event_type", list),
        directions=("direction", list),
        count=("event_type", "count"),
    )
    conflicts = dupes[dupes["count"] > 1]

    conflict_details = []
    for (sym, ts), row in conflicts.iterrows():
        dirs = row["directions"]
        if len(set(dirs)) > 1:  # different directions = real conflict
            conflict_details.append({
                "symbol": sym,
                "timestamp": ts,
                "event_types": row["event_types"],
                "directions": dirs,
                "action": "SKIP — direction conflict",
            })

    return {
        "total_trades_combined": len(combined),
        "overlapping_events": len(conflicts),
        "direction_conflicts": len(conflict_details),
        "details": conflict_details[:10],
        "verdict": "WARN" if conflict_details else "PASS",
        "recommended_priority": [
            "1. Deleveraging Reversal",
            "2. OI Shock Absorption (short squeeze)",
            "3. RS Shock",
        ],
        "conflict_rule": "Direction conflict → SKIP. Same direction → accept highest-score event.",
    }


# ── E. Shadow Fields ────────────────────────────────

def audit_shadow_fields(df: pd.DataFrame, name: str) -> dict:
    """Check which shadow mode fields are present."""
    required_fields = [
        "symbol", "event_type", "direction", "event_score",
        "regime", "entry_ts", "exit_ts",
        "gross_bps", "net_9bps",
    ]
    recommended_fields = [
        "regime_confidence", "entry_price_simulated", "exit_price_simulated",
        "expected_cost_bps", "reject_reason", "future_return", "net_after_cost",
        "oi_z", "vol_z", "close_loc",
    ]

    present = [f for f in required_fields if f in df.columns]
    present_rec = [f for f in recommended_fields if f in df.columns]

    missing = [f for f in required_fields if f not in df.columns]
    missing_rec = [f for f in recommended_fields if f not in df.columns]

    return {
        "required_present": len(present),
        "required_missing": missing,
        "recommended_present": len(present_rec),
        "recommended_missing": missing_rec,
        "verdict": "PASS" if not missing else "FAIL",
    }


# ── Main ────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("SHADOW READINESS AUDIT — 3 Event Types")
    print("=" * 70)

    all_dfs = run_all_trades()

    full_report = {}

    for name, df in all_dfs.items():
        print(f"\n{'─' * 70}")
        print(f"## {name}")
        print(f"   Trades: {len(df)}")

        # A. Extreme
        r_a = audit_extreme(df, name)
        print(f"\n   A. EXTREME CONTRIBUTION  [{r_a.get('verdict', '?')}]")
        print(f"      Total net (9bps): {r_a.get('total_net_9bps', 0):+.0f}")
        print(f"      Max single trade:  {r_a.get('max_single_trade', 0):+.0f}")
        print(f"      Top 3 = {r_a.get('top3_pct_of_net', 0):.0f}% of net")
        print(f"      -top1: {r_a.get('net_no_top1', 0):+.0f} ({r_a.get('pct_remain_no_top1', 0):.0f}% remains)")
        print(f"      -top3: {r_a.get('net_no_top3', 0):+.0f} ({r_a.get('pct_remain_no_top3', 0):.0f}% remains)")
        print(f"      -top5%: {r_a.get('net_no_top5pct', 0):+.0f} ({r_a.get('pct_remain_no_top5pct', 0):.0f}% remains)")
        for w in r_a.get("warnings", []):
            print(f"      ⚠ {w}")

        # B. Time Slice
        r_b = audit_time_slice(df, name)
        print(f"\n   B. TIME SLICE  [{r_b.get('verdict', '?')}]")
        for period in ["early", "middle", "late"]:
            s = r_b.get(period, {})
            if s.get("n", 0) > 0:
                print(f"      {period:8s}: n={s['n']:3d} Net={s['net']:+6.0f} PF={s['pf']:.2f} Hit={s['hit']}%  [{s.get('date_range','')}]")

        # C. Cost Stress
        r_c = audit_cost_stress(df, name)
        print(f"\n   C. COST STRESS  [{r_c.get('verdict', '?')}]")
        for k, v in r_c.items():
            if k.startswith("cost_") or "delay" in k or "worst" in k:
                mark = "✓" if v.get("alive") else "✗"
                print(f"      {k:25s}: Net={v['net']:+6.0f} PF={v['pf']:.2f} Hit={v['hit']:.0f}% {mark}")

        # E. Shadow Fields
        r_e = audit_shadow_fields(df, name)
        print(f"\n   E. SHADOW FIELDS  [{r_e.get('verdict')}]")
        if r_e.get("required_missing"):
            print(f"      MISSING REQUIRED: {r_e['required_missing']}")
        if r_e.get("recommended_missing"):
            print(f"      MISSING RECOMMENDED: {r_e['recommended_missing']}")

        full_report[name] = {"extreme": r_a, "time_slice": r_b, "cost_stress": r_c, "shadow_fields": r_e}

    # D. Conflicts (cross-event)
    print(f"\n{'─' * 70}")
    print(f"## D. EVENT CONFLICTS (cross-event)")
    r_d = audit_conflicts(all_dfs)
    print(f"   Overlapping events: {r_d['overlapping_events']}")
    print(f"   Direction conflicts: {r_d['direction_conflicts']}")
    print(f"   Verdict: {r_d['verdict']}")
    print(f"   Priority: {r_d['recommended_priority']}")
    if r_d["details"]:
        print(f"   Examples:")
        for d in r_d["details"][:5]:
            print(f"      {d['symbol']} @ {d['timestamp']}: {d['event_types']} dirs={d['directions']} → {d['action']}")

    full_report["conflicts"] = r_d

    # Save report
    report_path = ROOT / "logs" / "shadow_readiness_audit.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Report saved: {report_path}")
    print(f"{'=' * 70}")
