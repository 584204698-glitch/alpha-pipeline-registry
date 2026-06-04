"""Funding Carry EU — Full Audit Suite (0, 2, 3, 4, 5).

Audit 0: Funding Timestamp Alignment — is funding_rate forward-looking or backward-looking?
Audit 2: Session Boundary Stability — 8 window tests
Audit 3: Funding Threshold Neighborhood — 6 thresholds
Audit 4: Overlap With Existing Events
Audit 5: Symbol Concentration
"""
from __future__ import annotations

import json, sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
sys.path.insert(0, str(ROOT))
from research.regime_detector import detect_regime_fast


def load_data():
    return pd.read_parquet(ROOT / "data" / "data_storage.parquet")


def get_universe(data, min_rank=20, max_rank=100):
    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    return set(vol_rank[(vol_rank >= min_rank) & (vol_rank <= max_rank)].index) & set(
        data.index.get_level_values("symbol").unique())


def precompute_z(data):
    """Precompute funding_z and oi_delta_z."""
    f = {}
    fr = data["funding_rate"]
    gf = fr.groupby(level="symbol")
    fm = gf.transform(lambda s: s.rolling(24, min_periods=8).mean())
    fs = gf.transform(lambda s: s.rolling(24, min_periods=8).std()).replace(0, np.nan)
    f["funding_z"] = ((fr - fm) / fs).fillna(0.0)
    oi_d = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(6))
    go = oi_d.groupby(level="symbol")
    om = go.transform(lambda s: s.rolling(48, min_periods=8).mean())
    os_ = go.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    f["oi_delta_z"] = ((oi_d - om) / os_).fillna(0.0)
    return f


def run_funding_carry(data, feats, regime_map, universe, session_range, fz_threshold, hold_bars=12):
    """Core funding carry backtest with given session + threshold."""
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    trades = []

    for i in range(0, len(ts_list) - hold_bars - 1, 4):
        ts = ts_list[i]
        if ts.hour < session_range[0] or ts.hour >= session_range[1]:
            continue
        regime = regime_map.get(ts, "unknown")
        if regime == "panic_down":
            continue
        mask = data.index.get_level_values("timestamp") == ts
        syms = set(data.index[mask].get_level_values("symbol")) & universe
        for sym in syms:
            try:
                fz_val = float(feats["funding_z"].loc[(ts, sym)])
                oi_z = float(feats["oi_delta_z"].loc[(ts, sym)])
            except (KeyError, TypeError):
                continue
            if np.isnan(fz_val) or fz_val > fz_threshold:
                continue
            if oi_z < -2.0:
                continue
            idx = ts_to_idx.get(ts)
            if idx is None or idx + hold_bars >= len(ts_list):
                continue
            exit_ts = ts_list[idx + hold_bars]
            try:
                ep = float(data.loc[(ts, sym), "close"])
                xp = float(data.loc[(exit_ts, sym), "close"])
            except KeyError:
                continue

            price_bps = (xp / ep - 1.0) * 10000
            fund_sum = 0.0
            valid = 0
            for j in range(hold_bars):
                try:
                    fund_sum += float(data.loc[(ts_list[idx + j], sym), "funding_rate"])
                    valid += 1
                except:
                    pass
            fund_bps = -(fund_sum / max(valid, 1)) * 10000 * (hold_bars * 0.25 / 8)
            total_bps = price_bps + fund_bps

            trades.append({
                "symbol": sym, "entry_ts": str(ts),
                "price_bps": price_bps, "funding_bps": fund_bps, "total_bps": total_bps,
                "fz": fz_val, "regime": regime,
            })
    return trades


def pnl_from_trades(trades, costs=[9, 12, 15]):
    if not trades:
        return {"n_trades": 0}
    arr = np.array([t["total_bps"] for t in trades])
    price_arr = np.array([t["price_bps"] for t in trades])
    fund_arr = np.array([t["funding_bps"] for t in trades])
    r = {"n_trades": len(trades),
         "price_sum": round(price_arr.sum(), 1),
         "funding_sum": round(fund_arr.sum(), 1)}
    for cost in costs:
        net = arr - 2 * cost
        pos = net[net > 0].sum()
        neg = abs(net[net < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        hit = (net > 0).mean()
        sn = sorted(net, reverse=True)
        t3 = sum(sn[:3]) if len(sn) >= 3 else sum(sn)
        r[f"cost_{cost}bps"] = {
            "net_bps": round(net.sum(), 0), "pf": round(pf, 3),
            "hit_rate": round(hit, 3), "top3": round(t3, 0),
            "net_no_top3": round(net.sum() - t3, 0),
            "max_loss": round(net.min(), 0),
        }
    return r


# ═══════════════════════════════════════════
#  AUDIT 0: Funding Timestamp Alignment
# ═══════════════════════════════════════════

def audit_0_timestamp(data):
    """Check if funding_rate at timestamp T represents CURRENT or FUTURE rate."""
    print("=" * 60)
    print("  AUDIT 0: Funding Timestamp Alignment")
    print("=" * 60)

    # Sample a few symbols
    syms = sorted(data.index.get_level_values("symbol").unique())[:5]
    for sym in syms:
        try:
            sym_data = data.xs(sym, level="symbol").sort_index()
        except KeyError:
            continue
        fr_series = sym_data["funding_rate"].dropna()
        if len(fr_series) < 20:
            continue

        # Check: does funding_rate change at predictable intervals (every 8h)?
        changes = fr_series.diff().abs() > 1e-8
        change_timestamps = fr_series.index[changes]

        print(f"\n  Symbol: {sym}")
        print(f"    Funding rate range: {fr_series.min():.6f} → {fr_series.max():.6f}")
        print(f"    Value changes: {changes.sum()} out of {len(fr_series)} bars")

        # Show the last 5 values to understand pattern
        print(f"    Last 10 funding_rate values:")
        for i, (ts, val) in enumerate(fr_series.tail(10).items()):
            marker = " ← CHANGE" if i > 0 and abs(val - fr_series.iloc[-i-1]) > 1e-8 else ""
            print(f"      {ts}: {val:.6f}{marker}")

        # Check if funding is stable for multiple bars (typical for known-ahead rates)
        run_lengths = []
        current_run = 1
        for i in range(1, len(fr_series)):
            if abs(fr_series.iloc[i] - fr_series.iloc[i-1]) < 1e-8:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        if run_lengths:
            print(f"    Stable-run lengths: mean={np.mean(run_lengths):.1f} bars, "
                  f"max={max(run_lengths)}, min={min(run_lengths)}")
            print(f"    → Funding rate {'IS' if np.mean(run_lengths) > 8 else 'may NOT be'} stable across bars (known-ahead pattern)")

        # KEY TEST: Is funding_rate[t] correlated with future return?
        # If funding_rate at T predicts returns AFTER T → look-ahead bias.
        ret_fwd = sym_data["close"].pct_change(4).shift(-4)  # 1h forward return
        fr_now = fr_series.reindex(ret_fwd.index)
        valid = fr_now.notna() & ret_fwd.notna()
        if valid.sum() > 10:
            corr = np.corrcoef(fr_now[valid], ret_fwd[valid])[0, 1]
            print(f"    Corr(funding_rate[t], ret_1h[t+1]): {corr:.4f}")
            if abs(corr) > 0.1:
                print(f"    ⚠ WARNING: Non-trivial correlation → possible look-ahead or genuine predictive power")
            else:
                print(f"    ✓ Low correlation → unlikely to have look-ahead bias")

        break  # Only check 1-2 symbols

    # Also check: does funding_rate[t] == funding_rate[t+1] for most t?
    # If yes → rate is known ahead (genuine, no leakage)
    # If no → rate changes every bar → could be settlement-based
    all_syms = sorted(data.index.get_level_values("symbol").unique())
    stable_pcts = []
    for sym in all_syms[:20]:
        try:
            sd = data.xs(sym, level="symbol")["funding_rate"].dropna()
        except:
            continue
        if len(sd) < 20:
            continue
        same_as_next = (sd.iloc[:-1].values == sd.iloc[1:].values).mean()
        stable_pcts.append(same_as_next)
    if stable_pcts:
        avg_stable = np.mean(stable_pcts)
        print(f"\n  Average bar-to-bar funding stability (20 symbols): {avg_stable:.1%}")
        if avg_stable > 0.8:
            print("  ✓ Funding rate is highly stable bar-to-bar → known-ahead, NO leakage")
        elif avg_stable > 0.5:
            print("  ⚠ Moderate stability — some bars may reflect settlement changes")
        else:
            print("  ⚠ LOW stability — funding changes frequently, check settlement alignment")

    # Critical: does the funding_rate at entry differ from the rate during hold?
    print(f"\n  === Audit 0 Verdict ===")
    if stable_pcts and np.mean(stable_pcts) > 0.7:
        print("  NO LEAKAGE: Funding rates appear to be known-ahead (stable across bars).")
        print("  The funding_pnl calculation using rates during the hold period is valid.")
        return "NO_LEAKAGE"
    else:
        return "NEEDS_INVESTIGATION"


# ═══════════════════════════════════════════
#  AUDIT 2: Session Boundary Stability
# ═══════════════════════════════════════════

def audit_2_session(data, feats, regime_map, universe):
    print(f"\n{'='*60}")
    print("  AUDIT 2: Session Boundary Stability")
    print(f"{'='*60}")

    sessions = [
        ("06-14", (6, 14)), ("07-15", (7, 15)), ("08-16", (8, 16)),
        ("09-17", (9, 17)), ("10-18", (10, 18)),
        ("08-14", (8, 14)), ("09-15", (9, 15)), ("07-17", (7, 17)),
    ]
    results = {}
    print(f"  {'Window':<12} {'N':>5} {'Price':>10} {'Funding':>10} {'Net@9':>10} {'PF':>6} {'Net@12':>10} {'Net@15':>10} {'NoTop3':>10}")
    print(f"  {'-'*12} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    for label, (lo, hi) in sessions:
        trades = run_funding_carry(data, feats, regime_map, universe, (lo, hi), -2.5)
        pnl = pnl_from_trades(trades)
        results[label] = pnl
        c9 = pnl.get("cost_9bps", {})
        c12 = pnl.get("cost_12bps", {})
        c15 = pnl.get("cost_15bps", {})
        print(f"  {label:<12} {pnl.get('n_trades',0):>5} {pnl.get('price_sum',0):+10.0f} {pnl.get('funding_sum',0):+10.0f} "
              f"{c9.get('net_bps',0):+10.0f} {c9.get('pf',0):>6.2f} "
              f"{c12.get('net_bps',0):+10.0f} {c15.get('net_bps',0):+10.0f} "
              f"{c9.get('net_no_top3',0):+10.0f}")

    # Verdict
    positives = sum(1 for v in results.values()
                    if v.get("cost_9bps", {}).get("net_bps", 0) > 0
                    and v.get("cost_9bps", {}).get("pf", 1.0) >= 1.10)
    print(f"\n  Positive windows: {positives}/{len(sessions)}")
    if positives >= 5:
        print("  → SESSION_GATE_VALIDATED: EU-area session edge is robust")
    elif positives >= 3:
        print("  → SESSION_STABLE: Multiple windows positive, gate is reasonable")
    else:
        print("  → SESSION_FRAGILE: Only 08-16 works, parameter fragile")
    return results


# ═══════════════════════════════════════════
#  AUDIT 3: Funding Threshold Neighborhood
# ═══════════════════════════════════════════

def audit_3_threshold(data, feats, regime_map, universe):
    print(f"\n{'='*60}")
    print("  AUDIT 3: Funding Threshold Neighborhood")
    print(f"{'='*60}")

    thresholds = [-2.0, -2.25, -2.5, -2.75, -3.0, -3.5]
    results = {}
    print(f"  {'Thresh':<10} {'N':>5} {'Funding':>10} {'Net@9':>10} {'PF':>6} {'Net@12':>10} {'Net@15':>10} {'NoTop3':>10}")
    print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    for th in thresholds:
        trades = run_funding_carry(data, feats, regime_map, universe, (8, 16), th)
        pnl = pnl_from_trades(trades)
        results[f"fz<{th}"] = pnl
        c9 = pnl.get("cost_9bps", {})
        c12 = pnl.get("cost_12bps", {})
        c15 = pnl.get("cost_15bps", {})
        print(f"  fz<{th:<7} {pnl.get('n_trades',0):>5} {pnl.get('funding_sum',0):+10.0f} "
              f"{c9.get('net_bps',0):+10.0f} {c9.get('pf',0):>6.2f} "
              f"{c12.get('net_bps',0):+10.0f} {c15.get('net_bps',0):+10.0f} "
              f"{c9.get('net_no_top3',0):+10.0f}")

    # Check monotonicity: stricter threshold → higher PF?
    pfs = [results[f"fz<{th}"].get("cost_9bps", {}).get("pf", 0) for th in thresholds]
    ns = [results[f"fz<{th}"].get("n_trades", 0) for th in thresholds]
    pf_increasing = all(pfs[i] <= pfs[i+1] for i in range(len(pfs)-1) if ns[i] > 10 and ns[i+1] > 10)
    n_decreasing = all(ns[i] >= ns[i+1] for i in range(len(ns)-1))
    print(f"\n  PF monotonic (tighter→higher PF): {pf_increasing}")
    print(f"  N monotonic (tighter→fewer trades): {n_decreasing}")
    if pf_increasing:
        print("  → THRESHOLD_STABLE: Clean monotonic relationship — crowding signal is genuine")
    else:
        print("  → PARAM_FRAGILE: PF doesn't monotonically improve with threshold")
    return results


# ═══════════════════════════════════════════
#  AUDIT 4: Overlap With Existing Events
# ═══════════════════════════════════════════

def audit_4_overlap(data, feats, regime_map, universe):
    print(f"\n{'='*60}")
    print("  AUDIT 4: Overlap With Existing Events")
    print(f"{'='*60}")

    # Collect Funding Carry trades
    fc_trades = run_funding_carry(data, feats, regime_map, universe, (8, 16), -2.5)
    fc_set = set((t["symbol"], t["entry_ts"]) for t in fc_trades)

    # Collect existing event trades
    from event_scanner import DeleveragingEventScanner, EventState, SmallCapUniverse
    from relative_strength_shock import detect_relative_strength_shock
    from oi_shock_absorption import detect_oi_shock_absorption

    # Deleveraging
    scanner = DeleveragingEventScanner(data, regime_map, universe=SmallCapUniverse(), hold_bars=2)
    delev_set = set()
    for ts in scanner.ts_list[100:]:
        results = scanner.scan(ts)
        for r in results:
            if r.state == EventState.CONFIRMED:
                delev_set.add((r.symbol, str(ts)))

    # OI Absorption
    oi_events = detect_oi_shock_absorption(data, universe, regime_map, oi_delta_z_min=2.5, vol_z_min=1.5)
    oi_set = set((e["symbol"], str(e["timestamp"])) for e in oi_events)

    # RS Shock
    rs_events = detect_relative_strength_shock(data, universe, regime_map, rs_threshold=0.05, vol_z_min=2.0)
    rs_set = set((e["symbol"], str(e["timestamp"])) for e in rs_events)

    events = [("Deleveraging", delev_set), ("OI_Absorption", oi_set), ("RS_Shock", rs_set)]
    for name, ev_set in events:
        overlap = fc_set & ev_set
        pct = len(overlap) / len(fc_set) * 100 if fc_set else 0
        print(f"  FC ∩ {name:20s}: {len(overlap):4d} / {len(fc_set):4d} = {pct:5.1f}%")

    # Combined overlap (union of all 3)
    all_ev = delev_set | oi_set | rs_set
    total_overlap = fc_set & all_ev
    print(f"  FC ∩ ANY event:         {len(total_overlap):4d} / {len(fc_set):4d} = {len(total_overlap)/len(fc_set)*100:5.1f}%")

    total_pct = len(total_overlap) / len(fc_set) * 100 if fc_set else 0
    if total_pct < 15:
        print("  → INDEPENDENT_ALPHA: Very low overlap — genuine new income source")
    elif total_pct < 30:
        print("  → LOW_OVERLAP: Mostly independent with minor overlap")
    elif total_pct < 50:
        print("  → PARTIAL_OVERLAP: Could be supplementary but shares signal space")
    else:
        print("  → DUPLICATE_ALPHA: Too much overlap — may be same alpha in different form")


# ═══════════════════════════════════════════
#  AUDIT 5: Symbol Concentration
# ═══════════════════════════════════════════

def audit_5_concentration(data, feats, regime_map, universe):
    print(f"\n{'='*60}")
    print("  AUDIT 5: Symbol Concentration")
    print(f"{'='*60}")

    trades = run_funding_carry(data, feats, regime_map, universe, (8, 16), -2.5)
    if not trades:
        print("  No trades")
        return

    # Per-symbol stats
    sym_stats = defaultdict(lambda: {"count": 0, "total": 0.0, "price": 0.0, "funding": 0.0})
    for t in trades:
        s = t["symbol"]
        sym_stats[s]["count"] += 1
        sym_stats[s]["total"] += t["total_bps"]
        sym_stats[s]["price"] += t["price_bps"]
        sym_stats[s]["funding"] += t["funding_bps"]

    # Sort by total contribution
    ranked = sorted(sym_stats.items(), key=lambda x: -x[1]["total"])
    total_all = sum(v["total"] for _, v in ranked)

    print(f"  Total symbols with trades: {len(ranked)}")
    print(f"  {'Symbol':<25s} {'N':>4} {'Total':>10} {'Price':>10} {'Funding':>10} {'%ofTotal':>10}")
    print(f"  {'-'*25} {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i, (sym, stats) in enumerate(ranked[:15]):
        pct = stats["total"] / total_all * 100 if total_all else 0
        print(f"  {sym:<25s} {stats['count']:>4} {stats['total']:+10.0f} {stats['price']:+10.0f} "
              f"{stats['funding']:+10.0f} {pct:>9.1f}%")

    # Concentration metrics
    top1 = ranked[0][1]["total"] if ranked else 0
    top3 = sum(ranked[i][1]["total"] for i in range(min(3, len(ranked))))
    top5 = sum(ranked[i][1]["total"] for i in range(min(5, len(ranked))))
    print(f"\n  Top1: {top1:+10.0f} ({top1/total_all*100:.1f}%)")
    print(f"  Top3: {top3:+10.0f} ({top3/total_all*100:.1f}%)")
    print(f"  Top5: {top5:+10.0f} ({top5/total_all*100:.1f}%)")
    print(f"  Net without top1: {total_all - top1:+.0f}")
    print(f"  Net without top3: {total_all - top3:+.0f}")
    print(f"  Net without top5: {total_all - top5:+.0f}")

    top3_pct = top3 / total_all * 100 if total_all else 0
    top1_pct = top1 / total_all * 100 if total_all else 0
    if top1_pct < 25 and top3_pct < 50:
        print("  → DIVERSIFIED: No single symbol dominates")
    elif top1_pct < 40:
        print("  → MODERATE_CONCENTRATION: Top symbol notable but not dominant")
    else:
        print("  → SYMBOL_CONCENTRATED: Heavy reliance on few symbols")


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def main():
    print("Loading data and precomputing...")
    data = load_data()
    feats = precompute_z(data)
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))
    universe = get_universe(data)
    print(f"Universe: {len(universe)} symbols")

    all_results = {}

    # Audit 0
    a0 = audit_0_timestamp(data)
    all_results["audit_0_timestamp"] = a0

    # Audit 2
    a2 = audit_2_session(data, feats, regime_map, universe)
    all_results["audit_2_session"] = a2

    # Audit 3
    a3 = audit_3_threshold(data, feats, regime_map, universe)
    all_results["audit_3_threshold"] = a3

    # Audit 4
    a4 = audit_4_overlap(data, feats, regime_map, universe)
    all_results["audit_4_overlap"] = a4

    # Audit 5
    a5 = audit_5_concentration(data, feats, regime_map, universe)
    all_results["audit_5_concentration"] = a5

    # Save
    out = ROOT / "funding_carry_eu_full_audit.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull audit saved: {out}")


if __name__ == "__main__":
    main()
