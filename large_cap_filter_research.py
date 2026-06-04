"""Large-Cap Regime / Filter Research — A through E.

Tests large-cap state variables as filters/regime gates for 3 small-cap event types.
Produces: large_cap_regime_filter_report.json
"""

from __future__ import annotations

import json, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
sys.path.insert(0, str(ROOT))

from backtest_engine import BacktestEngine
from research.regime_detector import detect_regime_fast
from event_scanner import backtest_scanner, DeleveragingEventScanner, EventState
from relative_strength_shock import detect_relative_strength_shock
from oi_shock_absorption import detect_oi_shock_absorption


# ═══════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════

def load_data():
    engine = BacktestEngine(str(ROOT))
    data = engine._load_data()
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))

    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    small_syms = set(vol_rank[(vol_rank >= 20) & (vol_rank <= 100)].index)

    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}

    return data, regimes, regime_map, small_syms, ts_list, ts_to_idx


# ═══════════════════════════════════════════════════════
# TRADE COLLECTION (all 3 events)
# ═══════════════════════════════════════════════════════

def collect_all_trades(data, regime_map, small_syms, ts_list, ts_to_idx):
    trades = []

    # 1. Deleveraging Reversal
    scanner = DeleveragingEventScanner(data, regime_map, hold_bars=2)
    for i, ts in enumerate(ts_list):
        results = scanner.scan(ts)
        for r in results:
            if r.state != EventState.CONFIRMED:
                continue
            entry_ts = ts
            exit_idx = i + 2
            if exit_idx >= len(ts_list):
                continue
            exit_ts = ts_list[exit_idx]
            try:
                entry_px = float(data.loc[(entry_ts, r.symbol), "close"])
                exit_px = float(data.loc[(exit_ts, r.symbol), "close"])
            except KeyError:
                continue
            ret = (exit_px / entry_px) - 1.0
            trades.append({
                "symbol": r.symbol,
                "entry_ts": ts,
                "exit_ts": exit_ts,
                "event_type": "DeleveragingReversal",
                "direction": "long",
                "regime": regime_map.get(ts, "unknown"),
                "event_score": r.event_score,
                "gross_bps": ret * 10000,
                "oi_z": r.metrics.get("oi_z", 0),
                "ret_1h": r.metrics.get("ret_1h", 0),
                "vol_z": r.metrics.get("vol_z", 0),
            })

    # 2. RS Shock
    rs_events = detect_relative_strength_shock(
        data, small_syms, regime_map,
        rs_threshold=0.05, vol_z_min=2.0, oi_delta_z_min=0.5,
        close_loc_min=0.50, event_score_min=0.60, cooldown_bars=8,
    )
    for ev in rs_events:
        ts = ev["timestamp"]; sym = ev["symbol"]; direction = ev["direction"]
        ts_idx = ts_to_idx.get(ts)
        if ts_idx is None or ts_idx >= len(ts_list) - 2:
            continue
        try:
            ep = float(data.loc[(ts, sym), "close"])
            xp = float(data.loc[(ts_list[ts_idx + 2], sym), "close"])
        except KeyError:
            continue
        ret = (xp / ep) - 1.0
        if direction == "short":
            ret = -ret
        trades.append({
            "symbol": sym,
            "entry_ts": ts,
            "exit_ts": ts_list[ts_idx + 2],
            "event_type": "RelativeStrengthShock",
            "direction": direction,
            "regime": ev.get("regime", "unknown"),
            "event_score": ev["event_score"],
            "gross_bps": ret * 10000,
            "rs_pct": ev.get("rs_pct", 0),
            "vol_z": ev.get("vol_z", 0),
            "oi_z": ev.get("oi_delta_z", 0),
            "ret_1h": ev.get("ret_1h_pct", 0) / 100,
        })

    # 3. OI Shock Absorption
    oi_events = detect_oi_shock_absorption(
        data, small_syms, regime_map,
        oi_delta_z_min=2.5, vol_z_min=1.5, cooldown_bars=8,
    )
    for ev in oi_events:
        ts = ev["timestamp"]; sym = ev["symbol"]
        ts_idx = ts_to_idx.get(ts)
        if ts_idx is None or ts_idx >= len(ts_list) - 2:
            continue
        try:
            ep = float(data.loc[(ts, sym), "close"])
            xp = float(data.loc[(ts_list[ts_idx + 2], sym), "close"])
        except KeyError:
            continue
        ret = (xp / ep) - 1.0
        if ev["direction"] == "short":
            ret = -ret
        trades.append({
            "symbol": sym,
            "entry_ts": ts,
            "exit_ts": ts_list[ts_idx + 2],
            "event_type": "OIShockAbsorption",
            "direction": "long",
            "regime": ev.get("regime", "unknown"),
            "event_score": ev["event_score"],
            "gross_bps": ret * 10000,
            "oi_z": ev.get("oi_z", 0),
            "vol_z": ev.get("vol_z", 0),
        })

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════
# METRICS COMPUTATION
# ═══════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame, cost_bps: float = 9.0) -> dict:
    """Compute full metrics for a trade DataFrame."""
    if df.empty:
        return {"n_trades": 0, "net_pnl": 0, "pf": 0, "hit_rate": 0, "max_loss": 0,
                "top1": 0, "top3": 0, "top3_pct": 0, "net_no_top3": 0, "avg_win": 0, "avg_loss": 0}

    net = df["gross_bps"].values - cost_bps
    n = len(net)
    pos = net[net > 0].sum()
    neg = abs(net[net < 0].sum())
    pf = pos / neg if neg > 0 else float("inf")

    sorted_net = np.sort(net)[::-1]
    top1 = sorted_net[0] if n >= 1 else 0
    top3 = sorted_net[:3].sum() if n >= 3 else sorted_net.sum()

    return {
        "n_trades": n,
        "net_pnl": round(net.sum(), 0),
        "pf": round(pf, 3),
        "hit_rate": round((net > 0).mean(), 3),
        "max_loss": round(net.min(), 0) if n > 0 else 0,
        "top1_contribution": round(top1, 0),
        "top3_contribution": round(top3, 0),
        "top3_pct_of_net": round(top3 / net.sum() * 100, 0) if net.sum() > 0 else 0,
        "net_without_top3": round(sorted_net[3:].sum(), 0) if n >= 3 else 0,
        "avg_win": round(net[net > 0].mean(), 1) if (net > 0).any() else 0,
        "avg_loss": round(net[net < 0].mean(), 1) if (net < 0).any() else 0,
    }


def compare_metrics(before: pd.DataFrame, after: pd.DataFrame, cost: float = 9.0) -> dict:
    """Compare before/after metrics for a filter."""
    b = compute_metrics(before, cost)
    a = compute_metrics(after, cost)
    rejected = before[~before.index.isin(after.index)] if not before.empty and not after.empty else pd.DataFrame()

    return {
        "before": b,
        "after": a,
        "delta_n": a["n_trades"] - b["n_trades"],
        "delta_net": a["net_pnl"] - b["net_pnl"],
        "delta_pf": round(a["pf"] - b["pf"], 3),
        "delta_hit": round(a["hit_rate"] - b["hit_rate"], 3),
        "delta_max_loss": a["max_loss"] - b["max_loss"],
        "delta_net_no_top3": a["net_without_top3"] - b["net_without_top3"],
        "rejected_count": len(rejected),
        "rejected_hindsight_net": round(rejected["gross_bps"].sum() - len(rejected) * cost, 0) if not rejected.empty else 0,
    }


# ═══════════════════════════════════════════════════════
# LARGE-CAP FEATURE BUILDER
# ═══════════════════════════════════════════════════════

def build_large_cap_features(data, ts_list):
    """Build large-cap features indexed by timestamp."""
    # BTC
    btc = data.loc[data.index.get_level_values("symbol") == "Binance:BTCUSDT", ["close", "volume", "open_interest", "funding_rate", "high", "low"]].copy()
    btc = btc.droplevel("symbol").sort_index()
    btc = btc[~btc.index.duplicated(keep="last")]

    btc_ret_15m = btc["close"].pct_change(1)
    btc_ret_1h = btc["close"].pct_change(4)
    btc_ret_4h = btc["close"].pct_change(16)
    btc_vol = btc_ret_15m.rolling(48).std()
    btc_ema = btc["close"].ewm(span=24, adjust=False).mean()
    btc_ema_slope = btc_ema.pct_change(4)
    btc_low_12 = btc["low"].rolling(12).min()
    btc_new_low = btc["close"] < btc_low_12.shift(1)

    # Funding
    btc_funding = btc["funding_rate"]
    btc_funding_z = (btc_funding - btc_funding.rolling(24).mean()) / btc_funding.rolling(24).std().replace(0, np.nan)

    # Market breadth
    ret_all = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(1))
    breadth = ret_all.groupby(level="timestamp").apply(lambda x: (x > 0).mean())

    # Market OI
    oi_delta = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(4))
    med_oi_delta = oi_delta.groupby(level="timestamp").median()
    med_funding = data["funding_rate"].groupby(level="timestamp").median()

    # OI collapse count (symbols with oi_delta_z < -1.5 per timestamp)
    def roll_z_per_symbol(series, window=48):
        g = series.groupby(level="symbol")
        m = g.transform(lambda s: s.rolling(window, min_periods=8).mean())
        s = g.transform(lambda s: s.rolling(window, min_periods=8).std()).replace(0, np.nan)
        return ((series - m) / s).fillna(0)
    oi_delta_z_all = roll_z_per_symbol(oi_delta)
    oi_collapse = (oi_delta_z_all < -1.5).groupby(level="timestamp").sum()
    oi_collapse = oi_collapse.reindex(ts_list).fillna(0)

    # Event density (need to precompute — done later per-event)

    features = pd.DataFrame(index=ts_list)
    features["btc_ret_15m"] = btc_ret_15m.reindex(ts_list)
    features["btc_ret_1h"] = btc_ret_1h.reindex(ts_list)
    features["btc_ret_4h"] = btc_ret_4h.reindex(ts_list)
    features["btc_realized_vol"] = btc_vol.reindex(ts_list)
    features["btc_ema_slope"] = btc_ema_slope.reindex(ts_list)
    features["btc_new_low"] = btc_new_low.reindex(ts_list)
    features["btc_funding_z"] = btc_funding_z.reindex(ts_list)
    features["breadth"] = breadth.reindex(ts_list)
    features["med_oi_delta"] = med_oi_delta.reindex(ts_list)
    features["med_funding"] = med_funding.reindex(ts_list)
    features["oi_collapse_count"] = oi_collapse
    features = features.ffill().fillna(0)

    return features


# ═══════════════════════════════════════════════════════
# FILTER DEFINITIONS
# ═══════════════════════════════════════════════════════

def apply_filter(trades_df: pd.DataFrame, features: pd.DataFrame,
                 filter_fn, filter_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a filter function. Returns (accepted, rejected)."""
    mask = trades_df["entry_ts"].map(lambda ts: filter_fn(ts, features)) if hasattr(filter_fn, '__call__') else pd.Series(True, index=trades_df.index)
    accepted = trades_df[mask].copy()
    rejected = trades_df[~mask].copy()
    return accepted, rejected


# ═══════════════════════════════════════════════════════
# MAIN RESEARCH
# ═══════════════════════════════════════════════════════

def run_filter_research():
    print("Loading data...")
    data, regimes, regime_map, small_syms, ts_list, ts_to_idx = load_data()
    features = build_large_cap_features(data, ts_list)

    print("Collecting trades...")
    all_trades = collect_all_trades(data, regime_map, small_syms, ts_list, ts_to_idx)
    print(f"  Total trades: {len(all_trades)}")
    for et in ["DeleveragingReversal", "RelativeStrengthShock", "OIShockAbsorption"]:
        sub = all_trades[all_trades["event_type"] == et]
        m = compute_metrics(sub)
        print(f"  {et:25s}: n={m['n_trades']:4d} Net={m['net_pnl']:+6.0f} PF={m['pf']:.2f} "
              f"Hit={m['hit_rate']*100:.0f}% top3={m['top3_pct_of_net']:.0f}% noTop3={m['net_without_top3']:+6.0f}")

    report = {"filters": []}

    # ═══ A. BTC Panic Down Filter ═══
    print("\n=== A. BTC Panic Down Filter ===")
    for label, btc_ret_thresh, btc_vol_thresh, breadth_thresh, oi_collapse_thresh in [
        ("mild", -0.02, None, None, None),
        ("moderate", -0.03, None, None, None),
        ("severe", -0.05, None, None, None),
        ("vol_spike", -0.02, 0.01, None, None),
        ("breadth_collapse", -0.02, None, 0.35, None),
        ("full_panic", -0.03, 0.01, 0.35, 8),
    ]:
        def make_panic_filter(ret_t, vol_t, br_t, oi_t):
            def f(ts, feats):
                if ts not in feats.index: return True
                row = feats.loc[ts]
                conditions = [row["btc_ret_1h"] < ret_t]
                if vol_t is not None:
                    conditions.append(row["btc_realized_vol"] > vol_t)
                if br_t is not None:
                    conditions.append(row["breadth"] < br_t)
                if oi_t is not None:
                    conditions.append(row["oi_collapse_count"] > oi_t)
                return not all(conditions)  # keep if NOT panic
            return f

        filt = make_panic_filter(btc_ret_thresh, btc_vol_thresh, breadth_thresh, oi_collapse_thresh)

        for et in all_trades["event_type"].unique():
            sub = all_trades[all_trades["event_type"] == et]
            acc, rej = apply_filter(sub, features, filt, f"A_panic_{label}")
            cmp = compare_metrics(sub, acc)
            if cmp["delta_n"] != 0:
                print(f"  A_panic_{label:20s} {et:25s}: n={cmp['before']['n_trades']:3d}→{cmp['after']['n_trades']:3d} "
                      f"Net={cmp['before']['net_pnl']:+5.0f}→{cmp['after']['net_pnl']:+5.0f} "
                      f"PF={cmp['before']['pf']:.2f}→{cmp['after']['pf']:.2f} "
                      f"noTop3={cmp['before']['net_without_top3']:+5.0f}→{cmp['after']['net_without_top3']:+5.0f} "
                      f"rej_n={cmp['rejected_count']} rej_net={cmp['rejected_hindsight_net']:+5.0f}")
                report["filters"].append({
                    "filter_name": f"A_panic_{label}",
                    "hypothesis": "BTC panic down → skip all small-cap longs",
                    "variables_used": [f"btc_ret_1h<{btc_ret_thresh}"] + 
                        ([f"btc_vol>{btc_vol_thresh}"] if btc_vol_thresh else []) +
                        ([f"breadth<{breadth_thresh}"] if breadth_thresh else []) +
                        ([f"oi_collapse>{oi_collapse_thresh}"] if oi_collapse_thresh else []),
                    "rule_definition": f"Reject if btc_ret_1h < {btc_ret_thresh}" + 
                        (f" AND vol > {btc_vol_thresh}" if btc_vol_thresh else "") +
                        (f" AND breadth < {breadth_thresh}" if breadth_thresh else "") +
                        (f" AND oi_collapse > {oi_collapse_thresh}" if oi_collapse_thresh else ""),
                    "affected_event_type": et,
                    "comparison": {str(k): v for k, v in cmp.items() if not isinstance(v, dict)},
                    "before_metrics": cmp["before"],
                    "after_metrics": cmp["after"],
                    "decision": "TO BE FILLED AFTER ANALYSIS",
                })

    # ═══ C. Systemic Deleveraging Filter ═══
    print("\n=== C. Systemic Deleveraging Filter ===")
    for N in [3, 5, 8, 10]:
        def make_sys_filter(n):
            def f(ts, feats):
                if ts not in feats.index: return True
                return feats.loc[ts, "oi_collapse_count"] <= n
            return f
        filt = make_sys_filter(N)
        for et in all_trades["event_type"].unique():
            sub = all_trades[all_trades["event_type"] == et]
            acc, rej = apply_filter(sub, features, filt, f"C_systemic_N{N}")
            cmp = compare_metrics(sub, acc)
            if cmp["delta_n"] != 0:
                print(f"  C_systemic_N{N:<3d}       {et:25s}: n={cmp['before']['n_trades']:3d}→{cmp['after']['n_trades']:3d} "
                      f"Net={cmp['before']['net_pnl']:+5.0f}→{cmp['after']['net_pnl']:+5.0f} "
                      f"PF={cmp['before']['pf']:.2f}→{cmp['after']['pf']:.2f} "
                      f"noTop3={cmp['before']['net_without_top3']:+5.0f}→{cmp['after']['net_without_top3']:+5.0f} "
                      f"rej_net={cmp['rejected_hindsight_net']:+5.0f}")
                report["filters"].append({
                    "filter_name": f"C_systemic_N{N}",
                    "hypothesis": f"Market-wide OI collapse > {N} coins → systemic deleveraging, skip small-cap longs",
                    "variables_used": [f"oi_collapse_count > {N}"],
                    "rule_definition": f"Reject if marketwide_oi_collapse_count > {N}",
                    "tested_thresholds": [N],
                    "affected_event_type": et,
                    "comparison": {str(k): v for k, v in cmp.items() if not isinstance(v, dict)},
                    "before_metrics": cmp["before"],
                    "after_metrics": cmp["after"],
                    "decision": "TO BE FILLED AFTER ANALYSIS",
                })

    # ═══ D. Funding Crowding Filter ═══
    print("\n=== D. Funding Crowding Filter ===")
    for fz_thresh in [1.5, 2.0, 2.5, 3.0]:
        def make_funding_filter(t):
            def f(ts, feats):
                if ts not in feats.index: return True
                return abs(feats.loc[ts, "btc_funding_z"]) <= t
            return f
        filt = make_funding_filter(fz_thresh)
        for et in all_trades["event_type"].unique():
            sub = all_trades[all_trades["event_type"] == et]
            acc, rej = apply_filter(sub, features, filt, f"D_funding_z_{fz_thresh}")
            cmp = compare_metrics(sub, acc)
            if cmp["delta_n"] != 0:
                print(f"  D_funding_z_{fz_thresh:<4.1f}      {et:25s}: n={cmp['before']['n_trades']:3d}→{cmp['after']['n_trades']:3d} "
                      f"Net={cmp['before']['net_pnl']:+5.0f}→{cmp['after']['net_pnl']:+5.0f} "
                      f"PF={cmp['before']['pf']:.2f}→{cmp['after']['pf']:.2f} "
                      f"noTop3={cmp['before']['net_without_top3']:+5.0f}→{cmp['after']['net_without_top3']:+5.0f} "
                      f"rej_net={cmp['rejected_hindsight_net']:+5.0f}")
                report["filters"].append({
                    "filter_name": f"D_funding_z_{fz_thresh}",
                    "hypothesis": "Extreme BTC funding → crowded positioning, deleveraging may still be in progress",
                    "variables_used": [f"|btc_funding_z| > {fz_thresh}"],
                    "rule_definition": f"Reject if |btc_funding_z| > {fz_thresh}",
                    "tested_thresholds": [fz_thresh],
                    "affected_event_type": et,
                    "comparison": {str(k): v for k, v in cmp.items() if not isinstance(v, dict)},
                    "before_metrics": cmp["before"],
                    "after_metrics": cmp["after"],
                    "decision": "TO BE FILLED AFTER ANALYSIS",
                })

    # ═══ E. Event Density Filter ═══
    print("\n=== E. Event Density as Market Temperature ===")
    # Count events per timestamp
    event_counts = all_trades.groupby("entry_ts").size()
    event_density = event_counts.reindex(ts_list).fillna(0)

    for density_thresh in [3, 5, 8, 10]:
        def make_density_filter(t):
            def f(ts, feats):
                return event_density.get(ts, 0) <= t
            return f
        filt = make_density_filter(density_thresh)
        for et in all_trades["event_type"].unique():
            sub = all_trades[all_trades["event_type"] == et]
            acc, rej = apply_filter(sub, features, filt, f"E_density_{density_thresh}")
            cmp = compare_metrics(sub, acc)
            if cmp["delta_n"] != 0:
                print(f"  E_density_{density_thresh:<3d}       {et:25s}: n={cmp['before']['n_trades']:3d}→{cmp['after']['n_trades']:3d} "
                      f"Net={cmp['before']['net_pnl']:+5.0f}→{cmp['after']['net_pnl']:+5.0f} "
                      f"PF={cmp['before']['pf']:.2f}→{cmp['after']['pf']:.2f} "
                      f"noTop3={cmp['before']['net_without_top3']:+5.0f}→{cmp['after']['net_without_top3']:+5.0f} "
                      f"rej_net={cmp['rejected_hindsight_net']:+5.0f}")
                report["filters"].append({
                    "filter_name": f"E_density_{density_thresh}",
                    "hypothesis": "High event density → systemic stress, not opportunity",
                    "variables_used": [f"simultaneous_events > {density_thresh}"],
                    "rule_definition": f"Reject if > {density_thresh} events fire at same timestamp",
                    "tested_thresholds": [density_thresh],
                    "affected_event_type": et,
                    "comparison": {str(k): v for k, v in cmp.items() if not isinstance(v, dict)},
                    "before_metrics": cmp["before"],
                    "after_metrics": cmp["after"],
                    "decision": "TO BE FILLED AFTER ANALYSIS",
                })

    # B. Regime classifier — already in report from earlier audit
    print("\n=== B. Regime Classifier (from existing regime detector) ===")
    for et in all_trades["event_type"].unique():
        sub = all_trades[all_trades["event_type"] == et]
        print(f"  {et}:")
        for reg in ["trend_up", "trend_down", "range", "chop", "panic_down"]:
            rsub = sub[sub["regime"] == reg]
            if len(rsub) > 0:
                m = compute_metrics(rsub)
                print(f"    {reg:12s}: n={m['n_trades']:3d} Net={m['net_pnl']:+6.0f} PF={m['pf']:.2f} Hit={m['hit_rate']*100:.0f}% noTop3={m['net_without_top3']:+6.0f}")
        report["filters"].append({
            "filter_name": "B_regime_classifier",
            "hypothesis": "Different regimes have different event performance profiles",
            "variables_used": ["btc_ema_slope", "breadth", "btc_ret_4h", "btc_realized_vol"],
            "rule_definition": "Use existing regime_detector.py (5-state: trend_up/down/range/chop/panic_down)",
            "affected_event_type": et,
            "regime_breakdown": {reg: compute_metrics(sub[sub["regime"] == reg]) for reg in sub["regime"].unique()},
            "decision": "TO BE FILLED AFTER ANALYSIS",
        })

    # Save report
    report["metadata"] = {
        "generated_at": str(pd.Timestamp.now()),
        "total_trades_analyzed": len(all_trades),
        "event_types": list(all_trades["event_type"].unique()),
        "cost_assumption": "9bps per side",
        "hold_bars": 2,
    }

    report_path = ROOT / "large_cap_regime_filter_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")
    return report


if __name__ == "__main__":
    run_filter_research()
