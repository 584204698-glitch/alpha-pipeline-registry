"""Test existing 3 events on extended universe (rank 101-200).

Existing events:
  1. Deleveraging Reversal (long-only)
  2. OI Shock Absorption (long-only short squeeze)
  3. RS Shock (regime-dependent)

Test on rank 101-200 with adjusted thresholds for smaller-cap behavior.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
sys.path.insert(0, str(ROOT))
from research.regime_detector import detect_regime_fast
from event_scanner import DeleveragingEventScanner, EventState, SmallCapUniverse
from relative_strength_shock import detect_relative_strength_shock
from oi_shock_absorption import detect_oi_shock_absorption


def load_data():
    return pd.read_parquet(ROOT / "data" / "data_storage.parquet")


def get_universe(data, min_rank, max_rank):
    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    return set(vol_rank[(vol_rank >= min_rank) & (vol_rank <= max_rank)].index) & set(
        data.index.get_level_values("symbol").unique()
    )


def paper_trade_from_scan(results_list, data, hold_bars=2, costs=[9,12,15]):
    """Convert ScanResult list to paper trades."""
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    trades = []
    for r in results_list:
        if getattr(r, 'state', None) not in (EventState.CONFIRMED, EventState.QUALIFIED):
            continue
        ts = getattr(r, 'timestamp', None)
        if ts is None:
            continue
        idx = ts_to_idx.get(ts)
        if idx is None or idx + hold_bars >= len(ts_list):
            continue
        exit_ts = ts_list[idx + hold_bars]
        sym = getattr(r, 'symbol', '')
        try:
            entry_px = float(data.loc[(ts, sym), "close"])
            exit_px = float(data.loc[(exit_ts, sym), "close"])
        except KeyError:
            continue
        ret = (exit_px / entry_px) - 1.0
        gross_bps = ret * 10000
        trades.append({"symbol": sym, "gross_bps": gross_bps, "entry_ts": str(ts)})
    if not trades:
        return {"n_trades": 0}
    gross_arr = np.array([t["gross_bps"] for t in trades])
    results = {"n_trades": len(trades)}
    for cost in costs:
        net_arr = gross_arr - 2 * cost
        pos = net_arr[net_arr > 0].sum()
        neg = abs(net_arr[net_arr < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        sorted_net = sorted(net_arr, reverse=True)
        top3 = sum(sorted_net[:3]) if len(sorted_net) >= 3 else sum(sorted_net)
        net_no_top3 = net_arr.sum() - top3
        results[f"cost_{cost}bps"] = {
            "net_bps": round(net_arr.sum(), 0), "pf": round(pf, 3),
            "hit_rate": round((net_arr > 0).mean(), 3),
            "top3_contribution": round(top3, 0),
            "net_without_top3": round(net_no_top3, 0),
            "top3_pct_of_net": round(top3/net_arr.sum()*100, 0) if net_arr.sum() != 0 else 0,
        }
    return results


def test_deleveraging_ext(data, regime_map, universe, hold_bars=2):
    """Run Deleveraging scanner on extended universe."""
    scanner = DeleveragingEventScanner(data, regime_map, universe=SmallCapUniverse(
        min_rank=101, max_rank=200), hold_bars=hold_bars)
    ts_list = scanner.ts_list[100:]  # skip warmup
    all_results = []
    for ts in ts_list:
        results = scanner.scan(ts)
        all_results.extend(results)
    confirmed = [r for r in all_results if r.state == EventState.CONFIRMED]
    qualified = [r for r in all_results if r.state == EventState.QUALIFIED]
    rejected = [r for r in all_results if r.state == EventState.REJECTED]
    # Paper trade on confirmed
    pt = paper_trade_from_scan(confirmed, data, hold_bars=hold_bars)
    pt["confirmed_signals"] = len(confirmed)
    pt["qualified_signals"] = len(qualified)
    pt["rejected_signals"] = len(rejected)
    pt["watch_signals"] = len([r for r in all_results if r.state == EventState.WATCH])
    return pt


def test_oi_absorption_ext(data, regime_map, universe):
    """Run OI Shock Absorption on extended universe."""
    events = detect_oi_shock_absorption(data, universe, regime_map,
                                         oi_delta_z_min=2.5, vol_z_min=1.5, cooldown_bars=8)
    trades = []
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    for ev in events:
        ts = ev["timestamp"]
        idx = ts_to_idx.get(ts)
        if idx is None or idx + 2 >= len(ts_list):
            continue
        exit_ts = ts_list[idx + 2]
        sym = ev["symbol"]
        try:
            entry_px = float(data.loc[(ts, sym), "close"])
            exit_px = float(data.loc[(exit_ts, sym), "close"])
        except KeyError:
            continue
        ret = (exit_px / entry_px) - 1.0
        trades.append({"symbol": sym, "gross_bps": ret * 10000, "entry_ts": str(ts)})
    if not trades:
        return {"n_trades": 0, "raw_events": len(events)}
    gross_arr = np.array([t["gross_bps"] for t in trades])
    results = {"n_trades": len(trades), "raw_events": len(events)}
    for cost in [9, 12, 15]:
        net_arr = gross_arr - 2 * cost
        pos = net_arr[net_arr > 0].sum()
        neg = abs(net_arr[net_arr < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        top3 = sum(sorted(net_arr, reverse=True)[:3])
        results[f"cost_{cost}bps"] = {
            "net_bps": round(net_arr.sum(), 0), "pf": round(pf, 3),
            "hit_rate": round((net_arr > 0).mean(), 3),
            "top3_contribution": round(top3, 0),
            "net_without_top3": round(net_arr.sum() - top3, 0),
            "top3_pct": round(top3/net_arr.sum()*100, 0) if net_arr.sum() != 0 else 0,
        }
    return results


def test_rs_shock_ext(data, regime_map, universe):
    """Run RS Shock on extended universe."""
    events = detect_relative_strength_shock(data, universe, regime_map,
                                              rs_threshold=0.05, vol_z_min=2.0,
                                              oi_delta_z_min=0.5, close_loc_min=0.50,
                                              event_score_min=0.60, cooldown_bars=8)
    trades = []
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    for ev in events:
        ts = ev["timestamp"]
        idx = ts_to_idx.get(ts)
        if idx is None or idx + 2 >= len(ts_list):
            continue
        exit_ts = ts_list[idx + 2]
        sym = ev["symbol"]
        try:
            entry_px = float(data.loc[(ts, sym), "close"])
            exit_px = float(data.loc[(exit_ts, sym), "close"])
        except KeyError:
            continue
        ret = (exit_px / entry_px) - 1.0
        if ev.get("direction") == "short":
            ret = -ret
        trades.append({"symbol": sym, "gross_bps": ret * 10000, "entry_ts": str(ts)})
    if not trades:
        return {"n_trades": 0, "raw_events": len(events)}
    gross_arr = np.array([t["gross_bps"] for t in trades])
    results = {"n_trades": len(trades), "raw_events": len(events)}
    for cost in [9, 12, 15]:
        net_arr = gross_arr - 2 * cost
        pos = net_arr[net_arr > 0].sum()
        neg = abs(net_arr[net_arr < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        top3 = sum(sorted(net_arr, reverse=True)[:3])
        results[f"cost_{cost}bps"] = {
            "net_bps": round(net_arr.sum(), 0), "pf": round(pf, 3),
            "hit_rate": round((net_arr > 0).mean(), 3),
            "top3_contribution": round(top3, 0),
            "net_without_top3": round(net_arr.sum() - top3, 0),
            "top3_pct": round(top3/net_arr.sum()*100, 0) if net_arr.sum() != 0 else 0,
        }
    return results


def _verdict(name, res):
    if res.get("n_trades", 0) == 0:
        print(f"  {name:45s} → 0 trades")
        return
    c9 = res.get("cost_9bps", {})
    n = res["n_trades"]
    net9 = c9.get("net_bps", 0)
    pf9 = c9.get("pf", 0)
    top3_pct = c9.get("top3_pct", c9.get("top3_pct_of_net", 100))
    net_no_top3 = c9.get("net_without_top3", 0)
    c12_net = res.get("cost_12bps", {}).get("net_bps", -1)

    if n < 10:
        v = "SPARSE"
    elif net9 <= 0:
        v = "KILL"
    elif pf9 < 1.10:
        v = "KILL_PF"
    elif top3_pct > 80:
        v = "TOO_CONCENTRATED"
    elif c12_net <= 0:
        v = "COST_SENSITIVE"
    elif net_no_top3 <= 0:
        v = "PAPER_CANDIDATE"
    else:
        v = "SHADOW_CANDIDATE"

    raw = res.get("raw_events", res.get("confirmed_signals", "?"))
    print(f"  {name:45s} n={n:4d} raw={str(raw):5s} Net9={net9:+7.0f} PF={pf9:.2f} "
          f"Top3%={top3_pct:4.0f}% NoT3={net_no_top3:+7.0f} → {v}")


def main():
    print("Loading data...")
    data = load_data()
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))
    uni_mid = get_universe(data, 20, 100)
    uni_ext = get_universe(data, 101, 200)
    print(f"Mid: {len(uni_mid)}, Ext: {len(uni_ext)}")

    print("\n=== 1. Deleveraging Reversal on Extended (101-200) ===")
    res_d = test_deleveraging_ext(data, regime_map, uni_ext)
    _verdict("Deleveraging_ext", res_d)

    print("\n=== 2. OI Shock Absorption on Extended (101-200) ===")
    res_o = test_oi_absorption_ext(data, regime_map, uni_ext)
    _verdict("OI_Absorption_ext", res_o)

    print("\n=== 3. RS Shock on Extended (101-200) ===")
    res_r = test_rs_shock_ext(data, regime_map, uni_ext)
    _verdict("RS_Shock_ext", res_r)

    all_res = {"Deleveraging_ext": res_d, "OI_Absorption_ext": res_o, "RS_Shock_ext": res_r}
    out_path = ROOT / "extended_universe_existing_events.json"
    with open(out_path, "w") as f:
        json.dump(all_res, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    survivors = [k for k, v in all_res.items()
                 if v.get("cost_9bps", {}).get("net_bps", 0) > 0]
    print(f"Survivors: {survivors if survivors else 'NONE'}")


if __name__ == "__main__":
    main()
