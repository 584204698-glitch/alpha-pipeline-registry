"""Symbol Alpha Profile Audit — per-symbol event suitability analysis.

For each of the 81 rank 20-100 symbols, across 4 alphas:
  - How many times did each event fire?
  - What was the PnL per event per symbol?
  - Which events should be enabled/disabled/weight-reduced?

Output: symbol_alpha_profile.json
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
from event_scanner import DeleveragingEventScanner, EventState, SmallCapUniverse
from relative_strength_shock import detect_relative_strength_shock
from oi_shock_absorption import detect_oi_shock_absorption


def load_data():
    return pd.read_parquet(ROOT / "data" / "data_storage.parquet")


def get_universe(data):
    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    return set(vol_rank[(vol_rank >= 20) & (vol_rank <= 100)].index) & set(
        data.index.get_level_values("symbol").unique())


def compute_spread_proxy(data):
    """Estimate spread from high-low range."""
    spread = (data["high"] - data["low"]) / data["close"].clip(lower=1e-8) * 10000  # bps
    return spread.groupby(level="symbol").mean()


def compute_data_quality(data):
    """Fraction of bars with valid OHLCV+OI+funding."""
    valid = data[["open","high","low","close","volume","open_interest","funding_rate"]].notna().all(axis=1)
    return valid.groupby(level="symbol").mean()


def collect_deleveraging_trades(data, regime_map, universe):
    """Collect per-symbol Deleveraging trades."""
    scanner = DeleveragingEventScanner(data, regime_map, universe=SmallCapUniverse(), hold_bars=2)
    ts_list = scanner.ts_list
    sym_trades = defaultdict(list)
    for ts in ts_list[100:]:
        results = scanner.scan(ts)
        for r in results:
            if r.state != EventState.CONFIRMED:
                continue
            idx = scanner.ts_to_idx.get(ts)
            if idx is None or idx + 2 >= len(ts_list):
                continue
            exit_ts = ts_list[idx + 2]
            try:
                ep = float(data.loc[(ts, r.symbol), "close"])
                xp = float(data.loc[(exit_ts, r.symbol), "close"])
            except KeyError:
                continue
            gross = (xp / ep - 1.0) * 10000
            sym_trades[r.symbol].append(gross)
    return {sym: {"trades": arr} for sym, arr in sym_trades.items()}


def collect_oi_absorption_trades(data, regime_map, universe):
    events = detect_oi_shock_absorption(data, universe, regime_map, oi_delta_z_min=2.5, vol_z_min=1.5)
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    sym_trades = defaultdict(list)
    seen = set()
    for ev in events:
        key = (ev["symbol"], ev["timestamp"])
        if key in seen:
            continue
        seen.add(key)
        idx = ts_to_idx.get(ev["timestamp"])
        if idx is None or idx + 2 >= len(ts_list):
            continue
        try:
            ep = float(data.loc[(ev["timestamp"], ev["symbol"]), "close"])
            xp = float(data.loc[(ts_list[idx+2], ev["symbol"]), "close"])
        except KeyError:
            continue
        gross = (xp / ep - 1.0) * 10000
        sym_trades[ev["symbol"]].append(gross)
    return {sym: {"trades": arr} for sym, arr in sym_trades.items()}


def collect_rs_shock_trades(data, regime_map, universe):
    events = detect_relative_strength_shock(data, universe, regime_map, rs_threshold=0.05, vol_z_min=2.0,
                                             oi_delta_z_min=0.5, close_loc_min=0.50, event_score_min=0.60)
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    sym_trades = defaultdict(list)
    seen = set()
    for ev in events:
        key = (ev["symbol"], ev["timestamp"])
        if key in seen:
            continue
        seen.add(key)
        idx = ts_to_idx.get(ev["timestamp"])
        if idx is None or idx + 2 >= len(ts_list):
            continue
        try:
            ep = float(data.loc[(ev["timestamp"], ev["symbol"]), "close"])
            xp = float(data.loc[(ts_list[idx+2], ev["symbol"]), "close"])
        except KeyError:
            continue
        ret = (xp / ep - 1.0)
        if ev.get("direction") == "short":
            ret = -ret
        gross = ret * 10000
        sym_trades[ev["symbol"]].append(gross)
    return {sym: {"trades": arr} for sym, arr in sym_trades.items()}


def collect_funding_carry_trades(data, regime_map, universe):
    from funding_carry_scanner import FundingCarryScanner, EventState as FCState
    scanner = FundingCarryScanner(data, regime_map, universe)
    sym_trades = defaultdict(list)
    for ts in scanner.ts_list:
        if ts.hour < 7 or ts.hour >= 17:
            continue
        results = scanner.scan(ts)
        for r in results:
            if r.state != FCState.SHADOW_SIGNAL:
                continue
            idx = scanner.ts_to_idx.get(ts)
            if idx is None or idx + 12 >= len(scanner.ts_list):
                continue
            exit_ts = scanner.ts_list[idx + 12]
            try:
                ep = float(data.loc[(ts, r.symbol), "close"])
                xp = float(data.loc[(exit_ts, r.symbol), "close"])
            except KeyError:
                continue
            price_bps = (xp / ep - 1.0) * 10000
            fund_sum = 0.0
            valid = 0
            for j in range(12):
                try:
                    fund_sum += float(data.loc[(scanner.ts_list[idx+j], r.symbol), "funding_rate"])
                    valid += 1
                except:
                    pass
            fund_bps = -(fund_sum / max(valid, 1)) * 10000 * (12 * 0.25 / 8)
            sym_trades[r.symbol].append(price_bps + fund_bps)
    return {sym: {"trades": arr} for sym, arr in sym_trades.items()}


def analyze_symbol(sym, alpha_results, spread_bps, data_quality):
    """Generate symbol alpha profile from per-alpha trade data."""
    profile = {
        "symbol": sym,
        "spread_bps_mean": round(spread_bps.get(sym, 0), 1),
        "data_quality": round(data_quality.get(sym, 0), 3),
        "alphas": {},
        "enabled_events": [],
        "disabled_events": [],
        "weight_multiplier": 1.0,
        "risk_tag": "normal",
        "recommendation": "ENABLE_NORMAL",
    }

    for alpha_name in ["DeleveragingReversal", "OIShockAbsorption", "RelativeStrengthShock", "FundingCarryEU"]:
        trades = alpha_results.get(alpha_name, {}).get(sym, {}).get("trades", [])
        if not trades:
            profile["alphas"][alpha_name] = {"n": 0, "status": "NO_DATA"}
            profile["disabled_events"].append(alpha_name)
            continue

        arr = np.array(trades)
        net9 = (arr - 18).sum()
        pos = arr[arr > 0].sum()
        neg = abs(arr[arr < 0].sum())
        pf = pos / neg if neg > 0 else (999 if pos > 0 else 0)
        hit = (arr > 0).mean()

        # Top contribution check
        sorted_arr = sorted(arr, reverse=True)
        top1_pct = sorted_arr[0] / arr.sum() * 100 if arr.sum() != 0 else 0

        alpha_profile = {
            "n": len(arr),
            "net_9bps": round(net9, 0),
            "pf": round(pf, 2),
            "hit_rate": round(hit, 2),
            "avg_win": round(arr[arr > 0].mean(), 1) if (arr > 0).any() else 0,
            "avg_loss": round(arr[arr < 0].mean(), 1) if (arr < 0).any() else 0,
            "max_loss": round(arr.min(), 0),
            "top1_pct": round(top1_pct, 0),
        }

        # Decision rules
        if len(arr) < 3:
            alpha_profile["status"] = "INSUFFICIENT_DATA"
            profile["disabled_events"].append(alpha_name)
        elif net9 <= 0 and len(arr) >= 5:
            alpha_profile["status"] = "DISABLE"
            alpha_profile["reason"] = "net_negative"
            profile["disabled_events"].append(alpha_name)
        elif top1_pct > 60 and len(arr) < 10:
            alpha_profile["status"] = "REDUCED_WEIGHT"
            alpha_profile["reason"] = "top1_concentrated_small_sample"
            profile["enabled_events"].append(alpha_name)
            profile["risk_tag"] = "top_contributor"
            profile["weight_multiplier"] = min(profile["weight_multiplier"], 0.5)
        elif pf < 1.0 and len(arr) >= 5:
            alpha_profile["status"] = "DISABLE"
            alpha_profile["reason"] = "pf_below_1"
            profile["disabled_events"].append(alpha_name)
        elif pf < 1.15:
            alpha_profile["status"] = "REDUCED_WEIGHT"
            alpha_profile["reason"] = "marginal_pf"
            profile["enabled_events"].append(alpha_name)
            profile["weight_multiplier"] = min(profile["weight_multiplier"], 0.7)
        else:
            alpha_profile["status"] = "ENABLE"
            profile["enabled_events"].append(alpha_name)

        profile["alphas"][alpha_name] = alpha_profile

    # Global symbol rules — only disable for truly bad data
    dq = profile["data_quality"]
    if dq < 0.85:
        profile["recommendation"] = "DISABLE_SYMBOL"
        profile["risk_tag"] = "poor_data_quality"
        profile["enabled_events"] = []
        profile["disabled_events"] = ["ALL"]
    elif len(profile["enabled_events"]) == 0:
        profile["recommendation"] = "SHADOW_ONLY"
    elif profile["weight_multiplier"] < 0.7:
        profile["recommendation"] = "ENABLE_REDUCED_WEIGHT"

    return profile


def main():
    print("Loading data...")
    data = load_data()
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))
    universe = get_universe(data)
    print(f"Universe: {len(universe)} symbols")
    syms_sorted = sorted(universe)

    spread = compute_spread_proxy(data)
    dq = compute_data_quality(data)

    print("Collecting Deleveraging trades...")
    delev = collect_deleveraging_trades(data, regime_map, universe)

    print("Collecting OI Absorption trades...")
    oi_abs = collect_oi_absorption_trades(data, regime_map, universe)

    print("Collecting RS Shock trades...")
    rs_shock = collect_rs_shock_trades(data, regime_map, universe)

    print("Collecting Funding Carry trades...")
    fc = collect_funding_carry_trades(data, regime_map, universe)

    alpha_results = {
        "DeleveragingReversal": delev,
        "OIShockAbsorption": oi_abs,
        "RelativeStrengthShock": rs_shock,
        "FundingCarryEU": fc,
    }

    print("Building symbol profiles...")
    profiles = {}
    summary = {
        "ENABLE_NORMAL": 0, "ENABLE_REDUCED_WEIGHT": 0,
        "SHADOW_ONLY": 0, "DISABLE_SYMBOL": 0,
        "total_symbols": len(syms_sorted),
        "symbols_by_event": {
            "DeleveragingReversal": {"enabled": 0, "disabled": 0, "reduced": 0},
            "OIShockAbsorption": {"enabled": 0, "disabled": 0, "reduced": 0},
            "RelativeStrengthShock": {"enabled": 0, "disabled": 0, "reduced": 0},
            "FundingCarryEU": {"enabled": 0, "disabled": 0, "reduced": 0},
        },
    }

    for sym in syms_sorted:
        p = analyze_symbol(sym, alpha_results, spread, dq)
        profiles[sym] = p
        summary[p["recommendation"]] += 1

        for et in ["DeleveragingReversal", "OIShockAbsorption", "RelativeStrengthShock", "FundingCarryEU"]:
            status = p["alphas"].get(et, {}).get("status", "NO_DATA")
            if status in ("ENABLE",):
                summary["symbols_by_event"][et]["enabled"] += 1
            elif status in ("DISABLE", "INSUFFICIENT_DATA", "NO_DATA"):
                summary["symbols_by_event"][et]["disabled"] += 1
            elif status == "REDUCED_WEIGHT":
                summary["symbols_by_event"][et]["reduced"] += 1

    output = {
        "_meta": {
            "generated_by": "Symbol Alpha Profile Audit",
            "principle": "4 unified alphas + 81 symbol profiles. No per-symbol custom thresholds.",
            "rules": "DISABLE if net_negative/pf<1.0. REDUCED_WEIGHT if top1>60% or marginal PF. DISABLE_SYMBOL if spread>15bps or data_quality<85%.",
        },
        "summary": summary,
        "profiles": profiles,
    }

    out_path = ROOT / "symbol_alpha_profile.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary
    print(f"\n=== Profile Summary ===")
    print(f"Total symbols: {summary['total_symbols']}")
    print(f"  ENABLE_NORMAL:        {summary['ENABLE_NORMAL']}")
    print(f"  ENABLE_REDUCED_WEIGHT: {summary['ENABLE_REDUCED_WEIGHT']}")
    print(f"  SHADOW_ONLY:          {summary['SHADOW_ONLY']}")
    print(f"  DISABLE_SYMBOL:       {summary['DISABLE_SYMBOL']}")

    for et, counts in summary["symbols_by_event"].items():
        print(f"  {et:25s}: enabled={counts['enabled']:3d} reduced={counts['reduced']:3d} disabled={counts['disabled']:3d}")


if __name__ == "__main__":
    main()
