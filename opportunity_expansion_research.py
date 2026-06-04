"""Opportunity Expansion Research — 6 new event types + extended universe.

Sections:
  A. OI Build-up → Price Breakout (rank 20-200)
  B. Vol Compression Break (rank 20-200)
  C. Failed Breakout Reversal (rank 20-200)
  D. Mid-Cap 4h Low-Frequency Events (rank 20-100, 4h bars)
  E. Extended Universe (rank 101-200, best candidates)
  F. Cross-Event Combos (new events only)

All shadow/paper only. No live. No continuous ranking.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
sys.path.insert(0, str(ROOT))

from research.regime_detector import detect_regime_fast


# ── Load Data ──────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "data_storage.parquet")
    return df


def get_universe(data: pd.DataFrame, min_rank: int, max_rank: int) -> set[str]:
    """Get symbols by volume rank range."""
    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    syms = set(vol_rank[(vol_rank >= min_rank) & (vol_rank <= max_rank)].index)
    return syms & set(data.index.get_level_values("symbol").unique())


# ── Feature Precompute ─────────────────────────────

def precompute_features(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Precompute all rolling features needed across event types."""
    feats: dict[str, pd.Series] = {}

    # Returns (1h = 4 bars at 15m)
    feats["ret_1h"] = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(4))
    feats["ret_4h"] = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(16))

    # Volume z-score
    vol = data["volume"]
    gv = vol.groupby(level="symbol")
    vol_mean = gv.transform(lambda s: s.rolling(48, min_periods=8).mean())
    vol_std = gv.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    feats["vol_z"] = ((vol - vol_mean) / vol_std).fillna(0.0)

    # Vol compression: 24-bar trailing vol / 96-bar trailing vol
    vol_24 = gv.transform(lambda s: s.rolling(24, min_periods=8).std())
    vol_96 = gv.transform(lambda s: s.rolling(96, min_periods=16).std())
    feats["vol_compression_ratio"] = (vol_24 / vol_96.replace(0, np.nan)).fillna(1.0)

    # OI delta z-score (6-bar = 1.5h OI change, 48-bar z-score)
    oi_delta = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(6))
    go = oi_delta.groupby(level="symbol")
    oi_mean = go.transform(lambda s: s.rolling(48, min_periods=8).mean())
    oi_std = go.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    feats["oi_delta_z"] = ((oi_delta - oi_mean) / oi_std).fillna(0.0)

    # OI build-up: cumulative OI change over 12 bars (3h) / trailing OI std
    oi_cum_delta = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(12))
    oi_std_48 = go.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    feats["oi_buildup_z"] = (oi_cum_delta / oi_std_48).fillna(0.0)

    # Price stall: abs(ret_1h) is small
    feats["price_stall"] = abs(feats["ret_1h"]).fillna(0)

    # Close location
    hl_range = (data["high"] - data["low"]).clip(lower=1e-8)
    feats["close_loc"] = (data["close"] - data["low"]) / hl_range

    # Funding z-score
    funding = data["funding_rate"]
    gf = funding.groupby(level="symbol")
    f_mean = gf.transform(lambda s: s.rolling(24, min_periods=8).mean())
    f_std = gf.transform(lambda s: s.rolling(24, min_periods=8).std()).replace(0, np.nan)
    feats["funding_z"] = ((funding - f_mean) / f_std).fillna(0.0)

    # Breakout features: 24-bar high/low channel
    high_24 = data["high"].groupby(level="symbol").transform(lambda s: s.rolling(24, min_periods=8).max())
    low_24 = data["low"].groupby(level="symbol").transform(lambda s: s.rolling(24, min_periods=8).min())
    channel_width = (high_24 - low_24).clip(lower=1e-8)
    feats["breakout_up"] = (data["close"] - high_24.shift(1)) / channel_width.shift(1)
    feats["breakout_down"] = (data["close"] - low_24.shift(1)) / channel_width.shift(1)

    # Taker volume ratio
    taker_total = data.get("taker_volume", data["volume"] * 0.5)
    feats["taker_ratio"] = (taker_total / data["volume"].clip(lower=1e-8)).fillna(0.5)

    return feats


# ── Paper Trading ──────────────────────────────────

def paper_trade(
    signals: list[dict],
    data: pd.DataFrame,
    hold_bars: int = 4,
    costs_bps: list[float] = [9, 12, 15],
) -> dict[str, Any]:
    """Run paper trading on signals. Each signal: {symbol, entry_ts, direction, ...}"""
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}

    trades = []
    for sig in signals:
        entry_ts = pd.Timestamp(sig["entry_ts"])
        sym = sig["symbol"]
        direction = sig.get("direction", "long")

        entry_idx = ts_to_idx.get(entry_ts)
        if entry_idx is None:
            continue
        exit_idx = entry_idx + hold_bars
        if exit_idx >= len(ts_list):
            continue
        exit_ts = ts_list[exit_idx]

        try:
            entry_px = float(data.loc[(entry_ts, sym), "close"])
            exit_px = float(data.loc[(exit_ts, sym), "close"])
        except KeyError:
            continue

        ret = (exit_px / entry_px) - 1.0
        if direction == "short":
            ret = -ret

        gross_bps = ret * 10000
        trades.append({
            "symbol": sym,
            "entry_ts": str(entry_ts),
            "exit_ts": str(exit_ts),
            "direction": direction,
            "gross_bps": gross_bps,
            **{k: sig.get(k) for k in ["event_type", "event_score", "regime"] if k in sig},
        })

    if not trades:
        return {"n_trades": 0, "error": "No trades"}

    df_t = pd.DataFrame(trades)
    results: dict[str, Any] = {"n_trades": len(df_t)}
    gross_arr = df_t["gross_bps"].values

    for cost in costs_bps:
        net_arr = gross_arr - cost * (1 if "direction" not in df_t else 1)  # cost per round trip
        # Actually cost is per side for entry+exit, so 2x
        net_arr = gross_arr - (2 * cost)
        pos = net_arr[net_arr > 0].sum()
        neg = abs(net_arr[net_arr < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        hit = (net_arr > 0).mean()

        # Top contributions
        sorted_net = sorted(net_arr, reverse=True)
        top1 = sorted_net[0] if len(sorted_net) > 0 else 0
        top3 = sum(sorted_net[:3]) if len(sorted_net) >= 3 else sum(sorted_net)
        net_no_top3 = net_arr.sum() - top3

        results[f"cost_{cost}bps"] = {
            "net_bps": round(net_arr.sum(), 0),
            "pf": round(pf, 3),
            "hit_rate": round(hit, 3),
            "median_bps": round(np.median(net_arr), 1),
            "avg_win_bps": round(net_arr[net_arr > 0].mean(), 1) if (net_arr > 0).any() else 0,
            "avg_loss_bps": round(net_arr[net_arr < 0].mean(), 1) if (net_arr < 0).any() else 0,
            "top1_contribution": round(top1, 0),
            "top3_contribution": round(top3, 0),
            "net_without_top3": round(net_no_top3, 0),
            "top3_pct_of_net": round(top3 / net_arr.sum() * 100, 0) if net_arr.sum() != 0 else 0,
        }

    # Regime breakdown
    if "regime" in df_t.columns:
        regime_bd = {}
        for reg in df_t["regime"].unique():
            sub = df_t[df_t["regime"] == reg]
            net9 = (sub["gross_bps"].values - 18).sum()
            regime_bd[reg] = {"n": len(sub), "net_9bps": round(net9, 0)}
        results["regime_breakdown"] = regime_bd

    return results


# ════════════════════════════════════════════════════
#  EVENT TYPE A: OI Build-up → Price Breakout
# ════════════════════════════════════════════════════
# Mechanism: Sustained OI increase without price movement
#   → liquidity building → eventual breakout.
# Entry: OI buildup > 2σ + price stall (<|1%| 1h) + volume normal
#   → enter on next bar, direction = breakout direction
# Hold: 4 bars (1h)

def research_oi_buildup(
    data: pd.DataFrame,
    feats: dict[str, pd.Series],
    regime_map: dict,
    universe: set[str],
) -> tuple[list[dict], dict]:
    """OI Build-up event scanner + paper trade."""
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    signals = []
    rejects = []

    for i in range(1, len(ts_list) - 8):  # skip first/last bars
        ts = ts_list[i]
        mask = data.index.get_level_values("timestamp") == ts
        syms_at_ts = set(data.index.get_level_values("symbol")[mask]) & universe

        regime = regime_map.get(ts, "unknown")
        if regime in ("panic_down", "chop"):
            continue

        for sym in syms_at_ts:
            try:
                oi_bz = float(feats["oi_buildup_z"].loc[(ts, sym)])
                stall = float(feats["price_stall"].loc[(ts, sym)])
                vz = float(feats["vol_z"].loc[(ts, sym)])
                ret_1h = float(feats["ret_1h"].loc[(ts, sym)])
                cl = float(feats["close_loc"].loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            # Gates
            if oi_bz < 2.0:  # OI not building enough
                rejects.append({"symbol": sym, "ts": ts, "reason": "oi_buildup_z < 2.0"})
                continue
            if stall > 0.015:  # price moving too much
                rejects.append({"symbol": sym, "ts": ts, "reason": "price_not_stalling"})
                continue
            if vz > 2.0 or vz < -1.0:  # vol should be normal, not elevated
                rejects.append({"symbol": sym, "ts": ts, "reason": "vol_not_normal"})
                continue
            if np.isnan(oi_bz) or np.isnan(stall):
                continue

            # Direction = sign of ret_1h (micro-direction)
            direction = "long" if ret_1h > 0 else "short"

            # Event score
            oi_score = min(1.0, (oi_bz - 2.0) / 2.0)  # 2-4 → 0-1
            stall_score = 1.0 - stall / 0.015
            score = 0.6 * oi_score + 0.4 * stall_score

            if score < 0.4:
                rejects.append({"symbol": sym, "ts": ts, "reason": f"score={score:.2f}"})
                continue

            signals.append({
                "symbol": sym,
                "entry_ts": ts,
                "direction": direction,
                "event_type": "OIBuildupBreakout",
                "event_score": round(score, 3),
                "regime": regime,
            })

    results = paper_trade(signals, data, hold_bars=4)
    results["raw_signals"] = len(signals)
    results["rejected"] = len(rejects)
    results["universe_size"] = len(universe)
    return signals, results


# ════════════════════════════════════════════════════
#  EVENT TYPE B: Vol Compression Break
# ════════════════════════════════════════════════════
# Mechanism: Extended low-vol period → compression → expansion break.
# Entry: vol_compression_ratio < 0.5 (24-period vol < 50% of 96-period)
#   + moderate volume increase (>0.5σ) + direction established
# Hold: 4 bars (1h)

def research_vol_compression(
    data: pd.DataFrame,
    feats: dict[str, pd.Series],
    regime_map: dict,
    universe: set[str],
) -> tuple[list[dict], dict]:
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    signals = []
    rejects = []

    for i in range(1, len(ts_list) - 8):
        ts = ts_list[i]
        mask = data.index.get_level_values("timestamp") == ts
        syms_at_ts = set(data.index.get_level_values("symbol")[mask]) & universe

        regime = regime_map.get(ts, "unknown")
        if regime in ("panic_down", "chop"):
            continue

        for sym in syms_at_ts:
            try:
                vcr = float(feats["vol_compression_ratio"].loc[(ts, sym)])
                vz = float(feats["vol_z"].loc[(ts, sym)])
                ret_1h = float(feats["ret_1h"].loc[(ts, sym)])
                cl = float(feats["close_loc"].loc[(ts, sym)])
                oi_z = float(feats["oi_delta_z"].loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            if vcr > 0.5:  # not compressed enough
                continue
            if vz < 0.5:  # no expansion starting
                rejects.append({"symbol": sym, "ts": ts, "reason": "vol_not_expanding"})
                continue
            if abs(ret_1h) < 0.01:  # no direction
                rejects.append({"symbol": sym, "ts": ts, "reason": "no_direction"})
                continue
            if np.isnan(vcr):
                continue

            direction = "long" if ret_1h > 0 else "short"

            # Score: compression strength + expansion conviction
            comp_score = 1.0 - vcr / 0.5  # 0.5→0, 0.1→0.8
            exp_score = min(1.0, vz / 2.0)  # vol_z 0.5→0.25, 2.0→1.0
            dir_score = min(1.0, abs(ret_1h) / 0.03)  # 1%→0.33, 3%→1.0
            score = 0.4 * comp_score + 0.3 * exp_score + 0.3 * dir_score

            if score < 0.4:
                rejects.append({"symbol": sym, "ts": ts, "reason": f"score={score:.2f}"})
                continue

            signals.append({
                "symbol": sym,
                "entry_ts": ts,
                "direction": direction,
                "event_type": "VolCompressionBreak",
                "event_score": round(score, 3),
                "regime": regime,
            })

    results = paper_trade(signals, data, hold_bars=4)
    results["raw_signals"] = len(signals)
    results["rejected"] = len(rejects)
    results["universe_size"] = len(universe)
    return signals, results


# ════════════════════════════════════════════════════
#  EVENT TYPE C: Failed Breakout Reversal
# ════════════════════════════════════════════════════
# Mechanism: Price breaks 24-bar channel, then IMMEDIATELY reverses
#   → trapped breakout traders → reversal trade.
# Entry: breakout_up > 0.02 (broke above 24h high) + close_loc < 0.3
#   (rejected at high, closing near low) → short
#   OR breakout_down < -0.02 + close_loc > 0.7 → long
# Hold: 2 bars (30m — fast reversal)

def research_failed_breakout(
    data: pd.DataFrame,
    feats: dict[str, pd.Series],
    regime_map: dict,
    universe: set[str],
) -> tuple[list[dict], dict]:
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    signals = []
    rejects = []

    for i in range(1, len(ts_list) - 5):
        ts = ts_list[i]
        mask = data.index.get_level_values("timestamp") == ts
        syms_at_ts = set(data.index.get_level_values("symbol")[mask]) & universe

        regime = regime_map.get(ts, "unknown")

        for sym in syms_at_ts:
            try:
                bu = float(feats["breakout_up"].loc[(ts, sym)])
                bd = float(feats["breakout_down"].loc[(ts, sym)])
                cl = float(feats["close_loc"].loc[(ts, sym)])
                vz = float(feats["vol_z"].loc[(ts, sym)])
                ret_1h = float(feats["ret_1h"].loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            # Failed upside breakout → short
            if bu > 0.02 and cl < 0.35 and vz > 0.5:
                score = min(1.0, bu / 0.05) * 0.5 + (1.0 - cl / 0.35) * 0.5
                if score > 0.4:
                    signals.append({
                        "symbol": sym, "entry_ts": ts, "direction": "short",
                        "event_type": "FailedBreakoutReversal",
                        "event_score": round(score, 3), "regime": regime,
                    })
                else:
                    rejects.append({"symbol": sym, "ts": ts, "reason": f"upside_score={score:.2f}"})

            # Failed downside breakout → long
            elif bd < -0.02 and cl > 0.65 and vz > 0.5:
                score = min(1.0, abs(bd) / 0.05) * 0.5 + (cl - 0.65) / 0.35 * 0.5
                if score > 0.4:
                    signals.append({
                        "symbol": sym, "entry_ts": ts, "direction": "long",
                        "event_type": "FailedBreakoutReversal",
                        "event_score": round(score, 3), "regime": regime,
                    })
                else:
                    rejects.append({"symbol": sym, "ts": ts, "reason": f"downside_score={score:.2f}"})

    results = paper_trade(signals, data, hold_bars=2)
    results["raw_signals"] = len(signals)
    results["rejected"] = len(rejects)
    results["universe_size"] = len(universe)
    return signals, results


# ════════════════════════════════════════════════════
#  EVENT TYPE D: Mid-Cap 4h Low-Frequency
# ════════════════════════════════════════════════════
# Uses 4h bars (resampled from 15m) for mid-cap symbols.
# Event: OI buildup + vol compression → breakout on 4h timeframe.
# Lower frequency, fewer signals, longer holds.

def research_4h_midcap(
    data: pd.DataFrame,
    regime_map: dict,
    universe: set[str],
) -> tuple[list[dict], dict]:
    """Resample to 4h, run OI buildup + vol compression on mid-cap."""
    # Resample to 4h
    data_4h = data.groupby("symbol").resample("4h", level="timestamp").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "open_interest": "last", "funding_rate": "last",
    }).dropna(subset=["close"])

    feats_4h = precompute_features(data_4h)

    # Build regime map for 4h timestamps (use nearest)
    ts_4h = sorted(data_4h.index.get_level_values("timestamp").unique())
    regime_4h = {}
    for ts in ts_4h:
        regime_4h[ts] = regime_map.get(ts, "unknown")

    # Run OI Buildup on 4h
    sigs_oi, res_oi = research_oi_buildup(data_4h, feats_4h, regime_4h, universe)
    res_oi["timeframe"] = "4h"

    # Run Vol Compression on 4h
    sigs_vc, res_vc = research_vol_compression(data_4h, feats_4h, regime_4h, universe)
    res_vc["timeframe"] = "4h"

    return sigs_oi + sigs_vc, {"OI_Buildup_4h": res_oi, "VolCompression_4h": res_vc}


# ════════════════════════════════════════════════════
#  EVENT TYPE E: Extended Universe (rank 101-200)
# ════════════════════════════════════════════════════
# Test best event types from A-D on the extended (smaller) universe.

def research_extended_universe(
    data: pd.DataFrame,
    feats: dict[str, pd.Series],
    regime_map: dict,
    universe: set[str],
    best_event_types: list[str],
) -> dict:
    """Run best event types on rank 101-200 universe."""
    results = {}
    for et in best_event_types:
        if et == "OIBuildupBreakout":
            _, res = research_oi_buildup(data, feats, regime_map, universe)
        elif et == "VolCompressionBreak":
            _, res = research_vol_compression(data, feats, regime_map, universe)
        elif et == "FailedBreakoutReversal":
            _, res = research_failed_breakout(data, feats, regime_map, universe)
        else:
            continue
        results[et] = res
    return results


# ════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════

def main():
    print("Loading data...")
    data = load_data()
    print(f"  {len(data)} rows, {data.index.get_level_values('symbol').nunique()} symbols")

    print("Precomputing features...")
    feats = precompute_features(data)

    print("Computing regimes...")
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))

    # Universes
    universe_mid = get_universe(data, 20, 100)
    universe_ext = get_universe(data, 101, 200)
    universe_full = get_universe(data, 20, 200)
    print(f"Mid-cap universe (20-100): {len(universe_mid)}")
    print(f"Extended universe (101-200): {len(universe_ext)}")
    print(f"Full research universe (20-200): {len(universe_full)}")

    all_results: dict[str, Any] = {}

    # A. OI Build-up (mid-cap)
    print("\n=== A. OI Build-up Breakout (mid-cap) ===")
    sigs_a, res_a = research_oi_buildup(data, feats, regime_map, universe_mid)
    all_results["A_OIBuildupBreakout"] = res_a
    _print_results(res_a, "OIBuildupBreakout")

    # B. Vol Compression Break (mid-cap)
    print("\n=== B. Vol Compression Break (mid-cap) ===")
    sigs_b, res_b = research_vol_compression(data, feats, regime_map, universe_mid)
    all_results["B_VolCompressionBreak"] = res_b
    _print_results(res_b, "VolCompressionBreak")

    # C. Failed Breakout Reversal (mid-cap)
    print("\n=== C. Failed Breakout Reversal (mid-cap) ===")
    sigs_c, res_c = research_failed_breakout(data, feats, regime_map, universe_mid)
    all_results["C_FailedBreakoutReversal"] = res_c
    _print_results(res_c, "FailedBreakoutReversal")

    # D. 4h Mid-Cap
    print("\n=== D. Mid-Cap 4h Low-Frequency ===")
    sigs_d, res_d = research_4h_midcap(data, regime_map, universe_mid)
    all_results["D_4hMidCap"] = res_d
    for k, v in res_d.items():
        print(f"  {k}: n={v.get('n_trades',0)} net9={v.get('cost_9bps',{}).get('net_bps','?')} PF={v.get('cost_9bps',{}).get('pf','?')}")

    # Determine best event types for extended universe
    best_ets = []
    for name, res in [
        ("OIBuildupBreakout", res_a),
        ("VolCompressionBreak", res_b),
        ("FailedBreakoutReversal", res_c),
    ]:
        cost9 = res.get("cost_9bps", {})
        if cost9.get("net_bps", -999) > 0 and cost9.get("pf", 0) > 1.1:
            best_ets.append(name)
    print(f"\nBest event types for extended universe: {best_ets}")

    # E. Extended Universe
    if best_ets:
        print("\n=== E. Extended Universe (rank 101-200) ===")
        res_e = research_extended_universe(data, feats, regime_map, universe_ext, best_ets)
        all_results["E_ExtendedUniverse"] = res_e
        for k, v in res_e.items():
            _print_results(v, f"Extended_{k}")

    # Save
    out_path = ROOT / "opportunity_expansion_report.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Summary verdict
    print("\n" + "=" * 60)
    print("VERDICTS")
    print("=" * 60)
    for name, res in all_results.items():
        if name.startswith("D_"):
            for k, v in res.items():
                _verdict(k, v)
        elif name.startswith("E_"):
            for k, v in res.items():
                _verdict(f"Extended_{k}", v)
        else:
            _verdict(name, res)


def _print_results(res: dict, label: str):
    if "n_trades" not in res or res["n_trades"] == 0:
        print(f"  {label}: 0 trades")
        return
    c9 = res.get("cost_9bps", {})
    print(f"  {label}: n={res['n_trades']} Net9={c9.get('net_bps','?')} PF={c9.get('pf','?')} "
          f"Hit={c9.get('hit_rate','?')} Net_no_top3={c9.get('net_without_top3','?')} "
          f"Top3%={c9.get('top3_pct_of_net','?')}%")


def _verdict(name: str, res: dict):
    if "n_trades" not in res or res["n_trades"] == 0:
        return
    c9 = res.get("cost_9bps", {})
    c12 = res.get("cost_12bps", {})
    c15 = res.get("cost_15bps", {})
    net9 = c9.get("net_bps", 0)
    pf9 = c9.get("pf", 0)
    net_no_top3 = c9.get("net_without_top3", 0)
    top3_pct = c9.get("top3_pct_of_net", 100)
    n = res["n_trades"]

    if n < 20:
        verdict = "KILL — too few signals"
    elif net9 <= 0:
        verdict = "KILL — net negative at 9bps"
    elif pf9 < 1.10:
        verdict = "KILL — PF too low"
    elif top3_pct > 80:
        verdict = "TOO_CONCENTRATED — top3 > 80% of net"
    elif c12.get("net_bps", -1) <= 0:
        verdict = "COST_SENSITIVE — fails at 12bps"
    elif net_no_top3 <= 0:
        verdict = "PAPER_CANDIDATE — needs more trades for top3 confidence"
    elif n >= 30 and pf9 >= 1.15 and net_no_top3 > 0:
        verdict = "SHADOW_CANDIDATE — meets promotion criteria"
    else:
        verdict = "PAPER_CANDIDATE"

    print(f"  {name:50s} n={n:4d} Net9={net9:+6.0f} PF={pf9:.2f} "
          f"Top3%={top3_pct:3.0f}%  → {verdict}")


if __name__ == "__main__":
    main()
