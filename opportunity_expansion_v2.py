"""Opportunity Expansion V2 — Tighter parameters + directional + extended universe.

Changes from V1:
  - OI Build-up: long-only, tighter OI threshold, add volume expansion
  - Failed Breakout: regime filter, tighter thresholds
  - Vol Compression: long-only variant
  - Extended universe: test even if mid-cap fails
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
sys.path.insert(0, str(ROOT))

from research.regime_detector import detect_regime_fast


def load_data() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "data_storage.parquet")


def get_universe(data: pd.DataFrame, min_rank: int, max_rank: int) -> set[str]:
    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    return set(vol_rank[(vol_rank >= min_rank) & (vol_rank <= max_rank)].index) & set(
        data.index.get_level_values("symbol").unique()
    )


def precompute_features(data: pd.DataFrame) -> dict[str, pd.Series]:
    feats: dict[str, pd.Series] = {}
    feats["ret_1h"] = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(4))
    feats["ret_4h"] = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(16))

    vol = data["volume"]
    gv = vol.groupby(level="symbol")
    vol_mean = gv.transform(lambda s: s.rolling(48, min_periods=8).mean())
    vol_std = gv.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    feats["vol_z"] = ((vol - vol_mean) / vol_std).fillna(0.0)

    vol_24 = gv.transform(lambda s: s.rolling(24, min_periods=8).std())
    vol_96 = gv.transform(lambda s: s.rolling(96, min_periods=16).std())
    feats["vol_compression_ratio"] = (vol_24 / vol_96.replace(0, np.nan)).fillna(1.0)

    oi_delta = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(6))
    go = oi_delta.groupby(level="symbol")
    oi_mean = go.transform(lambda s: s.rolling(48, min_periods=8).mean())
    oi_std = go.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    feats["oi_delta_z"] = ((oi_delta - oi_mean) / oi_std).fillna(0.0)

    oi_cum_delta = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(12))
    feats["oi_buildup_z"] = (oi_cum_delta / oi_std).fillna(0.0)

    feats["price_stall"] = abs(feats["ret_1h"]).fillna(0)

    hl_range = (data["high"] - data["low"]).clip(lower=1e-8)
    feats["close_loc"] = (data["close"] - data["low"]) / hl_range

    funding = data["funding_rate"]
    gf = funding.groupby(level="symbol")
    f_mean = gf.transform(lambda s: s.rolling(24, min_periods=8).mean())
    f_std = gf.transform(lambda s: s.rolling(24, min_periods=8).std()).replace(0, np.nan)
    feats["funding_z"] = ((funding - f_mean) / f_std).fillna(0.0)

    high_24 = data["high"].groupby(level="symbol").transform(lambda s: s.rolling(24, min_periods=8).max())
    low_24 = data["low"].groupby(level="symbol").transform(lambda s: s.rolling(24, min_periods=8).min())
    channel_width = (high_24 - low_24).clip(lower=1e-8)
    feats["breakout_up"] = (data["close"] - high_24.shift(1)) / channel_width.shift(1)
    feats["breakout_down"] = (data["close"] - low_24.shift(1)) / channel_width.shift(1)

    return feats


def paper_trade(signals, data, hold_bars=4, costs_bps=[9, 12, 15]):
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
        trades.append({"symbol": sym, "entry_ts": str(entry_ts), "exit_ts": str(exit_ts),
                        "direction": direction, "gross_bps": gross_bps,
                        "regime": sig.get("regime", "?"), "event_score": sig.get("event_score", 0)})
    if not trades:
        return {"n_trades": 0}
    df_t = pd.DataFrame(trades)
    gross_arr = df_t["gross_bps"].values
    results = {"n_trades": len(df_t)}
    for cost in costs_bps:
        net_arr = gross_arr - 2 * cost
        pos = net_arr[net_arr > 0].sum()
        neg = abs(net_arr[net_arr < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        hit = (net_arr > 0).mean()
        sorted_net = sorted(net_arr, reverse=True)
        top1 = sorted_net[0] if sorted_net else 0
        top3 = sum(sorted_net[:3]) if len(sorted_net) >= 3 else sum(sorted_net)
        net_no_top3 = net_arr.sum() - top3
        results[f"cost_{cost}bps"] = {
            "net_bps": round(net_arr.sum(), 0), "pf": round(pf, 3),
            "hit_rate": round(hit, 3), "median_bps": round(np.median(net_arr), 1),
            "avg_win_bps": round(net_arr[net_arr > 0].mean(), 1) if (net_arr > 0).any() else 0,
            "avg_loss_bps": round(net_arr[net_arr < 0].mean(), 1) if (net_arr < 0).any() else 0,
            "top1_contribution": round(top1, 0), "top3_contribution": round(top3, 0),
            "net_without_top3": round(net_no_top3, 0),
            "top3_pct_of_net": round(top3 / net_arr.sum() * 100, 0) if net_arr.sum() != 0 else 0,
        }
    if "regime" in df_t.columns:
        regime_bd = {}
        for reg in df_t["regime"].unique():
            sub = df_t[df_t["regime"] == reg]
            net9 = (sub["gross_bps"].values - 18).sum()
            regime_bd[reg] = {"n": len(sub), "net_9bps": round(net9, 0)}
        results["regime_breakdown"] = regime_bd
    return results


# ═══════════════════════════════════════════
#  V2: OI Build-up → LONG ONLY, tighter
# ═══════════════════════════════════════════

def oi_buildup_v2(data, feats, regime_map, universe, label=""):
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    signals = []
    for i in range(1, len(ts_list) - 8):
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
                fz = float(feats["funding_z"].loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            # Tighter gates V2
            if oi_bz < 2.5:  # stricter OI threshold
                continue
            if stall > 0.01:  # tighter stall: < 1%
                continue
            if abs(vz) > 1.5:  # vol must be calm
                continue
            if abs(fz) > 2.0:  # no extreme funding
                continue
            if np.isnan(oi_bz):
                continue

            # LONG ONLY — OI building = accumulation = upside
            score = min(1.0, (oi_bz - 2.5) / 2.0) * 0.5 + (1.0 - stall / 0.01) * 0.3 + (1.0 - abs(vz) / 1.5) * 0.2
            if score < 0.3:
                continue

            signals.append({"symbol": sym, "entry_ts": ts, "direction": "long",
                           "event_type": f"OIBuildupLong{label}", "event_score": round(score, 3), "regime": regime})
    return paper_trade(signals, data, hold_bars=4)


# ═══════════════════════════════════════════
#  V2: Failed Breakout — regime filtered
# ═══════════════════════════════════════════

def failed_breakout_v2(data, feats, regime_map, universe, label=""):
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    signals = []
    for i in range(1, len(ts_list) - 5):
        ts = ts_list[i]
        mask = data.index.get_level_values("timestamp") == ts
        syms_at_ts = set(data.index.get_level_values("symbol")[mask]) & universe
        regime = regime_map.get(ts, "unknown")
        # Only trade in range/trend_down — trend_up breakouts are real
        if regime not in ("range", "trend_down"):
            continue
        for sym in syms_at_ts:
            try:
                bu = float(feats["breakout_up"].loc[(ts, sym)])
                bd = float(feats["breakout_down"].loc[(ts, sym)])
                cl = float(feats["close_loc"].loc[(ts, sym)])
                vz = float(feats["vol_z"].loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            # Tighter V2: larger breakout + stronger reversal
            if bu > 0.04 and cl < 0.25 and vz > 1.0:
                score = min(1.0, bu / 0.08) * 0.4 + (1.0 - cl / 0.25) * 0.4 + min(1.0, vz / 2.0) * 0.2
                if score > 0.5:
                    signals.append({"symbol": sym, "entry_ts": ts, "direction": "short",
                                   "event_type": f"FailedBreakout{label}", "event_score": round(score, 3), "regime": regime})
            elif bd < -0.04 and cl > 0.75 and vz > 1.0:
                score = min(1.0, abs(bd) / 0.08) * 0.4 + (cl - 0.75) / 0.25 * 0.4 + min(1.0, vz / 2.0) * 0.2
                if score > 0.5:
                    signals.append({"symbol": sym, "entry_ts": ts, "direction": "long",
                                   "event_type": f"FailedBreakout{label}", "event_score": round(score, 3), "regime": regime})
    return paper_trade(signals, data, hold_bars=2)


# ═══════════════════════════════════════════
#  V2: Vol Compression → LONG ONLY
# ═══════════════════════════════════════════

def vol_compression_v2(data, feats, regime_map, universe, label=""):
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    signals = []
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
                oi_z = float(feats["oi_delta_z"].loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            # Tighter: deeper compression + long only (OI confirming)
            if vcr > 0.35:  # deeper compression required
                continue
            if vz < 0.8:  # higher expansion trigger
                continue
            if ret_1h < 0.005:  # long-only: need positive micro-direction
                continue
            if oi_z < 0:  # OI must be expanding too
                continue

            comp_score = 1.0 - vcr / 0.35
            exp_score = min(1.0, vz / 2.0)
            dir_score = min(1.0, ret_1h / 0.02)
            oi_score = min(1.0, oi_z / 2.0)
            score = 0.35 * comp_score + 0.25 * exp_score + 0.25 * dir_score + 0.15 * oi_score
            if score < 0.4:
                continue

            signals.append({"symbol": sym, "entry_ts": ts, "direction": "long",
                           "event_type": f"VolCompressionLong{label}", "event_score": round(score, 3), "regime": regime})
    return paper_trade(signals, data, hold_bars=4)


# ═══════════════════════════════════════════
#  NEW: Funding Extreme Mean-Reversion
# ═══════════════════════════════════════════
# Extreme negative funding + OI stable → short squeeze → long
# Not continuous ranking — only extreme thresholds.

def funding_squeeze(data, feats, regime_map, universe, label=""):
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    signals = []
    for i in range(1, len(ts_list) - 8):
        ts = ts_list[i]
        mask = data.index.get_level_values("timestamp") == ts
        syms_at_ts = set(data.index.get_level_values("symbol")[mask]) & universe
        regime = regime_map.get(ts, "unknown")
        if regime in ("panic_down",):
            continue
        for sym in syms_at_ts:
            try:
                fz = float(feats["funding_z"].loc[(ts, sym)])
                oi_z = float(feats["oi_delta_z"].loc[(ts, sym)])
                ret_1h = float(feats["ret_1h"].loc[(ts, sym)])
                vz = float(feats["vol_z"].loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            # Extreme negative funding (< -2.5σ) + OI not collapsing → squeeze
            if fz > -2.5:
                continue
            if oi_z < -1.0:  # OI collapsing = real liquidation, not squeeze setup
                continue
            if ret_1h > 0.02:  # already recovering
                continue
            if vz < 1.0:  # need volume confirmation
                continue

            fz_score = min(1.0, abs(fz + 2.5) / 1.5)  # -2.5→0, -4.0→1.0
            score = 0.5 * fz_score + 0.25 * min(1.0, vz / 2.0) + 0.25 * min(1.0, -ret_1h / 0.05)
            if score < 0.3:
                continue

            signals.append({"symbol": sym, "entry_ts": ts, "direction": "long",
                           "event_type": f"FundingSqueeze{label}", "event_score": round(score, 3), "regime": regime})
    return paper_trade(signals, data, hold_bars=4)


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def _verdict(name, res):
    if "n_trades" not in res or res["n_trades"] == 0:
        return
    c9 = res.get("cost_9bps", {})
    c12 = res.get("cost_12bps", {})
    net9 = c9.get("net_bps", 0)
    pf9 = c9.get("pf", 0)
    net_no_top3 = c9.get("net_without_top3", 0)
    top3_pct = c9.get("top3_pct_of_net", 100)
    n = res["n_trades"]

    if n < 15:
        verdict = "SPARSE — too few signals for event type"
    elif net9 <= 0:
        verdict = "KILL — net negative at 9bps"
    elif pf9 < 1.10:
        verdict = "KILL — PF too low"
    elif top3_pct > 80:
        verdict = "TOO_CONCENTRATED"
    elif c12.get("net_bps", -1) <= 0:
        verdict = "COST_SENSITIVE"
    elif net_no_top3 <= 0:
        verdict = "PAPER_CANDIDATE"
    elif n >= 20 and pf9 >= 1.15 and net_no_top3 > 0:
        verdict = "SHADOW_CANDIDATE"
    else:
        verdict = "PAPER_CANDIDATE"

    print(f"  {name:45s} n={n:4d} Net9={net9:+7.0f} PF={pf9:.2f} "
          f"Top3%={top3_pct:4.0f}% NoT3={net_no_top3:+7.0f} → {verdict}")


def main():
    print("Loading...")
    data = load_data()
    feats = precompute_features(data)
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))

    uni_mid = get_universe(data, 20, 100)
    uni_ext = get_universe(data, 101, 200)
    uni_all = get_universe(data, 20, 200)
    print(f"Mid: {len(uni_mid)}, Ext: {len(uni_ext)}, All: {len(uni_all)}")

    all_res = {}

    print("\n=== OI Buildup V2 (long-only, tighter) ===")
    for uni_name, uni in [("mid", uni_mid), ("ext", uni_ext), ("all", uni_all)]:
        res = oi_buildup_v2(data, feats, regime_map, uni, f"_{uni_name}")
        all_res[f"OI_Buildup_V2_{uni_name}"] = res
        _verdict(f"OI_Buildup_V2_{uni_name}", res)

    print("\n=== Failed Breakout V2 (regime filtered) ===")
    for uni_name, uni in [("mid", uni_mid), ("ext", uni_ext), ("all", uni_all)]:
        res = failed_breakout_v2(data, feats, regime_map, uni, f"_{uni_name}")
        all_res[f"FailedBreakout_V2_{uni_name}"] = res
        _verdict(f"FailedBreakout_V2_{uni_name}", res)

    print("\n=== Vol Compression V2 (long-only) ===")
    for uni_name, uni in [("mid", uni_mid), ("ext", uni_ext), ("all", uni_all)]:
        res = vol_compression_v2(data, feats, regime_map, uni, f"_{uni_name}")
        all_res[f"VolCompression_V2_{uni_name}"] = res
        _verdict(f"VolCompression_V2_{uni_name}", res)

    print("\n=== Funding Squeeze (new event type) ===")
    for uni_name, uni in [("mid", uni_mid), ("ext", uni_ext), ("all", uni_all)]:
        res = funding_squeeze(data, feats, regime_map, uni, f"_{uni_name}")
        all_res[f"FundingSqueeze_{uni_name}"] = res
        _verdict(f"FundingSqueeze_{uni_name}", res)

    # Save
    out_path = ROOT / "opportunity_expansion_v2_report.json"
    with open(out_path, "w") as f:
        json.dump(all_res, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Overall conclusion
    survivors = [k for k, v in all_res.items()
                 if v.get("cost_9bps", {}).get("net_bps", 0) > 0
                 and v.get("cost_9bps", {}).get("pf", 1.0) >= 1.10]
    print(f"\nSurvivors (PASS or better): {survivors if survivors else 'NONE'}")


if __name__ == "__main__":
    main()
