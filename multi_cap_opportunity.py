"""Multi-Cap Opportunity Exploration — 5 new research directions.

Directions:
  1. Funding Carry Decomposition: funding_pnl vs price_pnl, extreme funding mean-reversion
  2. OI Cascade + Wick Confirmation: forced liquidation absorption (different from Deleveraging)
  3. Taker Flow Imbalance: taker_buy/sell ratio extremes → directional alpha
  4. Session-Enhanced Events: Asia/EU/US session filtering of existing signals
  5. Mid-Cap 101-300: funding carry + low-vol events on mid-cap layer

Paper-only, shadow JSONL, 9/12/15bps cost. No live.
"""
from __future__ import annotations

import json, sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
sys.path.insert(0, str(ROOT))
from research.regime_detector import detect_regime_fast


def load_data():
    return pd.read_parquet(ROOT / "data" / "data_storage.parquet")


def get_universe(data, min_rank, max_rank):
    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    return set(vol_rank[(vol_rank >= min_rank) & (vol_rank <= max_rank)].index) & set(
        data.index.get_level_values("symbol").unique())


def precompute(data):
    f = {}
    f["ret_1h"] = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(4))
    f["ret_4h"] = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(16))
    vol = data["volume"]
    gv = vol.groupby(level="symbol")
    vm = gv.transform(lambda s: s.rolling(48, min_periods=8).mean())
    vs = gv.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    f["vol_z"] = ((vol - vm) / vs).fillna(0.0)
    oi_d = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(6))
    go = oi_d.groupby(level="symbol")
    om = go.transform(lambda s: s.rolling(48, min_periods=8).mean())
    os = go.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    f["oi_delta_z"] = ((oi_d - om) / os).fillna(0.0)
    hl = (data["high"] - data["low"]).clip(lower=1e-8)
    f["close_loc"] = (data["close"] - data["low"]) / hl
    f["lower_wick"] = (data["open"].clip(upper=data["close"]) - data["low"]) / hl
    f["upper_wick"] = (data["high"] - data["open"].clip(lower=data["close"])) / hl
    fr = data["funding_rate"]
    gf = fr.groupby(level="symbol")
    fm = gf.transform(lambda s: s.rolling(24, min_periods=8).mean())
    fs = gf.transform(lambda s: s.rolling(24, min_periods=8).std()).replace(0, np.nan)
    f["funding_z"] = ((fr - fm) / fs).fillna(0.0)
    # Taker flow
    tbv = data.get("taker_buy_volume", data["volume"] * 0.5)
    tsv = data.get("taker_sell_volume", data["volume"] * 0.5)
    f["taker_ratio"] = (tbv / (tbv + tsv).clip(lower=1e-8)).fillna(0.5)
    tz = f["taker_ratio"]
    gtz = tz.groupby(level="symbol")
    tzm = gtz.transform(lambda s: s.rolling(48, min_periods=8).mean())
    tzs = gtz.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    f["taker_z"] = ((tz - tzm) / tzs).fillna(0.0)
    # Spread proxy: (high-low)/close, z-scored
    spread_px = (data["high"] - data["low"]) / data["close"].clip(lower=1e-8)
    gsp = spread_px.groupby(level="symbol")
    spm = gsp.transform(lambda s: s.rolling(48, min_periods=8).mean())
    sps = gsp.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    f["spread_z"] = ((spread_px - spm) / sps).fillna(0.0)
    # Session: 0=Asia(0-8UTC), 1=EU(8-16), 2=US(16-24)
    hours = data.index.get_level_values("timestamp").hour
    f["session"] = pd.Series(np.where(hours < 8, 0, np.where(hours < 16, 1, 2)),
                              index=data.index)
    return f


def pnl_from_arr(arr, costs=[9,12,15]):
    n = len(arr)
    if n == 0:
        return {"n_trades": 0}
    r = {"n_trades": n}
    for cost in costs:
        net = arr - 2 * cost
        pos = net[net > 0].sum()
        neg = abs(net[net < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        sn = sorted(net, reverse=True)
        t3 = sum(sn[:3]) if len(sn) >= 3 else sum(sn)
        r[f"cost_{cost}bps"] = {
            "net_bps": round(net.sum(), 0), "pf": round(pf, 3),
            "hit_rate": round((net > 0).mean(), 3),
            "avg_win": round(net[net > 0].mean(), 1) if (net > 0).any() else 0,
            "avg_loss": round(net[net < 0].mean(), 1) if (net < 0).any() else 0,
            "max_loss": round(net.min(), 0),
            "top3": round(t3, 0), "net_no_top3": round(net.sum() - t3, 0),
        }
    return r


def verdict(name, res, min_n=10):
    if res.get("n_trades", 0) < min_n:
        return "SPARSE"
    c9 = res.get("cost_9bps", {})
    n = res["n_trades"]; net9 = c9.get("net_bps", 0); pf9 = c9.get("pf", 0)
    if net9 <= 0: return "KILL"
    if pf9 < 1.10: return "KILL_PF"
    c12 = res.get("cost_12bps", {})
    if c12.get("net_bps", -1) <= 0: return "COST_SENSITIVE"
    if c9.get("net_no_top3", -1) <= 0: return "PAPER_CANDIDATE"
    return "NEW_OPPORTUNITY_PASS"


# ═══════════════════════════════════════════
#  1. FUNDING CARRY DECOMPOSITION
# ═══════════════════════════════════════════

def research_funding_carry(data, feats, regime_map, universe, hold_bars=8):
    """Funding carry trade: extreme negative funding → long, earn funding + price recovery."""
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    trades_price = []
    trades_funding = []
    trades_combined = []

    for i in range(0, len(ts_list) - hold_bars - 1, 4):  # scan every 1h
        ts = ts_list[i]
        mask = data.index.get_level_values("timestamp") == ts
        syms = set(data.index.get_level_values("symbol")[mask]) & universe
        regime = regime_map.get(ts, "unknown")
        if regime in ("panic_down",):
            continue

        for sym in syms:
            try:
                fz = float(feats["funding_z"].loc[(ts, sym)])
            except (KeyError, TypeError):
                continue
            if np.isnan(fz):
                continue

            # Entry: extreme negative funding (< -2.5σ)
            if fz > -2.5:
                continue
            idx = ts_to_idx.get(ts)
            if idx is None or idx + hold_bars >= len(ts_list):
                continue
            exit_ts = ts_list[idx + hold_bars]
            try:
                entry_px = float(data.loc[(ts, sym), "close"])
                exit_px = float(data.loc[(exit_ts, sym), "close"])
            except KeyError:
                continue

            # Price PnL
            price_ret = (exit_px / entry_px - 1.0)
            price_bps = price_ret * 10000

            # Funding PnL: avg funding_rate * hours/8 * 10000 bps
            funding_sum = 0.0
            valid_bars = 0
            for j in range(hold_bars):
                ft = ts_list[idx + j]
                try:
                    fr = float(data.loc[(ft, sym), "funding_rate"])
                    funding_sum += fr
                    valid_bars += 1
                except (KeyError, TypeError):
                    pass
            funding_bps = (funding_sum / max(valid_bars, 1)) * 10000 * (hold_bars * 0.25 / 8)

            combined_bps = price_bps + funding_bps * 0.5  # halve funding for realism

            trades_price.append(price_bps)
            trades_funding.append(funding_bps)
            trades_combined.append(combined_bps)

    return {
        "price_only": pnl_from_arr(np.array(trades_price) if trades_price else np.array([])),
        "funding_only": pnl_from_arr(np.array(trades_funding) if trades_funding else np.array([])),
        "combined": pnl_from_arr(np.array(trades_combined) if trades_combined else np.array([])),
    }


# Simpler: funding carry as standalone trade
def research_funding_carry_simple(data, feats, regime_map, universe, hold_bars=8):
    """Just the funding carry: hold a position purely for funding, minimal price exposure."""
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    # Use 8h funding cycles
    all_trades = []
    for et in ["long", "short"]:
        trades = []
        for i in range(0, len(ts_list) - hold_bars - 1, 16):  # scan every 4h
            ts = ts_list[i]
            mask = data.index.get_level_values("timestamp") == ts
            syms = set(data.index[mask].get_level_values("symbol")) & universe
            regime = regime_map.get(ts, "unknown")
            if regime == "panic_down":
                continue
            for sym in syms:
                try:
                    fz = float(feats["funding_z"].loc[(ts, sym)])
                except: continue
                if np.isnan(fz): continue

                if et == "long" and fz > -2.0:  # less strict for carry
                    continue
                if et == "short" and fz < 2.0:
                    continue
                idx = ts_to_idx.get(ts)
                if idx is None or idx + hold_bars >= len(ts_list):
                    continue
                exit_ts = ts_list[idx + hold_bars]
                try:
                    ep = float(data.loc[(ts, sym), "close"])
                    xp = float(data.loc[(exit_ts, sym), "close"])
                except: continue

                ret = (xp / ep - 1.0)
                if et == "short":
                    ret = -ret
                price_bps = ret * 10000

                # Earn funding over hold. funding_rate is per 8h.
                # hold_bars * 0.25 = hours held. funding earned = avg_rate * hours/8 * 10000 bps
                fund_sum = 0.0
                valid_bars = 0
                for j in range(hold_bars):
                    try:
                        fund_sum += float(data.loc[(ts_list[idx + j], sym), "funding_rate"])
                        valid_bars += 1
                    except: pass
                fund_bps = (fund_sum / max(valid_bars, 1)) * 10000 * (hold_bars * 0.25 / 8)
                if et == "short":
                    fund_bps = -fund_bps
                total = price_bps + fund_bps
                trades.append(total)
        all_trades.append((et, trades))

    results = {}
    for et, trades in all_trades:
        arr = np.array(trades) if trades else np.array([])
        results[et] = pnl_from_arr(arr)
    return results


# ═══════════════════════════════════════════
#  2. OI CASCADE + WICK CONFIRMATION
# ═══════════════════════════════════════════

def research_oi_cascade_wick(data, feats, regime_map, universe):
    """OI collapse + long lower wick = forced sells being absorbed by buyers.
    Different from Deleveraging: doesn't require volume surge or extreme return.
    Focus on the wick pattern: buyers stepping in at the low."""
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    trades = {"long": [], "short": []}

    for i in range(1, len(ts_list) - 5):
        ts = ts_list[i]
        mask = data.index.get_level_values("timestamp") == ts
        syms = set(data.index.get_level_values("symbol")[mask]) & universe
        regime = regime_map.get(ts, "unknown")
        if regime in ("panic_down", "chop"):
            continue

        for sym in syms:
            try:
                oi_z = float(feats["oi_delta_z"].loc[(ts, sym)])
                lw = float(feats["lower_wick"].loc[(ts, sym)])
                uw = float(feats["upper_wick"].loc[(ts, sym)])
                ret_1h = float(feats["ret_1h"].loc[(ts, sym)])
            except (KeyError, TypeError):
                continue
            if np.isnan(oi_z) or np.isnan(lw):
                continue

            # Long setup: OI collapse + long lower wick (buyers absorbing)
            if oi_z < -1.5 and lw > 0.6 and uw < 0.2:
                # Price dropped but buyers stepped in hard at the low
                idx = ts_to_idx.get(ts)
                if idx is None or idx + 3 >= len(ts_list):
                    continue
                try:
                    ep = float(data.loc[(ts, sym), "close"])
                    xp = float(data.loc[(ts_list[idx + 3], sym), "close"])
                except KeyError:
                    continue
                trades["long"].append((xp / ep - 1.0) * 10000)

            # Short setup: OI surge + long upper wick (sellers rejecting)
            if oi_z > 1.5 and uw > 0.6 and lw < 0.2:
                idx = ts_to_idx.get(ts)
                if idx is None or idx + 3 >= len(ts_list):
                    continue
                try:
                    ep = float(data.loc[(ts, sym), "close"])
                    xp = float(data.loc[(ts_list[idx + 3], sym), "close"])
                except KeyError:
                    continue
                trades["short"].append((ep / xp - 1.0) * 10000)

    results = {}
    for d, arr_list in trades.items():
        results[d] = pnl_from_arr(np.array(arr_list) if arr_list else np.array([]))
    return results


# ═══════════════════════════════════════════
#  3. TAKER FLOW IMBALANCE
# ═══════════════════════════════════════════

def research_taker_flow(data, feats, regime_map, universe):
    """Extreme taker buy/sell ratio → directional alpha."""
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    trades = []

    for i in range(1, len(ts_list) - 5):
        ts = ts_list[i]
        mask = data.index.get_level_values("timestamp") == ts
        syms = set(data.index.get_level_values("symbol")[mask]) & universe
        regime = regime_map.get(ts, "unknown")
        if regime in ("panic_down", "chop"):
            continue

        for sym in syms:
            try:
                tz = float(feats["taker_z"].loc[(ts, sym)])
                ret_1h = float(feats["ret_1h"].loc[(ts, sym)])
                vz = float(feats["vol_z"].loc[(ts, sym)])
                oi_z = float(feats["oi_delta_z"].loc[(ts, sym)])
            except (KeyError, TypeError):
                continue
            if np.isnan(tz):
                continue

            # Taker buying extreme (>2σ) + OI confirming
            if tz > 2.0 and oi_z > 0 and vz > 0.5:
                idx = ts_to_idx.get(ts)
                if idx is None or idx + 4 >= len(ts_list):
                    continue
                try:
                    ep = float(data.loc[(ts, sym), "close"])
                    xp = float(data.loc[(ts_list[idx + 4], sym), "close"])
                except KeyError:
                    continue
                trades.append({"gross": (xp / ep - 1.0) * 10000, "dir": "long"})

            # Taker selling extreme
            if tz < -2.0 and oi_z < 0 and vz > 0.5:
                idx = ts_to_idx.get(ts)
                if idx is None or idx + 4 >= len(ts_list):
                    continue
                try:
                    ep = float(data.loc[(ts, sym), "close"])
                    xp = float(data.loc[(ts_list[idx + 4], sym), "close"])
                except KeyError:
                    continue
                trades.append({"gross": (ep / xp - 1.0) * 10000, "dir": "short"})

    arr = np.array([t["gross"] for t in trades]) if trades else np.array([])
    return pnl_from_arr(arr)


# ═══════════════════════════════════════════
#  4. SESSION-ENHANCED EVENTS
# ═══════════════════════════════════════════

def research_session_alpha(data, feats, regime_map, universe):
    """Test if event alpha varies by trading session."""
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}

    # Session labels
    session_names = {0: "Asia", 1: "EU", 2: "US"}

    # Simple test: funding extreme mean-reversion by session
    all_results = {}
    for sess_id, sess_name in session_names.items():
        trades = []
        for i in range(1, len(ts_list) - 5):
            ts = ts_list[i]
            if ts.hour % 24 not in (range(0, 8) if sess_id == 0 else range(8, 16) if sess_id == 1 else range(16, 24)):
                continue
            mask = data.index.get_level_values("timestamp") == ts
            syms = set(data.index.get_level_values("symbol")[mask]) & universe
            regime = regime_map.get(ts, "unknown")
            if regime == "panic_down":
                continue
            for sym in syms:
                try:
                    fz = float(feats["funding_z"].loc[(ts, sym)])
                    oi_z = float(feats["oi_delta_z"].loc[(ts, sym)])
                except: continue
                if np.isnan(fz): continue

                # Extreme negative funding → long
                if fz < -2.5 and oi_z > -2.0:
                    idx = ts_to_idx.get(ts)
                    if idx is None or idx + 4 >= len(ts_list): continue
                    try:
                        ep = float(data.loc[(ts, sym), "close"])
                        xp = float(data.loc[(ts_list[idx + 4], sym), "close"])
                    except: continue
                    trades.append((xp / ep - 1.0) * 10000)

        all_results[sess_name] = pnl_from_arr(np.array(trades) if trades else np.array([]))
    return all_results


# ═══════════════════════════════════════════
#  5. MID-CAP 101-300
# ═══════════════════════════════════════════

def research_midcap_new(data, feats, regime_map):
    """Mid-cap (101-300): funding carry + OI cascade with wick."""
    uni_mid = get_universe(data, 101, 300)
    print(f"  Mid-cap universe (101-300): {len(uni_mid)} symbols")

    results = {}

    # Funding carry on mid-cap
    print("  Testing funding carry...")
    fc = research_funding_carry_simple(data, feats, regime_map, uni_mid, hold_bars=12)
    results["funding_carry"] = fc

    # OI cascade + wick on mid-cap
    print("  Testing OI cascade + wick...")
    ow = research_oi_cascade_wick(data, feats, regime_map, uni_mid)
    results["oi_cascade_wick"] = ow

    # Taker flow on mid-cap
    print("  Testing taker flow...")
    tf = research_taker_flow(data, feats, regime_map, uni_mid)
    results["taker_flow"] = tf

    return results


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def psection(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def pverdict(label, res, min_n=10):
    v = verdict(label, res, min_n)
    if res.get("n_trades", 0) == 0:
        print(f"  {label:45s} → 0 trades")
        return
    c9 = res.get("cost_9bps", {})
    n = res["n_trades"]; net9 = c9.get("net_bps", 0); pf9 = c9.get("pf", 0)
    print(f"  {label:45s} n={n:4d} Net9={net9:+7.0f} PF={pf9:.2f} → {v}")


def main():
    print("Loading data...")
    data = load_data()
    feats = precompute(data)
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))
    uni_small = get_universe(data, 20, 100)
    print(f"Small-cap (20-100): {len(uni_small)}")

    all_results = {}

    # 1. Funding Carry Decomposition
    psection("1. FUNDING CARRY DECOMPOSITION (extreme funding → earn funding + price recovery)")
    fc = research_funding_carry(data, feats, regime_map, uni_small, hold_bars=12)
    all_results["FundingCarry"] = fc
    for k in ["price_only", "funding_only", "combined"]:
        pverdict(f"FundingCarry_{k}", fc.get(k, {}))
    fcs = research_funding_carry_simple(data, feats, regime_map, uni_small, hold_bars=12)
    all_results["FundingCarrySimple"] = fcs
    for et in ["long", "short"]:
        pverdict(f"FundingCarrySimple_{et}", fcs.get(et, {}))

    # 2. OI Cascade + Wick
    psection("2. OI CASCADE + WICK CONFIRMATION (long wick = buyers absorbing forced sells)")
    ow = research_oi_cascade_wick(data, feats, regime_map, uni_small)
    all_results["OICascadeWick"] = ow
    for d in ["long", "short"]:
        pverdict(f"OICascadeWick_{d}", ow.get(d, {}))

    # 3. Taker Flow Imbalance
    psection("3. TAKER FLOW IMBALANCE (extreme taker buy/sell ratio)")
    tf = research_taker_flow(data, feats, regime_map, uni_small)
    all_results["TakerFlow"] = tf
    pverdict("TakerFlow", tf)

    # 4. Session-Enhanced
    psection("4. SESSION-ENHANCED (funding extreme by Asia/EU/US)")
    se = research_session_alpha(data, feats, regime_map, uni_small)
    all_results["SessionEnhanced"] = se
    for sess in ["Asia", "EU", "US"]:
        pverdict(f"Session_{sess}", se.get(sess, {}))

    # 5. Mid-Cap 101-300
    psection("5. MID-CAP 101-300 (new events on mid-cap layer)")
    mc = research_midcap_new(data, feats, regime_map)
    all_results["MidCap"] = mc
    for k, v in mc.items():
        if isinstance(v, dict) and "cost_9bps" in v:
            pverdict(f"MidCap_{k}", v)
        elif isinstance(v, dict):
            for subk, subv in v.items():
                pverdict(f"MidCap_{k}/{subk}", subv)

    # Save
    out = ROOT / "multi_cap_opportunity_report.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out}")

    # Summary
    psection("SURVIVORS (NEW_OPPORTUNITY_PASS)")
    survivors = []
    def _collect(name, obj):
        if isinstance(obj, dict) and "cost_9bps" in obj:
            v = verdict(name, obj)
            if "PASS" in v:
                survivors.append((name, obj, v))
        elif isinstance(obj, dict):
            for k, sv in obj.items():
                _collect(f"{name}/{k}", sv)
    for k, v in all_results.items():
        _collect(k, v)
    if survivors:
        for n, r, v in survivors:
            c9 = r["cost_9bps"]
            print(f"  {n:50s} n={r['n_trades']:4d} Net9={c9['net_bps']:+7.0f} PF={c9['pf']:.2f} → {v}")
    else:
        print("  NONE")


if __name__ == "__main__":
    main()
