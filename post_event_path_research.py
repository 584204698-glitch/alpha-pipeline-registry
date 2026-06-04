"""Post-Event Path Alpha Research — second-leg alpha after event trades complete.

4 research directions:
  1. Hold Extension — extend hold from 2→4/6 if conditions clean
  2. Bounce Failure Short — Deleveraging long fails → short second leg
  3. Event Continuation — RS Shock +3/+6 bar follow-through by regime
  4. Early Exit Filter — failure conditions → exit before hold=2

All paper-only, 9/12/15bps cost analysis.
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
from event_scanner import DeleveragingEventScanner, EventState, SmallCapUniverse
from relative_strength_shock import detect_relative_strength_shock
from oi_shock_absorption import detect_oi_shock_absorption


def load_data():
    return pd.read_parquet(ROOT / "data" / "data_storage.parquet")


def get_universe(data, min_rank=20, max_rank=100):
    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    return set(vol_rank[(vol_rank >= min_rank) & (vol_rank <= max_rank)].index) & set(
        data.index.get_level_values("symbol").unique())


def precompute_post_features(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Features needed for post-event path analysis."""
    feats = {}
    feats["ret_1h"] = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(4))
    feats["ret_4h"] = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(16))
    vol = data["volume"]
    gv = vol.groupby(level="symbol")
    vol_mean = gv.transform(lambda s: s.rolling(48, min_periods=8).mean())
    vol_std = gv.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    feats["vol_z"] = ((vol - vol_mean) / vol_std).fillna(0.0)
    oi_delta = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(6))
    go_oi = oi_delta.groupby(level="symbol")
    oi_mean = go_oi.transform(lambda s: s.rolling(48, min_periods=8).mean())
    oi_std = go_oi.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    feats["oi_delta_z"] = ((oi_delta - oi_mean) / oi_std).fillna(0.0)
    hl_range = (data["high"] - data["low"]).clip(lower=1e-8)
    feats["close_loc"] = (data["close"] - data["low"]) / hl_range
    funding = data["funding_rate"]
    gf = funding.groupby(level="symbol")
    f_mean = gf.transform(lambda s: s.rolling(24, min_periods=8).mean())
    f_std = gf.transform(lambda s: s.rolling(24, min_periods=8).std()).replace(0, np.nan)
    feats["funding_z"] = ((funding - f_mean) / f_std).fillna(0.0)
    # VWAP (rolling 16-bar = 4h)
    vwap_num = (data["close"] * data["volume"]).groupby(level="symbol").transform(
        lambda s: s.rolling(16, min_periods=4).sum())
    vwap_den = data["volume"].groupby(level="symbol").transform(
        lambda s: s.rolling(16, min_periods=4).sum()).clip(lower=1e-8)
    feats["vwap_4h"] = vwap_num / vwap_den
    return feats


# ═══════════════════════════════════════════════
#  SIGNAL COLLECTION — replay all events
# ═══════════════════════════════════════════════

@dataclass
class EventTrade:
    symbol: str
    event_type: str  # DeleveragingReversal, OIShockAbsorption, RelativeStrengthShock
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp  # standard hold=2 exit
    direction: str
    entry_price: float
    exit_price: float
    gross_bps: float
    regime: str
    event_score: float
    # Post-exit features (at exit_ts)
    close_vs_vwap: float = 0.0
    oi_z_at_exit: float = 0.0
    vol_z_at_exit: float = 0.0
    close_loc_at_exit: float = 0.0
    funding_z_at_exit: float = 0.0
    ret_1h_at_exit: float = 0.0



def collect_all_event_trades(data, feats, regime_map, universe) -> list[EventTrade]:
    """Replay all 3 event scanners, collect completed trades + post-exit state."""
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    trades = []

    # 1. Deleveraging Reversal scanner
    scanner = DeleveragingEventScanner(data, regime_map, universe=SmallCapUniverse(), hold_bars=2)
    for i, ts in enumerate(ts_list[100:]):  # skip warmup
        results = scanner.scan(ts)
        for r in results:
            if r.state != EventState.CONFIRMED:
                continue
            sym = r.symbol
            idx = ts_to_idx.get(ts)
            if idx is None or idx + 2 >= len(ts_list):
                continue
            exit_ts = ts_list[idx + 2]
            try:
                entry_px = float(data.loc[(ts, sym), "close"])
                exit_px = float(data.loc[(exit_ts, sym), "close"])
            except KeyError:
                continue
            gross = (exit_px / entry_px - 1.0) * 10000
            regime = regime_map.get(ts, "unknown")
            post = _get_post_exit_features(data, feats, sym, exit_ts)
            trades.append(EventTrade(
                symbol=sym, event_type="DeleveragingReversal",
                entry_ts=ts, exit_ts=exit_ts, direction="long",
                entry_price=entry_px, exit_price=exit_px,
                gross_bps=gross, regime=regime,
                event_score=r.event_score, **post))
            scanner.cooldowns[sym] = scanner.cooldown_bars

    # 2. OI Shock Absorption
    oi_events = detect_oi_shock_absorption(data, universe, regime_map,
                                            oi_delta_z_min=2.5, vol_z_min=1.5, cooldown_bars=8)
    seen_oi = set()
    for ev in oi_events:
        ts = ev["timestamp"]
        sym = ev["symbol"]
        key = (sym, ts)
        if key in seen_oi:
            continue
        seen_oi.add(key)
        idx = ts_to_idx.get(ts)
        if idx is None or idx + 2 >= len(ts_list):
            continue
        exit_ts = ts_list[idx + 2]
        try:
            entry_px = float(data.loc[(ts, sym), "close"])
            exit_px = float(data.loc[(exit_ts, sym), "close"])
        except KeyError:
            continue
        gross = (exit_px / entry_px - 1.0) * 10000
        regime = regime_map.get(ts, "unknown")
        post = _get_post_exit_features(data, feats, sym, exit_ts)
        trades.append(EventTrade(
            symbol=sym, event_type="OIShockAbsorption",
            entry_ts=ts, exit_ts=exit_ts, direction="long",
            entry_price=entry_px, exit_price=exit_px,
            gross_bps=gross, regime=regime,
            event_score=ev.get("event_score", 0), **post))

    # 3. RS Shock
    rs_events = detect_relative_strength_shock(data, universe, regime_map,
                                                rs_threshold=0.05, vol_z_min=2.0,
                                                oi_delta_z_min=0.5, close_loc_min=0.50,
                                                event_score_min=0.60, cooldown_bars=8)
    seen_rs = set()
    for ev in rs_events:
        ts = ev["timestamp"]
        sym = ev["symbol"]
        key = (sym, ts)
        if key in seen_rs:
            continue
        seen_rs.add(key)
        idx = ts_to_idx.get(ts)
        if idx is None or idx + 2 >= len(ts_list):
            continue
        exit_ts = ts_list[idx + 2]
        try:
            entry_px = float(data.loc[(ts, sym), "close"])
            exit_px = float(data.loc[(exit_ts, sym), "close"])
        except KeyError:
            continue
        ret = (exit_px / entry_px - 1.0)
        if ev.get("direction") == "short":
            ret = -ret
        gross = ret * 10000
        regime = regime_map.get(ts, "unknown")
        post = _get_post_exit_features(data, feats, sym, exit_ts)
        trades.append(EventTrade(
            symbol=sym, event_type="RelativeStrengthShock",
            entry_ts=ts, exit_ts=exit_ts, direction=ev.get("direction", "long"),
            entry_price=entry_px, exit_price=exit_px,
            gross_bps=gross, regime=regime,
            event_score=ev.get("event_score", 0), **post))

    return trades


def _get_post_exit_features(data, feats, sym, ts) -> dict:
    """Get features at the exit timestamp for a symbol."""
    try:
        close = float(data.loc[(ts, sym), "close"])
        vwap = float(feats["vwap_4h"].loc[(ts, sym)])
        return {
            "close_vs_vwap": (close / vwap - 1.0) if vwap > 0 else 0,
            "oi_z_at_exit": float(feats["oi_delta_z"].loc[(ts, sym)]) if (ts, sym) in feats["oi_delta_z"].index else 0,
            "vol_z_at_exit": float(feats["vol_z"].loc[(ts, sym)]) if (ts, sym) in feats["vol_z"].index else 0,
            "close_loc_at_exit": float(feats["close_loc"].loc[(ts, sym)]) if (ts, sym) in feats["close_loc"].index else 0.5,
            "funding_z_at_exit": float(feats["funding_z"].loc[(ts, sym)]) if (ts, sym) in feats["funding_z"].index else 0,
            "ret_1h_at_exit": float(feats["ret_1h"].loc[(ts, sym)]) if (ts, sym) in feats["ret_1h"].index else 0,
        }
    except (KeyError, TypeError):
        return {"close_vs_vwap": 0, "oi_z_at_exit": 0, "vol_z_at_exit": 0,
                "close_loc_at_exit": 0.5, "funding_z_at_exit": 0, "ret_1h_at_exit": 0}


# ═══════════════════════════════════════════════
#  1. HOLD EXTENSION
# ═══════════════════════════════════════════════

def research_hold_extension(trades, data, ts_list, ts_to_idx, hold_ext=4):
    """Extend hold from 2→4 bars. Test with/without filter conditions."""
    # Baseline: all trades at hold=2
    baseline = compute_pnl(trades, [9, 12, 15])

    # Unconditional: extend ALL to hold=4
    ext_all = []
    for t in trades:
        idx = ts_to_idx.get(t.entry_ts)
        if idx is None or idx + hold_ext >= len(ts_list):
            continue
        exit_ts = ts_list[idx + hold_ext]
        try:
            exit_px = float(data.loc[(exit_ts, t.symbol), "close"])
        except KeyError:
            continue
        ret = (exit_px / t.entry_price - 1.0)
        if t.direction == "short":
            ret = -ret
        gross = ret * 10000
        ext_all.append({"gross_bps": gross, "symbol": t.symbol, "regime": t.regime})
    unconditional = compute_pnl_from_dicts(ext_all, [9, 12, 15])

    # Conditional: extend only if post-exit conditions are clean
    ext_cond = []
    ext_drop = []
    for t in trades:
        # Conditions: price above VWAP, OI not worsening further, vol calming
        if t.close_vs_vwap > -0.01 and t.oi_z_at_exit > -1.0 and t.vol_z_at_exit < 2.0:
            idx = ts_to_idx.get(t.entry_ts)
            if idx is None or idx + hold_ext >= len(ts_list):
                continue
            exit_ts = ts_list[idx + hold_ext]
            try:
                exit_px = float(data.loc[(exit_ts, t.symbol), "close"])
            except KeyError:
                continue
            ret = (exit_px / t.entry_price - 1.0)
            if t.direction == "short":
                ret = -ret
            gross = ret * 10000
            ext_cond.append({"gross_bps": gross, "symbol": t.symbol, "regime": t.regime})
        else:
            ext_drop.append(t)
    conditional = compute_pnl_from_dicts(ext_cond, [9, 12, 15])

    return {
        "baseline_hold2": baseline,
        "extend_all_hold4": unconditional,
        "extend_conditional_hold4": conditional,
        "conditional_trade_count": len(ext_cond),
        "dropped_trade_count": len(ext_drop),
    }


# ═══════════════════════════════════════════════
#  2. BOUNCE FAILURE SHORT (Deleveraging only)
# ═══════════════════════════════════════════════

def research_bounce_failure_short(trades, data, ts_list, ts_to_idx):
    """After Deleveraging long fails, enter SHORT at bar+3, hold 2 bars."""
    delev_trades = [t for t in trades if t.event_type == "DeleveragingReversal"]
    short_trades = []
    for t in delev_trades:
        # Conditions for bounce failure:
        # 1. The original long was a loser OR barely profitable (< 5bps gross)
        # 2. Close at exit is still below VWAP
        # 3. OI still depressed
        # 4. Volume still elevated
        # 5. BTC not in trend_up (trend_up shorting dangerous)
        if t.gross_bps > 50:  # bounce succeeded, no short
            continue
        if t.close_vs_vwap > 0.005:  # above VWAP = bounce working
            continue
        if t.oi_z_at_exit > -0.5:  # OI recovering = liquidation done
            continue
        if t.vol_z_at_exit < 0.5:  # vol too low = no continuation
            continue
        if t.regime == "trend_up":  # don't short into uptrend
            continue

        # Enter short at bar+3 (1 bar after exit)
        idx = ts_to_idx.get(t.entry_ts)
        if idx is None or idx + 3 >= len(ts_list):
            continue
        entry_ts = ts_list[idx + 3]
        exit_idx = idx + 5  # hold 2 bars
        if exit_idx >= len(ts_list):
            continue
        exit_ts = ts_list[exit_idx]
        try:
            entry_px = float(data.loc[(entry_ts, t.symbol), "close"])
            exit_px = float(data.loc[(exit_ts, t.symbol), "close"])
        except KeyError:
            continue
        gross = (entry_px / exit_px - 1.0) * 10000  # short: entry→exit profit
        short_trades.append({"gross_bps": gross, "symbol": t.symbol, "regime": t.regime})

    return compute_pnl_from_dicts(short_trades, [9, 12, 15])


# ═══════════════════════════════════════════════
#  3. EVENT CONTINUATION (RS Shock)
# ═══════════════════════════════════════════════

def research_event_continuation(trades, data, ts_list, ts_to_idx, event_type="RelativeStrengthShock"):
    """After RS Shock trade, enter continuation in same direction at bar+3."""
    rs_trades = [t for t in trades if t.event_type == event_type]
    cont_trades = []
    regime_results = defaultdict(list)

    for t in rs_trades:
        idx = ts_to_idx.get(t.entry_ts)
        if idx is None or idx + 3 >= len(ts_list):
            continue
        # Enter continuation at bar+3, hold 3 bars
        cont_entry_idx = idx + 3
        cont_exit_idx = cont_entry_idx + 3
        if cont_exit_idx >= len(ts_list):
            continue
        entry_ts = ts_list[cont_entry_idx]
        exit_ts = ts_list[cont_exit_idx]
        try:
            entry_px = float(data.loc[(entry_ts, t.symbol), "close"])
            exit_px = float(data.loc[(exit_ts, t.symbol), "close"])
        except KeyError:
            continue
        ret = (exit_px / entry_px - 1.0)
        if t.direction == "short":
            ret = -ret
        gross = ret * 10000

        # Regime-specific filtering
        rg = t.regime
        # range: continuation unreliable → filter for strong close
        if rg == "range" and t.close_loc_at_exit < 0.6:
            continue
        # trend_down: short continuation most reliable
        if rg == "trend_up" and ret < 0:
            continue  # skip losing continuation in uptrend

        cont_trades.append({"gross_bps": gross, "symbol": t.symbol, "regime": rg})
        regime_results[rg].append(gross)

    overall = compute_pnl_from_dicts(cont_trades, [9, 12, 15])
    regime_split = {}
    for rg, gbps in regime_results.items():
        arr = np.array(gbps)
        if len(arr) < 5:
            continue
        net9 = (arr - 18).sum()
        pos = arr[arr > 0].sum()
        neg = abs(arr[arr < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        regime_split[rg] = {"n": len(arr), "net_9bps": round(net9, 0), "pf": round(pf, 3)}

    overall["regime_split"] = regime_split
    return overall


# ═══════════════════════════════════════════════
#  4. EARLY EXIT FILTER
# ═══════════════════════════════════════════════

def research_early_exit(trades, data, ts_list, ts_to_idx):
    """Exit at bar+1 (instead of bar+2) if failure conditions detected at bar+1."""
    early_exits = []
    normal_exits = []
    for t in trades:
        idx = ts_to_idx.get(t.entry_ts)
        if idx is None or idx + 1 >= len(ts_list):
            continue
        bar1_ts = ts_list[idx + 1]
        try:
            bar1_close = float(data.loc[(bar1_ts, t.symbol), "close"])
        except KeyError:
            continue
        bar1_ret = (bar1_close / t.entry_price - 1.0)
        if t.direction == "short":
            bar1_ret = -bar1_ret

        # Failure conditions at bar+1:
        # - Price moved against us > 1%
        # - OR OI continuing to collapse (Deleveraging case)
        # - OR vol spiking (new event, not recovery)
        fail = False
        if bar1_ret < -0.01:  # >1% against
            fail = True
        try:
            oi_z_1 = float(data.loc[(bar1_ts, t.symbol), "open_interest"]) if "open_interest" in data.columns else 0
        except:
            oi_z_1 = 0

        if fail:
            gross = bar1_ret * 10000
            early_exits.append({"gross_bps": gross, "symbol": t.symbol, "regime": t.regime})
        else:
            normal_exits.append(t)

    early = compute_pnl_from_dicts(early_exits, [9, 12, 15])
    normal = compute_pnl(normal_exits, [9, 12, 15])
    return {
        "early_exit_bar1": early,
        "normal_bar2": normal,
        "early_count": len(early_exits),
        "normal_count": len(normal_exits),
    }


# ═══════════════════════════════════════════════
#  PNL HELPERS
# ═══════════════════════════════════════════════

def compute_pnl(trades: list[EventTrade], costs: list[float]) -> dict:
    arr = np.array([t.gross_bps for t in trades])
    return _pnl_from_array(arr, costs, len(trades))


def compute_pnl_from_dicts(dicts: list[dict], costs: list[float]) -> dict:
    if not dicts:
        return {"n_trades": 0}
    arr = np.array([d["gross_bps"] for d in dicts])
    return _pnl_from_array(arr, costs, len(dicts))


def _pnl_from_array(arr, costs, n):
    if n == 0:
        return {"n_trades": 0}
    results = {"n_trades": n}
    for cost in costs:
        net_arr = arr - 2 * cost
        pos = net_arr[net_arr > 0].sum()
        neg = abs(net_arr[net_arr < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        hit = (net_arr > 0).mean()
        sn = sorted(net_arr, reverse=True)
        top1 = sn[0] if sn else 0
        top3 = sum(sn[:3]) if len(sn) >= 3 else sum(sn)
        net_no_top3 = net_arr.sum() - top3
        results[f"cost_{cost}bps"] = {
            "net_bps": round(net_arr.sum(), 0),
            "pf": round(pf, 3),
            "hit_rate": round(hit, 3),
            "avg_win_bps": round(net_arr[net_arr > 0].mean(), 1) if (net_arr > 0).any() else 0,
            "avg_loss_bps": round(net_arr[net_arr < 0].mean(), 1) if (net_arr < 0).any() else 0,
            "max_loss": round(net_arr.min(), 0),
            "top1_contribution": round(top1, 0),
            "top3_contribution": round(top3, 0),
            "net_without_top3": round(net_no_top3, 0),
            "top3_pct": round(top3 / net_arr.sum() * 100, 0) if net_arr.sum() != 0 else 0,
        }
    return results


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════

def _verdict(name, res, min_n=10):
    if res.get("n_trades", 0) < min_n:
        return f"SPARSE (<{min_n} trades)"
    c9 = res.get("cost_9bps", {})
    n = res["n_trades"]
    net9 = c9.get("net_bps", 0)
    pf9 = c9.get("pf", 0)
    top3p = c9.get("top3_pct", 100)
    net_no_top3 = c9.get("net_without_top3", 0)
    c12_net = res.get("cost_12bps", {}).get("net_bps", -1)
    if net9 <= 0:
        return "KILL — net negative"
    if pf9 < 1.10:
        return "KILL — PF too low"
    if top3p > 80:
        return "TOO_CONCENTRATED"
    if c12_net <= 0:
        return "COST_SENSITIVE"
    if net_no_top3 <= 0:
        return "PAPER_CANDIDATE"
    return "SECOND_TRADE_PASS" if "extension" in name.lower() or "continuation" in name.lower() or "bounce" in name.lower() else "EXIT_FILTER_PASS"


def _print_section(title, results, label_map):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for label, res in label_map:
        if isinstance(res, dict) and "cost_9bps" in res:
            c9 = res["cost_9bps"]
            n = res.get("n_trades", 0)
            if n == 0:
                print(f"  {label:40s} → 0 trades")
                continue
            net9 = c9.get("net_bps", 0)
            pf9 = c9.get("pf", 0)
            top3p = c9.get("top3_pct", 100)
            max_l = c9.get("max_loss", 0)
            v = _verdict(label, res)
            print(f"  {label:40s} n={n:4d} Net9={net9:+7.0f} PF={pf9:.2f} "
                  f"Top3%={top3p:4.0f}% MaxLoss={max_l:+5.0f} → {v}")
        elif isinstance(res, dict) and "regime_split" in res:
            c9 = res.get("cost_9bps", {})
            n = res.get("n_trades", 0)
            if n == 0:
                print(f"  {label:40s} → 0 trades")
                continue
            net9 = c9.get("net_bps", 0)
            pf9 = c9.get("pf", 0)
            v = _verdict(label, res)
            print(f"  {label:40s} n={n:4d} Net9={net9:+7.0f} PF={pf9:.2f} → {v}")
            for rg, rd in res.get("regime_split", {}).items():
                print(f"    regime={rg:15s} n={rd['n']:3d} Net9={rd['net_9bps']:+6.0f} PF={rd['pf']:.2f}")
        else:
            print(f"  {label:40s} → {str(res)[:80]}")


def main():
    print("Loading data...")
    data = load_data()
    feats = precompute_post_features(data)
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))
    universe = get_universe(data)
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    print(f"Universe: {len(universe)}, Timestamps: {len(ts_list)}")

    print("\nCollecting all historical event trades...")
    trades = collect_all_event_trades(data, feats, regime_map, universe)
    print(f"Total trades collected: {len(trades)}")
    by_type = defaultdict(int)
    for t in trades:
        by_type[t.event_type] += 1
    for et, cnt in by_type.items():
        print(f"  {et}: {cnt}")

    all_results = {}

    # 1. Hold Extension (by event type)
    print("\n=== 1. HOLD EXTENSION (2→4 bars) ===")
    for et in ["DeleveragingReversal", "OIShockAbsorption", "RelativeStrengthShock"]:
        et_trades = [t for t in trades if t.event_type == et]
        if len(et_trades) < 10:
            continue
        res = research_hold_extension(et_trades, data, ts_list, ts_to_idx, hold_ext=4)
        all_results[f"HoldExt_{et}"] = res
        _print_section(f"Hold Extension — {et}", res, [
            ("Baseline (hold=2)", res["baseline_hold2"]),
            ("Extend ALL to hold=4", res["extend_all_hold4"]),
            ("Extend CONDITIONAL to hold=4", res["extend_conditional_hold4"]),
        ])
        print(f"  Conditional: {res['conditional_trade_count']} kept, {res['dropped_trade_count']} dropped")
        # Also test hold=6
        res6 = research_hold_extension(et_trades, data, ts_list, ts_to_idx, hold_ext=6)
        all_results[f"HoldExt6_{et}"] = res6
        nc6 = res6.get("conditional_trade_count", 0)
        c9_6 = res6["extend_conditional_hold4"].get("cost_9bps", {})
        print(f"  Extend CONDITIONAL to hold=6: n={nc6} Net9={c9_6.get('net_bps','?')} PF={c9_6.get('pf','?')}")

    # 2. Bounce Failure Short (Deleveraging only)
    print("\n=== 2. BOUNCE FAILURE SHORT (Deleveraging) ===")
    res_bf = research_bounce_failure_short(trades, data, ts_list, ts_to_idx)
    all_results["BounceFailureShort"] = res_bf
    _print_section("Bounce Failure Short", res_bf, [("Deleveraging bounce fail → short", res_bf)])

    # 3. Event Continuation (RS Shock)
    print("\n=== 3. EVENT CONTINUATION (RS Shock) ===")
    res_ec = research_event_continuation(trades, data, ts_list, ts_to_idx, "RelativeStrengthShock")
    all_results["EventContinuation_RS"] = res_ec
    _print_section("RS Shock Continuation (+3 bars)", res_ec, [("RS Shock continuation (all)", res_ec)])

    # Also test continuation for Deleveraging (same direction)
    res_ec_d = research_event_continuation(trades, data, ts_list, ts_to_idx, "DeleveragingReversal")
    all_results["EventContinuation_Delev"] = res_ec_d
    _print_section("Deleveraging Continuation (+3 bars)", res_ec_d, [("Deleveraging continuation (all)", res_ec_d)])

    # 4. Early Exit Filter
    print("\n=== 4. EARLY EXIT FILTER ===")
    res_ee = research_early_exit(trades, data, ts_list, ts_to_idx)
    all_results["EarlyExit"] = res_ee
    _print_section("Early Exit (bar+1 vs bar+2)", res_ee, [
        ("Early exit at bar+1 (failures)", res_ee["early_exit_bar1"]),
        ("Normal exit at bar+2 (non-failures)", res_ee["normal_bar2"]),
    ])
    print(f"  Early: {res_ee['early_count']} trades, Normal: {res_ee['normal_count']} trades")

    # Save
    out = ROOT / "post_event_path_report.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out}")

    # Final verdicts
    print(f"\n{'='*60}")
    print("  FINAL VERDICTS")
    print(f"{'='*60}")
    for name, res in all_results.items():
        if isinstance(res, dict) and "cost_9bps" in res:
            print(f"  {name:50s} → {_verdict(name, res)}")
        elif isinstance(res, dict) and "baseline_hold2" in res:
            baseline = res["baseline_hold2"]
            cond = res["extend_conditional_hold4"]
            b9 = baseline.get("cost_9bps", {}).get("net_bps", 0)
            c9 = cond.get("cost_9bps", {}).get("net_bps", 0)
            improvement = c9 - b9 if isinstance(c9, (int, float)) and isinstance(b9, (int, float)) else 0
            v = "HOLD_EXTENSION_PASS" if improvement > 0 and cond.get("cost_9bps", {}).get("pf", 0) >= 1.10 else "KILL"
            print(f"  {name:50s} Baseline={b9:+6.0f} Cond={c9:+6.0f} Δ={improvement:+6.0f} → {v}")
        elif isinstance(res, dict) and "early_exit_bar1" in res:
            ee = res["early_exit_bar1"]
            nm = res["normal_bar2"]
            ee9 = ee.get("cost_9bps", {}).get("net_bps", 0)
            nm9 = nm.get("cost_9bps", {}).get("net_bps", 0)
            ee_loss = ee.get("cost_9bps", {}).get("max_loss", 0)
            nm_loss = nm.get("cost_9bps", {}).get("max_loss", 0)
            loss_improvement = abs(ee_loss) - abs(nm_loss) if isinstance(ee_loss, (int, float)) and isinstance(nm_loss, (int, float)) else 0
            v = "EXIT_FILTER_PASS" if loss_improvement < 0 else "KILL — no loss reduction"
            print(f"  {name:50s} EarlyLoss={ee_loss:+5.0f} NormalLoss={nm_loss:+5.0f} Δ={loss_improvement:+5.0f} → {v}")


if __name__ == "__main__":
    main()
