"""
Small-cap event alpha — Batch A/B/C per specification.
OI Shock Absorption, Deleveraging Reversal, Funding Extreme Failure.
"""
import json, sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_engine import BacktestEngine
from research.regime_detector import detect_regimes


@dataclass
class Event:
    timestamp: pd.Timestamp
    symbol: str
    event_type: str
    direction: str
    strength: float
    metrics: dict = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────

def _roll_z(series, window=48):
    """Per-symbol rolling z-score."""
    g = series.groupby(level="symbol")
    mean = g.transform(lambda s: s.rolling(window, min_periods=8).mean())
    std = g.transform(lambda s: s.rolling(window, min_periods=8).std()).replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


# ── Event Detectors ───────────────────────────────────

def detect_oi_shock_absorption(data, vol_z, oi_z, close_loc, small_syms, btc_regime):
    """OI surges, price fails to continue → trapped traders.

    Spec: oi_delta_z > 1.5, volume_z > 1.0, abs(return_15m) < 1% or abs(return_1h) < 2%
    Direction: close_location > 0.6 → long (shorts trapped), < -0.6 → short
    """
    ret_15m = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(1))
    ret_1h = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(4))

    events = []
    ts_list = data.index.get_level_values("timestamp").unique()

    for ts in ts_list:
        mask = data.index.get_level_values("timestamp") == ts
        syms = data.index.get_level_values("symbol")[mask]

        for sym in syms:
            if sym not in small_syms:
                continue
            try:
                oz = float(oi_z.loc[(ts, sym)])
                vz = float(vol_z.loc[(ts, sym)])
                cl = float(close_loc.loc[(ts, sym)])
                r15 = abs(float(ret_15m.loc[(ts, sym)]))
                r1h = abs(float(ret_1h.loc[(ts, sym)]))
            except (KeyError, TypeError):
                continue

            if oz < 1.5 or vz < 1.0:
                continue
            if r15 >= 0.01 and r1h >= 0.02:  # price IS moving → no failure
                continue

            if cl > 0.6:
                events.append(Event(ts, sym, "OI_Shock_Absorption", "long",
                                    min(oz / 3.0, 1.0),
                                    {"oi_z": oz, "vol_z": vz, "close_loc": cl}))
            elif cl < 0.4:
                events.append(Event(ts, sym, "OI_Shock_Absorption", "short",
                                    min(oz / 3.0, 1.0),
                                    {"oi_z": oz, "vol_z": vz, "close_loc": cl}))
    return events


def detect_funding_extreme_failure(data, vol_z, oi_z, small_syms, btc_regime):
    """Funding extreme but price not confirming → crowded side failing.

    Spec: |funding_z| > 2.0, price failure, oi_delta_z >= -0.5
    Direction: positive funding + price stale/down → short; negative + price stale/up → long
    """
    funding = data["funding_rate"]
    funding_z = _roll_z(funding, 24)
    ret_1h = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(4))
    ret_6h = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(24))

    events = []
    ts_list = data.index.get_level_values("timestamp").unique()

    for ts in ts_list:
        mask = data.index.get_level_values("timestamp") == ts
        syms = data.index.get_level_values("symbol")[mask]

        # BTC regime check
        btc_reg = btc_regime.get(ts, "unknown") if btc_regime is not None else "unknown"

        for sym in syms:
            if sym not in small_syms:
                continue
            try:
                fz = float(funding_z.loc[(ts, sym)])
                oz = float(oi_z.loc[(ts, sym)])
                r1h = float(ret_1h.loc[(ts, sym)])
                r6h = float(ret_6h.loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            if abs(fz) < 2.0 or oz < -0.5:
                continue

            if fz > 2.0:  # positive funding → longs are paying
                # Price failure: price not going up (flat or down)
                if r1h <= 0.001:  # flat or negative 1h return
                    if btc_reg == "trend_up" and r6h > 0.02:
                        continue  # BTC strong up, don't fight it
                    events.append(Event(ts, sym, "Funding_Extreme_Failure", "short",
                                        min(abs(fz) / 4.0, 1.0),
                                        {"funding_z": fz, "ret_1h": r1h}))
            elif fz < -2.0:  # negative funding → shorts are paying
                if r1h >= -0.001:  # flat or positive 1h return
                    if btc_reg == "panic_down":
                        continue
                    events.append(Event(ts, sym, "Funding_Extreme_Failure", "long",
                                        min(abs(fz) / 4.0, 1.0),
                                        {"funding_z": fz, "ret_1h": r1h}))
    return events


def detect_deleveraging_reversal(data, vol_z, oi_z, small_syms, btc_regime):
    """OI collapse + sharp move + volume surge → forced liquidation bounce.

    Spec: oi_delta_z < -1.5, abs(return_1h) > 2.5%, volume_z > 1.0
    """
    ret_1h = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(4))

    events = []
    ts_list = data.index.get_level_values("timestamp").unique()

    for ts in ts_list:
        mask = data.index.get_level_values("timestamp") == ts
        syms = data.index.get_level_values("symbol")[mask]

        btc_reg = btc_regime.get(ts, "unknown") if btc_regime is not None else "unknown"

        for sym in syms:
            if sym not in small_syms:
                continue
            try:
                oz = float(oi_z.loc[(ts, sym)])
                vz = float(vol_z.loc[(ts, sym)])
                r1h = float(ret_1h.loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            if oz > -1.5 or vz < 1.0 or abs(r1h) < 0.025:
                continue

            if r1h < -0.025:
                if btc_reg == "panic_down":
                    continue
                events.append(Event(ts, sym, "Deleveraging_Reversal", "long",
                                    min(abs(r1h) / 0.10, 1.0),
                                    {"oi_z": oz, "vol_z": vz, "ret_1h": r1h}))
            elif r1h > 0.025:
                events.append(Event(ts, sym, "Deleveraging_Reversal", "short",
                                    min(r1h / 0.10, 1.0),
                                    {"oi_z": oz, "vol_z": vz, "ret_1h": r1h}))
    return events


# ── Paper Trading ─────────────────────────────────────

def paper_trade_events(data, events, hold_bars, cost_bps):
    """Simulate event trading with forward returns."""
    ts_values = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_values)}
    round_trip = cost_bps / 10000.0

    trades = []
    for ev in events:
        ts = ev.timestamp
        sym = ev.symbol
        if ts not in ts_to_idx:
            continue
        entry_idx = ts_to_idx[ts]
        try:
            entry_px = data.loc[(ts, sym), "close"]
        except KeyError:
            continue

        for h in range(1, hold_bars + 1):
            if entry_idx + h >= len(ts_values):
                break
            exit_ts = ts_values[entry_idx + h]
            try:
                exit_px = data.loc[(exit_ts, sym), "close"]
            except KeyError:
                continue

            ret = (exit_px / entry_px) - 1.0
            if ev.direction == "short":
                ret = -ret
            gbp = ret * 10000
            nbp = gbp - round_trip * 10000
            hit = ((ev.direction == "long" and ret > 0) or (ev.direction == "short" and ret < 0))

            trades.append({
                "event_type": ev.event_type,
                "symbol": sym, "direction": ev.direction,
                "gross_bps": gbp, "cost_bps": round_trip * 10000,
                "net_bps": nbp, "direction_hit": hit,
                "strength": ev.strength, "hold_bar": h,
                "entry_ts": ts,
            })
    return trades


def classify(gross, net, pf, n_trades, cost_to_gross, median_bps):
    if n_trades < 10:
        return "RETEST_ON_1000BAR"
    if gross < 0:
        return "KILL"
    if net > 0 and pf > 1.1 and cost_to_gross < 150 and median_bps > 0:
        return "PAPER_PASS"
    if gross > 0 and net < 0:
        return "TRADE_RULE_REPAIR"
    if pf < 0.9:
        return "KILL"
    return "FILTER_ONLY"


# ── Main ───────────────────────────────────────────────

def main():
    ROOT = Path("/mnt/e/alpha_pipeline")
    engine = BacktestEngine(ROOT)
    data = engine._load_data()

    # BTC regime
    regimes = detect_regimes(data)
    btc_regime_map = dict(zip(regimes.index, regimes))

    # Pre-compute shared metrics
    vol_z = _roll_z(data["volume"], 48)
    oi_delta = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(6))
    oi_z = _roll_z(oi_delta, 48)

    eps = 1e-8
    hl_range = (data["high"] - data["low"]).clip(lower=eps)
    close_loc = (data["close"] - data["low"]) / hl_range

    # Small-cap universe: volume rank 20-100
    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    small_syms = set(vol_rank[(vol_rank >= 20) & (vol_rank <= 100)].index)
    print(f"Small-cap universe: {len(small_syms)} symbols")

    # Detect events
    print("\n=== Detecting Events ===")
    events_oi = detect_oi_shock_absorption(data, vol_z, oi_z, close_loc, small_syms, btc_regime_map)
    events_fund = detect_funding_extreme_failure(data, vol_z, oi_z, small_syms, btc_regime_map)
    events_del = detect_deleveraging_reversal(data, vol_z, oi_z, small_syms, btc_regime_map)

    print(f"OI Shock Absorption:     {len(events_oi)} events")
    print(f"Funding Extreme Failure: {len(events_fund)} events")
    print(f"Deleveraging Reversal:   {len(events_del)} events")

    # Paper trade across full grid
    print("\n=== Paper Trading ===")
    print(f"{'Event':<28} {'Hold':>5} {'Cost':>5} {'Trades':>7} {'Gross':>10} {'Net':>10} {'C/G%':>7} {'PF':>6} {'Hit%':>6} {'Med':>8} {'Verdict':<25}")
    print("-" * 140)

    all_results = []

    for hold in [1, 2, 3, 6]:
        for cost in [9, 6, 4]:
            for ev_list, ev_name in [
                (events_oi, "OI_Shock_Absorption"),
                (events_fund, "Funding_Extreme_Failure"),
                (events_del, "Deleveraging_Reversal"),
            ]:
                trades = paper_trade_events(data, ev_list, hold_bars=hold, cost_bps=float(cost))
                if not trades:
                    continue
                df = pd.DataFrame(trades)

                gross = df["gross_bps"].sum()
                cost_total = df["cost_bps"].sum()
                net = df["net_bps"].sum()
                pos = df.loc[df["net_bps"] > 0, "net_bps"].sum()
                neg = abs(df.loc[df["net_bps"] < 0, "net_bps"].sum())
                pf = pos / neg if neg > 0 else float("inf")
                cg = cost_total / abs(gross) * 100 if abs(gross) > 0 else float("inf")
                hit_rate = df["direction_hit"].mean()
                med = df["net_bps"].median()
                n = len(df)
                verdict = classify(gross, net, pf, n, cg, med)

                print(f"{ev_name:<28} {hold:>5} {cost:>4}bps {n:>7} {gross:>10.1f} {net:>10.1f} {cg:>6.0f}% {pf:>6.2f} {hit_rate:>5.1%} {med:>8.1f} {verdict:<25}")

                all_results.append({
                    "event_type": ev_name,
                    "hold_bars": hold,
                    "cost_bps": cost,
                    "trade_count": n,
                    "gross_pnl": round(gross, 1),
                    "net_pnl": round(net, 1),
                    "cost_to_gross": round(cg, 1),
                    "pf": round(pf, 3),
                    "hit_rate": round(hit_rate, 3),
                    "median_bps": round(med, 1),
                    "verdict": verdict,
                })

    # Save
    out = ROOT / "logs" / "small_cap_event_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {out}")

    # Final summary
    print("\n=== Final Summary ===")
    for ev_name in ["OI_Shock_Absorption", "Funding_Extreme_Failure", "Deleveraging_Reversal"]:
        subset = [r for r in all_results if r["event_type"] == ev_name]
        best = max(subset, key=lambda r: r["net_pnl"])
        has_pass = any(r["verdict"] == "PAPER_PASS" for r in subset)
        has_repair = any(r["verdict"] == "TRADE_RULE_REPAIR" for r in subset)
        print(f"{ev_name}: best Net={best['net_pnl']:.1f} (h={best['hold_bars']} c={best['cost_bps']}bps)")
        print(f"  PAPER_PASS={has_pass} TRADE_RULE_REPAIR={has_repair}")


if __name__ == "__main__":
    main()
