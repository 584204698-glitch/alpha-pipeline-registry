"""OI Shock Absorption — Fixed Event Detector.

Detects: OI surges aggressively but price fails to continue → trapped traders.
The absorption reveals who's getting trapped, enabling a fade.

Financial logic (FIXED — directions were inverted in original code):
- OI surge + price at LOW (close_loc < 0.4): New SHORTS opening at range low,
  expecting breakdown. Price fails → shorts trapped → they buy back → price
  rises → LONG the short squeeze. ← THIS IS THE ONLY VIABLE DIRECTION.
- OI surge + price at HIGH (close_loc > 0.6): New LONGS opening at range high.
  Shorting trapped longs does NOT work — long flushes at range highs are less
  mechanical than short squeezes at range lows. ASYMMETRY CONFIRMED.

Key finding: short squeezes work (trapped shorts → up), long flushes don't.
This is by design: short covering creates price-insensitive buying pressure
(shorts MUST buy), while long liquidations are absorbed by value buyers.

Design:
- Event-triggered (not continuous)
- Fade-the-trap direction (reverse of the crowded side)
- Regime-aware: best in range, skip in panic_down
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def detect_oi_shock_absorption(
    data: pd.DataFrame,
    small_sym_set: set[str],
    btc_regime_map: dict | None = None,
    oi_delta_z_min: float = 2.5,
    vol_z_min: float = 1.5,
    ret_15m_max: float = 0.01,
    ret_1h_max: float = 0.02,
    close_loc_long: float = 0.60,
    close_loc_short: float = 0.40,
    allow_short: bool = False,  # short side is dead — asymmetry confirmed
    cooldown_bars: int = 8,
) -> list[dict[str, Any]]:
    """Detect OI Shock Absorption events.

    OI surges (new positions) but price stalls → the side that's trapped
    gets squeezed out → fade them.

    Args:
        data: MultiIndex (timestamp, symbol) DataFrame
        small_sym_set: set of small-cap symbols
        btc_regime_map: dict mapping timestamp → regime
        oi_delta_z_min: minimum OI delta z-score (default 1.5)
        vol_z_min: minimum volume z-score (default 1.0)
        ret_15m_max: max abs 15m return for price failure
        ret_1h_max: max abs 1h return for price failure
        close_loc_long: threshold for trapped SHORTS → LONG (price near low)
        close_loc_short: threshold for trapped LONGS → SHORT (price near high)
        cooldown_bars: bars to wait before re-entering same symbol

    Returns:
        List of event dicts
    """
    # Precompute
    ret_15m = data["close"].groupby(level="symbol").transform(
        lambda s: s.pct_change(1)
    )
    ret_1h = data["close"].groupby(level="symbol").transform(
        lambda s: s.pct_change(4)
    )

    # OI z-score
    oi_delta = data["open_interest"].groupby(level="symbol").transform(
        lambda s: s.diff(6)
    )
    def roll_z(series, window=48):
        g = series.groupby(level="symbol")
        mean = g.transform(lambda s: s.rolling(window, min_periods=8).mean())
        std = g.transform(lambda s: s.rolling(window, min_periods=8).std()).replace(0, np.nan)
        return ((series - mean) / std).fillna(0.0)
    oi_z = roll_z(oi_delta, 48)
    vol_z = roll_z(data["volume"], 48)

    hl_range = (data["high"] - data["low"]).clip(lower=1e-8)
    close_loc = (data["close"] - data["low"]) / hl_range

    # Scan
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    events = []
    cooldowns: dict[str, pd.Timestamp] = {}

    for ts_idx, ts in enumerate(ts_list):
        btc_reg = btc_regime_map.get(ts, "unknown") if btc_regime_map else "unknown"

        # Skip panic_down — nothing is a good fade
        if btc_reg in ("panic_down", "chop"):
            continue

        mask = data.index.get_level_values("timestamp") == ts
        syms_at_ts = set(data.index.get_level_values("symbol")[mask])

        for sym in small_sym_set & syms_at_ts:
            try:
                oz = float(oi_z.loc[(ts, sym)])
                vz = float(vol_z.loc[(ts, sym)])
                cl = float(close_loc.loc[(ts, sym)])
                r15 = abs(float(ret_15m.loc[(ts, sym)]))
                r1h = abs(float(ret_1h.loc[(ts, sym)]))
            except (KeyError, TypeError):
                continue

            # ── Hard gates ──
            if oz < oi_delta_z_min or vz < vol_z_min:
                continue
            # Price failure: at least one timeframe shows price not moving
            if r15 >= ret_15m_max and r1h >= ret_1h_max:
                continue

            # ── Cooldown ──
            if sym in cooldowns:
                last_ts = cooldowns[sym]
                if ts_idx - ts_to_idx.get(last_ts, 0) < cooldown_bars:
                    continue
                else:
                    del cooldowns[sym]

            # ── Direction (FIXED) ──
            # close_loc < 0.4 → price at LOW, OI surging = new SHORTS trapped → LONG
            # close_loc > 0.6 → price at HIGH, OI surging = new LONGS trapped → SHORT (only if allow_short)
            if cl < close_loc_short:
                direction = "long"
                strength = min(oz / 3.0, 1.0) * (0.5 - cl) / 0.5
            elif allow_short and cl > close_loc_long:
                direction = "short"
                strength = min(oz / 3.0, 1.0) * (cl - 0.5) / 0.5
            else:
                continue  # mid-range, no clear trap

            cooldowns[sym] = ts

            events.append({
                "symbol": sym,
                "timestamp": ts,
                "event_type": "OI_Shock_Absorption",
                "direction": direction,
                "event_score": round(strength, 4),
                "oi_z": round(oz, 2),
                "vol_z": round(vz, 2),
                "close_loc": round(cl, 3),
                "ret_15m_pct": round(r15 * 100, 2),
                "ret_1h_pct": round(r1h * 100, 2),
                "regime": btc_reg,
            })

    return events


def paper_trade_oi_absorption(
    data: pd.DataFrame,
    events: list[dict],
    hold_bars: int = 2,
    entry_mode: str = "current_close",
    cost_bps_per_side: float = 9.0,
) -> dict[str, Any]:
    """Paper trade OI Shock Absorption events."""
    if not events:
        return {"error": "No events"}

    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    trades = []

    for ev in events:
        ts = ev["timestamp"]
        sym = ev["symbol"]
        ts_idx = ts_to_idx.get(ts)
        if ts_idx is None:
            continue

        if entry_mode == "current_close":
            entry_ts = ts
        elif entry_mode == "next_bar_open":
            if ts_idx + 1 >= len(ts_list):
                continue
            entry_ts = ts_list[ts_idx + 1]
        else:
            entry_ts = ts

        try:
            entry_px = float(data.loc[(entry_ts, sym), "close"])
        except KeyError:
            continue

        exit_idx = ts_idx + hold_bars
        if exit_idx >= len(ts_list):
            continue
        exit_ts = ts_list[exit_idx]
        try:
            exit_px = float(data.loc[(exit_ts, sym), "close"])
        except KeyError:
            continue

        ret = (exit_px / entry_px) - 1.0
        if ev["direction"] == "short":
            ret = -ret
        gross_bps = ret * 10000
        net_bps = gross_bps - cost_bps_per_side

        trades.append({
            "symbol": sym,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "direction": ev["direction"],
            "regime": ev["regime"],
            "event_score": ev["event_score"],
            "gross_bps": gross_bps,
            "net_bps": net_bps,
        })

    if not trades:
        return {"error": "No completed trades"}

    df = pd.DataFrame(trades)
    net_arr = df["net_bps"].values
    gross_sum = df["gross_bps"].sum()
    pos = net_arr[net_arr > 0].sum()
    neg = abs(net_arr[net_arr < 0].sum())
    pf = pos / neg if neg > 0 else float("inf")
    cg = (len(net_arr) * cost_bps_per_side) / abs(gross_sum) * 100 if abs(gross_sum) > 0 else float("inf")

    results = {
        "n_events": len(events),
        "n_trades": len(trades),
        "gross_bps": round(gross_sum, 0),
        "net_bps": round(net_arr.sum(), 0),
        "pf": round(pf, 3),
        "cost_to_gross_pct": round(cg, 0),
        "hit_rate": round((net_arr > 0).mean(), 3),
        "median_bps": round(np.median(net_arr), 1),
        "entry_mode": entry_mode,
        "cost_bps": cost_bps_per_side,
    }

    # Direction breakdown
    for dir_label in ["long", "short"]:
        sub = df[df["direction"] == dir_label]
        if len(sub) > 0:
            sn = sub["net_bps"].sum()
            results[f"{dir_label}_n"] = len(sub)
            results[f"{dir_label}_net"] = round(sn, 0)
            results[f"{dir_label}_hit"] = round((sub["net_bps"] > 0).mean(), 3)

    # Regime breakdown
    regime_bd = {}
    for reg in df["regime"].unique():
        sub = df[df["regime"] == reg]
        regime_bd[reg] = {
            "n": len(sub),
            "net": round(sub["net_bps"].sum(), 0),
            "hit": round((sub["net_bps"] > 0).mean(), 3),
        }
    results["regime_breakdown"] = regime_bd

    return results


# ── Self-Test ───────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parent))

    from backtest_engine import BacktestEngine
    from research.regime_detector import detect_regime_fast

    ROOT = P("/mnt/e/alpha_pipeline")
    engine = BacktestEngine(ROOT)
    data = engine._load_data()
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))

    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    small_syms = set(vol_rank[(vol_rank >= 20) & (vol_rank <= 100)].index)
    print(f"Small-cap pool: {len(small_syms)}")

    print("\n=== OI Shock Absorption (FIXED direction) ===")
    events = detect_oi_shock_absorption(data, small_syms, regime_map)
    print(f"Events detected: {len(events)}")

    if events:
        # Direction distribution
        longs = sum(1 for e in events if e["direction"] == "long")
        shorts = sum(1 for e in events if e["direction"] == "short")
        print(f"  Long: {longs}, Short: {shorts}")

        top = sorted(events, key=lambda e: -e["event_score"])[:5]
        print("\nTop 5 events:")
        for e in top:
            print(f"  {e['timestamp']} {e['symbol']:20s} dir={e['direction']:5s} "
                  f"oi_z={e['oi_z']:+.1f} cl={e['close_loc']:.2f} score={e['event_score']:.3f}")

        print("\nPaper Trading:")
        for mode in ["current_close", "next_bar_open"]:
            r = paper_trade_oi_absorption(data, events, hold_bars=2, entry_mode=mode)
            print(f"  {mode}: n={r.get('n_trades',0):4d} "
                  f"Net={r.get('net_bps',0):+6.0f} PF={r.get('pf',0):.2f} "
                  f"Hit={r.get('hit_rate',0)*100:.1f}% C/G={r.get('cost_to_gross_pct',0):.0f}%")
            if "long_net" in r:
                print(f"    long:  n={r['long_n']} Net={r['long_net']:+6.0f} Hit={r['long_hit']*100:.0f}%")
            if "short_net" in r:
                print(f"    short: n={r['short_n']} Net={r['short_net']:+6.0f} Hit={r['short_hit']*100:.0f}%")
            if r.get("regime_breakdown"):
                for reg, info in r["regime_breakdown"].items():
                    print(f"    {reg:12s}: n={info['n']:2d} Net={info['net']:+6.0f} Hit={info['hit']*100:.0f}%")
