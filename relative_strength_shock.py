"""Relative Strength Shock — Event Detector.

Detects: small-cap coins showing sudden, significant outperformance vs BTC
with volume + OI confirmation. The signal is that real money is entering
the coin (not just market-beta noise).

Financial logic:
- Small-cap coin moves up sharply while BTC is flat/weak → accumulation
- Elevated volume = real interest, not thin-market noise
- OI increasing = new positions, not short covering
- Strong close = momentum sustained through bar

Design:
- Event-triggered (not continuous ranking)
- Long-only (short RS events are different — mean-reversion, not momentum)
- Regime-aware: best in range/trend_up, avoid in panic_down
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd


# ── Helpers ──────────────────────────────────────────

def _roll_z(series, window=12):  # was 48 for 15m bars
    """Per-symbol rolling z-score."""
    g = series.groupby(level="symbol")
    mean = g.transform(lambda s: s.rolling(window, min_periods=8).mean())
    std = g.transform(lambda s: s.rolling(window, min_periods=8).std()).replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


# ── Event Detector ──────────────────────────────────

def detect_relative_strength_shock(
    data: pd.DataFrame,
    small_sym_set: set[str],
    btc_regime_map: dict | None = None,
    btc_symbol: str = "Binance:BTCUSDT",
    rs_threshold: float = 0.05,
    vol_z_min: float = 2.0,
    oi_delta_z_min: float = 0.5,
    close_loc_min: float = 0.50,
    btc_ret_floor: float = -0.01,
    funding_z_max: float = 2.5,
    event_score_min: float = 0.60,
    cooldown_bars: int = 8,  # ~2h cooldown per symbol
) -> list[dict[str, Any]]:
    """Detect Relative Strength Shock events.

    A small-cap coin shows significant outperformance vs BTC with volume + OI
    confirmation. Direction is REGIME-DEPENDENT:
    - range: LONG (real accumulation, not noise)
    - trend_down: SHORT (dead cat bounce, fade the pump)
    - trend_up: SKIP (no edge)
    - panic_down/chop: SKIP

    Args:
        data: MultiIndex (timestamp, symbol) DataFrame
        small_sym_set: set of symbols in small-cap universe
        btc_regime_map: dict mapping timestamp → regime label
        btc_symbol: BTC symbol identifier

    Returns:
        List of event dicts with symbol, timestamp, metrics, score
    """
    # Precompute (1h bars: pct_change(1)=1h return, was pct_change(4) for 15m→1h)
    ret_1h = data["close"].groupby(level="symbol").transform(
        lambda s: s.pct_change(1)
    )

    # BTC return
    btc_data = data.loc[data.index.get_level_values("symbol") == btc_symbol, ["close"]]
    if btc_data.empty:
        print("WARNING: BTC data not found")
        return []
    btc_close = btc_data.droplevel("symbol").sort_index()
    btc_close = btc_close[~btc_close.index.duplicated(keep="last")]["close"]
    btc_ret_1h = btc_close.pct_change(1)  # 1h return on 1h bars

    # Metrics (12-bar windows for 1h bars, was 48-bar for 15m)
    vol_z = _roll_z(data["volume"], 12)
    oi_delta = data["open_interest"].groupby(level="symbol").transform(
        lambda s: s.diff(2)  # 2h OI change (was diff(6))
    )
    oi_delta_z = _roll_z(oi_delta, 12)
    funding_z = _roll_z(data["funding_rate"], 6)  # was 24

    hl_range = (data["high"] - data["low"]).clip(lower=1e-8)
    close_loc = (data["close"] - data["low"]) / hl_range

    # Scan
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    events = []
    cooldowns: dict[str, pd.Timestamp] = {}
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}

    for ts_idx, ts in enumerate(ts_list):
        # BTC regime
        btc_reg = btc_regime_map.get(ts, "unknown") if btc_regime_map else "unknown"

        # Skip panic_down (nothing is a good buy) and chop
        if btc_reg in ("panic_down", "chop"):
            continue

        # Skip trend_up — no edge for RS shock
        if btc_reg == "trend_up":
            continue

        # Determine direction based on regime
        if btc_reg == "range":
            direction = "long"   # real accumulation
        elif btc_reg == "trend_down":
            direction = "short"  # dead cat bounce
        else:
            direction = "long"   # default (shouldn't reach here)

        # BTC return at this timestamp
        try:
            btc_r1h = float(btc_ret_1h.get(ts, 0.0))
        except (TypeError, KeyError):
            btc_r1h = 0.0

        # Skip if BTC is crashing (then everything is relative strength, meaningless)
        if btc_r1h < btc_ret_floor:
            continue

        # Get symbols present at this timestamp
        mask = data.index.get_level_values("timestamp") == ts
        syms_at_ts = set(data.index.get_level_values("symbol")[mask])

        for sym in small_sym_set & syms_at_ts:
            try:
                r1h = float(ret_1h.loc[(ts, sym)])
                vz = float(vol_z.loc[(ts, sym)])
                oz = float(oi_delta_z.loc[(ts, sym)])
                cl = float(close_loc.loc[(ts, sym)])
                fz = float(funding_z.loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            # ── Cooldown check ──
            if sym in cooldowns:
                last_ts = cooldowns[sym]
                if ts_idx - ts_to_idx.get(last_ts, 0) < cooldown_bars:
                    continue
                else:
                    del cooldowns[sym]

            # Compute relative strength vs BTC
            rs = r1h - btc_r1h

            # ── Hard gates ──
            if rs < rs_threshold:
                continue
            if vz < vol_z_min:
                continue
            if oz < oi_delta_z_min:
                continue
            if cl < close_loc_min:
                continue
            if abs(fz) > funding_z_max:
                continue

            # ── Score ──
            rs_score = min(rs / 0.08, 1.0)  # cap at 8% RS
            vol_score = min(vz / 4.0, 1.0)  # cap at 4σ
            oi_score = min(oz / 2.0, 1.0)   # cap at 2σ
            cl_score = min(cl / 0.85, 1.0)  # cap at 85% close_location

            event_score = (
                0.35 * rs_score
                + 0.25 * vol_score
                + 0.20 * oi_score
                + 0.20 * cl_score
            )

            # Event score threshold
            if event_score < event_score_min:
                continue

            # Regime adjustment: trend_up → higher bar, trend_down → only best
            if btc_reg == "trend_up":
                event_score *= 0.9  # BTC is rising anyway, RS less meaningful
            elif btc_reg == "trend_down":
                if rs < 0.05:  # need even stronger RS in downtrend
                    continue
                event_score *= 1.1  # RS in downtrend is more impressive

            # Set cooldown
            cooldowns[sym] = ts

            events.append({
                "symbol": sym,
                "timestamp": ts,
                "event_type": "RelativeStrengthShock",
                "direction": direction,
                "event_score": round(event_score, 4),
                "rs_pct": round(rs * 100, 2),
                "ret_1h_pct": round(r1h * 100, 2),
                "btc_ret_1h_pct": round(btc_r1h * 100, 2),
                "vol_z": round(vz, 2),
                "oi_delta_z": round(oz, 2),
                "close_loc": round(cl, 3),
                "funding_z": round(fz, 2),
                "regime": btc_reg,
            })

    return events


# ── Paper Trading for RS Shock ──────────────────────

def paper_trade_rs_shock(
    data: pd.DataFrame,
    events: list[dict],
    hold_bars: int = 2,
    entry_mode: str = "current_close",
    cost_bps_per_side: float = 9.0,
) -> dict[str, Any]:
    """Run paper trading on RS Shock events.

    Args:
        data: MultiIndex DataFrame
        events: list of event dicts from detect_relative_strength_shock
        hold_bars: bars to hold position
        entry_mode: "current_close" or "next_bar_open"
        cost_bps_per_side: cost in bps per side

    Returns:
        Dict with PnL metrics
    """
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

        # Entry
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

        # Exit
        exit_idx = ts_idx + hold_bars
        if exit_idx >= len(ts_list):
            continue
        exit_ts = ts_list[exit_idx]
        try:
            exit_px = float(data.loc[(exit_ts, sym), "close"])
        except KeyError:
            continue

        ret = (exit_px / entry_px) - 1.0
        # Apply direction: short means we profit when price drops
        if ev.get("direction") == "short":
            ret = -ret
        gross_bps = ret * 10000
        net_bps = gross_bps - cost_bps_per_side

        trades.append({
            "symbol": sym,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "event_score": ev["event_score"],
            "rs_pct": ev["rs_pct"],
            "regime": ev["regime"],
            "gross_bps": gross_bps,
            "net_bps": net_bps,
        })

    if not trades:
        return {"error": "No completed trades"}

    df = pd.DataFrame(trades)

    # Metrics
    net_arr = df["net_bps"].values
    gross_sum = df["gross_bps"].sum()
    net_sum = net_arr.sum()
    pos = net_arr[net_arr > 0].sum()
    neg = abs(net_arr[net_arr < 0].sum())
    pf = pos / neg if neg > 0 else float("inf")
    cg = (len(net_arr) * cost_bps_per_side) / abs(gross_sum) * 100 if abs(gross_sum) > 0 else float("inf")
    hit = (net_arr > 0).mean()

    results = {
        "n_events": len(events),
        "n_trades": len(trades),
        "gross_bps": round(gross_sum, 0),
        "net_bps": round(net_sum, 0),
        "pf": round(pf, 3),
        "cost_to_gross_pct": round(cg, 0),
        "hit_rate": round(hit, 3),
        "median_bps": round(np.median(net_arr), 1),
        "avg_win": round(net_arr[net_arr > 0].mean(), 1) if (net_arr > 0).any() else 0,
        "avg_loss": round(net_arr[net_arr < 0].mean(), 1) if (net_arr < 0).any() else 0,
        "entry_mode": entry_mode,
        "cost_bps": cost_bps_per_side,
    }

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

    print("Regime distribution:")
    for reg, cnt in regimes.value_counts().items():
        print(f"  {reg}: {cnt} ({cnt/len(regimes)*100:.1f}%)")

    # Build small-cap universe
    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    small_syms = set(vol_rank[
        (vol_rank >= 20) & (vol_rank <= 100)
    ].index)
    print(f"\nSmall-cap pool: {len(small_syms)} symbols")

    # Detect events
    events = detect_relative_strength_shock(data, small_syms, regime_map)
    print(f"\nRS Shock events detected: {len(events)}")

    if events:
        # Show top events
        top = sorted(events, key=lambda e: -e["event_score"])[:5]
        print("\nTop 5 events:")
        for ev in top:
            print(f"  {ev['timestamp']} {ev['symbol']:20s} "
                  f"RS={ev['rs_pct']:+.1f}% score={ev['event_score']:.3f} "
                  f"regime={ev['regime']}")

        # Paper trade
        print("\n=== Paper Trading ===")
        for mode in ["current_close", "next_bar_open"]:
            r = paper_trade_rs_shock(data, events, hold_bars=2, entry_mode=mode)
            print(f"  {mode}: n={r.get('n_trades', 0)} "
                  f"Net={r.get('net_bps', 0):+.0f} "
                  f"PF={r.get('pf', 0):.2f} "
                  f"Hit={r.get('hit_rate', 0)*100:.1f}% "
                  f"C/G={r.get('cost_to_gross_pct', 0):.0f}%")
            if r.get("regime_breakdown"):
                for reg, info in r["regime_breakdown"].items():
                    print(f"    {reg}: n={info['n']} Net={info['net']:+.0f} Hit={info['hit']*100:.0f}%")
    else:
        print("No events detected — try looser thresholds")
