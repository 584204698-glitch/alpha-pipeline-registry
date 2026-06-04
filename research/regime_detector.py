"""Market regime detector using rule-based state machine.

Classifies each timestamp into one of 5 regimes:
- trend_up: BTC trending up, market breadth positive
- trend_down: BTC trending down, breadth negative
- range: BTC sideways, volatility normal
- panic_down: sharp BTC drop, high vol, OI collapsing
- chop: directionless, high turnover, low conviction

Uses BTC as anchor + market breadth indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_regimes(
    data: pd.DataFrame,
    btc_symbol: str = "Binance:BTCUSDT",
    ema_span: int = 24,
    vol_window: int = 48,
    min_periods: int = 20,
) -> pd.Series:
    """Assign regime labels to each timestamp.

    Args:
        data: MultiIndex (timestamp, symbol) DataFrame with columns:
              close, volume, open_interest, funding_rate
        btc_symbol: BTC perpetual identifier
        ema_span: span for BTC EMA
        vol_window: window for realized volatility
        min_periods: minimum periods for rolling metrics

    Returns:
        Series indexed by timestamp with regime labels
    """
    all_ts = data.index.get_level_values("timestamp").unique().sort_values()

    # --- Find BTC symbol ---
    symbols = data.index.get_level_values("symbol").unique()
    btc_sym = btc_symbol if btc_symbol in symbols else None
    if btc_sym is None:
        for s in symbols:
            if "BTC" in str(s):
                btc_sym = s
                break
    if btc_sym is None:
        return pd.Series("unknown", index=all_ts)

    # --- Extract BTC as Series indexed by timestamp ---
    btc = data.loc[data.index.get_level_values("symbol") == btc_sym, ["close"]].copy()
    # Drop symbol level so we have plain timestamp index
    btc = btc.droplevel("symbol").sort_index()
    btc = btc[~btc.index.duplicated(keep="last")]

    close = btc["close"]
    ret_1h = close.pct_change(1)
    ret_4h = close.pct_change(4)
    ema = close.ewm(span=ema_span, adjust=False).mean()
    ema_slope = ema.pct_change(4)  # slope over 4h
    realized_vol = ret_1h.rolling(vol_window, min_periods=min_periods).std()

    # --- Market breadth ---
    ret_all = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(1))
    breadth = ret_all.groupby(level="timestamp").apply(
        lambda x: (x > 0).mean() if len(x) > 10 else 0.5
    )

    # --- Median OI delta (4h) ---
    oi_delta = data["open_interest"].groupby(level="symbol").transform(
        lambda s: s.diff(4)
    )
    med_oi_delta = oi_delta.groupby(level="timestamp").median()

    # --- Build frame ---
    frame = pd.DataFrame(index=all_ts)
    frame["ret_4h"] = ret_4h.reindex(all_ts)
    frame["ema_slope"] = ema_slope.reindex(all_ts)
    frame["realized_vol"] = realized_vol.reindex(all_ts)
    frame["breadth"] = breadth.reindex(all_ts)
    frame["med_oi_delta"] = med_oi_delta.reindex(all_ts)

    # Forward-fill: use closest previous valid value
    frame = frame.ffill()

    # Drop rows where ALL metrics are still NaN (leading edge)
    frame = frame.dropna(how="all")

    # --- Thresholds (computed from valid data only) ---
    ret_std = frame["ret_4h"].dropna().std()
    vol_median = frame["realized_vol"].dropna().median()
    vol_high = vol_median * 2.0
    vol_chop = vol_median * 1.5

    # --- Assign regimes ---
    regimes: list[tuple[pd.Timestamp, str]] = []

    for ts, row in frame.iterrows():
        ret = float(row["ret_4h"])
        slope = float(row["ema_slope"])
        vol = float(row["realized_vol"])
        breadth_val = float(row["breadth"])
        oi_d = float(row["med_oi_delta"])

        # 1. Panic: sharp BTC drop + high vol + OI collapse
        if (
            ret < -2 * ret_std
            and vol > vol_high
            and oi_d < 0
        ):
            regimes.append((ts, "panic_down"))
            continue

        # 2. Chop: high vol but no clear direction
        if vol > vol_chop and abs(ret) < ret_std * 0.5:
            regimes.append((ts, "chop"))
            continue

        # 3. Trend up: positive slope + breadth confirms
        if slope > 0.001 and breadth_val > 0.48:
            regimes.append((ts, "trend_up"))
            continue

        # 4. Trend down: negative slope + breadth confirms
        if slope < -0.001 and breadth_val < 0.48:
            regimes.append((ts, "trend_down"))
            continue

        # 5. Default: range
        regimes.append((ts, "range"))

    return pd.Series(
        dict(regimes), name="regime"
    ).reindex(all_ts).fillna("range")


def detect_regime_fast(
    data: pd.DataFrame,
    btc_symbol: str = "Binance:BTCUSDT",
    ema_span: int = 24,
    vol_window: int = 48,
) -> pd.Series:
    """Vectorized version — faster but same logic."""
    all_ts = data.index.get_level_values("timestamp").unique().sort_values()

    symbols = data.index.get_level_values("symbol").unique()
    btc_sym = btc_symbol if btc_symbol in symbols else None
    if btc_sym is None:
        for s in symbols:
            if "BTC" in str(s):
                btc_sym = s
                break
    if btc_sym is None:
        return pd.Series("unknown", index=all_ts)

    btc = (
        data.loc[data.index.get_level_values("symbol") == btc_sym, ["close"]]
        .droplevel("symbol")
        .sort_index()
    )
    btc = btc[~btc.index.duplicated(keep="last")]
    close = btc["close"]

    ret_1h = close.pct_change(1)
    ret_4h = close.pct_change(4)
    ema = close.ewm(span=ema_span, adjust=False).mean()
    ema_slope = ema.pct_change(4)
    realized_vol = ret_1h.rolling(vol_window, min_periods=20).std()

    # Breadth
    ret_all = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(1))
    breadth = ret_all.groupby(level="timestamp").apply(
        lambda x: (x > 0).mean() if len(x) > 10 else 0.5
    )

    # OI delta
    oi_delta = data["open_interest"].groupby(level="symbol").transform(
        lambda s: s.diff(4)
    )
    med_oi_delta = oi_delta.groupby(level="timestamp").median()

    frame = pd.DataFrame(index=all_ts)
    frame["ret_4h"] = ret_4h.reindex(all_ts)
    frame["ema_slope"] = ema_slope.reindex(all_ts)
    frame["realized_vol"] = realized_vol.reindex(all_ts)
    frame["breadth"] = breadth.reindex(all_ts)
    frame["med_oi_delta"] = med_oi_delta.reindex(all_ts)
    frame = frame.ffill().dropna(how="all")

    ret_std = frame["ret_4h"].std()
    vol_median = frame["realized_vol"].median()

    # Vectorized conditions
    is_panic = (
        (frame["ret_4h"] < -2 * ret_std)
        & (frame["realized_vol"] > vol_median * 2.0)
        & (frame["med_oi_delta"] < 0)
    )
    is_chop = (
        (frame["realized_vol"] > vol_median * 1.5)
        & (abs(frame["ret_4h"]) < ret_std * 0.5)
    )
    is_trend_up = (frame["ema_slope"] > 0.001) & (frame["breadth"] > 0.48)
    is_trend_down = (frame["ema_slope"] < -0.001) & (frame["breadth"] < 0.48)

    result = pd.Series("range", index=frame.index)
    result[is_panic] = "panic_down"
    result[is_chop & ~is_panic] = "chop"
    result[is_trend_up & ~is_panic & ~is_chop] = "trend_up"
    result[is_trend_down & ~is_panic & ~is_chop] = "trend_down"

    return result.reindex(all_ts).fillna("range").rename("regime")


def regime_summary(regimes: pd.Series) -> dict:
    """Summarize regime distribution."""
    counts = regimes.value_counts().to_dict()
    total = len(regimes)
    return {
        "n_total": total,
        "distribution": {
            k: {"count": int(v), "pct": round(v / total * 100, 1)}
            for k, v in counts.items()
        },
    }


def get_regime_mask(regimes: pd.Series, allowed: list[str]) -> pd.Index:
    """Return timestamp index where regime is in allowed list."""
    return regimes[regimes.isin(allowed)].index


# ── Regime-Based Trading Rules ────────────────────────

# Per-regime trading rules for Deleveraging Reversal (long-only event strategy)
REGIME_RULES = {
    "trend_up": {
        "allow_long": True,
        "description": "BTC tailwind — relaxed entry conditions",
        "event_score_threshold": 0.55,  # lower threshold
        "position_pct": 1.0,  # full size
    },
    "range": {
        "allow_long": True,
        "description": "Sideways — default conditions",
        "event_score_threshold": 0.60,
        "position_pct": 1.0,
    },
    "trend_down": {
        "allow_long": True,
        "description": "BTC headwind — tighter entry, smaller size",
        "event_score_threshold": 0.70,  # higher bar
        "position_pct": 0.5,  # half size
    },
    "chop": {
        "allow_long": False,
        "description": "Directionless high-vol — no trading",
        "event_score_threshold": 0.99,  # effectively disabled
        "position_pct": 0.0,
    },
    "panic_down": {
        "allow_long": False,
        "description": "Systemic panic — NO longs",
        "event_score_threshold": 0.99,
        "position_pct": 0.0,
    },
}


def get_regime_rule(regime: str) -> dict:
    """Return trading rule for a given regime."""
    return REGIME_RULES.get(regime, REGIME_RULES["range"])


if __name__ == "__main__":
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parent.parent))

    from backtest_engine import BacktestEngine

    ROOT = str(P(__file__).resolve().parent.parent)
    engine = BacktestEngine(ROOT)
    data = engine._load_data()

    print("detect_regimes (loop):")
    r1 = detect_regimes(data)
    print(regime_summary(r1))

    print("\ndetect_regime_fast (vectorized):")
    r2 = detect_regime_fast(data)
    print(regime_summary(r2))

    # Verify they match
    common = r1.index.intersection(r2.index)
    match = (r1[common] == r2[common]).mean()
    print(f"\nMatch rate: {match:.1%}")
