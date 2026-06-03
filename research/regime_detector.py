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
    breadth_window: int = 6,
) -> pd.Series:
    """Assign regime labels to each timestamp.

    Args:
        data: MultiIndex (timestamp, symbol) DataFrame with columns:
              close, volume, open_interest, funding_rate
        btc_symbol: BTC perpetual identifier
        ema_span: span for BTC EMA
        vol_window: window for realized volatility
        breadth_window: window for breadth indicators

    Returns:
        Series indexed by timestamp with regime labels
    """
    ts_index = data.index.get_level_values("timestamp").unique().sort_values()

    # Extract BTC data
    btc_mask = data.index.get_level_values("symbol") == btc_symbol
    btc = data.loc[btc_mask].copy()
    if btc.empty:
        # Fallback: try to find any BTC-like symbol
        for s in data.index.get_level_values("symbol").unique():
            if "BTC" in str(s):
                btc = data.loc[data.index.get_level_values("symbol") == s].copy()
                break
    if btc.empty:
        return pd.Series("unknown", index=ts_index)

    # Sort BTC by timestamp
    btc = btc.sort_index()
    btc["ret_4h"] = btc["close"].pct_change(4)  # ~4-period return
    btc["ret_1h"] = btc["close"].pct_change(1)
    btc["ema"] = btc["close"].ewm(span=ema_span, adjust=False).mean()
    btc["ema_slope"] = btc["ema"].pct_change(4)  # EMA direction
    btc["realized_vol"] = btc["ret_1h"].rolling(vol_window).std()

    # Market breadth: fraction of symbols with positive return
    ret_all = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(1))
    breadth = ret_all.groupby(level="timestamp").apply(
        lambda x: (x > 0).mean() if len(x) > 10 else 0.5
    )

    # Median funding rate
    med_funding = data["funding_rate"].groupby(level="timestamp").median()

    # Median OI change
    oi_delta = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(4))
    med_oi_delta = oi_delta.groupby(level="timestamp").median()

    # Build regime frame
    regime_frame = pd.DataFrame(index=ts_index)
    regime_frame["btc_ret_4h"] = btc["ret_4h"].reindex(ts_index)
    regime_frame["btc_ema_slope"] = btc["ema_slope"].reindex(ts_index)
    regime_frame["btc_realized_vol"] = btc["realized_vol"].reindex(ts_index)
    regime_frame["breadth"] = breadth.reindex(ts_index)
    regime_frame["med_funding"] = med_funding.reindex(ts_index)
    regime_frame["med_oi_delta"] = med_oi_delta.reindex(ts_index)
    regime_frame = regime_frame.ffill().fillna(0.0)

    # Thresholds
    vol_median = regime_frame["btc_realized_vol"].median()
    vol_high = vol_median * 2.0  # 2x median = panic
    btc_ret_std = regime_frame["btc_ret_4h"].std()

    regimes: list[str] = []
    for _, row in regime_frame.iterrows():
        ret = row["btc_ret_4h"]
        slope = row["btc_ema_slope"]
        vol = row["btc_realized_vol"]
        breadth_val = row["breadth"]
        oi_d = row["med_oi_delta"]

        # Panic: sharp BTC drop + high vol + OI collapse
        if ret < -2 * btc_ret_std and vol > vol_high and oi_d < 0:
            regimes.append("panic_down")
            continue

        # Trend up: positive slope + breadth confirms
        if slope > 0.001 and breadth_val > 0.48:
            regimes.append("trend_up")
            continue

        # Trend down: negative slope + breadth confirms
        if slope < -0.001 and breadth_val < 0.48:
            regimes.append("trend_down")
            continue

        # Chop: high vol but no clear direction, breadth around 0.5
        if vol > vol_median * 1.5 and abs(ret) < btc_ret_std * 0.5:
            regimes.append("chop")
            continue

        # Default: range
        regimes.append("range")

    return pd.Series(regimes, index=ts_index, name="regime")


def regime_summary(regimes: pd.Series) -> dict:
    """Summarize regime distribution."""
    counts = regimes.value_counts().to_dict()
    total = len(regimes)
    return {
        "n_total": total,
        "distribution": {k: {"count": int(v), "pct": round(v / total * 100, 1)} for k, v in counts.items()},
    }


def get_regime_mask(regimes: pd.Series, allowed: list[str]) -> pd.Index:
    """Return timestamp index where regime is in allowed list."""
    return regimes[regimes.isin(allowed)].index
