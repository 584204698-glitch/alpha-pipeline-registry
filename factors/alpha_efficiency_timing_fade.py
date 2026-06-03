"""
EfficiencyTimingFade — Bar Efficiency Timing × Body Direction

Logic (柱体效率择时反转):
Bar efficiency = abs(close-open)/(high-low) measures how much of the bar's
range was a clean directional move vs noise. High efficiency bars represent
conviction-driven price discovery — the kind that exhausts quickly.

When bar efficiency is high (conviction bar) AND body is at extreme →
the move is overdone → fade it.

This is a novel timing mechanism:
- Vol-accel captures multi-bar volatility acceleration (proven: ICIR~0.3)
- Bar efficiency captures single-bar "conviction intensity" (NEW)
- Both are timing mechanisms but structurally orthogonal:
  high bar efficiency can happen in both accelerating and decelerating vol

Why this avoids known failures:
- Pure OHLC internal structure (no volume/taker/OI dependency)
- zscore*rank_pct both centered → symmetric signal
- No multi-bar momentum involved → uncorrelated with VolAccelFade family

Formula:
-1 * zscore(abs(close-open)/(high-low+1e-8), 24)
    * (rank_pct((2*close-high-low)/(high-low+1e-8), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from factors.base import FactorRegistry


class EfficiencyTimingFade(FactorRegistry):
    factor_name = "EfficiencyTimingFade"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "Bar efficiency (abs(c-o)/(h-l)) captures single-bar conviction intensity. "
        "High efficiency = clean directional price discovery that exhausts quickly. "
        "When bar is both efficient AND body at extreme (rank_pct body_bias-0.5) → "
        "fade the move. Novel timing mechanism orthogonal to vol-accel: efficiency "
        "measures bar QUALITY, not multi-bar acceleration."
    )
    mathematical_formula = (
        "-1 * zscore(abs(close-open)/(high-low+1e-8), 24) "
        "* (rank_pct((2*close-high-low)/(high-low+1e-8), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index
        w = self.parameters["window"]

        hl = (data["high"] - data["low"]).clip(lower=eps)
        bar_efficiency = (data["close"] - data["open"]).abs() / hl
        body_bias = (2 * data["close"] - data["high"] - data["low"]) / hl

        def _zs(ser, win):
            m = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, pd.NA)
            return ((ser - m) / s).fillna(0.0)

        z_efficiency = _zs(bar_efficiency, w)

        def _rank_centered(ser, win):
            ranked = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
            )
            return ranked - 0.5

        body_rank_c = _rank_centered(body_bias, w)

        signal = pd.Series(
            -1.0 * z_efficiency.values * body_rank_c.values, index=idx
        )
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
