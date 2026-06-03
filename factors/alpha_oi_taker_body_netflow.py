"""
OITakerBodyNetflow — 3-Component OHLC Absorption Signal

Logic (OI主动量柱体净流):
The prompt's combined absorption formula with rank_pct-0.5 replacing indicator
gates for guaranteed signal symmetry. When taker aggression, OI buildup direction,
and bar body direction all align → strong directional signal.

Components (all naturally symmetric):
- zscore(taker_vol/vol, 24)      — aggressive flow, centered at 0
- rank_pct(delta(OI,6), 24)-0.5  — OI buildup direction, [-0.5, +0.5]
- (2c-h-l)/(h-l)                 — body bias, [-1, +1]

Why this avoids known failures:
- No indicator(x>0) gates → no long_ratio collapse
- rank_pct-0.5 for OI → symmetric, unlike OIDeltaBodyFlow which started
  from already-biased zscore(delta(OI))
- Body bias is naturally [-1,+1] symmetric

When all 3 agree: long_absorption = taker selling + OI drop + bearish body → bullish.
When all 3 agree: short_absorption = taker buying + OI rise + bullish body → bearish
(we FOLLOW the flow — positive body_bias + positive OI rank + positive taker zscore → LONG)

Formula:
zscore(taker_volume/(volume+1e-8), 24)
  * (rank_pct(delta(open_interest, 6), 24) - 0.5)
  * (2*close-high-low)/(high-low+1e-8)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from factors.base import FactorRegistry


class OITakerBodyNetflow(FactorRegistry):
    factor_name = "OITakerBodyNetflow"
    parameters = {"window": 24, "oi_delta": 6}
    inputs = ["close", "high", "low", "volume", "taker_volume", "open_interest"]
    timeframes = ["1h"]
    rationale = (
        "3-component OHLC absorption: taker aggression zscore × OI buildup rank "
        "× body bias. All components centered/symmetric — no indicator gates. "
        "When taker flow, OI change, and bar body all agree → strong flow-following "
        "signal. Captures absorption dynamics: OI building + aggressive taker "
        "+ body direction = conviction about which side is winning."
    )
    mathematical_formula = (
        "zscore(taker_volume/(volume+1e-8), 24) "
        "* (rank_pct(delta(open_interest, 6), 24) - 0.5) "
        "* (2*close-high-low)/(high-low+1e-8)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index
        w = self.parameters["window"]

        # Taker flow ratio zscore
        taker_ratio = data["taker_volume"] / (data["volume"].clip(lower=eps))
        taker_ratio[taker_ratio > 2.0] = 2.0  # clip extreme outliers

        def _zs(ser, win):
            m = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, pd.NA)
            return ((ser - m) / s).fillna(0.0)

        z_taker = _zs(taker_ratio, w)

        # OI delta rank_pct centered at 0.5
        di = self.parameters["oi_delta"]
        oi_delta = data["open_interest"].groupby(level="symbol", group_keys=False).diff(di)

        def _rank_centered(ser, win):
            """Rolling rank percentile, centered at 0.5: rank_pct - 0.5"""
            ranked = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
            )
            return ranked - 0.5

        oi_rank_c = _rank_centered(oi_delta, w)

        # Body bias: (2c-h-l)/(h-l) ∈ [-1, +1]
        hl = (data["high"] - data["low"]).clip(lower=eps)
        body_bias = (2 * data["close"] - data["high"] - data["low"]) / hl

        # 3-way product
        signal = pd.Series(
            z_taker.values * oi_rank_c.values * body_bias.values, index=idx
        )
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
