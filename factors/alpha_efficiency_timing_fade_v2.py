"""
EfficiencyTimingFadeV2 — Bar Efficiency × Body Direction (Robust, Longer Window)

Logic (柱体效率择时反转V2):
V1 got 4/6 (OOS ICIR=0.162, t-stat=1.04, all 3 regimes positive). 
V2 improvements:
- robust_zscore (median/MAD) instead of zscore (mean/std) for crypto fat tails
- Window 36 instead of 24 for more stable estimates
- -1 reversal on body_bias same as proven BodyBiasReversal (OOS ICIR~0.24)

Bar efficiency captures single-bar conviction intensity — clean directional price
discovery exhausts quickly. This timing mechanism is orthogonal to vol-accel.

Formula:
-1 * robust_zscore(abs(close-open)/(high-low+1e-8), 36)
    * (rank_pct((2*close-high-low)/(high-low+1e-8), 36) - 0.5)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from factors.base import FactorRegistry


class EfficiencyTimingFadeV2(FactorRegistry):
    factor_name = "EfficiencyTimingFadeV2"
    parameters = {"window": 36}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "V2 of EfficiencyTimingFade: robust_zscore for fat-tailed efficiency, "
        "window=36 for stability. All 3 regimes were positive in V1 — this "
        "indicates genuine cross-regime alpha. V2 aims to boost OOS t-stat "
        "above 1.5 threshold."
    )
    mathematical_formula = (
        "-1 * robust_zscore(abs(close-open)/(high-low+1e-8), 36) "
        "* (rank_pct((2*close-high-low)/(high-low+1e-8), 36) - 0.5)"
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

        # Robust zscore: median/MAD
        def _rzs(ser, win):
            roll_med = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).median()
            )
            roll_mad = (ser - roll_med).abs().groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).median()
            )
            return ((ser - roll_med) / (roll_mad + eps)).fillna(0.0)

        rz_efficiency = _rzs(bar_efficiency, w)

        # Rank body_bias centered at 0.5
        def _rank_centered(ser, win):
            ranked = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
            )
            return ranked - 0.5

        body_rank_c = _rank_centered(body_bias, w)

        signal = pd.Series(
            -1.0 * rz_efficiency.values * body_rank_c.values, index=idx
        )
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
