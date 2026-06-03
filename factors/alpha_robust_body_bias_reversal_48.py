"""
RobustBodyBiasReversal48 — Pure Body Bias Reversal, Robust, Window=48

Logic (稳健实体偏向反转48):
The empirical finding: zscore(body_bias, 24) with -1 reversal gives OOS ICIR~0.24
but fails regime invariance (high/med vol negative IC). 

V2: robust_zscore (median/MAD) with window=48 for maximum stability.
Longer window smooths out noise and may improve high-vol regime performance
by capturing more extreme relative to a longer baseline.

Single component, no multiplication — pure reversal of bar extremes.

Formula:
-1 * robust_zscore((2*close-high-low)/(high-low+1e-8), 48)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from factors.base import FactorRegistry


class RobustBodyBiasReversal48(FactorRegistry):
    factor_name = "RobustBodyBiasReversal48"
    parameters = {"window": 48}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "Pure body bias reversal with robust_zscore (median/MAD) and longer "
        "window=48. BodyBiasReversal family has proven OOS ICIR~0.24 but fails "
        "regime invariance. Longer window with robust normalization may improve "
        "high-vol regime performance by capturing more extreme extremes."
    )
    mathematical_formula = (
        "-1 * robust_zscore((2*close-high-low)/(high-low+1e-8), 48)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index
        w = self.parameters["window"]

        hl = (data["high"] - data["low"]).clip(lower=eps)
        body_bias = (2 * data["close"] - data["high"] - data["low"]) / hl

        def _rzs(ser, win):
            roll_med = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).median()
            )
            diff_abs = (ser - roll_med).abs()
            roll_mad = diff_abs.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).median()
            )
            return ((ser - roll_med) / (roll_mad + eps)).fillna(0.0)

        rz_body = _rzs(body_bias, w)

        signal = pd.Series(-1.0 * rz_body.values, index=idx)
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
