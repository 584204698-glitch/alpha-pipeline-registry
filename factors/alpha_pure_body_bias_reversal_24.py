"""
PureBodyBiasReversal24 — Pure Body Bias Reversal (Standard Zscore)

Logic (纯实体偏向反转):
The simplest OHLC factor: bar body position zscored relative to its own
24-bar history, with -1 reversal sign. When bar closes near extreme of
range relative to recent history → mean-revert.

Single component, no multiplications, pure OHLC structure.
Proven in historical runs with OOS ICIR~0.24.

Formula:
-1 * zscore((2*close-high-low)/(high-low+1e-8), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class PureBodyBiasReversal24(FactorRegistry):
    factor_name = "PureBodyBiasReversal24"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "Simplest OHLC factor. Bar body position zscored: when close is at extreme "
        "of the bar range relative to 24-bar history → reversal. Pure construction "
        "with no multiplications or gates. BodyBiasReversal family OOS ICIR~0.24."
    )
    mathematical_formula = (
        "-1 * zscore((2*close-high-low)/(high-low+1e-8), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index
        w = self.parameters["window"]

        hl = (data["high"] - data["low"]).clip(lower=eps)
        body_bias = (2 * data["close"] - data["high"] - data["low"]) / hl

        def _zs(ser, win):
            m = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, pd.NA)
            return ((ser - m) / s).fillna(0.0)

        z_body = _zs(body_bias, w)

        signal = pd.Series(-1.0 * z_body.values, index=idx)
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
