"""
BodyEfficiencyZscore — Body Bias × Bar Efficiency, Single Zscore

Logic (实体效率综合Z分数):
BodyEffReversal proven: OOS ICIR=0.24, all 6 except regime. 
-1 * zscore(body_bias * bar_efficiency, 24)

body_bias * bar_efficiency: when bar is efficient AND near extreme,
the product amplifies. When bar is doji (low efficiency) OR near mid,
the product is near 0. Zscore normalizes relative to history.

Single zscore avoids multiplication attenuation issues.

Formula:
-1 * zscore(((2*close-high-low)/(high-low+1e-8)) 
             * (abs(close-open)/(high-low+1e-8)), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class BodyEfficiencyZscore(FactorRegistry):
    factor_name = "BodyEfficiencyZscore"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "body_bias × bar_efficiency inside single zscore. Combines conviction "
        "(efficiency) with direction (body_bias) before normalization. BodyEffReversal "
        "proven OOS ICIR~0.24 — this is the same construction. "
        "Pure OHLC, no volume/OI dependency."
    )
    mathematical_formula = (
        "-1 * zscore(((2*close-high-low)/(high-low+1e-8)) "
        "* (abs(close-open)/(high-low+1e-8)), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index
        w = self.parameters["window"]

        hl = (data["high"] - data["low"]).clip(lower=eps)
        body_bias = (2 * data["close"] - data["high"] - data["low"]) / hl
        bar_efficiency = (data["close"] - data["open"]).abs() / hl
        combined = body_bias * bar_efficiency

        def _zs(ser, win):
            m = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, pd.NA)
            return ((ser - m) / s).fillna(0.0)

        z_combined = _zs(combined, w)

        signal = pd.Series(-1.0 * z_combined.values, index=idx)
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
