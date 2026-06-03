"""
EfficiencyTimingFadeV6 — Body×Eff Product, Single Zscore, Window=36

Logic (柱体效率择时反转V6):
BodyEfficiencyZscore (w=24) had OOS ICIR=0.239, t-stat=1.53 but all regimes
negative. V4 (double zscore, body_w=36) had all regimes positive but t-stat=1.07.
V6 combines: single zscore on body_bias×efficiency product (V4's regime structure
was from the product interaction, not double zscore) with window=36 for stability.

Formula:
-1 * zscore(((2*close-high-low)/(high-low+1e-8)) 
             * (abs(close-open)/(high-low+1e-8)), 36)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class EfficiencyTimingFadeV6(FactorRegistry):
    factor_name = "EfficiencyTimingFadeV6"
    parameters = {"window": 36}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "V6: Single zscore on body×eff product, window=36. Bridges V4 (regime+) "
        "and BodyEfficiencyZscore (OOS+). The product weights body bias by bar "
        "efficiency before normalization, capturing conviction-weighted extremes. "
        "Window 36 for stability."
    )
    mathematical_formula = (
        "-1 * zscore(((2*close-high-low)/(high-low+1e-8)) "
        "* (abs(close-open)/(high-low+1e-8)), 36)"
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
