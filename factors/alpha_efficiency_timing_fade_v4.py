"""
EfficiencyTimingFadeV4 — Pure Zscore Version

Logic (柱体效率择时反转V4):
Revert to standard zscore (mean/std) after V3 robust_zscore bug. 
Double zscore construction:
- zscore(bar_efficiency, 24): timing — when bar is exceptionally efficient
- zscore(body_bias, 36): direction — which side the bar favors

Longer body window (36 vs 24) for stability since body_bias is noisier
than bar efficiency.

Formula:
-1 * zscore(abs(close-open)/(high-low+1e-8), 24)
    * zscore((2*close-high-low)/(high-low+1e-8), 36)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class EfficiencyTimingFadeV4(FactorRegistry):
    factor_name = "EfficiencyTimingFadeV4"
    parameters = {"eff_window": 24, "body_window": 36}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "V4: Standard zscore (not robust), double zscore construction. "
        "Bar efficiency zscore=24 for timing, body bias zscore=36 for direction. "
        "V2 had OOS ICIR=0.203 with all regimes positive — V4 uses zscore for "
        "broader signal range."
    )
    mathematical_formula = (
        "-1 * zscore(abs(close-open)/(high-low+1e-8), 24) "
        "* zscore((2*close-high-low)/(high-low+1e-8), 36)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index
        ew, bw = self.parameters["eff_window"], self.parameters["body_window"]

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

        z_eff = _zs(bar_efficiency, ew)
        z_body = _zs(body_bias, bw)

        signal = pd.Series(-1.0 * z_eff.values * z_body.values, index=idx)
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
