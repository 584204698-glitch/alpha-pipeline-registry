"""
EfficiencyTimingFadeV8 — V4 Structure with Wider Efficiency Window (Eff=48, Body=36)

Logic (柱体效率择时反转V8):
V4 structure (eff_zscore × body_zscore) with efficiency window=48 for more
selective timing. Longer efficiency window captures more extreme conviction bars,
reducing noise while maintaining regime-positive structure.

Formula:
-1 * zscore(abs(close-open)/(high-low+1e-8), 48)
    * zscore((2*close-high-low)/(high-low+1e-8), 36)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class EfficiencyTimingFadeV8(FactorRegistry):
    factor_name = "EfficiencyTimingFadeV8"
    parameters = {"eff_window": 48, "body_window": 36}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "V8: V4 structure with eff_window=48 (more selective timing). "
        "Longer efficiency zscore captures rarer, more significant conviction bars. "
        "V4 had all regimes positive but OOS t-stat=1.07; slower timing may reduce "
        "noise and push t-stat above 1.5 threshold."
    )
    mathematical_formula = (
        "-1 * zscore(abs(close-open)/(high-low+1e-8), 48) "
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
