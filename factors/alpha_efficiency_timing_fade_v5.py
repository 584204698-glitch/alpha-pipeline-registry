"""
EfficiencyTimingFadeV5 — Bar Efficiency Zscore × Body Bias Zscore (Window=24)

Logic (柱体效率择时反转V5):
V4 used body_window=36 and got all 3 regimes positive but OOS t-stat=1.07.
V5 reduces body_window to 24 to increase signal responsiveness while
keeping the double-zscore structure that delivered regime invariance.

Formula:
-1 * zscore(abs(close-open)/(high-low+1e-8), 24)
    * zscore((2*close-high-low)/(high-low+1e-8), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class EfficiencyTimingFadeV5(FactorRegistry):
    factor_name = "EfficiencyTimingFadeV5"
    parameters = {"eff_window": 24, "body_window": 24}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "V5: V4 structure with body_window reduced from 36→24 for sharper "
        "signals. V4 had all 3 regimes positive (0.072/0.026/0.043) but "
        "t-stat=1.07. Shorter body window preserves regime structure while "
        "boosting OOS responsiveness."
    )
    mathematical_formula = (
        "-1 * zscore(abs(close-open)/(high-low+1e-8), 24) "
        "* zscore((2*close-high-low)/(high-low+1e-8), 24)"
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
