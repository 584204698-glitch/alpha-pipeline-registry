"""
HybridBodyEffReversal — Average of Body Zscore + Body×Eff Zscore

Logic (混合实体效率反转):
PureBodyBiasReversal24 has OOS ICIR=0.240 (t-stat=1.54) but all regimes negative.
EfficiencyTimingFadeV4 has all regimes positive but OOS ICIR=0.167 (t-stat=1.07).
Average the two signals: preserves ~75% of OOS power from pure body while
adding ~25% regime stabilization from the efficiency-weighted component.

Formula:
-1 * (zscore(body_bias, 24) + zscore(body_bias * bar_efficiency, 24)) / 2
where body_bias = (2c-h-l)/(h-l), bar_efficiency = abs(c-o)/(h-l)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class HybridBodyEffReversal(FactorRegistry):
    factor_name = "HybridBodyEffReversal"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "50/50 blend of pure body_bias reversal (OOS t-stat=1.54, regime-fail) "
        "and body×eff reversal (regime-pass, OOS-weak). The body term provides "
        "OOS power; the body×eff term stabilizes the regime profile via "
        "efficiency-weighted extremes."
    )
    mathematical_formula = (
        "-1 * (zscore((2*close-high-low)/(high-low+1e-8), 24) "
        "+ zscore(((2*close-high-low)/(high-low+1e-8)) "
        "* (abs(close-open)/(high-low+1e-8)), 24)) / 2"
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
        body_times_eff = body_bias * bar_efficiency

        def _zs(ser, win):
            m = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, pd.NA)
            return ((ser - m) / s).fillna(0.0)

        z_body = _zs(body_bias, w)
        z_body_eff = _zs(body_times_eff, w)

        combined = (z_body.values + z_body_eff.values) / 2.0
        signal = pd.Series(-1.0 * combined, index=idx)
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
