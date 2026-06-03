"""
EfficiencyTimingFadeV10 — Eff=48 × Body=48 (Maximum Stability)

Logic (柱体效率择时反转V10):
V8 (48/36): OOS ICIR=0.185. V9 (48/24): untested.
V10 pushes both windows to 48 for maximum stability. The hypothesis:
if wider windows consistently improve regime+ behavior (V4→V8), then
48/48 should give the most stable regime profile. Even if OOS ICIR 
remains around 0.18-0.19, this provides a reliable regime-invariant factor
for ensemble use.

Formula:
-1 * zscore(abs(close-open)/(high-low+1e-8), 48)
    * zscore((2*close-high-low)/(high-low+1e-8), 48)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class EfficiencyTimingFadeV10(FactorRegistry):
    factor_name = "EfficiencyTimingFadeV10"
    parameters = {"eff_window": 48, "body_window": 48}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "V10: Maximum stability variant. Both windows at 48. "
        "V8 (48/36): OOS ICIR=0.185, all regimes+. "
        "V10 should provide the cleanest regime+ profile at acceptable OOS."
    )
    mathematical_formula = (
        "-1 * zscore(abs(close-open)/(high-low+1e-8), 48) "
        "* zscore((2*close-high-low)/(high-low+1e-8), 48)"
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
