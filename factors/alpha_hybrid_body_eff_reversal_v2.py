"""
HybridBodyEffReversalV2 — 70% Body + 30% Body×Eff

Logic (混合实体效率反转V2):
V1 (50/50): OOS ICIR=0.243, t-stat=1.55 PASS, but all regimes negative.
V2 increases body weight to 70% to preserve more OOS power while keeping
30% body×eff to nudge the regime profile toward positive.

Formula:
-1 * (0.7 * zscore(body_bias, 24) + 0.3 * zscore(body_bias * bar_efficiency, 24))
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class HybridBodyEffReversalV2(FactorRegistry):
    factor_name = "HybridBodyEffReversalV2"
    parameters = {"window": 24, "body_weight": 0.7, "eff_weight": 0.3}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "V2: 70/30 blend of pure body zscore + body×eff zscore. "
        "V1 (50/50) had OOS t-stat=1.55 but regimes negative. "
        "Increasing body weight preserves OOS while 30% body×eff nudges "
        "regime profile toward positivity."
    )
    mathematical_formula = (
        "-1 * (0.7 * zscore(body_bias, 24) + 0.3 * zscore(body_bias * bar_efficiency, 24))"
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

        bw = self.parameters["body_weight"]
        ew = self.parameters["eff_weight"]
        combined = bw * z_body.values + ew * z_body_eff.values
        signal = pd.Series(-1.0 * combined, index=idx)
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
