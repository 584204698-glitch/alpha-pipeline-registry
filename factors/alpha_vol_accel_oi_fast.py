"""
VolAccelOIFadeFast — Vol Accel + OI Fade (Fast Variant)

Logic (波动率加速+持仓量反转-快变体):
VolAccelOIFade with shorter/faster vol-accel windows (8/4/16 vs 12/6/24).
The faster vol-accel captures more immediate regime shifts and produces
a structurally different signal from VolAccelFade, reducing correlation
while maintaining the core vol-accel × OI-delta interaction.

Formula:
-1 * zscore(delta(rolling_std(pct_change(close,1), 8), 4), 16) * zscore(delta(open_interest, 6), 16)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolAccelOIFadeFast(FactorRegistry):
    factor_name = "VolAccelOIFadeFast"
    parameters = {"vol_window": 8, "accel_lag": 4, "window": 16, "oi_delta": 6}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Faster vol-accel variant (8/4/16) structurally decorrelated from "
        "VolAccelFade (12/6/24). Captures more immediate regime shifts while "
        "maintaining the vol-accel × OI-delta interaction."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_std(pct_change(close, 1), 8), 4), 16) "
        "* zscore(delta(open_interest, 6), 16)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
