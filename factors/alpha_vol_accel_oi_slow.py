"""
VolAccelOIFadeSlow — Vol Accel + OI Fade (Slow Variant)

Logic (波动率加速+持仓量反转-慢变体):
VolAccelOIFade with longer/slower vol-accel windows (20/10/30 vs 12/6/24).
The slower vol-accel captures sustained regime transitions over longer
horizons, producing a structurally distinct signal from both VolAccelFade
and the fast variant.

Formula:
-1 * zscore(delta(rolling_std(pct_change(close,1), 20), 10), 30) * zscore(delta(open_interest, 10), 30)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolAccelOIFadeSlow(FactorRegistry):
    factor_name = "VolAccelOIFadeSlow"
    parameters = {"vol_window": 20, "accel_lag": 10, "window": 30, "oi_delta": 10}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Slower vol-accel variant (20/10/30) structurally decorrelated from "
        "VolAccelFade (12/6/24). Captures sustained regime transitions over "
        "longer horizons with less noise."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_std(pct_change(close, 1), 20), 10), 30) "
        "* zscore(delta(open_interest, 10), 30)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
