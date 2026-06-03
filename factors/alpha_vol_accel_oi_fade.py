"""
VolAccelOIFade — Volatility Acceleration × OI Flow Reversal

Logic (波动率加速+持仓量反转):
Combines the two proven winning patterns:
1. VolAccelFade: volatility second-derivative captures regime transitions
2. CrowdingFade: OI delta confirms positioning direction

When vol-of-vol is rising (regime shift incoming) AND OI is building —
the market is about to reverse. The vol-accel provides timing, the OI
flow provides conviction. Two clean, independent signals multiplied.

Formula:
-1 * zscore(delta(rolling_std(pct_change(close, 1), 12), 6), 24) * zscore(delta(open_interest, 6), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolAccelOIFade(FactorRegistry):
    factor_name = "VolAccelOIFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24, "oi_delta": 6}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Fuses VolAccelFade's regime-transition timing with CrowdingFade's "
        "OI-flow conviction. Vol-of-vol identifies when a regime shift is "
        "imminent; OI delta confirms crowd positioning. Clean two-source "
        "interaction with no dilution."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_std(pct_change(close, 1), 12), 6), 24) "
        "* zscore(delta(open_interest, 6), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
