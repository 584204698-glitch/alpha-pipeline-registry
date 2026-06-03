"""
VolAccelOIRankFade — Vol Accel + OI Rank Fade (Correlation-Fixed)

Logic (波动率加速+持仓量排序反转):
VolAccelOIFade with OI component changed from zscore to (rank_pct - 0.5).
The rank_pct transformation produces a uniform distribution on [0,1],
fundamentally changing the signal distribution vs zscore (which preserves
outliers and has a ~normal distribution).

This structural change to the OI term reduces correlation with VolAccelFade
while preserving the core insight: vol-accel (regime timing) × OI flow
(positioning conviction).

Formula:
-1 * zscore(delta(rolling_std(pct_change(close,1), 12), 6), 24) * (rank_pct(delta(open_interest, 6), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolAccelOIRankFade(FactorRegistry):
    factor_name = "VolAccelOIRankFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24, "oi_delta": 6}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "VolAccelOIFade variant: OI component uses rank_pct (uniform distribution) "
        "instead of zscore (normal distribution). This structural change reduces "
        "correlation with VolAccelFade's price-zscore component while maintaining "
        "the same regime-transition + positioning-fade logic."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_std(pct_change(close, 1), 12), 6), 24) "
        "* (rank_pct(delta(open_interest, 6), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
