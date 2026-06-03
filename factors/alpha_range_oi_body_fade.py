"""
RangeOIBodyFade — Bar range expansion timing × OI rank direction

Logic (K线波幅扩张×持仓变化反转):
When bar range (high-low) expands, the market is in an emotional regime where
directional bets become fragile. OI delta rank provides the directional anchor:
when OI is building (positive rank) during range expansion, positions are
being established at extremes → fade them.

Formula:
-1 * zscore((high-low)/(rolling_mean(high-low,48)+1e-8), 24) * (rank_pct(delta(open_interest, 6), 24) - 0.5)

Timing: range expansion zscore (orthogonal to close-to-close vol)
Direction: OI delta rank (symmetric via rank_pct-0.5)
Fade: -1 multiplier
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class RangeOIBodyFade(FactorRegistry):
    factor_name = "RangeOIBodyFade"
    parameters = {"window": 24, "range_mean": 48, "oi_delta": 6}
    inputs = ["high", "low", "open_interest"]
    timeframes = ["1h"]
    rationale = (
        "Range expansion timing (intra-bar extremes) × OI delta direction. "
        "Orthogonal to VolAccelFade (uses bar range, not close-to-close vol). "
        "OI rank provides symmetric directional anchor. Fade positions built at extremes."
    )
    mathematical_formula = (
        "-1 * zscore((high-low)/(rolling_mean(high-low, 48)+1e-8), 24) "
        "* (rank_pct(delta(open_interest, 6), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
