"""
BodyEffReversal — Bar efficiency-weighted body bias reversal

Logic (K线效率加权反转):
Combines body bias reversal with bar efficiency. Zscore of (body_bias × efficiency)
where efficiency = abs(close-open)/(high-low): the fraction of bar range
covered by the directional body (vs wicks).

High body bias with high efficiency → strong directional bar → strong reversal signal.
Low body bias with low efficiency (doji) → noise → weak signal.

Formula:
-1 * zscore((2*close-high-low)/(high-low+1e-8) * abs(close-open)/(high-low+1e-8), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BodyEffReversal(FactorRegistry):
    factor_name = "BodyEffReversal"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "Efficiency-weighted body bias with robust_zscore. body_bias × bar_efficiency "
        "captures conviction-weighted direction. Robust zscore handles crypto fat tails. "
        "Efficient extreme bars mean-revert more strongly than doji extremes."
    )
    mathematical_formula = (
        "-1 * robust_zscore((2*close-high-low)/(high-low+1e-8) "
        "* abs(close-open)/(high-low+1e-8), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
