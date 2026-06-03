"""
BodyBiasReversalRobust — Bar body reversal with robust_zscore

Logic (K线实体稳健反转):
Same as BodyBiasReversal but uses robust_zscore (median/MAD) instead of
zscore (mean/std). Robust to fat-tailed crypto returns that can cause
zscore to be dominated by a single extreme bar.

Formula:
-1 * robust_zscore((2*close-high-low)/(high-low+1e-8), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BodyBiasReversalRobust(FactorRegistry):
    factor_name = "BodyBiasReversalRobust"
    parameters = {"window": 36}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "Robust zscore version of body bias reversal, window=36. "
        "median/MAD handles fat tails in crypto better than mean/std. "
        "Tuned from 24 to 36 for optimal t-stat. Same direction: bar extremes mean-revert."
    )
    mathematical_formula = (
        "-1 * robust_zscore((2*close-high-low)/(high-low+1e-8), 36)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
