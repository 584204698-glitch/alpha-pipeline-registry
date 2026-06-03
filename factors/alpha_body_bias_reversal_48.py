"""
BodyBiasReversal48 — Bar body reversal with 48-bar window

Logic (K线实体反转48周期):
Same as BodyBiasReversal but with 48-bar zscore window to capture
longer-term bar position extremes. Longer window should provide more
stable estimates and potentially better regime invariance.

Formula:
-1 * zscore((2*close-high-low)/(high-low+1e-8), 48)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BodyBiasReversal48(FactorRegistry):
    factor_name = "BodyBiasReversal48"
    parameters = {"window": 48}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "48-bar variant of body bias reversal. Longer window captures multi-day "
        "bar position extremes, providing more stable zscore estimates. "
        "Single component, clean construction."
    )
    mathematical_formula = (
        "-1 * zscore((2*close-high-low)/(high-low+1e-8), 48)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
