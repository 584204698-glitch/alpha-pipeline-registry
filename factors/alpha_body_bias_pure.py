"""
BodyBiasPure — Pure bar body bias zscore

Logic (纯K线实体偏离):
The simplest OHLC factor: where does the bar close within its range?
zscore over 24 bars per symbol captures when the bar position is at an
extreme relative to its own history.

When body bias is extremely positive (close near high) → the bar shows
strong buying conviction → continuation. When extremely negative → strong
selling → continuation (momentum effect at intra-bar level).

Formula:
zscore((2*close-high-low)/(high-low+1e-8), 24)

Single component — no multiplication to destroy symmetry.
zscore provides natural centering and scaling.
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BodyBiasPure(FactorRegistry):
    factor_name = "BodyBiasPure"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "Single-component OHLC factor: zscore of bar body position. "
        "When close is extremely near the high or low relative to history, "
        "the bar shows conviction → follow the move (momentum). "
        "No multiplication, no gates — pure signal with natural symmetry."
    )
    mathematical_formula = (
        "zscore((2*close-high-low)/(high-low+1e-8), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
