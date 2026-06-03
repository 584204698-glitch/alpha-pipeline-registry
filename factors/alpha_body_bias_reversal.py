"""
BodyBiasReversal — Pure OHLC bar body mean-reversion

Logic (K线实体均值回归):
Ground truth discovery: zscore(body_bias) has OOS ICIR = -0.2435.
Bar body position MEAN-REVERTS, not trends! When bar closes extremely
near its high → subsequent returns are lower (fade). When bar closes
extremely near its low → subsequent returns are higher (fade).

This is the simplest, cleanest OHLC factor: single component, no
multiplication, no gates. Pure reversal of intra-bar extremes.

Formula:
-1 * zscore((2*close-high-low)/(high-low+1e-8), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BodyBiasReversal(FactorRegistry):
    factor_name = "BodyBiasReversal"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "Empirically discovered: zscore(body_bias) has OOS ICIR=-0.24. "
        "Bar closes at extremes mean-revert. Single component, no gates, "
        "no multiplications — pure reversal signal. Natural symmetry via zscore."
    )
    mathematical_formula = (
        "-1 * robust_zscore((2*close-high-low)/(high-low+1e-8), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
