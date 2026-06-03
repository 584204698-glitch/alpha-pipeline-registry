"""
BodyBiasReversalVol — Bar body reversal × volume conviction

Logic (K线实体反转×成交量确认):
Extends BodyBiasReversal (OOS ICIR -0.24 discovered) with volume timing.
High volume during extreme bar closes = high conviction reversal signal.
Low volume = noise, signal dampened.

Volume acts as a conviction amplifier, not a directional term — it scales
the reversal signal strength without changing its sign.

Formula:
-1 * zscore((2*close-high-low)/(high-low+1e-8), 24) * zscore(volume, 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BodyBiasReversalVol(FactorRegistry):
    factor_name = "BodyBiasReversalVol"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close", "volume"]
    timeframes = ["1h"]
    rationale = (
        "Body bias reversal (empirically OOS ICIR -0.24) with volume conviction. "
        "Volume is a non-directional amplifier: high volume on extreme bars = high "
        "conviction reversal. Both zscore terms centered at 0 for symmetry."
    )
    mathematical_formula = (
        "-1 * zscore((2*close-high-low)/(high-low+1e-8), 24) "
        "* zscore(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
