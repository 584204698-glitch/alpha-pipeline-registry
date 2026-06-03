"""
BodyBiasReversalTaker — Bar body reversal × taker flow confirmation

Logic (K线实体反转×主动成交确认):
Body bias reversal with taker flow as an orthogonal directional confirmation.
When bar closes at extreme AND taker flow is aggressive in the SAME direction,
the reversal signal strength is maximized (both components agree the move is overdone).

Uses abs(taker zscore) to capture flow intensity regardless of direction,
multiplying with the directional reversal signal.

Formula:
-1 * zscore((2*close-high-low)/(high-low+1e-8), 24) * zscore(taker_volume/(volume+1e-8), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BodyBiasReversalTaker(FactorRegistry):
    factor_name = "BodyBiasReversalTaker"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close", "volume", "taker_volume"]
    timeframes = ["1h"]
    rationale = (
        "Bar body reversal with taker flow direction. When bar extreme AND "
        "taker flow confirms (both zscores agree in sign), the reversal signal "
        "is strongest — conviction + direction alignment. Both zscore-centered."
    )
    mathematical_formula = (
        "-1 * zscore((2*close-high-low)/(high-low+1e-8), 24) "
        "* zscore(taker_volume/(volume+1e-8), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
