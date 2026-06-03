"""
FundingBodyFlow — Funding rate × bar body bias

Logic (资金费率×K线实体跟随):
Funding rate captures the cost of maintaining directional positions — a pure
sentiment measure orthogonal to price action and volume. When funding is extreme
positive (longs paying heavily) AND bar closes near high, the bullish sentiment
has real conviction → follow the flow (LONG).

zscore(funding_rate) provides centered timing, body_bias provides [-1,+1] direction.
Both are naturally symmetric — no indicator gates, no rank transforms needed.

Formula:
zscore(funding_rate, 24) * (2*close-high-low)/(high-low+1e-8)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class FundingBodyFlow(FactorRegistry):
    factor_name = "FundingBodyFlow"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close", "funding_rate"]
    timeframes = ["1h"]
    rationale = (
        "Funding rate × bar body bias: funding captures pure sentiment (orthogonal to "
        "price/volume), bar body captures intra-bar direction. When funding extreme "
        "aligns with bar direction → conviction continuation. Both symmetric by construction."
    )
    mathematical_formula = (
        "zscore(funding_rate, 24) "
        "* (2*close-high-low)/(high-low+1e-8)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
