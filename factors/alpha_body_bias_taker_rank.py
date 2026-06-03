"""
BodyBiasTakerRank — Body bias zscore × taker flow rank

Logic (K线实体×主动成交):
Bar body bias captures intra-bar conviction. Taker flow ratio captures
aggressive trading direction. When both align → strong directional signal.

Unlike V1 versions, uses only 2 components, both centered at 0:
zscore(body_bias) naturally centered; rank_pct(taker)-0.5 strictly symmetric.

Formula:
zscore((2*close-high-low)/(high-low+1e-8), 24) * (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BodyBiasTakerRank(FactorRegistry):
    factor_name = "BodyBiasTakerRank"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close", "volume", "taker_volume"]
    timeframes = ["1h"]
    rationale = (
        "Two symmetric components: zscore(body_bias) × (rank_pct(taker_vol/vol)-0.5). "
        "Bar body conviction × aggressive flow direction. When both align → "
        "follow the flow. Both components centered at 0 for signal symmetry."
    )
    mathematical_formula = (
        "zscore((2*close-high-low)/(high-low+1e-8), 24) "
        "* (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
