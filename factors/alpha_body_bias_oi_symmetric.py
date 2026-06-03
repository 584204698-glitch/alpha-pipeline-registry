"""
BodyBiasOISymmetric — Body bias × OI delta rank (both symmetric)

Logic (K线实体×持仓变化):
Two symmetric components multiplied: bar body bias and OI delta rank.
Both centered at 0 by construction (zscore and rank_pct-0.5).

When both agree on direction (both positive or both negative), the signal
is strong: bar closing at extreme + OI building/declining → conviction.
When they disagree, the signal is weak — market is conflicted.

Formula:
zscore((2*close-high-low)/(high-low+1e-8), 24) * (rank_pct(delta(open_interest, 6), 24) - 0.5)

Only 2 components, both symmetric. No gates, no 3-way interactions.
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BodyBiasOISymmetric(FactorRegistry):
    factor_name = "BodyBiasOISymmetric"
    parameters = {"window": 24, "oi_delta": 6}
    inputs = ["open", "high", "low", "close", "open_interest"]
    timeframes = ["1h"]
    rationale = (
        "Two symmetric components: zscore(body_bias) × rank_pct(OI_delta)-0.5. "
        "Both centered at 0. When they agree → strong directional signal. "
        "When they disagree → weak signal. No gates, clean 2-way interaction."
    )
    mathematical_formula = (
        "zscore((2*close-high-low)/(high-low+1e-8), 24) "
        "* (rank_pct(delta(open_interest, 6), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
