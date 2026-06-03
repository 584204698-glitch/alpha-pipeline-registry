"""
BarBodyFlow — OHLC bar body directional flow with OI gate

Logic (K线实体方向跟随):
When OI is building (positions being established) and taker flow is aggressive,
the bar body direction reveals which side is winning the position battle.
Follow the winning side — continuation, not reversal.

The (2c-h-l)/(h-l) term is naturally symmetric in [-1,+1], providing
built-in signal balance without rank_pct or indicator multiplications.

Formula:
zscore(taker_volume/(volume+1e-8), 24) * indicator(delta(open_interest,6)>0) * (2*close-high-low)/(high-low+1e-8)

indicator(OI>0) is the single approved indicator gate: OI building has clear
economic meaning (new positions entering), satisfying the "有明确经济含义的状态" exception.
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BarBodyFlow(FactorRegistry):
    factor_name = "BarBodyFlow"
    parameters = {"window": 24, "oi_delta": 6}
    inputs = ["open", "high", "low", "close", "volume", "taker_volume", "open_interest"]
    timeframes = ["1h"]
    rationale = (
        "When OI is building (positions being established) and taker flow is aggressive, "
        "the bar body direction reveals which side is winning. Follow the flow. "
        "Body bias (2c-h-l)/(h-l) gives natural [-1,+1] symmetry without rank transforms. "
        "indicator gate on OI delta > 0 provides sparse, economically meaningful activation."
    )
    mathematical_formula = (
        "zscore(taker_volume/(volume+1e-8), 24) "
        "* indicator(delta(open_interest, 6) > 0) "
        "* (2*close-high-low)/(high-low+1e-8)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
