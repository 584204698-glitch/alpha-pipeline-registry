"""
BarBodyFlowV2 — Taker flow × bar body bias, WITHOUT OI indicator gate

Logic (K线实体方向跟随V2):
V1 used indicator(OI>0) gate which caused extreme short bias (long_ratio=17.43%)
because OI-rising periods in crypto skew toward downtrends with closes near lows.

V2 removes the OI gate entirely: pure taker flow × bar body bias.
zscore(taker/vol) provides conviction weighting. (2c-h-l)/(h-l) provides
natural [-1,+1] symmetry for the directional signal.

Formula:
zscore(taker_volume/(volume+1e-8), 24) * (2*close-high-low)/(high-low+1e-8)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BarBodyFlowV2(FactorRegistry):
    factor_name = "BarBodyFlowV2"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close", "volume", "taker_volume"]
    timeframes = ["1h"]
    rationale = (
        "V2 removes indicator(OI>0) gate from V1 after it caused extreme short bias "
        "(long=17.4%). Pure taker flow × body bias: when aggressive trading direction "
        "aligns with bar body position, follow the flow. Body_bias provides natural symmetry."
    )
    mathematical_formula = (
        "zscore(taker_volume/(volume+1e-8), 24) "
        "* (2*close-high-low)/(high-low+1e-8)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
