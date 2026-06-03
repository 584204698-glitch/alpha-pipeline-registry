"""
OIFundingSignFade — OI Delta + Funding Sign + Price Sign Reversal

Logic (持仓量+资金费率方向+价格方向反转):
Three directional signals: OI delta (magnitude), funding_rate sign (sentiment
direction), price-trend sign (trend direction). Both sign terms are ±1, providing
clean directional filtering without magnitude dilution.

When OI is building AND funding_rate and price_trend agree on direction →
the consensus is crowded → fade. This is CrowdingFade without the volume
amplifier — using funding+price agreement as a stronger directional filter.

Formula:
-1 * zscore(delta(open_interest, 6), 24) * sign(funding_rate) * sign(close - rolling_mean(close, 48))
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OIFundingSignFade(FactorRegistry):
    factor_name = "OIFundingSignFade"
    parameters = {"oi_delta": 6, "window": 24, "trend_window": 48}
    inputs = ["open_interest", "funding_rate", "close"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "CrowdingFade simplified: OI delta for magnitude, funding_rate sign and "
        "price-trend sign for directional agreement. Both signs = ±1 → product "
        "is ±1 → clean direction. When both signs agree AND OI is building → "
        "strong consensus → fade. No vol or volume component."
    )
    mathematical_formula = (
        "-1 * zscore(delta(open_interest, 6), 24) "
        "* sign(funding_rate) "
        "* sign(close - rolling_mean(close, 48))"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
