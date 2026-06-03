"""
OITrendCrowdingFade — OI + Price Trend + Volume Crowding Reversal

Logic (持仓量趋势拥挤反转):
CrowdingFade variant that replaces funding_rate with price trend as the
directional anchor. When OI builds rapidly AND price is strongly trending
AND volume is elevated — the crowd is positioned in the trend direction.
Fade the crowded consensus.

Unlike OIVolumeDivergence (which used LOW volume as signal), this uses
HIGH volume as confirmation — matching the CrowdingFade pattern.

Formula:
-1 * zscore(delta(open_interest, 6), 24) * sign(close - rolling_mean(close, 48)) * rank_pct(volume, 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OITrendCrowdingFade(FactorRegistry):
    factor_name = "OITrendCrowdingFade"
    parameters = {"oi_delta": 6, "window": 24, "trend_window": 48}
    inputs = ["open_interest", "close", "volume"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "CrowdingFade variant: OI flow + price trend + volume. "
        "When OI is growing AND price is trending AND volume is high, "
        "the crowd is committed in one direction. Fade the consensus. "
        "Uses sign(price_trend) as directional anchor (not zscore, to preserve magnitude)."
    )
    mathematical_formula = (
        "-1 * zscore(delta(open_interest, 6), 24) "
        "* sign(close - rolling_mean(close, 48)) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
