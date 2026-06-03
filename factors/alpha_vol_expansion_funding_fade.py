"""
VolExpansionFundingFade — Vol Expansion + Funding Direction Fade

Logic (波动率扩张+资金费率反转):
Vol expansion ratio captures regime fragility. Funding rate direction
provides the crowd-positioning anchor. When vol is expanding (short > long)
AND funding rate signals extreme positioning → fade.

This is a CrowdingFade re-interpretation: replace OI delta with vol expansion
as the fragility signal, keep funding rate as directional anchor.

Formula:
-1 * zscore(rolling_std(pct_change(close,1), 6) / rolling_std(pct_change(close,1), 24) - 1, 24) * sign(funding_rate) * rank_pct(volume, 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolExpansionFundingFade(FactorRegistry):
    factor_name = "VolExpansionFundingFade"
    parameters = {"short_vol": 6, "long_vol": 24, "window": 24}
    inputs = ["close", "funding_rate", "volume"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Vol expansion replaces OI delta as the fragility signal in a CrowdingFade "
        "structure. Vol expansion ratio captures regime transitions; funding rate sign "
        "provides directional anchor; volume confirms conviction. Three independent "
        "sources, structurally distinct from both CrowdingFade and VolAccelFade."
    )
    mathematical_formula = (
        "-1 * zscore(rolling_std(pct_change(close, 1), 6) / (rolling_std(pct_change(close, 1), 24) + 1e-8) - 1, 24) "
        "* sign(funding_rate) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
