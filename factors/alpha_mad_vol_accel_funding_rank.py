"""
MADVolAccelFundingRank — MAD Vol Accel + Funding Rank + Volume

Logic:
Same as FundingFade but using (rank_pct(funding_rate) - 0.5) instead of
sign(funding_rate). The continuous rank provides better long/short balance
(~50% each) vs sign() which was asymmetric (29.5% long).

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 6), 24) * (rank_pct(funding_rate, 24) - 0.5) * rank_pct(volume, 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelFundingRank(FactorRegistry):
    factor_name = "MADVolAccelFundingRank"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24}
    inputs = ["close", "funding_rate", "volume"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "FundingFade variant using continuous rank instead of discrete sign. "
        "rank_pct(funding) centers at 0.5 → (rank-0.5) centers at 0 → "
        "balanced long/short ratio. Three clean independent sources."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) "
        "* (rank_pct(funding_rate, 24) - 0.5) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
