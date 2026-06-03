"""
FundingAccelPriceReversal — Funding Rate Acceleration + Price Reversal

Logic (资金费率加速反转):
Rather than using funding_rate level (which is persistent/slow-changing),
use the ACCELERATION of funding_rate — how fast sentiment is shifting.
Combined with price momentum and volume confirmation.

When funding_rate is accelerating (delta of funding_rate growing) AND
price is trending AND volume is high → sentiment shift is happening
but price hasn't adjusted yet → reversal signal.

This is the first factor to use funding_rate DELTA (acceleration)
instead of funding_rate LEVEL — capturing regime shifts in sentiment.

Formula:
-1 * zscore(delta(funding_rate, 6), 24) * zscore(pct_change(close, 6), 24) * rank_pct(volume, 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class FundingAccelPriceReversal(FactorRegistry):
    factor_name = "FundingAccelPriceReversal"
    parameters = {"funding_delta": 6, "mom_lag": 6, "window": 24}
    inputs = ["funding_rate", "close", "volume"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "First factor to use funding_rate ACCELERATION (delta), not level. "
        "When funding is accelerating AND price is trending AND volume confirms — "
        "sentiment is shifting faster than price reflects. Fade the trend. "
        "Funding acceleration captures regime shifts that funding level misses."
    )
    mathematical_formula = (
        "-1 * zscore(delta(funding_rate, 6), 24) "
        "* zscore(pct_change(close, 6), 24) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
