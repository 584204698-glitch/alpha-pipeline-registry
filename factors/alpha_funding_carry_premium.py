from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class FundingCarryPremium(FactorRegistry):
    factor_name = 'FundingCarryPremium'
    parameters = {'window': 48}
    inputs = ['close', 'funding_rate']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Rank-normalized funding rate gated by price-trend sign reversal: high funding + above trend = bearish, high funding + below trend = bullish. Two-sided carry signal.'
    mathematical_formula = 'rank_pct(funding_rate, 48) * (indicator(close < rolling_mean(close, 48)) - indicator(close > rolling_mean(close, 48)))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
