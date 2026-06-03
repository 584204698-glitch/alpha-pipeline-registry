from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Fundingcrowdingfade(FactorRegistry):
    factor_name = 'FundingCrowdingFade'
    parameters = {'funding_window': 24, 'price_window': 24}
    inputs = ['funding_rate', 'close']
    timeframes = ['1h', '4h', '1d']
    rationale = 'Extreme funding rates near price extremes reveal crowded positioning that tends to fade.'
    mathematical_formula = 'zscore(funding_rate, 24) * (indicator(close > rolling_max(close,24)) - indicator(close < rolling_min(close,24))) * -1'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
