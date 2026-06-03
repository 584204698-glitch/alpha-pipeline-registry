from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class TestFundingDivergence(FactorRegistry):
    factor_name = 'TestFundingDivergence'
    parameters = {'oi_lookback': 4, 'window': 12, 'quantile_window': 24, 'threshold': 0.2}
    inputs = ['funding_rate', 'open_interest', 'close']
    timeframes = ['15m', '4h']
    rationale = 'test'
    mathematical_formula = 'zscore(delta(open_interest, 4), 12) * indicator(funding_rate < rolling_quantile(funding_rate, 24, 0.2))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
