from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class FundingRateCurveSteepness(FactorRegistry):
    factor_name = 'FundingRateCurveSteepness'
    parameters = {'window': 24}
    inputs = ['funding_rate']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Two-sided: funding_rate above its rolling max signals extreme bullish sentiment (bearish revert); below rolling min signals extreme bearish (bullish revert).'
    mathematical_formula = '(funding_rate - rolling_min(funding_rate, 24)) / (rolling_max(funding_rate, 24) - rolling_min(funding_rate, 24) + 1e-8) - 0.5'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
