from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class PureFundingCarry(FactorRegistry):
    factor_name = 'PureFundingCarry'
    parameters = {'window': 48}
    inputs = ['funding_rate']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Funding rate mean-reversion: extreme positive funding signals overcrowded longs (bearish), extreme negative signals overcrowded shorts (bullish). Simplest and most robust crypto alpha.'
    mathematical_formula = '-1 * zscore(funding_rate, 48)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
