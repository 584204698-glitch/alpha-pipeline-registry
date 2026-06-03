from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Fundingratezscore(FactorRegistry):
    factor_name = 'FundingRateZscore'
    parameters = {'window': 24}
    inputs = ['funding_rate']
    timeframes = ['1h', '4h', '1d']
    rationale = 'Extreme funding rates signal overcrowded positions likely to reverse'
    mathematical_formula = 'zscore(funding_rate, 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
