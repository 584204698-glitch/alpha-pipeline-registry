from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class VolSurpriseReversal(FactorRegistry):
    factor_name = 'VolSurpriseReversal'
    parameters = {'window': 24}
    inputs = ['high', 'low', 'close', 'volume']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Realized range/close volatility spike, confirmed by volume expansion, tends to reverse as panic exhausts.'
    mathematical_formula = '-1 * zscore((high - low) / close, 24) * zscore(volume, 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
