from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Factorname(FactorRegistry):
    factor_name = 'FactorName'
    parameters = {'lag': 4, 'window': 12}
    inputs = ['close', 'volume', 'taker_volume', 'funding_rate', 'open_interest']
    timeframes = ['15m', '2h', '4h']
    rationale = 'One sentence rationale'
    mathematical_formula = 'zscore(delta(open_interest, 4), 12)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
