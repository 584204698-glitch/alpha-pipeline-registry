from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class CloseMomentumZscore(FactorRegistry):
    factor_name = 'Close_Momentum_Zscore'
    parameters = {'lag': 4, 'window': 12}
    inputs = ['close']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Captures recent price momentum normalized by volatility using z-score of price changes.'
    mathematical_formula = 'zscore(delta(close, 4), 12)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
