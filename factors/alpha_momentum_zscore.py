from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class MomentumZscore(FactorRegistry):
    factor_name = 'momentum_zscore'
    parameters = {'period': 1, 'window': 20}
    inputs = ['close']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Captures short-term momentum by measuring deviation of recent price changes from their historical distribution.'
    mathematical_formula = 'zscore(pct_change(close, 1), 20)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
