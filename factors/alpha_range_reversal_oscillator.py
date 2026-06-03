from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class RangeReversalOscillator(FactorRegistry):
    factor_name = 'RangeReversalOscillator'
    parameters = {'window': 24}
    inputs = ['high', 'low', 'close']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Price near rolling max signals overbought (bearish); near rolling min signals oversold (bullish). Two-sided mean-reversion oscillator.'
    mathematical_formula = '(close - rolling_min(close, 24)) / (rolling_max(close, 24) - rolling_min(close, 24) + 1e-8) - 0.5'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
