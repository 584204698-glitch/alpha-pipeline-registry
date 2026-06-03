from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class RangeContractionBreakout(FactorRegistry):
    factor_name = 'RangeContractionBreakout'
    parameters = {'window': 24, 'lag': 3}
    inputs = ['high', 'low', 'close']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Low realized range (vol compression) precedes breakout in the direction of recent drift. High range signals exhaustion — fade the extreme. Classic vol-regime factor.'
    mathematical_formula = '-1 * zscore((high - low) / close, 24) * sign(pct_change(close, 3))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
