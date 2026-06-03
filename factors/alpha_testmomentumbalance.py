from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class TestMomentumBalance(FactorRegistry):
    factor_name = 'TestMomentumBalance'
    parameters = {'window': 8, 'lag': 6}
    inputs = ['close', 'volume', 'taker_volume']
    timeframes = ['15m', '2h']
    rationale = 'test'
    mathematical_formula = '(pct_change(close, 6) * zscore(volume, 8)) - zscore(taker_volume, 8)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
