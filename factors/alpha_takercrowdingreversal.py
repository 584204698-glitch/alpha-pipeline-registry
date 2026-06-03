from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Takercrowdingreversal(FactorRegistry):
    factor_name = 'TakerCrowdingReversal'
    parameters = {'lag': 3, 'window': 12, 'trend_window': 24}
    inputs = ['close', 'taker_volume']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Uses taker volume momentum with price-mean sign to produce regime-dependent signals, reversing when above mean and continuing when below.'
    mathematical_formula = '-1 * zscore(delta(taker_volume,3),12) * sign(close - rolling_mean(close,24))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
