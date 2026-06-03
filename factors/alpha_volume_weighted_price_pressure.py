from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class VolumeWeightedPricePressure(FactorRegistry):
    factor_name = 'VolumeWeightedPricePressure'
    parameters = {'short_window': 8, 'long_window': 24}
    inputs = ['close', 'volume']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Short-term price change weighted by relative volume vs its average, then compared to long-term trend. Two-sided momentum/mean-reversion hybrid.'
    mathematical_formula = 'zscore(pct_change(close, 3) * (volume / (rolling_mean(volume, 24) + 1e-8)), 12)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
