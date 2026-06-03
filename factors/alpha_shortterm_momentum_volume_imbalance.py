from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class ShorttermMomentumVolumeImbalance(FactorRegistry):
    factor_name = 'ShortTerm_Momentum_Volume_Imbalance'
    parameters = {'lag': 3, 'lookback': 1}
    inputs = ['close', 'taker_volume']
    timeframes = ['15m', '1h']
    rationale = 'Captures momentum reversal when price change is positive but taker volume declines'
    mathematical_formula = 'pct_change(close, 5) * sign(delta(taker_volume, 1))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
