from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Volnormpricereversal(FactorRegistry):
    factor_name = 'VolNormPriceReversal'
    parameters = {'trend_window': 50, 'std_window': 50, 'zscore_window': 24, 'volume_rank_window': 24}
    inputs = ['close', 'volume', 'funding_rate']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Mean-reversion of price normalized by volatility, using volume intensity and funding rate sentiment.'
    mathematical_formula = 'zscore((close - rolling_mean(close, 50)) / rolling_std(close, 50), 24) * (rank_pct(volume, 24) - 0.5) * sign(funding_rate) * -1'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
