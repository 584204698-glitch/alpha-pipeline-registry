from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Stochastictakerfunding(FactorRegistry):
    factor_name = 'StochasticTakerFunding'
    parameters = {'stoch_minmax_window': 48, 'stoch_zscore_window': 24, 'taker_delta_lag': 3, 'taker_rank_window': 24}
    inputs = ['close', 'taker_volume', 'funding_rate']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Stochastic oscillator extremes with taker volume momentum and funding sentiment confirm reversal.'
    mathematical_formula = 'zscore((close - rolling_min(close, 48)) / (rolling_max(close, 48) - rolling_min(close, 48) + 1e-8), 24) * (rank_pct(delta(taker_volume, 3), 24) - 0.5) * sign(funding_rate) * -1'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
