from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class VolumeFundingDivergence(FactorRegistry):
    factor_name = 'Volume-Funding Divergence'
    parameters = {'lookback': 20, 'z_score_threshold': 2.0}
    inputs = ['close', 'volume', 'funding_rate']
    timeframes = ['15m', '2h', '4h']
    rationale = 'Divergence between volume and funding rate signals potential reversals. High volume with negative funding suggests aggressive shorting, often preceding a squeeze.'
    mathematical_formula = 'Z_Volume = (volume - rolling_mean(volume, 20)) / rolling_std(volume, 20); Z_Funding = (funding_rate - rolling_mean(funding_rate, 20)) / rolling_std(funding_rate, 20); factor = Z_Volume * (-Z_Funding)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
