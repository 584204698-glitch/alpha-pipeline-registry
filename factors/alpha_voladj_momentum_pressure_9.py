from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class VoladjMomentumPressure9(FactorRegistry):
    factor_name = 'VolAdj_Momentum_Pressure_9'
    parameters = {'lag': 6, 'window': 18}
    inputs = ['close', 'volume', 'open_interest']
    timeframes = ['15m', '2h', '4h']
    rationale = 'Uses volatility-adjusted momentum to reduce regime fragility observed in (none).'
    mathematical_formula = 'zscore(pct_change(close, lag), window) * zscore(delta(open_interest, lag), window) - zscore(volume, window)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
