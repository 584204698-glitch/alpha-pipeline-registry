from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Volatilityscaledfundingreversal(FactorRegistry):
    factor_name = 'VolatilityScaledFundingReversal'
    parameters = {'vol_std_window': 24, 'vol_zscore_window': 48, 'funding_window': 24, 'trend_window': 100}
    inputs = ['close', 'funding_rate']
    timeframes = ['15m', '1h', '4h']
    rationale = 'High volatility amplifies the predictive power of funding rate extremes, fading them against the macro trend.'
    mathematical_formula = '-1 * zscore(rolling_std(pct_change(close,1), 24), 48) * zscore(funding_rate, 24) * sign(close - rolling_mean(close, 100))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
