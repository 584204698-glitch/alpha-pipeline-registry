from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Oidivergencebreak(FactorRegistry):
    factor_name = 'OIDivergenceBreak'
    parameters = {'extreme_window': 20}
    inputs = ['close', 'open_interest', 'funding_rate']
    timeframes = ['1h', '4h']
    rationale = 'Detects bearish/bullish divergences between price extremes, open interest changes, and funding rate sign to predict reversals.'
    mathematical_formula = '(-1 * indicator(close > rolling_max(close,20)) * indicator(funding_rate > 0) * indicator(delta(open_interest,1) < 0)) + (indicator(close < rolling_min(close,20)) * indicator(funding_rate < 0) * indicator(delta(open_interest,1) > 0))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
