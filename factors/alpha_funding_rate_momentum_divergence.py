from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class FundingRateMomentumDivergence(FactorRegistry):
    factor_name = 'Funding Rate Momentum Divergence'
    parameters = {'fast_period': 3, 'slow_period': 10, 'roc_period': 5, 'vol_period': 20}
    inputs = ['close', 'funding_rate', 'volume']
    timeframes = ['15m', '2h', '4h']
    rationale = 'Funding rate reflects perpetual futures market sentiment. When funding rate diverges from price momentum (e.g., price rising but funding rate falling), it may signal weakening trend and potential reversal. Combining with volume changes improves robustness.'
    mathematical_formula = 'Zscore(ema(funding_rate, fast_period) - ema(funding_rate, slow_period)) * sign(roc(close, roc_period)) * zscore(volume / sma(volume, vol_period))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
