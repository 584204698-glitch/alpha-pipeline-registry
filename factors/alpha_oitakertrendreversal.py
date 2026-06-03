from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Oitakertrendreversal(FactorRegistry):
    factor_name = 'OITakerTrendReversal'
    parameters = {'oi_lag': 5, 'oi_window': 20, 'taker_lag': 5, 'taker_window': 20, 'trend_window': 50}
    inputs = ['open_interest', 'taker_volume', 'close']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Rising open interest and taker volume together, when price is extended above its mean, signal an overheated market ready to reverse.'
    mathematical_formula = '-1 * zscore(delta(open_interest, 5), 20) * zscore(pct_change(taker_volume, 5), 20) * sign(close - rolling_mean(close, 50))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
