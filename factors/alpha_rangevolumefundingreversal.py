from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Rangevolumefundingreversal(FactorRegistry):
    factor_name = 'RangeVolumeFundingReversal'
    parameters = {'range_window': 12, 'volume_window': 24}
    inputs = ['high', 'low', 'close', 'volume', 'funding_rate']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Wide price ranges with heavy volume and one-sided funding indicate exhaustion, leading to mean reversion.'
    mathematical_formula = '-1 * zscore((high - low) / close, 12) * rank_pct(volume, 24) * sign(funding_rate)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
