from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Takerfundingdirectionreversal(FactorRegistry):
    factor_name = 'TakerFundingDirectionReversal'
    parameters = {'funding_window': 24, 'taker_ratio_window': 24}
    inputs = ['funding_rate', 'taker_volume', 'volume', 'close', 'open']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Extreme funding rates combined with aggressive taker participation and intra-candle direction signal imminent reversals.'
    mathematical_formula = '-1 * zscore(funding_rate, 24) * rank_pct(taker_volume/volume, 24) * sign(close - open)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
