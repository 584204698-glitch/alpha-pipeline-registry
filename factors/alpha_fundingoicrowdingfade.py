from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Fundingoicrowdingfade(FactorRegistry):
    factor_name = 'FundingOICrowdingFade'
    parameters = {'funding_rank_window': 24, 'oi_delta_lag': 6, 'oi_rank_window': 24, 'trend_window': 100}
    inputs = ['close', 'funding_rate', 'open_interest']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Fades crowded positions when both funding rate and OI changes are extreme, with price trend confirmation.'
    mathematical_formula = '(rank_pct(funding_rate, 24) - 0.5) * (rank_pct(delta(open_interest, 6), 24) - 0.5) * sign(close - rolling_mean(close, 100)) * -1'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
