from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Oifundingpressurereversal(FactorRegistry):
    factor_name = 'OIFundingPressureReversal'
    parameters = {'oi_lag': 10, 'oi_window': 20, 'funding_window': 24, 'clip_limit': 2, 'quantile_window': 60}
    inputs = ['open_interest', 'funding_rate', 'close']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Rising open interest with extreme funding and price above its historical median signals a crowded market due for a correction.'
    mathematical_formula = '-1 * zscore(delta(open_interest, 10), 20) * clip(zscore(funding_rate, 24), -2, 2) * (close / rolling_quantile(close, 60, 0.5) - 1)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
