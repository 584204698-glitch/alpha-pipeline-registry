from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Oimomentumdivergence(FactorRegistry):
    factor_name = 'OIMomentumDivergence'
    parameters = {'oi_delta_lag': 5, 'zscore_window': 20, 'close_lag': 1}
    inputs = ['open_interest', 'close']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Open interest changes combined with price direction often signal positioning imbalances that revert.'
    mathematical_formula = 'zscore(delta(open_interest,5), 20) * zscore(pct_change(close,1), 20) * -1'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
