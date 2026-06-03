from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class OIPriceDivergence(FactorRegistry):
    factor_name = 'OIPriceDivergence'
    parameters = {'window': 20, 'lag': 6}
    inputs = ['close', 'open_interest']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Classic divergence: price up + OI down = short covering (bullish), price down + OI up = accumulation (bullish), price up + OI up = overbought (bearish), price down + OI down = distribution (bearish).'
    mathematical_formula = '-1 * zscore(pct_change(close, 6), 20) * zscore(delta(open_interest, 6), 20)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
