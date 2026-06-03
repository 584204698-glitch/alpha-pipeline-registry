from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Oiexpansionbreakout(FactorRegistry):
    factor_name = 'OIExpansionBreakout'
    parameters = {'lag': 5}
    inputs = ['open_interest', 'close']
    timeframes = ['2h', '4h', '1d']
    rationale = 'Rising open interest with positive returns confirms trend strength'
    mathematical_formula = 'rank_pct(delta(open_interest, 5)*pct_change(close, 5))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
