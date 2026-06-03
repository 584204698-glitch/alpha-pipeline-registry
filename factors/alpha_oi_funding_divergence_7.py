from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class OiFundingDivergence7(FactorRegistry):
    factor_name = 'OI_Funding_Divergence_7'
    parameters = {'oi_lookback': 4, 'window': 12, 'quantile_window': 24, 'funding_quantile': 0.2}
    inputs = ['open_interest', 'funding_rate', 'close']
    timeframes = ['15m', '2h']
    rationale = 'Avoid prior failed motifs (none) by combining OI acceleration with funding stress.'
    mathematical_formula = 'zscore(delta(open_interest, oi_lookback), window) * indicator(funding_rate < rolling_quantile(funding_rate, quantile_window, funding_quantile))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
