from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Oivolatilityregime(FactorRegistry):
    factor_name = 'OIVolatilityRegime'
    parameters = {'oi_delta_lag': 6, 'oi_z_window': 24, 'vol_std_window': 24, 'vol_mean_window': 48, 'close_lag': 1}
    inputs = ['open_interest', 'close']
    timeframes = ['1h', '4h', '1d']
    rationale = 'Open interest shifts gain predictive power when volatility rises above its norm, signaling regime‑dependent reversals.'
    mathematical_formula = '-1 * zscore(delta(open_interest,6), 24) * indicator(rolling_std(pct_change(close,1), 24) > rolling_mean(rolling_std(pct_change(close,1), 24), 48))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
