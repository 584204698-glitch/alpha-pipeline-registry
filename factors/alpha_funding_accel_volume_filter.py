from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class FundingAccelVolumeFilter(FactorRegistry):
    factor_name = 'FundingAccelVolumeFilter'
    parameters = {'window': 12, 'vol_rank_window': 24}
    inputs = ['funding_rate', 'volume']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Funding rate deceleration (bearish flow shift) combined with elevated volume signals panic liquidation — reversal upward.'
    mathematical_formula = '-1 * zscore(delta(funding_rate, 3), 12) * rank_pct(volume, 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
