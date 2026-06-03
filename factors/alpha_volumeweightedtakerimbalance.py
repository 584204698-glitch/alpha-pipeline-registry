from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Volumeweightedtakerimbalance(FactorRegistry):
    factor_name = 'VolumeWeightedTakerImbalance'
    parameters = {'momentum_window': 12, 'taker_rank_window': 24, 'close_lag': 1}
    inputs = ['taker_volume', 'volume', 'close']
    timeframes = ['5m', '15m', '1h']
    rationale = 'Taker volume dominance amplified by total volume captures aggressive trading that tends to mean‑revert.'
    mathematical_formula = '-1 * zscore(pct_change(close,1), 12) * rank_pct(taker_volume/volume, 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
