from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class TakerImbalanceReversion8(FactorRegistry):
    factor_name = 'Taker_Imbalance_Reversion_8'
    parameters = {'window': 16}
    inputs = ['taker_volume', 'volume', 'close']
    timeframes = ['15m', '4h']
    rationale = 'Counteracts unstable directional bias seen in (none) by fading extreme taker flow imbalances.'
    mathematical_formula = '-1 * zscore((taker_volume / (volume + 1e-9)) - rolling_mean(taker_volume / (volume + 1e-9), window), window)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
