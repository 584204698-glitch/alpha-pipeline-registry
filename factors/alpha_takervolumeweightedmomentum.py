from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Takervolumeweightedmomentum(FactorRegistry):
    factor_name = 'TakerVolumeWeightedMomentum'
    parameters = {'window': 20}
    inputs = ['close', 'taker_volume']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Strong taker volume reinforces price direction, capturing institutional flow'
    mathematical_formula = 'rolling_sum(delta(close)*taker_volume, 20)/rolling_sum(taker_volume, 20)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
