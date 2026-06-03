from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class OpenInterestMomentumWithTakerVolume(FactorRegistry):
    factor_name = 'Open Interest Momentum with Taker Volume'
    parameters = {'oi_lookback': 1, 'taker_ratio_threshold': 0.55}
    inputs = ['close', 'open_interest', 'taker_volume']
    timeframes = ['15m', '2h', '4h']
    rationale = 'Rising open interest combined with aggressive taker buy volume indicates strong directional conviction, while divergence warns of exhaustion.'
    mathematical_formula = 'OI_change = (open_interest - open_interest.shift(1)) / open_interest.shift(1); Taker_ratio = taker_volume / volume; factor = OI_change * (Taker_ratio - 0.5)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
