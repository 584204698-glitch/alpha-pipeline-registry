from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Volumefadeextremes(FactorRegistry):
    factor_name = 'VolumeFadeExtremes'
    parameters = {'price_window': 24, 'volume_rank_window': 24}
    inputs = ['close', 'volume']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Thinly traded breakouts/breakdowns are unreliable and fade back into the range.'
    mathematical_formula = '(indicator(close > rolling_max(close,24)) - indicator(close < rolling_min(close,24))) * (1 - rank_pct(volume,24)) * -1'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
