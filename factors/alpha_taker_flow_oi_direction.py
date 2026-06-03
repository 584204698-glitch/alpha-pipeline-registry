from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class TakerFlowWithOiDirection(FactorRegistry):
    factor_name = 'TakerFlowWithOiDirection'
    parameters = {'window': 12, 'oi_lag': 3}
    inputs = ['taker_volume', 'volume', 'open_interest']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Taker-volume share z-scored, gated by OI trend direction: rising OI amplifies trend signal, falling OI flips to reversal.'
    mathematical_formula = 'zscore(taker_volume / (volume + 1e-8), 12) * sign(delta(open_interest, 3))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
