from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class TakerOiCrossSignal(FactorRegistry):
    factor_name = 'TakerOiCrossSignal'
    parameters = {'window': 12}
    inputs = ['taker_volume', 'volume', 'open_interest']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Two-sided: taker share above avg combined with OI expansion = bullish continuation; taker share below avg with OI contraction = bearish. Both directions.'
    mathematical_formula = '(zscore(taker_volume / (volume + 1e-8), 12) + zscore(delta(open_interest, 3), 12)) / 2'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
