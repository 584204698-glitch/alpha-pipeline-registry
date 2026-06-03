from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class TakerFlowImbalance(FactorRegistry):
    factor_name = 'TakerFlowImbalance'
    parameters = {'window': 24}
    inputs = ['taker_volume', 'volume']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Taker buy/sell ratio above 50% signals aggressive buying; below 50% signals aggressive selling. Mean-reverts over medium horizon as aggressive flow exhausts. Two-sided microstructure factor.'
    mathematical_formula = '-1 * zscore(taker_volume / (volume + 1e-8), 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
