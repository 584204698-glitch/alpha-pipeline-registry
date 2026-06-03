from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula

class AbsorptionContinuation(FactorRegistry):
    """v6: rank-split吸收。rank_pct-0.5天然50-50双向，吸收强度由|taker_change| * stall_gate提供。"""
    factor_name = 'AbsorptionContinuation'
    parameters = {'taker_lag': 3, 'zscore_window': 24, 'trend_window': 48}
    inputs = ['taker_volume', 'close']
    timeframes = ['1h', '4h']
    rationale = 'v6 rank-split: direction via rank_pct(taker)-0.5 (mathematically 50-50), magnitude via absorption intensity |taker_change| * stall_gate. Taker flow + price stall = hidden order flow continuation.'
    mathematical_formula = '(rank_pct(pct_change(taker_volume, 3), 24) - 0.5) * zscore(abs(zscore(pct_change(taker_volume, 3), 24)) * (1 - rank_pct(abs(pct_change(close, 1)), 48)), 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
