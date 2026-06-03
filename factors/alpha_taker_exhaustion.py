from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula

class TakerExhaustion(FactorRegistry):
    """主动量飙升但价格停滞 = 买方/卖方力量耗尽 → 反转"""
    factor_name = 'TakerExhaustion'
    parameters = {'taker_delta_lag': 3, 'zscore_window': 24, 'price_stall_window': 48}
    inputs = ['taker_volume', 'close']
    timeframes = ['1h', '4h']
    rationale = 'When taker volume surges but price barely moves, the aggressive side is exhausted and a reversal follows.'
    mathematical_formula = 'zscore(pct_change(taker_volume, 3), 24) * (1 - rank_pct(abs(pct_change(close, 1)), 48))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
