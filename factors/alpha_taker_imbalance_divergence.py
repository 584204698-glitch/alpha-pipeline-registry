from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula

class TakerImbalanceDivergence(FactorRegistry):
    """主动量失衡与价格方向背离=隐藏订单流。当taker量暴增但价格不跟=聪明钱在对面运作。"""
    factor_name = 'TakerImbalanceDivergence'
    parameters = {'taker_lag': 3, 'window': 24, 'trend_window': 48}
    inputs = ['taker_volume', 'volume', 'close']
    timeframes = ['1h', '4h']
    rationale = 'When taker_volume/volume ratio (aggressiveness) diverges from price direction, the visible price is misleading. High taker ratio + falling price = aggressive selling being absorbed = reversal up. Low taker ratio + rising price = passive buying = fragile. Cross product of taker aggressiveness and price momentum for bidirectional symmetry.'
    mathematical_formula = '-1 * zscore(taker_volume / (volume + 1e-8), 24) * zscore(pct_change(close, 1), 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
