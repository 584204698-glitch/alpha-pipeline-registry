from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula

class OIDeltaAcceleration(FactorRegistry):
    """OI变化加速+价格背离=机构调仓拐点。不是看OI在变（一阶导），是看OI变化的速度在变（二阶导）。"""
    factor_name = 'OIDeltaAcceleration'
    parameters = {'delta_lag': 6, 'accel_lag': 3, 'window': 24, 'trend_window': 48}
    inputs = ['open_interest', 'close']
    timeframes = ['1h', '4h']
    rationale = 'OI acceleration (second derivative) captures positioning inflection points that first-derivative signals miss. When OI growth is accelerating while price trends opposite = smart money repositioning → reversal. Cross product of acceleration and momentum ensures bidirectional symmetry.'
    mathematical_formula = '-1 * zscore(delta(delta(open_interest, 6), 3), 24) * zscore(pct_change(close, 1), 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
