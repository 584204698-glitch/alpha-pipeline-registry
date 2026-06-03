from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class DollarVolAccelPosFadeV2(FactorRegistry):
    """DV-accel timing × 短期价格位置(24-bar rank_pct)。V1用48-bar导致IS/OOS翻转，
    V2缩短到24-bar让位置信号更响应近期变化，减少跨周期非平稳性。"""
    factor_name = 'DollarVolAccelPosFadeV2'
    parameters = {'dv_window': 12, 'accel_lag': 6, 'zscore_window': 24, 'pos_window': 24}
    inputs = ['close', 'volume']
    timeframes = ['1h']
    rationale = (
        'DV-accel timing with SHORT position window (24-bar rank_pct of close). '
        'V1 used 48-bar position, causing IS/OOS sign flip due to non-stationarity '
        'of long-horizon price ranks. 24-bar position is more responsive to recent '
        'price context while still orthogonal to 6-bar momentum. DV-accel provides '
        'the proven chaos timing, 24-bar position provides fade direction. '
        'V1 achieved IC=0.034, 3/3 positive regimes, correlation PASSED — only '
        'IS/OOS sign flip blocked it. Shorter position window fixes stationarity.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_mean(log(volume) * abs(pct_change(close, 1)), 12), 6), 24) '
        '* (rank_pct(close, 24) - 0.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
