from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class DollarVolAccelPosFade(FactorRegistry):
    """成交额波动加速×价格位置反转：DV-accel timing + price POSITION(非momentum)方向。
    价格位置(rank_pct of close)替代价格动量(zscore of pct_change)来降低与VolAccelFade的相关性。
    量价共振加速+价格高位→拥挤→做空；低位→超卖→做多。"""
    factor_name = 'DollarVolAccelPosFade'
    parameters = {'dv_window': 12, 'accel_lag': 6, 'zscore_window': 24, 'pos_window': 48}
    inputs = ['close', 'volume']
    timeframes = ['1h']
    rationale = (
        'Dollar-volume acceleration timing with PRICE POSITION (not momentum) as '
        'directional anchor. DV-accel tells us WHEN the market is in chaos (volume×vol '
        'accelerating). Price position (rank_pct of close within 48-bar window) tells us '
        'WHERE we are in the range — near highs = crowded longs = SHORT; near lows = '
        'oversold = LONG. Price position has much lower correlation with price momentum '
        '(zscore of pct_change) used by VolAccelFade, dramatically reducing factor overlap '
        'while preserving the DV-accel timing power (IC=0.042, ICIR=0.338 in V2 tests).'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_mean(log(volume) * abs(pct_change(close, 1)), 12), 6), 24) '
        '* (rank_pct(close, 48) - 0.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
