from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OIAccelPriceFade(FactorRegistry):
    """OI波动加速×价格反转：OI变化率的vol-of-vol捕捉持仓异动加速→市场分歧加剧→反转。
    使用OI delta的波动率加速度作为timing，完全不同于close-based vol-accel。
    Price momentum提供被fade的方向。零volume/费率依赖。"""
    factor_name = 'OIAccelPriceFade'
    parameters = {'oi_vol_window': 12, 'accel_lag': 6, 'zscore_window': 24, 'mom_lag': 6}
    inputs = ['close', 'open_interest']
    timeframes = ['1h']
    rationale = (
        'OI volatility acceleration: when the volatility of OI changes is accelerating, '
        'positions are being rapidly opened AND closed — the market is deeply divided '
        'and directionally fragile. This is a different chaos signal than price vol-accel '
        'because it captures POSITIONING chaos, not PRICE chaos. Fade the recent price '
        'trend when OI vol is accelerating. Uses delta(OI) std-of-std to measure how '
        'fast position churn is intensifying. Two clean components: OI-vol-accel (when) '
        'and price momentum (what to fade). Zero overlap with close-based vol metrics.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_std(delta(open_interest, 1), 12), 6), 24) '
        '* zscore(pct_change(close, 6), 24)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
