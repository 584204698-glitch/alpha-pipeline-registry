from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolumeImpulseReversal(FactorRegistry):
    """成交量脉冲反转：异常放量(vol/rolling_mean)意味着情绪化交易→反转。
    zscore(相对成交量)捕捉放量异常程度，zscore(price)决定反转方向。
    放量涨→做空(贪婪见顶)，放量跌→做多(恐慌见底)。无OI无费率纯量价。"""
    factor_name = 'VolumeImpulseReversal'
    parameters = {'price_window': 24, 'vol_mean_window': 48, 'zscore_window': 24}
    inputs = ['close', 'volume']
    timeframes = ['1h']
    rationale = (
        'Volume impulse reversal: when volume spikes well above its rolling mean, '
        'the market is experiencing an emotional event — panic selling or FOMO buying. '
        'These volume-driven moves tend to reverse as the emotional energy dissipates. '
        'Uses volume/rolling_mean(volume) to detect relative volume anomalies, '
        'zscored to normalize across assets. Price direction determines fade direction. '
        'Unlike DollarVolAccelFade which measures volume×vol acceleration, this '
        'measures pure volume anomaly → naturally low correlation with vol-accel factors.'
    )
    mathematical_formula = (
        '-1 * zscore(pct_change(close, 1), 24) '
        '* zscore(volume / rolling_mean(volume, 48), 24)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
