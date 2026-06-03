from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class TakerImbalanceReversal(FactorRegistry):
    """主动量失衡反转：taker_ratio极端=情绪化交易高峰→均值回归。
    纯taker量信号，robust_zscore用MAD抗fat-tail。
    方向：taker极端高(贪婪)→做空，taker极端低(恐慌)→做多。
    与所有含价/OI/费率的因子天然低相关。"""
    factor_name = 'TakerImbalanceReversal'
    parameters = {'window': 24}
    inputs = ['taker_volume', 'volume']
    timeframes = ['1h', '4h']
    rationale = (
        'When taker_volume/volume ratio reaches extreme levels (captured by robust_zscore '
        'using median/MAD), the market is at peak emotional intensity — either panic selling '
        'or FOMO buying. These extremes revert. -1 inverts the robust_zscore: '
        'extreme high taker→SHORT (fade greed), extreme low taker→LONG (fade panic). '
        'robust_zscore handles fat-tailed crypto volume distributions better than standard zscore. '
        'Pure taker-volume signal with zero correlation to price/OI/funding-based factors.'
    )
    mathematical_formula = (
        '-1 * robust_zscore(taker_volume / (volume + 1e-8), 24)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
