from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class PriceAccelVolFade(FactorRegistry):
    """价格加速度反转×成交量确认：price的二阶导(加速度)捕捉趋势是否在加速→反转。
    abs(volume_zscore)作为conviction乘数，不做方向判断(避免乘积符号歧义)。
    价格加速涨+高量=SHORT(趋势加速赶顶)，加速跌+高量=LONG(恐慌加速见底)。"""
    factor_name = 'PriceAccelVolFade'
    parameters = {'accel_lag': 6, 'zscore_window': 24, 'vol_mean_window': 48}
    inputs = ['close', 'volume']
    timeframes = ['1h']
    rationale = (
        'Price ACCELERATION (second derivative, not velocity) reversal: when price '
        'movement is accelerating (getting faster), trends become increasingly fragile '
        'as they attract momentum chasers. Price accel = delta(pct_change), capturing '
        'whether the trend is speeding up or slowing down. Volume anomaly (abs zscore '
        'of volume/rolling_mean) provides conviction without introducing directional '
        'ambiguity — the sign comes purely from price acceleration. '
        'Fundamentally different from VolAccelFade which uses vol-of-vol + price momentum: '
        'this uses PRICE accel (fragile trend detection) + VOLUME conviction (crowding).'
    )
    mathematical_formula = (
        '-1 * zscore(delta(pct_change(close, 1), 6), 24) '
        '* abs(zscore(volume / rolling_mean(volume, 48), 24))'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
