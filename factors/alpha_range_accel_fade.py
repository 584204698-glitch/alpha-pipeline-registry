from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class RangeAccelFade(FactorRegistry):
    """价格振幅加速反转：用high/low振幅替代close-to-close vol，捕捉价格波动区间扩张→趋势脆弱→反转。
    振幅加速=市场进入混沌，price momentum=被fade的趋势方向。与VolAccelFade共享反转框架但用不同vol度量。"""
    factor_name = 'RangeAccelFade'
    parameters = {'range_window': 12, 'accel_lag': 6, 'zscore_window': 24, 'mom_lag': 6}
    inputs = ['high', 'low', 'close']
    timeframes = ['1h']
    rationale = (
        'Range-based volatility acceleration: when the high/low range is expanding '
        'faster and faster, the market is entering a chaotic regime. Unlike close-to-close '
        'vol, range captures intra-bar extremes missed by closing prices. '
        'Fade the recent price trend when range is accelerating — trends formed during '
        'range expansion are fragile and prone to sharp reversals. '
        'Uses (high/low - 1) as the range metric for log-normality.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_std(high / low - 1, 12), 6), 24) '
        '* zscore(pct_change(close, 6), 24)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
