from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class DollarVolAccelFade(FactorRegistry):
    """成交额波动加速反转：volume × |return| 的加速度捕捉量价共振加速→市场过热→反转。
    量价共振加速=情绪化交易高峰，比纯vol-accel多了成交额维度。fade price momentum。"""
    factor_name = 'DollarVolAccelFade'
    parameters = {'dv_window': 12, 'accel_lag': 6, 'zscore_window': 24, 'mom_lag': 6}
    inputs = ['close', 'volume']
    timeframes = ['1h']
    rationale = (
        'Dollar-volume (volume × |return|) acceleration: captures when BOTH volume '
        'and price movement are accelerating together — the hallmark of emotional, '
        'conviction-driven trading. When dollar-volume is accelerating, positions are '
        'being built on emotion rather than analysis → fragile and prone to reversal. '
        'Fades the recent price trend. More powerful than pure vol-accel because it '
        'requires volume confirmation (conviction) alongside volatility expansion. '
        'Uses abs(pct_change) × volume to measure capital at risk per bar.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_mean(volume * abs(pct_change(close, 1)), 12), 6), 24) '
        '* zscore(pct_change(close, 6), 24)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
