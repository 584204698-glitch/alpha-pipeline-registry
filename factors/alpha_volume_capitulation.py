from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolumeCapitulationReversal(FactorRegistry):
    """放量恐慌反转：极端放量(vol>P90) + 近期跌幅→LONG恐慌底；极端放量+近期涨幅→SHORT贪婪顶。
    indicator稀疏激活(仅极端放量时)、rank_pct决定方向、dropna=False保持对齐。"""
    factor_name = 'VolumeCapitulationReversal'
    parameters = {'vol_window': 48, 'price_window': 24, 'quantile': 0.9}
    inputs = ['close', 'volume']
    timeframes = ['1h']
    rationale = (
        'Sparse volume capitulation reversal: only active when volume exceeds the 90th '
        'percentile over 48 bars — genuine capitulation/euphoria events, not noise. '
        'Within these extreme volume events, the price direction determines the fade: '
        'extreme volume + price rising → greedy top → SHORT. Extreme volume + price '
        'falling → panic bottom → LONG. indicator() gate ensures the signal is zero '
        'during normal volume, creating sparse high-conviction entries. '
        'rank_pct forces symmetric distribution within active periods. '
        'Different from vol-accel (which uses continuous acceleration, not binary extremes).'
    )
    mathematical_formula = (
        'indicator(volume > rolling_quantile(volume, 48, 0.9)) '
        '* -1 '
        '* (rank_pct(pct_change(close, 24), 48) - 0.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
