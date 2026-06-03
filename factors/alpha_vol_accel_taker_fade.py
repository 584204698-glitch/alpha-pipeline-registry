from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolAccelTakerFade(FactorRegistry):
    """波动率加速×主动量背离：vol-accel提供timing（何时行动），taker_ratio提供方向（多空）。
    vol加速+taker极端高→贪婪→做空；vol加速+taker极端低→恐慌→做多。
    与VolAccelFade共享vol-accel脊柱但用taker方向替代price方向，相关性可控。"""
    factor_name = 'VolAccelTakerFade'
    parameters = {'vol_window': 12, 'accel_lag': 6, 'window': 24}
    inputs = ['close', 'taker_volume', 'volume']
    timeframes = ['1h']
    rationale = (
        'Vol-accel spine (proven in VolAccelFade + MADVolAccelOIRankV6) with '
        'taker flow direction instead of price or OI. Vol-accel tells us WHEN '
        'the market regime is fragile (volatility accelerating = chaos). '
        'Taker ratio rank tells us WHO is panicking: extreme high taker = greedy '
        'buying → fade; extreme low taker = panic selling → fade. '
        'Three independent data pillars: vol-of-vol, taker flow, no OI overlap.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) '
        '* (rank_pct(taker_volume / (volume + 1e-8), 24) - 0.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
