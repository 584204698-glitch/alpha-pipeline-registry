from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class FundingTrapV2(FactorRegistry):
    """资金费率陷阱V2：仅在费率极端负(≤P15)时激活，taker量决定方向。
    费率极端负+主动买入=LONG(陷阱→反向)，费率极端负+无买入=SHORT(真极端→顺势)。
    用indicator做稀疏激活，避免全区间噪声。与所有vol-accel因子低相关。"""
    factor_name = 'FundingTrapV2'
    parameters = {'funding_window': 24, 'taker_window': 24, 'quantile': 0.15}
    inputs = ['funding_rate', 'taker_volume', 'volume']
    timeframes = ['1h']
    rationale = (
        'Sparse activation funding trap: only active when funding_rate is in the '
        'bottom 15% (shorts paying heavily). Within this extreme regime, taker_volume '
        'ratio determines if it is a real extreme (low taker → SHORT with the trend) '
        'or a trap (high taker → smart money absorbing → LONG the reversal). '
        'Binary indicator gate eliminates noise from non-extreme funding periods. '
        'No vol-accel component → naturally low correlation with vol-accel factors.'
    )
    mathematical_formula = (
        'indicator(funding_rate < rolling_quantile(funding_rate, 24, 0.15)) '
        '* (rank_pct(taker_volume / (volume + 1e-8), 24) - 0.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
