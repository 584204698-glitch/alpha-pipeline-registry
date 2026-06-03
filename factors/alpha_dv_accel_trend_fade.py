from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class DollarVolAccelTrendFade(FactorRegistry):
    """DV-accel timing × medium-term TREND方向(24-bar，非短期6-bar momentum)。
    用rank_pct强制对称(~50% long/short)。24-bar trend与VolAccelFade的6-bar momentum
    相关性大幅降低。DV-accel(IC=0.042已验证)×trend强度方向。"""
    factor_name = 'DollarVolAccelTrendFade'
    parameters = {'dv_window': 12, 'accel_lag': 6, 'zscore_window': 24, 'trend_bars': 24, 'rank_window': 48}
    inputs = ['close', 'volume']
    timeframes = ['1h']
    rationale = (
        'DV-accel timing (proven IC=0.042, ICIR=0.338) with MEDIUM-TERM trend direction '
        '(24-bar ≈ 1 day in 1h bars) instead of short-term momentum (6-bar). The 24-bar '
        'trend captures the broader directional regime rather than recent noise. '
        'rank_pct forces symmetric distribution (~50% long/short) solving the persistent '
        '20% long-ratio problem. 24-bar pct_change is fundamentally different from '
        'VolAccelFade\'s 6-bar momentum, targeting <0.5 correlation while keeping '
        'the powerful DV-accel timing spine intact.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_mean(log(volume) * abs(pct_change(close, 1)), 12), 6), 24) '
        '* (rank_pct(pct_change(close, 24), 48) - 0.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
