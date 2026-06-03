from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class TailRiskReversal(FactorRegistry):
    """尾部风险反转：用zscore阈值(2.5σ)检测极端收益，rank_pct决定方向。
    仅在极端尾部事件时激活，做反转：极端正收益→做空，极端负收益→做多。
    indicator做稀疏门控，避免非极端区间的噪声。与vol-accel框架完全不同。"""
    factor_name = 'TailRiskReversal'
    parameters = {'window': 48, 'z_threshold': 2.5}
    inputs = ['close']
    timeframes = ['1h']
    rationale = (
        'Tail event reversal: when price returns exceed 2.5 robust standard deviations, '
        'the market has experienced a tail event (panic or euphoria). These extremes '
        'are driven by forced positioning, not rational revaluation → mean-revert. '
        'indicator() gate ensures the factor only fires during genuine tail events, '
        'producing sparse, high-conviction signals. rank_pct determines direction: '
        'extreme positive returns → SHORT, extreme negative → LONG. '
        'Completely different mechanism from vol-accel (which uses gradual acceleration, '
        'not tail thresholds) — ensuring low correlation.'
    )
    mathematical_formula = (
        '-1 * (rank_pct(pct_change(close, 1), 48) - 0.5) '
        '* indicator(abs(zscore(pct_change(close, 1), 48)) > 2.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
