from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class DollarVolAccelTakerFade(FactorRegistry):
    """成交额波动加速×主动量方向：dollar-vol-accel spine + taker ratio rank方向。
    量价共振加速(timing)×主动量极端(direction)→情绪化交易→反转。
    与VolAccelFade用price方向完全不同，与MADVolAccelOIRank用OI也不同。"""
    factor_name = 'DollarVolAccelTakerFade'
    parameters = {'dv_window': 12, 'accel_lag': 6, 'zscore_window': 24}
    inputs = ['close', 'volume', 'taker_volume']
    timeframes = ['1h']
    rationale = (
        'Dollar-volume acceleration (proven timing mechanism) × taker ratio rank '
        '(direction from aggressive flow, orthogonal to both price momentum and OI). '
        'When dollar-volume is accelerating (chaos regime) and taker_ratio is extreme '
        'high → FOMO buying → SHORT. When dollar-volume accelerating and taker_ratio '
        'extreme low → panic selling → LONG. Three clean pillars: DV-accel (when), '
        'taker flow (what direction), -1 (fade it). Zero overlap with price-based or '
        'OI-based directional anchors.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_mean(log(volume) * abs(pct_change(close, 1)), 12), 6), 24) '
        '* (rank_pct(taker_volume / (volume + 1e-8), 24) - 0.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
