from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class DollarVolAccelOIFade(FactorRegistry):
    """成交额波动加速×OI方向：dollar-vol-accel spine + OI delta rank方向。
    使用与VolAccelFade完全不同的方向锚(OI而非price)，保持DV-accel强timing。
    量价共振加速时，OI方向揭示哪方在被强制平仓→反向交易。"""
    factor_name = 'DollarVolAccelOIFade'
    parameters = {'dv_window': 12, 'accel_lag': 6, 'zscore_window': 24, 'oi_delta': 12, 'oi_rank_window': 48}
    inputs = ['close', 'volume', 'open_interest']
    timeframes = ['1h']
    rationale = (
        'Dollar-volume acceleration (proven timing mechanism) × OI delta rank (orthogonal '
        'to price momentum). When dollar-volume is accelerating AND OI is rising '
        '(positions building in chaos) → fade the position direction. When dollar-volume '
        'is accelerating AND OI is falling (positions closing in chaos) → fade the exit. '
        'OI direction replaces price momentum as the directional anchor, dramatically '
        'reducing correlation with VolAccelFade while preserving the DV-accel timing power.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_mean(log(volume) * abs(pct_change(close, 1)), 12), 6), 24) '
        '* (rank_pct(delta(open_interest, 12), 48) - 0.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
