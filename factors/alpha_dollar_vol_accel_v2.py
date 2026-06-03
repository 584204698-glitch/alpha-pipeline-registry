from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class DollarVolAccelFadeV2(FactorRegistry):
    """V2: 用log(volume)替代volume降低与VolAccelFade的相关性。量价共振加速→反转。
    log归一化成交量右尾，使dollar-vol accel更独立于纯vol-accel。"""
    factor_name = 'DollarVolAccelFadeV2'
    parameters = {'dv_window': 12, 'accel_lag': 6, 'zscore_window': 24, 'mom_lag': 6}
    inputs = ['close', 'volume']
    timeframes = ['1h']
    rationale = (
        'Dollar-volume acceleration with log-normalized volume to decorrelate from '
        'pure vol-accel (VolAccelFade). log(volume) compresses the heavy right tail '
        'of crypto volume distribution, making dollar-vol accel more about the '
        'synergy of vol×volume rather than just vol. Captures regime transitions '
        'where BOTH volatility and participation are accelerating — the most fragile '
        'market states where directional bets are most likely to reverse.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_mean(log(volume) * abs(pct_change(close, 1)), 12), 6), 24) '
        '* zscore(pct_change(close, 6), 24)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
