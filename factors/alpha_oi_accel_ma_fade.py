from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OIAccelMAFade(FactorRegistry):
    """OI波动加速×价格MA偏离反转：OI-vol-accel(持仓异动) timing + MA偏离方向。
    避免使用price momentum降低与VolAccelFade相关性。OI异动+价格偏离=分歧→反转。"""
    factor_name = 'OIAccelMAFade'
    parameters = {'oi_vol_window': 12, 'accel_lag': 6, 'zscore_window': 24, 'ma_window': 48}
    inputs = ['close', 'open_interest']
    timeframes = ['1h']
    rationale = (
        'OI volatility acceleration (position churn intensity) as timing mechanism, '
        'paired with price-to-MA deviation as directional anchor. OI-vol-accel captures '
        'when positions are being rapidly rotated — building and unwinding — signaling '
        'deep market disagreement. Price/MA deviation (zscored) provides the fade '
        'direction: above MA → positions being built at premium → SHORT; below MA → '
        'positions being dumped at discount → LONG. Price/MA deviation is mean-reverting '
        'by construction, creating a more stationary directional signal than raw price '
        'momentum. Zero correlation with volume-based or close-vol-based factors.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_std(delta(open_interest, 1), 12), 6), 24) '
        '* zscore(close / rolling_mean(close, 48) - 1, 24)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
