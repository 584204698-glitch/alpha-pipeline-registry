from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class DollarVolAccelMAFade(FactorRegistry):
    """成交额波动加速×价格偏离MA反转：V2修复IS/OOS翻转。用价格偏离均线(zscore of close/MA-1)
    替代rank_pct(close)，更平稳、均值回复性强。DV-accel timing + MA偏离方向。"""
    factor_name = 'DollarVolAccelMAFade'
    parameters = {'dv_window': 12, 'accel_lag': 6, 'zscore_window': 24, 'ma_window': 48}
    inputs = ['close', 'volume']
    timeframes = ['1h']
    rationale = (
        'DV-accel timing with price-to-MA deviation as directional anchor. '
        'Price/MA deviation is mean-reverting by construction — prices oscillate '
        'around their moving average. This creates a more stationary directional '
        'signal than raw price rank, reducing IS/OOS instability. '
        'DV-accel captures chaos regime, price-MA deviation (zscored) tells us '
        'which direction to fade: above MA → SHORT, below MA → LONG. '
        'V1 with rank_pct had IC=0.034 but IS/OOS sign flip; MA deviation should '
        'fix stationarity while keeping the strong timing signal.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_mean(log(volume) * abs(pct_change(close, 1)), 12), 6), 24) '
        '* zscore(close / rolling_mean(close, 48) - 1, 24)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
