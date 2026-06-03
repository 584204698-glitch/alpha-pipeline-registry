from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula

class OIVolumeDivergence(FactorRegistry):
    """OI攀升但成交量萎缩 = 无信念的被动建仓 → 持仓脆弱 → 趋势反转"""
    factor_name = 'OIVolumeDivergence'
    parameters = {'oi_delta_lag': 6, 'zscore_window': 24, 'trend_window': 48}
    inputs = ['open_interest', 'volume', 'close']
    timeframes = ['1h', '4h']
    rationale = 'OI growing briskly while volume shrinks = passive positioning without conviction. The crowded side is fragile and prone to reversal. Opposite of CrowdingFade (which uses high volume as crowd signal).'
    mathematical_formula = '-1 * zscore(delta(open_interest, 6), 24) * (1 - rank_pct(volume, 24)) * sign(close - rolling_mean(close, 48))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
