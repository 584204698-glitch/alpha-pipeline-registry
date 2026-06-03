from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class LongCrowdingExhaustion(FactorRegistry):
    """多头拥挤衰竭：funding高+OI增+回报转弱+taker衰竭=多头要炸。
    用indicator(x>0)*x实现max(x,0)的非线性门控。"""
    factor_name = 'LongCrowdingExhaustion'
    parameters = {'funding_window': 24, 'oi_window': 6, 'ret_window': 12, 'taker_window': 12}
    inputs = ['close', 'funding_rate', 'open_interest', 'taker_volume', 'volume']
    timeframes = ['1h', '4h']
    rationale = 'When funding is high (longs paying), OI is rising (more leverage entering), but returns are weakening and taker flow is fading — the bullish crowd is exhausted and fragile. Uses indicator-based nonlinear gating: only positive zscore(funding) and zscore(OI_delta) contribute, multiplied by negative return and taker signals. This is NOT a simple funding-reversal bet.'
    mathematical_formula = 'indicator(zscore(funding_rate, 24) > 0) * zscore(funding_rate, 24) * indicator(zscore(delta(open_interest, 6), 24) > 0) * zscore(delta(open_interest, 6), 24) * indicator(-zscore(pct_change(close, 1), 12) > 0) * -zscore(pct_change(close, 1), 12) * indicator(-zscore(taker_volume / (volume + 1e-8), 12) > 0) * -zscore(taker_volume / (volume + 1e-8), 12)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
