from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula

class FundingTermStructure(FactorRegistry):
    """短期资金费率极端偏离长期均值 = 暂时性过度反应 → 均值回归"""
    factor_name = 'FundingTermStructure'
    parameters = {'short_window': 6, 'long_window': 48, 'vol_rank_window': 24}
    inputs = ['funding_rate', 'volume']
    timeframes = ['1h', '4h']
    rationale = 'Short-term funding zscore far from long-term zscore = temporary panic/greed. High conviction when volume confirms.'
    mathematical_formula = '-1 * (zscore(funding_rate, 6) - zscore(funding_rate, 48)) * rank_pct(volume, 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
