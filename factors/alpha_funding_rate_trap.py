from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class FundingRateTrap(FactorRegistry):
    """资金费率陷阱：费率极端+taker量背离=假极端信号→反向。区分"真反转"和"假极端"。"""
    factor_name = 'FundingRateTrap'
    parameters = {'funding_window': 24, 'taker_window': 12, 'zscore_window': 24}
    inputs = ['funding_rate', 'taker_volume', 'volume']
    timeframes = ['1h', '4h']
    rationale = 'Extreme funding rate is NOT always a reversal signal. When funding is extreme negative (shorts paying) but taker volume is surging (aggressive buying), smart money is absorbing — the funding extreme is a TRAP. Fade the funding extreme when taker flow confirms it\'s a trap. Bidirectional via sign(funding_rate) × taker_ratio divergence.'
    mathematical_formula = '-1 * zscore(funding_rate, 24) * (rank_pct(taker_volume / (volume + 1e-8), 24) - 0.5)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
