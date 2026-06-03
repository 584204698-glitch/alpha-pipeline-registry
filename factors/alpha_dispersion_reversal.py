from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula

class CrossSectionalDispersionReversal(FactorRegistry):
    """横截面收益离散度极端+资金费率极端=恐慌/狂热→反转。市场层面的VIX-like信号。"""
    factor_name = 'CrossSectionalDispersionReversal'
    parameters = {'window': 24, 'zscore_window': 48}
    inputs = ['close', 'funding_rate']
    timeframes = ['1h', '4h']
    rationale = 'When cross-sectional return dispersion spikes (all coins moving chaotically) AND funding is at extremes, the market is in panic or euphoria. Both conditions together predict reversal. Uses rolling_std of cross-sectional returns as a market-level fear gauge.'
    mathematical_formula = '-1 * zscore(rolling_std(pct_change(close, 1), 24), 48) * zscore(funding_rate, 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
