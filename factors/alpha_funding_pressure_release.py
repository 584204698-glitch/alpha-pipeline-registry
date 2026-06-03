from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class FundingPressureRelease(FactorRegistry):
    """费率压力释放：高funding+OI降+价格跌=杠杆多头正在出逃→先跌后弹。
    分为两段：早期(OI还在增)偏空，后期(OI已降)偏反弹。"""
    factor_name = 'FundingPressureRelease'
    parameters = {'funding_window': 24, 'oi_window': 6, 'ret_window': 12}
    inputs = ['funding_rate', 'open_interest', 'close']
    timeframes = ['1h', '4h']
    rationale = 'When funding is positive (longs paying) but price is falling AND OI is dropping — leveraged longs are exiting. Early phase (OI still rising): pressure continues, bearish. Late phase (OI already declining): longs are gone, bounce likely. The sign(-delta_OI) term naturally separates the two phases: when OI is still rising, the signal flips to capture the continuation phase.'
    mathematical_formula = '-zscore(funding_rate, 24) * zscore(-pct_change(close, 1), 24) * sign(-pct_change(open_interest, 6))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
