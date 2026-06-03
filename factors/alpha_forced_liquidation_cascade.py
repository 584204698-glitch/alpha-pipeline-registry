from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class ForcedLiquidationCascade(FactorRegistry):
    """风控机器人被迫平仓V2：robust_zscore捕捉价格暴跌+OI骤降→清算压力耗尽后超卖反弹。V2修正方向：-1移除，价格跌+OI降=product正=LONG反弹。"""
    factor_name = 'ForcedLiquidationCascadeV2'
    parameters = {'price_window': 12, 'oi_window': 6, 'zscore_window': 24}
    inputs = ['close', 'open_interest']
    timeframes = ['1h', '4h']
    rationale = 'When price collapses (robust_zscore detects extreme drawdown) AND OI drops sharply, leveraged longs are being force-liquidated. The cascade exhausts as positions clear — creating oversold conditions primed for a bounce. Both components negative → product positive → LONG the reversal. robust_zscore uses median/MAD to handle fat-tailed crypto returns.'
    mathematical_formula = 'robust_zscore(pct_change(close, 1), 12) * zscore(delta(open_interest, 6), 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
