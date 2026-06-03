from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolumePriceDivergence(FactorRegistry):
    """成交量-价格背离V2：价格创新高/低但成交量萎缩=趋势衰竭→反转。V2修正：移除-1，price↑×vol↓=signal负=做空，price↓×vol↓=signal正=做多，双向自动捕捉假突破。纯量价零OI零费率。"""
    factor_name = 'VolumePriceDivergenceV2'
    parameters = {'price_window': 24, 'vol_window': 12, 'zscore_window': 24}
    inputs = ['close', 'volume']
    timeframes = ['1h', '4h']
    rationale = 'When price makes new highs but volume is shrinking (not confirming the move), the trend is losing fuel. The divergence between price momentum and volume momentum signals exhaustion. Bidirectional: price↑+vol↓→SHORT weak rally; price↓+vol↓→LONG weak selloff. Uses log(volume) to normalize skewed volume distribution. No OI noise.'
    mathematical_formula = 'zscore(pct_change(close, 1), 24) * (rank_pct(log(volume), 24) - 0.5)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
