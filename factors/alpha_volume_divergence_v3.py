from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolumePriceDivergenceV3(FactorRegistry):
    """V3: 用量变化(delta log vol)替代量水平，捕捉"量能衰竭"而非"绝对量低"。
    更长窗口(48)提高IS/OOS一致性。纯量价零OI零费率。"""
    factor_name = 'VolumePriceDivergenceV3'
    parameters = {'price_window': 24, 'vol_delta': 6, 'rank_window': 48}
    inputs = ['close', 'volume']
    timeframes = ['1h']
    rationale = (
        'Volume CHANGE divergence: when price is rising but volume growth is '
        'decelerating (delta log vol rank below median), the rally is losing '
        'fuel → SHORT. When price is falling but volume is not accelerating '
        '(delta log vol rank above median), selling is not intensifying '
        '→ LONG. Uses delta(log(volume)) instead of log(volume) level to '
        'capture the RATE of volume change — more predictive than static levels. '
        'Longer 48-bar rank window for regime stability.'
    )
    mathematical_formula = (
        'zscore(pct_change(close, 1), 24) '
        '* (rank_pct(delta(log(volume), 6), 48) - 0.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
