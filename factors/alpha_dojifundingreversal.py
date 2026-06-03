from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Dojifundingreversal(FactorRegistry):
    factor_name = 'DojiFundingReversal'
    parameters = {'candle_window': 24, 'volume_rank_window': 24}
    inputs = ['open', 'high', 'low', 'close', 'volume', 'funding_rate']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Fades indecision candles (low body/range) when funding rate is extreme and volume is expanding rapidly, capturing reversal anticipations.'
    mathematical_formula = '-1 * zscore((close - open) / (high - low + 1e-8), 24) * sign(funding_rate) * rank_pct(delta(volume,1), 24)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
