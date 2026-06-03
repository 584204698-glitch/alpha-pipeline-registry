from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Rangeexpansiontakeroi(FactorRegistry):
    factor_name = 'RangeExpansionTakerOI'
    parameters = {'range_pct_lag': 1, 'range_zscore_window': 12, 'taker_ratio_rank_window': 24, 'oi_delta_lag': 6}
    inputs = ['high', 'low', 'taker_volume', 'volume', 'open_interest']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Exploiting range acceleration when taker volume is dominant and OI trend is declining.'
    mathematical_formula = 'zscore(pct_change(high - low, 1), 12) * (rank_pct(taker_volume / volume, 24) - 0.5) * sign(delta(open_interest, 6)) * -1'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
