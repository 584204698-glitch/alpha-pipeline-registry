from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class VolumeTakerDivergence(FactorRegistry):
    factor_name = 'VolumeTakerDivergence'
    parameters = {
        'price_momentum_lag': 6,
        'price_momentum_zscore': 24,
        'volume_delta_lag': 6,
        'volume_delta_zscore': 24,
        'taker_ratio_window': 24,
    }
    inputs = ['close', 'volume', 'taker_volume']
    timeframes = ['15m', '1h', '4h']
    rationale = (
        'When total volume grows but taker participation shrinks, the market is driven by passive orders '
        'rather than aggressive traders. Combined with a trending price, this divergence signals '
        'low-conviction moves that are likely to reverse.'
    )
    mathematical_formula = (
        '-1 * zscore(pct_change(close, 6), 24)'
        ' * zscore(delta(volume, 6), 24)'
        ' * (1 - rank_pct(taker_volume / volume, 24))'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
