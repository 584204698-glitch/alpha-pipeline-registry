from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class RangeVolumeCompression(FactorRegistry):
    factor_name = 'RangeVolumeCompression'
    parameters = {
        'price_momentum_lag': 2,
        'price_momentum_zscore': 6,
        'range_mean_window': 24,
        'range_rank_window': 16,
        'volume_rank_window': 16,
    }
    inputs = ['high', 'low', 'close', 'volume']
    timeframes = ['15m', '1h', '4h']
    rationale = (
        'When the intra-bar range (high-low) contracts relative to its historical mean while volume stays '
        'elevated, the market is in a compression/coiling phase. Traders are indecisive and recent price '
        'direction lacks conviction. Fade the recent short-term move — range compression + high volume '
        'is mean-reverting rather than breakout-prone at these timeframes.'
    )
    mathematical_formula = (
        '-1 * zscore(pct_change(close, 2), 6)'
        ' * (1 - rank_pct((high - low) / rolling_mean(high - low, 24), 16))'
        ' * rank_pct(volume, 16)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
