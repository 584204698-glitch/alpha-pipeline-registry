from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class OIVolumeRatioReversal(FactorRegistry):
    factor_name = 'OIVolumeRatioReversal'
    parameters = {
        'oi_vol_ratio_smooth': 12,
        'ratio_zscore_window': 24,
        'price_momentum_lag': 3,
        'price_momentum_zscore': 12,
    }
    inputs = ['open_interest', 'volume', 'close']
    timeframes = ['15m', '1h', '4h']
    rationale = (
        'OI/Volume turnover ratio captures position churn speed. '
        'High ratio (slow turnover, accumulation) combined with trending price = conviction move '
        'that will continue. Low ratio (fast turnover, distribution) with trending price = weak hands '
        'churning, the move will reverse. The interaction of turnover speed and price direction '
        'produces asymmetric mean-reversion signals.'
    )
    mathematical_formula = (
        '-1 * zscore(rolling_mean(open_interest / (volume + 1), 12), 24)'
        ' * zscore(pct_change(close, 3), 12)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
