"""
MADVolAccelSignFade — MAD Vol Accel + Price Direction

Logic:
Simplest MAD vol-accel variant: vol-accel × sign(price trend). 
When vol regime is shifting AND price is trending → fade with sign as
clean ±1 directional anchor. Two clean signals, structurally distinct
from VolAccelFade's zscore(momentum).

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 6), 24) * sign(close - rolling_mean(close, 48))
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelSignFade(FactorRegistry):
    factor_name = "MADVolAccelSignFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24, "trend_window": 48}
    inputs = ["close"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Simplest vol-accel fade: MAD vol-accel × sign(price-mean). "
        "sign preserves ±1 directional strength without zscore dilution. "
        "Structurally distinct from VolAccelFade's zscore(momentum) approach."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) "
        "* sign(close - rolling_mean(close, 48))"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
