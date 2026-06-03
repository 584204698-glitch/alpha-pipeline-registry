"""
MADVolAccelFundingFade — MAD Vol Accel + Funding Direction + Volume

Logic:
MAD vol-accel (regime transition) × funding_rate sign (sentiment direction)
× volume rank (conviction). Three independent data sources in the proven
vol-accel framework. No OI component — replaces with funding+volume.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 6), 24) * sign(funding_rate) * rank_pct(volume, 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelFundingFade(FactorRegistry):
    factor_name = "MADVolAccelFundingFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24}
    inputs = ["close", "funding_rate", "volume"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "MAD vol-accel core with funding_rate direction and volume conviction. "
        "Three clean, independent data sources. Vol-accel provides regime timing, "
        "funding sign provides directional anchor, volume rank amplifies."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) "
        "* sign(funding_rate) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
