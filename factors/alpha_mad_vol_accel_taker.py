"""
MADVolAccelTakerRatio — MAD Vol Accel + Taker Ratio

Logic:
MAD vol-accel paired with taker buy/sell ratio. When vol-accel signals
regime transition AND taker ratio is extreme → fade the aggressive direction.
Uses (rank_pct-0.5) for balanced signals.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 6), 24) * (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelTakerRatio(FactorRegistry):
    factor_name = "MADVolAccelTakerRatio"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24}
    inputs = ["close", "taker_volume", "volume"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "MAD vol-accel core with taker ratio as directional anchor. "
        "When vol regime is shifting AND taker aggression is extreme → fade. "
        "Two clean, independent signals."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) "
        "* (rank_pct(taker_volume / (volume + 1e-8), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
