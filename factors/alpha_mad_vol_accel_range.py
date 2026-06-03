"""
MADVolAccelRangeFade — MAD Vol Accel + Range Compression

Logic:
MAD vol-accel (regime transition) × range_rank (compression/expansion).
When vol-accel signals regime transition AND range is extreme (narrow or wide),
the market is primed for reversal. (rank_pct - 0.5) centers the range signal.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 6), 24) * (rank_pct((high-low)/close, 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelRangeFade(FactorRegistry):
    factor_name = "MADVolAccelRangeFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24}
    inputs = ["close", "high", "low"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "MAD vol-accel core paired with range signal. Range (high-low)/close "
        "captures intra-bar price excursion — a different volatility dimension "
        "from the vol-accel's return-based measure. Two complementary perspectives "
        "on regime fragility."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) "
        "* (rank_pct((high - low) / close, 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
