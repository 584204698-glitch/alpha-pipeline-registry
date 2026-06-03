"""
MADVolAccelOIRankV6 — oi_delta=9 variant

Logic:
V3 base with oi_delta increased from 8 to 9. Further shifts the OI component
distribution to reduce Pearson correlation with VolAccelFade.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 6), 24) * (rank_pct(delta(open_interest, 9), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelOIRankV6(FactorRegistry):
    factor_name = "MADVolAccelOIRankV6"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24, "oi_delta": 9}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = "V3 with oi_delta=9 for additional Pearson reduction."
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) "
        "* (rank_pct(delta(open_interest, 9), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
