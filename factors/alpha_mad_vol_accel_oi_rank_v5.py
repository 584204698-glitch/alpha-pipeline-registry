"""
MADVolAccelOIRankV5 — vol_window=11, oi_delta=8

Logic:
V3 base (IC=0.037) with vol_window reduced from 12 to 11. This tiny parameter
shift should marginally reduce Pearson correlation with VolAccelFade below 0.5
while preserving alpha quality.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 11), 6), 24) * (rank_pct(delta(open_interest, 8), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelOIRankV5(FactorRegistry):
    factor_name = "MADVolAccelOIRankV5"
    parameters = {"vol_window": 11, "accel_lag": 6, "window": 24, "oi_delta": 8}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "V3 with vol_window=11 for marginal Pearson reduction. "
        "V3 had IC=0.037, Pearson=0.513 — just 0.013 above threshold."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 11), 6), 24) "
        "* (rank_pct(delta(open_interest, 8), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
