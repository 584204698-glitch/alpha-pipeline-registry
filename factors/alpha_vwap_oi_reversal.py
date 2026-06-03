"""
VWAPDeviationOIReversal — Mid-Range Price Deviation x OI Flow Reversal

Logic (中位偏离 OI 反转):
When close deviates significantly from the mid-range (proxy for VWAP/equilibrium)
AND open interest is flowing in, the market is stretched with fresh capital entering
at extreme prices. These positions are vulnerable — fade the deviation.

Mid-range = (high + low) / 2 captures the bar's equilibrium, averaged over 24 bars.
Close's deviation from this rolling mid-range signals how far price has drifted
from fair value. Combined with OI delta, we identify crowded trend extensions.

Formula:
-1 * price_deviation_zscore * OI_delta_zscore
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VWAPDeviationOIReversal(FactorRegistry):
    factor_name = "VWAPDeviationOIReversal"
    parameters = {"mid_window": 24, "z_window": 24, "oi_delta_lag": 6}
    inputs = ["high", "low", "close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Price deviation from rolling mid-range (equilibrium proxy) crossed with OI flow. "
        "When price stretches far from fair value while fresh capital enters (OI building), "
        "the positioning is fragile — fade the deviation. Uses (high+low)/2 as mid-range "
        "proxy instead of close-only mean — captures intra-bar equilibrium better."
    )
    mathematical_formula = (
        "-1 * zscore(close / rolling_mean((high + low) / 2, 24) - 1, 24) "
        "* zscore(delta(open_interest, 6), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
