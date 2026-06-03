"""
OI_MAD_CrowdingFade — OI delta × MAD vol accel × funding sign × volume rank.

Quad interaction: all four signals must agree for a strong signal.
- OI delta: positioning change
- MAD vol accel: robust volatility structure change
- funding sign: cost-of-carry direction
- volume rank: activity context

4-way AND dramatically reduces false signals at the cost of sparsity.
"""
from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OI_MAD_CrowdingFade(FactorRegistry):
    factor_name = "OI_MAD_CrowdingFade"
    parameters = {"oi_lag": 12, "mad_window": 24, "accel_lag": 12, "z_window": 48, "vol_window": 24}
    inputs = ["open_interest", "close", "funding_rate", "volume"]
    timeframes = ["1h", "4h"]
    rationale = (
        "Four-way crowding detection: OI momentum + MAD vol acceleration + funding "
        "direction + volume context. All four must align for a signal. This is a more "
        "stringent version of CrowdingFade, designed to produce fewer but higher-quality "
        "signals that should survive cost deduction better."
    )
    mathematical_formula = (
        "-1 * zscore(delta(open_interest, 12), 48) "
        "* robust_zscore(delta(rolling_mad(pct_change(close, 1), 24), 12), 48) "
        "* sign(funding_rate) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
