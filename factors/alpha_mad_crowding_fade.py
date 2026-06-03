"""
MADCrowdingFade — MAD vol accel + funding sign + volume rank (CrowdingFade structure with MAD).

Key difference from CrowdingFade:
- CrowdingFade: OI delta × funding_sign × vol_rank
- MADCrowdingFade: MAD_vol_accel × funding_sign × vol_rank

Uses MAD instead of rolling_std for robustness to crypto wicks/outliers.
Uses robust_zscore instead of zscore for fat-tailed returns.
Preserves the proven funding_sign + vol_rank structure.
"""
from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADCrowdingFade(FactorRegistry):
    factor_name = "MADCrowdingFade"
    parameters = {"mad_window": 24, "accel_lag": 12, "z_window": 48, "vol_window": 24}
    inputs = ["close", "funding_rate", "volume"]
    timeframes = ["1h", "4h"]
    rationale = (
        "MAD-based vol acceleration captures genuine volatility regime changes while "
        "ignoring wick/outlier noise. Combined with funding direction (sign) and volume "
        "context (rank), this detects when crowding is becoming fragile. "
        "Same proven 3-component structure as CrowdingFade, but with robust statistics."
    )
    mathematical_formula = (
        "-1 * robust_zscore(delta(rolling_mad(pct_change(close, 1), 24), 12), 48) "
        "* sign(funding_rate) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
