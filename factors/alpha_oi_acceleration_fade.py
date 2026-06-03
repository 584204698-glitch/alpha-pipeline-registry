"""
OIAccelerationFade — OI Second-Derivative Reversal Factor

Logic (持仓量加速反转):
When OI is accelerating (second derivative of OI turning sharply),
the market is entering a positioning-regime shift. Combined with elevated
volatility, this signals fragility in the current directional bet.
We fade: accelerating OI buildup during high vol → crowded positions → reversal.

This applies the VolAccelFade framework (second-derivative thinking) to OI
rather than volatility, creating a complementary signal.

Formula:
-1 * zscore(OI_acceleration, 24) * zscore(rolling_std(returns, 12), 24)
where OI_acceleration = delta(delta(open_interest, 4), 5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OIAccelerationFade(FactorRegistry):
    factor_name = "OIAccelerationFade"
    parameters = {"oi_delta_1": 4, "oi_delta_2": 5, "window": 24, "vol_window": 12}
    inputs = ["open_interest", "close"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Second-derivative of OI captures positioning regime shifts. "
        "When OI is accelerating (growing faster and faster) during high volatility, "
        "crowded positions are about to unwind — fade the direction implied by OI flow. "
        "Extends the VolAccelFade second-derivative framework to open interest."
    )
    mathematical_formula = (
        "-1 * zscore(delta(delta(open_interest, 4), 5), 24) "
        "* zscore(rolling_std(pct_change(close, 1), 12), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
