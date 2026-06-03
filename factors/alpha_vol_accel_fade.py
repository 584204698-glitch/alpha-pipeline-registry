"""
VolAccelFade — Volatility Acceleration Mean-Reversion Factor

Logic (波动率加速反转):
When volatility is accelerating (vol-of-vol rising), the market is entering
a chaotic regime where directional bets are fragile. Fade the recent price move:
trends formed during accelerating volatility tend to reverse sharply.

This is the second derivative of volatility — not just high vol, but vol that's
rising faster and faster. This captures regime transitions BEFORE they complete,
giving earlier entry than static vol-threshold approaches.

Formula:
-1 * vol_accel * price_momentum
where vol_accel = zscore(delta(rolling_std(returns), 6), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolAccelFade(FactorRegistry):
    factor_name = "VolAccelFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24, "mom_lag": 6}
    inputs = ["close"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Volatility acceleration captures regime transitions before they complete. "
        "When volatility is accelerating (vol-of-vol rising), directional trends are "
        "fragile — fade the recent price move. Unlike static vol filters, this uses "
        "the rate of change of volatility for earlier, more precise signals."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_std(pct_change(close, 1), 12), 6), 24) "
        "* zscore(pct_change(close, 6), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
