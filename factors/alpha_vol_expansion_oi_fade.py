"""
VolExpansionOIFade — Vol Expansion Ratio × OI Flow Reversal

Logic (波动率扩张+持仓量反转):
Instead of vol-of-vol (second derivative), use VOL EXPANSION RATIO:
short-term vol / long-term vol. When short-term vol spikes above its
long-term baseline, the market is entering a fragile regime. Combined
with OI delta to confirm crowd positioning.

The vol expansion ratio is structurally different from vol-of-vol
(delta of rolling std), producing a signal orthogonal to VolAccelFade
while still capturing regime transitions.

Formula:
-1 * zscore(rolling_std(pct_change(close,1), 6) / rolling_std(pct_change(close,1), 24) - 1, 24) * zscore(delta(open_interest, 6), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolExpansionOIFade(FactorRegistry):
    factor_name = "VolExpansionOIFade"
    parameters = {"short_vol": 6, "long_vol": 24, "window": 24, "oi_delta": 6}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Vol expansion ratio (short-vol / long-vol) captures regime shifts "
        "without the second-derivative computation that correlates with VolAccelFade. "
        "When short-term vol exceeds baseline AND OI is building → fade the crowd. "
        "Structurally orthogonal to VolAccelFade's delta-of-rolling-std approach."
    )
    mathematical_formula = (
        "-1 * zscore(rolling_std(pct_change(close, 1), 6) / (rolling_std(pct_change(close, 1), 24) + 1e-8) - 1, 24) "
        "* zscore(delta(open_interest, 6), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
