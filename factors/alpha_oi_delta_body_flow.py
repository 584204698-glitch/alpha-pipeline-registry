"""
OIDeltaBodyFlow — OI delta rank × bar body bias

Logic (OI变化×K线实体):
Uses symmetric rank_pct-0.5 for OI delta as a continuous, centered signal
(instead of binary indicator gate used in V1). Combined with bar body bias
for directional context.

When OI is rising sharply AND bar closes near the high → new longs entering,
follow the flow (LONG). When OI is dropping AND bar closes near the low →
positions exiting, follow the flow (SHORT).

Formula:
(rank_pct(delta(open_interest, 6), 24) - 0.5) * (2*close-high-low)/(high-low+1e-8)

Both components are naturally symmetric around 0, ensuring balanced long/short.
No zscore needed — rank_pct already provides cross-sectional normalization.
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OIDeltaBodyFlow(FactorRegistry):
    factor_name = "OIDeltaBodyFlow"
    parameters = {"window": 24, "oi_delta": 6}
    inputs = ["open", "high", "low", "close", "open_interest"]
    timeframes = ["1h"]
    rationale = (
        "Symmetric OI delta (rank_pct-0.5) × symmetric body bias ((2c-h-l)/(h-l)). "
        "Both components centered at 0 — no indicator gates needed. When OI is building "
        "AND bar closes near high → follow longs. When OI is dropping AND bar closes "
        "near low → follow shorts. Pure position flow following."
    )
    mathematical_formula = (
        "(rank_pct(delta(open_interest, 6), 24) - 0.5) "
        "* (2*close-high-low)/(high-low+1e-8)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
