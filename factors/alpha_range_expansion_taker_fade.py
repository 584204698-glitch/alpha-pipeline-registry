"""
RangeExpansionTakerFade — OHLC bar range expansion with taker flow direction

Logic (K线波幅扩张反转):
When the bar range (high-low) expands significantly relative to its rolling mean,
the market is entering a volatile/emotional regime. Combined with extreme taker flow
direction, this signals overextended moves that tend to reverse.

Unlike VolAccelFade (which uses close-to-close return volatility acceleration),
this uses the absolute bar range — capturing intra-bar extremes that close-to-close
returns miss. Should have lower correlation with vol-accel family factors.

Formula:
-1 * zscore((high-low)/(rolling_mean(high-low,48)+1e-8), 24) * (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5)

Timing: bar range expansion (zscore of range/mean_range)
Direction: taker flow rank (rank_pct-0.5, symmetric)
Fade: -1 multiplier
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class RangeExpansionTakerFade(FactorRegistry):
    factor_name = "RangeExpansionTakerFade"
    parameters = {"window": 24, "range_mean_window": 48}
    inputs = ["high", "low", "volume", "taker_volume"]
    timeframes = ["1h"]
    rationale = (
        "Bar range expansion captures intra-bar volatility extremes that close-to-close "
        "returns miss. When range is expanding (emotional market) and taker flow is extreme, "
        "fade the direction. Orthogonal to VolAccelFade which uses close-to-close vol. "
        "Naturally symmetric via rank_pct-0.5 — no indicator gates."
    )
    mathematical_formula = (
        "-1 * zscore((high-low)/(rolling_mean(high-low, 48)+1e-8), 24) "
        "* (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
