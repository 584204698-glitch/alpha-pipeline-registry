"""
BarBodyExhaustionFade — Fade extreme OHLC bar body positions

Logic (K线实体极端反转):
When the bar body position (close relative to bar range) reaches an extreme
and taker flow confirms the move, the directional move is overdone — fade it.

Uses zscore of body_bias for timing (how extreme is the bar position),
and (rank_pct(taker_vol/vol, 24)-0.5) for directional confirmation.
The -1 multiplier ensures fading: extreme bullish bars → sell, extreme bearish → buy.

Formula:
-1 * zscore((2*close-high-low)/(high-low+1e-8), 24) * (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5)

Both zscore (mean-centered) and rank_pct-0.5 are naturally symmetric,
ensuring balanced long/short signals without indicator gates.
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BarBodyExhaustionFade(FactorRegistry):
    factor_name = "BarBodyExhaustionFade"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close", "volume", "taker_volume"]
    timeframes = ["1h"]
    rationale = (
        "When bar body reaches extreme position (close at extreme of range) and taker flow "
        "confirms, the move is overdone — fade it. Uses naturally symmetric components: "
        "zscore(body_bias) centered at 0 + rank_pct-0.5 centered at 0. "
        "No indicator gates needed — pure exhaustion reversal."
    )
    mathematical_formula = (
        "-1 * zscore((2*close-high-low)/(high-low+1e-8), 24) "
        "* (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
