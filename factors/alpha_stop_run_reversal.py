"""
StopRunReversal — Stop-Hunt Fake Breakout Reversal

Logic (逆向思维):
When price approaches a recent swing high/low with elevated taker volume, this
often triggers clustered stop-loss orders. The initial breakout through the level
is a "stop run" — it executes standing orders, exhausts the directional flow,
and then reverses.

This exploits the STRUCTURAL behavior of stop-loss clustering, not any prediction
about asset value. Stop-loss orders are mechanically price-insensitive — they
convert to market orders when triggered, creating a brief cascade followed by
a vacuum that pulls price back.

Formula:
Price near rolling max/min (at extreme) + high taker volume → stop run → reverse
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class StopRunReversal(FactorRegistry):
    factor_name = "StopRunReversal"
    parameters = {"window": 48, "taker_window": 24}
    inputs = ["close", "taker_volume", "volume"]
    timeframes = ["1h", "4h"]
    rationale = (
        "Stop-loss orders cluster near recent swing highs/lows. When price reaches these "
        "levels with elevated taker volume, a stop cascade triggers a brief overshoot "
        "followed by reversal. Fade the stop run."
    )
    mathematical_formula = (
        "-1 * (close - rolling_mean(close, 48)) / (rolling_std(close, 48) + 1e-8) "
        "* zscore(taker_volume / (volume + 1e-8), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
