"""
OITakerDirectionFade — OI Delta + Taker Ratio + Price Direction Reversal

Logic (持仓量+主动量比率+价格方向反转):
Three clean signals: OI delta (positioning), taker_volume/volume ratio
(aggression direction), sign(price trend) (directional filter).
No volatility component → zero structural correlation with VolAccelFade.

Unlike prior taker+OI factors which used pct_change(taker_volume), this
uses the taker RATIO (taker_volume/volume) which directly measures
buyer-vs-seller aggression, not flow rate.

Formula:
-1 * zscore(delta(open_interest, 6), 24) * zscore(taker_volume/(volume+1e-8), 12) * sign(close - rolling_mean(close, 48))
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OITakerDirectionFade(FactorRegistry):
    factor_name = "OITakerDirectionFade"
    parameters = {"oi_delta": 6, "window": 24, "taker_window": 12, "trend_window": 48}
    inputs = ["open_interest", "taker_volume", "volume", "close"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "OI delta (crowding) × taker ratio (aggression direction) × price trend "
        "(directional filter). No vol component → structurally decorrelated from "
        "VolAccelFade. Uses taker RATIO not taker delta — ratio directly measures "
        "buyer/seller imbalance, not flow rate changes."
    )
    mathematical_formula = (
        "-1 * zscore(delta(open_interest, 6), 24) "
        "* zscore(taker_volume / (volume + 1e-8), 12) "
        "* sign(close - rolling_mean(close, 48))"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
