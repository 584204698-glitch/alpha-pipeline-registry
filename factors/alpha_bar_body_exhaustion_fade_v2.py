"""
BarBodyExhaustionFadeV2 — Fade extreme bar body with rank-based symmetry

Logic (K线实体极端反转V2):
V1 had IS/OOS sign flip (IS ICIR +0.048, OOS -0.156) suggesting the
body_bias zscore is not stationary across regimes.

V2 replaces zscore with rank_pct-0.5 for the body bias component, which
is more robust to distribution shifts (rank is always uniform). Sign
flipped to +1 (V1 used -1) based on V1's overall negative IC.

Formula:
(rank_pct((2*close-high-low)/(high-low+1e-8), 24) - 0.5) * (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5)

Double rank_pct-0.5 construction: both components are strictly symmetric around 0.
Fade direction determined by interaction sign: when body bias rank and taker rank
agree (both positive or both negative) → extreme → fade.
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class BarBodyExhaustionFadeV2(FactorRegistry):
    factor_name = "BarBodyExhaustionFadeV2"
    parameters = {"window": 24}
    inputs = ["open", "high", "low", "close", "volume", "taker_volume"]
    timeframes = ["1h"]
    rationale = (
        "V2 replaces zscore with rank_pct-0.5 for stationarity and uses double rank "
        "for guaranteed symmetry. When body bias rank and taker rank align → extreme "
        "conviction → fade. Sign flipped from V1 based on empirical IC direction."
    )
    mathematical_formula = (
        "(rank_pct((2*close-high-low)/(high-low+1e-8), 24) - 0.5) "
        "* (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
