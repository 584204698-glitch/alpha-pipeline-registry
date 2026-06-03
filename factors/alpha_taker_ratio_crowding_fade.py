"""
TakerRatioCrowdingFade — OI + Taker Ratio + Volume Crowding Reversal

Logic (主动量比率拥挤反转):
CrowdingFade variant replacing funding_rate with taker buy/sell ratio
as the directional anchor. When OI builds rapidly AND taker ratio is
extreme (dominant buying or selling) AND volume is elevated, the
directional bet is crowded. Fade the taker-driven consensus.

Unlike prior taker factors (all reversal-based with zscore), this uses
crowding-fade structure: zscore(OI) * direction * volume_amplifier.

Formula:
-1 * zscore(delta(open_interest, 6), 24) * (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5) * rank_pct(volume, 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class TakerRatioCrowdingFade(FactorRegistry):
    factor_name = "TakerRatioCrowdingFade"
    parameters = {"oi_delta": 6, "window": 24}
    inputs = ["open_interest", "taker_volume", "volume"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "CrowdingFade variant: OI flow + taker ratio direction + volume. "
        "When OI builds AND taker ratio is extreme AND volume confirms — "
        "the aggressive side is overcrowded. (rank_pct-0.5) provides the "
        "directional anchor centered at zero."
    )
    mathematical_formula = (
        "-1 * zscore(delta(open_interest, 6), 24) "
        "* (rank_pct(taker_volume / (volume + 1e-8), 24) - 0.5) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
