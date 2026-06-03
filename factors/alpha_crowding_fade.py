"""
CrowdingFade — Anti-Crowding Reversal Factor

Logic (逆向思维):
When OI growth accelerates AND funding rate reaches extreme AND volume spikes,
the market is overcrowded in one direction. The crowd's positioning becomes
fragile — any small catalyst triggers a mass exit. We fade the crowded direction.

This is a meta-factor: it doesn't predict fundamental value, it predicts when
OTHER traders' consensus is about to break.

Formula:
High OI growth + extreme |funding| + volume spike → crowded long → go short
Low OI growth + extreme |funding| + volume spike → crowded short → go long
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class CrowdingFade(FactorRegistry):
    factor_name = "CrowdingFade"
    parameters = {"lookback": 6, "window": 24, "vol_window": 48}
    inputs = ["open_interest", "funding_rate", "volume", "close"]
    timeframes = ["1h", "4h"]
    rationale = (
        "Fade overcrowded positioning: when OI, funding, and volume all spike together, "
        "the crowd is all-in one direction. The positioning becomes fragile and prone to "
        "sharp reversals — we take the opposite side."
    )
    mathematical_formula = (
        "-1 * zscore(delta(open_interest, 6), 24) "
        "* sign(funding_rate) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
