"""
CrossSectionalMomentumFade — Pure price cross-sectional momentum reversal.

Completely orthogonal to vol/OI/funding/taker families.
Uses only close prices. Fades extreme relative performers.
"""
from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class CrossSectionalMomentumFade(FactorRegistry):
    factor_name = "CrossSectionalMomentumFade"
    parameters = {"mom_lag": 6, "window": 24}
    inputs = ["close"]
    timeframes = ["1h"]
    rationale = (
        "Pure price momentum reversal in cross-section. Coins that have strongly "
        "outperformed over the mom_lag window tend to revert relative to the "
        "universe. No OI, funding, vol, or volume data — completely orthogonal "
        "to all existing factor families."
    )
    mathematical_formula = (
        "-1 * (rank_pct(pct_change(close, 6), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
