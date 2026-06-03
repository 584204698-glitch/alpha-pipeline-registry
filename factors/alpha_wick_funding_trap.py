"""
WickFundingTrap — OHLC wick-body ratio × funding extreme × OI confirmation

Logic (影线资金费率陷阱):
When the bar has a large wick relative to body and funding is extreme,
positions are being trapped:
- Large upper wick + extreme positive funding → longs trapped → SHORT
- Large lower wick + extreme negative funding → shorts trapped → LONG

Wick-body ratio = abs(high-low)/abs(close-open+1e-8) → larger values = more wick, less body.
Combined with funding rate direction (which side is paying/crowded) and OI changes.

Formula:
zscore(abs(high-low)/abs(close-open+1e-8), 24) * zscore(funding_rate, 24) * (rank_pct(delta(open_interest, 6), 24) - 0.5)

Wick ratio zscore: when large → trapped positions exist
Funding zscore: which side is crowded (+ = longs paying → short trap likely)
OI rank: confirmation that positions are being built
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class WickFundingTrap(FactorRegistry):
    factor_name = "WickFundingTrap"
    parameters = {"window": 24, "oi_delta": 6}
    inputs = ["open", "high", "low", "close", "open_interest", "funding_rate"]
    timeframes = ["1h"]
    rationale = (
        "Wick-body ratio (trapped positions) × funding direction (crowded side) × "
        "OI rank (position building). When wicks are large (trapped) and funding "
        "signals crowded side while OI builds → trap confirmed → trade opposite. "
        "Uses rank_pct-0.5 for OI symmetry + zscore(funding) for natural centering."
    )
    mathematical_formula = (
        "zscore(abs(high-low)/abs(close-open+1e-8), 24) "
        "* zscore(funding_rate, 24) "
        "* (rank_pct(delta(open_interest, 6), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
