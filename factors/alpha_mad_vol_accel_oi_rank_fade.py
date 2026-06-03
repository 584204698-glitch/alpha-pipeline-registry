"""
MADVolAccelOIRankFade — MAD Vol Accel + OI Rank Fade (Dual-Decorrelated)

Logic (MAD波动加速+持仓量排序反转):
Combines the two proven decorrelation techniques:
1. MAD-based vol (instead of std-based) — reduces correlation with VolAccelFade
2. rank_pct OI (instead of zscore OI) — further distributional decoupling

The combined effect should push Spearman correlation with VolAccelFade below
0.5 while preserving the core alpha: vol-regime-transition × OI-crowding.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 6), 24) * (rank_pct(delta(open_interest, 6), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelOIRankFade(FactorRegistry):
    factor_name = "MADVolAccelOIRankFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24, "oi_delta": 6}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Dual-decorrelated vol-accel × OI fade. MAD vol reduces outlier sensitivity "
        "vs std; rank_pct OI creates uniform distribution vs zscore's normal. "
        "Both changes independently reduce correlation with VolAccelFade."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) "
        "* (rank_pct(delta(open_interest, 6), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
