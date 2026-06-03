"""
TakerVolPriceReversal — Taker-Volume-Volatility-Price Triple Reversal

Logic (主动量波动率价格三重反转):
When taker buy ratio is extreme, volatility regime is extreme, AND price has moved
significantly, the market is dominated by aggressive directional traders in a
high-uncertainty environment. These conditions create fragile positioning —
fade the recent move.

All three components are z-scored for symmetry and to avoid indicator-gate cliffs.
No funding rate or OI involved — purely microstructure (taker flow + vol regime + price).

Formula:
-1 * taker_ratio_zscore * vol_regime_zscore * momentum_zscore
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class TakerVolPriceReversal(FactorRegistry):
    factor_name = "TakerVolPriceReversal"
    parameters = {"taker_window": 24, "vol_window": 24, "vol_z_window": 48, "mom_window": 12, "mom_lag": 3}
    inputs = ["taker_volume", "volume", "close"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Triple interaction of taker flow, volatility regime, and price momentum — "
        "all z-scored, no indicator gates. When aggressive traders dominate in extreme "
        "volatility while price trends, the positioning is fragile and prone to reversal. "
        "Pure microstructure: no funding rate, no OI — novel combination."
    )
    mathematical_formula = (
        "-1 * zscore(taker_volume / (volume + 1e-8), 24) "
        "* zscore(rolling_std(pct_change(close, 1), 24), 48) "
        "* zscore(pct_change(close, 3), 12)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
