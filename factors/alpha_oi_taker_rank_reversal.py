"""
OITakerRankReversal — OI Rank × Taker Rank Reversal

Logic (OI排名×主动量排名反转):
Pure 2-way symmetric interaction. No body_bias, no indicator gates.
Both components use rank_pct-0.5 → guaranteed symmetry [-0.25, +0.25].

When OI is building (OI delta rank > 0.5) AND taker flow is aggressive
(taker rank > 0.5) → positions being established with conviction → fade.

Simple, clean, and orthogonal to OHLC-only factors.

Formula:
-1 * (rank_pct(delta(open_interest, 6), 24) - 0.5)
    * (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from factors.base import FactorRegistry


class OITakerRankReversal(FactorRegistry):
    factor_name = "OITakerRankReversal"
    parameters = {"window": 24, "oi_delta": 6}
    inputs = ["volume", "taker_volume", "open_interest"]
    timeframes = ["1h"]
    rationale = (
        "Double rank-centered 2-way interaction: OI delta rank × taker ratio rank. "
        "Both symmetric centered at 0. When OI is building aggressively AND taker "
        "flow is extreme → conviction positions → fade. No OHLC dependency, fully "
        "orthogonal to body_bias/bar_structure factors."
    )
    mathematical_formula = (
        "-1 * (rank_pct(delta(open_interest, 6), 24) - 0.5) "
        "* (rank_pct(taker_volume/(volume+1e-8), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index
        w = self.parameters["window"]
        di = self.parameters["oi_delta"]

        taker_ratio = data["taker_volume"] / (data["volume"].clip(lower=eps))
        taker_ratio[taker_ratio > 2.0] = 2.0

        oi_delta = data["open_interest"].groupby(level="symbol", group_keys=False).diff(di)

        def _rank_centered(ser, win):
            ranked = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
            )
            return ranked - 0.5

        oi_rank_c = _rank_centered(oi_delta, w)
        taker_rank_c = _rank_centered(taker_ratio, w)

        signal = pd.Series(
            -1.0 * oi_rank_c.values * taker_rank_c.values, index=idx
        )
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
