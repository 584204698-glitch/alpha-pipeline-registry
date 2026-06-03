"""
EfficiencyTimingFadeV3 — Bar Efficiency × Body Bias (CS Rank, Window=48)

Logic (柱体效率择时反转V3):
V2 (robust_zscore + rolling_rank, w=36): OOS ICIR=0.2025, t-stat=1.30, all regimes+.
V3 improvements:
- Cross-sectional rank for body_bias instead of per-symbol rolling rank
  (CS rank is more stable and responsive to cross-sectional relative extremes)
- Window 48 for efficiency zscore (more stable)
- Window 24 for body_bias cs_rank (to avoid lookback dilution)

Cross-sectional rank: at each timestamp, rank body_bias across all symbols.
This captures "which bars are at extremes RELATIVE to peers right now" vs
"which bars are at extremes relative to their own history."

Formula:
-1 * robust_zscore(abs(close-open)/(high-low+1e-8), 48)
    * (cs_rank((2*close-high-low)/(high-low+1e-8)) - 0.5)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from factors.base import FactorRegistry


class EfficiencyTimingFadeV3(FactorRegistry):
    factor_name = "EfficiencyTimingFadeV3"
    parameters = {"eff_window": 48, "body_window": 24}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "V3: Cross-sectional rank for body_bias (captures peer-relative extremes) "
        "and longer efficiency window (48). V2 had OOS ICIR=0.2025 with all regimes "
        "positive — V3 aims to push t-stat above 1.5 via CS rank stability."
    )
    mathematical_formula = (
        "-1 * robust_zscore(abs(close-open)/(high-low+1e-8), 48) "
        "* (cs_rank((2*close-high-low)/(high-low+1e-8)) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index
        ew = self.parameters["eff_window"]

        hl = (data["high"] - data["low"]).clip(lower=eps)
        bar_efficiency = (data["close"] - data["open"]).abs() / hl
        body_bias = (2 * data["close"] - data["high"] - data["low"]) / hl

        # Robust zscore for efficiency
        def _rzs(ser, win):
            roll_med = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).median()
            )
            diff_abs = (ser - roll_med).abs()
            roll_mad = diff_abs.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).median()
            )
            return ((ser - roll_med) / (roll_mad + eps)).fillna(0.0)

        rz_efficiency = _rzs(bar_efficiency, ew)

        # Cross-sectional rank of body_bias per timestamp
        ts = data.index.get_level_values("timestamp")
        unique_ts = ts.unique()
        ts_map = {v: i for i, v in enumerate(unique_ts)}
        ts_codes = np.array([ts_map[t] for t in ts])

        cs_rank = np.zeros(len(idx))
        for ti in range(len(unique_ts)):
            mask = ts_codes == ti
            n = mask.sum()
            if n <= 1:
                continue
            vals = body_bias.values[mask]
            cs_rank[mask] = vals.argsort().argsort() / (n - 1)

        body_cs_rank_c = pd.Series(cs_rank - 0.5, index=idx).fillna(0.0)

        signal = pd.Series(
            -1.0 * rz_efficiency.values * body_cs_rank_c.values, index=idx
        )
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
