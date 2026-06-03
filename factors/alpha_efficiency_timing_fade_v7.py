"""
EfficiencyTimingFadeV7 — Eff Zscore × Body Rank (Window=24)

Logic (柱体效率择时反转V7):
Double-zscore suffered from signal dilution. V7 uses rank_pct-0.5 for body_bias
direction — more stable under fat tails and preserves efficiency timing strength.

Efficiency zscore captures WHEN (conviction bar), body rank captures WHICH SIDE.

Formula:
-1 * zscore(abs(close-open)/(high-low+1e-8), 24)
    * (rank_pct((2*close-high-low)/(high-low+1e-8), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class EfficiencyTimingFadeV7(FactorRegistry):
    factor_name = "EfficiencyTimingFadeV7"
    parameters = {"eff_window": 24, "body_window": 24}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "V7: Eff zscore × body rank (rank_pct-0.5). Replaces body zscore with "
        "rank for stability under crypto fat tails. Efficiency zscore captures "
        "conviction timing; body rank provides symmetric [-0.5,+0.5] direction. "
        "Previous versions had either regime-fail or OOS-fail — V7 targets both."
    )
    mathematical_formula = (
        "-1 * zscore(abs(close-open)/(high-low+1e-8), 24) "
        "* (rank_pct((2*close-high-low)/(high-low+1e-8), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index
        ew, bw = self.parameters["eff_window"], self.parameters["body_window"]

        hl = (data["high"] - data["low"]).clip(lower=eps)
        bar_efficiency = (data["close"] - data["open"]).abs() / hl
        body_bias = (2 * data["close"] - data["high"] - data["low"]) / hl

        def _zs(ser, win):
            m = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, pd.NA)
            return ((ser - m) / s).fillna(0.0)

        z_eff = _zs(bar_efficiency, ew)

        def _rank_centered(ser, win):
            ranked = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
            )
            return ranked - 0.5

        body_rank_c = _rank_centered(body_bias, bw)

        signal = pd.Series(-1.0 * z_eff.values * body_rank_c.values, index=idx)
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
