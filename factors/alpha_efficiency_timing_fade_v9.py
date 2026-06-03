"""
EfficiencyTimingFadeV9 — Eff=48 × Body=24 (Wider Timing, Sharper Direction)

Logic (柱体效率择时反转V9):
V8 (eff=48, body=36): OOS ICIR=0.185 (t-stat=1.18), all regimes positive.
V4 (eff=24, body=36): OOS ICIR=0.167, all regimes positive.
→ Wider efficiency window INCREASES OOS.
V5 (eff=24, body=24): OOS ICIR=0.132, regimes positive but weaker.
→ Body=36 better than body=24 for OOS with eff=24.

V9 tries eff=48 (proven wider is better) with body=24 (untested).
Rationale: wide efficiency gives stable timing; shorter body window may
provide sharper directional signal that doesn't get diluted.

Formula:
-1 * zscore(abs(close-open)/(high-low+1e-8), 48)
    * zscore((2*close-high-low)/(high-low+1e-8), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class EfficiencyTimingFadeV9(FactorRegistry):
    factor_name = "EfficiencyTimingFadeV9"
    parameters = {"eff_window": 48, "body_window": 24}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "V9: eff_window=48 (proven wider is better) × body_window=24 "
        "(sharper direction signal). V8 (48/36) had OOS ICIR=0.185, "
        "regime+. V5 (24/24) had OOS ICIR=0.132, regime+. "
        "Hypothesis: 48/24 preserves regime+ structure with sharper signals."
    )
    mathematical_formula = (
        "-1 * zscore(abs(close-open)/(high-low+1e-8), 48) "
        "* zscore((2*close-high-low)/(high-low+1e-8), 24)"
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
        z_body = _zs(body_bias, bw)

        signal = pd.Series(-1.0 * z_eff.values * z_body.values, index=idx)
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
