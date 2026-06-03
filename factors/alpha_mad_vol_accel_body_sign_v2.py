"""
MADVolAccelBodySignV2 — Vol-Accel Timing × Body Bias Direction (Sign-Only)

Logic (波动加速×柱体方向符号V2):
V1 (MADVolAccelBodyFade) lost vol-accel power because body_bias zscore 
diluted the vol-accel magnitude. V2 preserves vol-accel strength by using
body_bias only for SIGN determination, not magnitude:

signal = vol_accel_z * sign(body_bias)

When body closes near high (body_bias > 0): bears absorb overhead selling → fade LONG
When body closes near low (body_bias < 0): bulls absorb downside buying → fade SHORT

sign() preserves the vol-accel magnitude (proven ICIR~0.3 family) while 
body_bias only determines direction. Avoids the magnitude dilution issue.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24)
    * sign((2*close-high-low)/(high-low+1e-8))
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from factors.base import FactorRegistry


class MADVolAccelBodySignV2(FactorRegistry):
    factor_name = "MADVolAccelBodySignV2"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24}
    inputs = ["close", "high", "low"]
    timeframes = ["1h"]
    rationale = (
        "V1 (MADVolAccelBodyFade) had OOS ICIR=0.117 because multiplying two "
        "zscores diluted vol-accel power. V2 uses sign(body_bias) only for "
        "direction, preserving full vol-accel magnitude. Vol-accel determines "
        "WHEN to fade; body bias determines WHICH SIDE to fade."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) "
        "* sign((2*close-high-low)/(high-low+1e-8))"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index
        w = 24

        # Vol-accel spine (proven)
        ret = data["close"].groupby(level="symbol", group_keys=False).pct_change().fillna(0.0)
        abs_ret_ma = ret.abs().groupby(level="symbol", group_keys=False).transform(
            lambda s: s.rolling(12, min_periods=2).mean()
        )
        vol_accel = abs_ret_ma.groupby(level="symbol", group_keys=False).diff(6)

        def _zs(ser, win):
            m = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, pd.NA)
            return ((ser - m) / s).fillna(0.0)

        z_vol_accel = _zs(vol_accel, w)

        # Body bias sign: +1 if close > mid, -1 if close < mid
        hl = (data["high"] - data["low"]).clip(lower=eps)
        body_bias = (2 * data["close"] - data["high"] - data["low"]) / hl
        body_sign = np.sign(body_bias.values)
        body_sign[body_sign == 0] = 0.0

        signal = pd.Series(
            -1.0 * z_vol_accel.values * body_sign, index=idx
        )
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
