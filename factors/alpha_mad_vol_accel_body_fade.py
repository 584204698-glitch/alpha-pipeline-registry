"""
MADVolAccelBodyFade — Vol-Accel Timing × OHLC Body Direction

Logic (波动加速×柱体方向反转):
Vol-accel (proven timing mechanism, family ICIR~0.3) tells us WHEN the market
is in a fragile/chaotic regime. Body_bias = (2c-h-l)/(h-l) tells us WHICH SIDE
was dominant within the bar. When vol is accelerating and body is at extreme:
fade the intra-bar direction.

Why OHLC body_bias ≠ price momentum:
- price momentum (pct_change(close,6)) is multi-bar trend
- body_bias is single-bar OHLC internal structure
- They are structurally orthogonal: one bar can have a bullish body during
  a multi-bar downtrend (temporary bounce with absorption)

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 6), 24)
    * zscore((2*close-high-low)/(high-low+1e-8), 24)

Both zscore components centered at 0 → natural signal symmetry.
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class MADVolAccelBodyFade(FactorRegistry):
    factor_name = "MADVolAccelBodyFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24}
    inputs = ["close", "high", "low"]
    timeframes = ["1h"]
    rationale = (
        "Vol-accel from proven MADVolAccelOIRankV6 family as timing mechanism. "
        "OHLC body_bias = (2c-h-l)/(h-l) as directional anchor instead of price "
        "momentum or OI delta. Body bias captures single-bar internal structure "
        "orthogonal to multi-bar price trends. When vol-accel signals chaos, "
        "fade the intra-bar direction."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) "
        "* zscore((2*close-high-low)/(high-low+1e-8), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index

        # Vol-accel: same as MADVolAccelOIRankV6 spine
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

        z_vol_accel = _zs(vol_accel, 24)

        # Body bias: (2c-h-l)/(h-l) ∈ [-1, +1]
        hl = (data["high"] - data["low"]).clip(lower=eps)
        body_bias = (2 * data["close"] - data["high"] - data["low"]) / hl
        z_body = _zs(body_bias, 24)

        signal = pd.Series(-1.0 * z_vol_accel.values * z_body.values, index=idx)
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
