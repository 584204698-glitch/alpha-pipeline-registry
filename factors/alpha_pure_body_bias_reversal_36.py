"""
PureBodyBiasReversal36 — Body Bias Reversal with Window=36

Logic (纯实体偏向反转36):
PureBodyBiasReversal24 had OOS ICIR=0.240 (t-stat=1.54) but all regimes negative.
Longer window (36 vs 24) may shift the regime profile by capturing multi-day
bar extremes rather than single-session extremes. The longer baseline could
make high-vol and medium-vol extremes more mean-reverting.

Formula:
-1 * zscore((2*close-high-low)/(high-low+1e-8), 36)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry


class PureBodyBiasReversal36(FactorRegistry):
    factor_name = "PureBodyBiasReversal36"
    parameters = {"window": 36}
    inputs = ["open", "high", "low", "close"]
    timeframes = ["1h"]
    rationale = (
        "Variant of PureBodyBiasReversal24 with window=36. Longer window captures "
        "multi-day bar extremes — may shift regime profile from uniformly negative "
        "to more balanced. The 24-bar version captures single-day extremes which "
        "mean-revert differently across volatility regimes."
    )
    mathematical_formula = (
        "-1 * zscore((2*close-high-low)/(high-low+1e-8), 36)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index
        w = self.parameters["window"]

        hl = (data["high"] - data["low"]).clip(lower=eps)
        body_bias = (2 * data["close"] - data["high"] - data["low"]) / hl

        def _zs(ser, win):
            m = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level="symbol", group_keys=False).transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, pd.NA)
            return ((ser - m) / s).fillna(0.0)

        z_body = _zs(body_bias, w)

        signal = pd.Series(-1.0 * z_body.values, index=idx)
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
