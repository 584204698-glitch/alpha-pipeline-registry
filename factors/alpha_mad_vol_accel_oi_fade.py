"""
MADVolAccelOIFade — MAD-Based Vol Accel + OI Flow Fade

Logic (绝对偏离波动加速+持仓量反转):
Replaces standard-deviation-based vol with Mean Absolute Deviation (MAD).
MAD measures volatility using absolute returns instead of squared returns,
making it less sensitive to outlier bars and producing a structurally
different vol-accel signal.

The MAD-based vol-accel is decorrelated from VolAccelFade's std-based
vol-accel while capturing the same regime-transition dynamics.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 6), 24) * zscore(delta(open_interest, 6), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelOIFade(FactorRegistry):
    factor_name = "MADVolAccelOIFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24, "oi_delta": 6}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "MAD (mean absolute deviation) vol is structurally different from std vol: "
        "absolute vs squared returns, less outlier-sensitive, different distribution. "
        "MAD-based vol-accel captures regime transitions while being decorrelated "
        "from VolAccelFade's std-based vol-accel."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) "
        "* zscore(delta(open_interest, 6), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
