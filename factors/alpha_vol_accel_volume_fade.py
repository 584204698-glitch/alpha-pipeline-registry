"""
VolAccelVolumeDirectionFade — Vol Accel + Volume + Direction Reversal

Logic (波动率加速+量价方向反转):
VolAccelFade extended with volume and directional confirmation.
Vol-of-vol signals regime transition. Volume rank amplifies conviction.
Direction term (sign of price-vs-mean) provides orientation.

When vol is accelerating (regime fragile) AND volume is high (liquidity
present) AND price is trending away from mean — fade the directional move.
Three independent data sources, zero zscore dilution on direction.

Formula:
-1 * zscore(delta(rolling_std(pct_change(close, 1), 12), 6), 24) * rank_pct(volume, 24) * sign(close - rolling_mean(close, 48))
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolAccelVolumeDirectionFade(FactorRegistry):
    factor_name = "VolAccelVolumeDirectionFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24, "trend_window": 48}
    inputs = ["close", "volume"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "VolAccelFade + CrowdingFade hybrid. Vol-of-vol timing × volume conviction "
        "× price-trend direction. sign() preserves directional magnitude (no zscore "
        "dilution). Three clean, independent sources."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_std(pct_change(close, 1), 12), 6), 24) "
        "* rank_pct(volume, 24) "
        "* sign(close - rolling_mean(close, 48))"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
