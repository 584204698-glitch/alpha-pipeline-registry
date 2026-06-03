"""
RangeSqueezeBreakout — Range Compression Breakout Factor

Logic (区间压缩突破):
When the price range narrows (compresses) relative to its historical average,
the market is coiling like a spring. Combine with volume confirmation and
recent price direction to predict the breakout direction.

Unlike RangeVolumeCompression (which used reversal -1*), this is a PURE
CONTINUATION factor: compressed range + high volume + trending = breakout
continuation in the same direction.

Formula:
(1 - rank_pct(range_ratio, 24)) * zscore(pct_change(close, 6), 24) * rank_pct(volume, 24)
where range_ratio = (high - low) / rolling_mean(high - low, 48)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class RangeSqueezeBreakout(FactorRegistry):
    factor_name = "RangeSqueezeBreakout"
    parameters = {"squeeze_window": 24, "range_mean_window": 48, "mom_lag": 6, "window": 24}
    inputs = ["high", "low", "close", "volume"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Range compression signals coiling energy. When range is narrow vs history "
        "AND volume remains elevated AND price has direction — the breakout continues "
        "in the trending direction. Unlike prior range factors (all reversal-based), "
        "this is a pure continuation/breakout signal. The squeeze term amplifies "
        "signals when compression is extreme."
    )
    mathematical_formula = (
        "(1 - rank_pct((high - low) / rolling_mean(high - low, 48), 24)) "
        "* zscore(pct_change(close, 6), 24) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
