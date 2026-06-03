"""
OIDeviationVolumeFade — OI Delta + Price Deviation + Volume Reversal

Logic (持仓量+价格偏离+成交量反转):
Uses price DEVIATION from long-term mean (not momentum/pct_change) as the
directional anchor. When price is far from its 48-period mean AND OI is
building AND volume confirms — the deviation is crowded. Fade it.

Price deviation (close/mean - 1) captures "how far from normal" which is
structurally different from momentum (pct_change). A market can be far from
mean but not moving, or moving fast near mean — these are distinct signals.

Formula:
-1 * zscore(delta(open_interest, 6), 24) * zscore(close / rolling_mean(close, 48) - 1, 24) * rank_pct(volume, 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OIDeviationVolumeFade(FactorRegistry):
    factor_name = "OIDeviationVolumeFade"
    parameters = {"oi_delta": 6, "window": 24, "dev_window": 48}
    inputs = ["open_interest", "close", "volume"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Price deviation (not momentum) as directional anchor. When price deviates "
        "far from its long-term mean, the market is stretched — combine with OI "
        "building (crowd conviction) and volume confirmation. Deviation captures "
        "stretch, momentum captures speed — orthogonal signal components."
    )
    mathematical_formula = (
        "-1 * zscore(delta(open_interest, 6), 24) "
        "* zscore(close / rolling_mean(close, 48) - 1, 24) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
