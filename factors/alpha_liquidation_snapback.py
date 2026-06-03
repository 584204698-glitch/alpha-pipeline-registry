"""
LiquidationSnapback — Post-Liquidation Cascade Reversal

Logic (逆向思维):
When OI drops sharply AND price moves sharply in the same direction, forced
liquidations are cascading. The cascade mechanically overshoots fair value
because liquidations are price-insensitive — they execute at ANY price.

After the cascade subsides, price tends to partially snap back as the mechanical
selling pressure is absorbed. This is NOT mean reversion — it's specifically
about the mechanical aftermath of leveraged liquidations.

Key difference from standard reversal:
- Standard reversal: "price went down, it'll go back up" (too noisy)
- Liquidation snapback: "OI crashed + price crashed = forced selling → snapback" (clean signal)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class LiquidationSnapback(FactorRegistry):
    factor_name = "LiquidationSnapback"
    parameters = {"oi_lookback": 3, "window": 24}
    inputs = ["open_interest", "close"]
    timeframes = ["1h", "4h"]
    rationale = (
        "After a liquidation cascade (sharp OI drop + sharp price move), price mechanically "
        "overshoots and partially rebounds. Take the opposite direction of the cascade."
    )
    mathematical_formula = (
        "-1 * sign(pct_change(close, 1)) "
        "* zscore(delta(open_interest, 3), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
