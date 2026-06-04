"""
TakerImbalanceFade — Net taker volume imbalance mean-reversion.

五问:
1. 赚谁的钱？追taker方向的动量交易者
2. 对手方为何犯错？taker imbalance反映短期情绪，过度延伸后会回归
3. 为何不被套利？微观结构信号衰减快但信号清晰
4. 哪些regime有效？range, trend_exhaustion
5. 持仓周期2-4 bar

完全不同的机制: 用 taker_buy/sell_volume 比值，完全不用 OI/funding/vol。
"""
from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class TakerImbalanceFade(FactorRegistry):
    factor_name = "TakerImbalanceFade"
    parameters = {"lookback": 6, "window": 24}
    inputs = ["taker_buy_volume", "taker_sell_volume"]
    timeframes = ["1h"]
    rationale = (
        "Net taker volume imbalance reversal. When taker buys dominate heavily "
        "over a window, aggressive buying is exhausted → fade long. When taker "
        "sells dominate → fade short. Uses raw taker flow, completely different "
        "from OI/funding/volatility factors."
    )
    mathematical_formula = (
        "-1 * zscore("
        "rolling_sum(taker_buy_volume, 6) - rolling_sum(taker_sell_volume, 6)"
        ", 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
