"""
OICrowdingExhaustionFade — OI momentum + price direction + volume confirmation fade.

五问:
1. 赚谁的钱？追OI和价格同向运动但忽略成交量确认的交易者
2. 对手方为何犯错？只看OI方向不看量价配合，容易被假突破吸引
3. 为何不被套利？三信号AND条件降低假阳性
4. 哪些regime有效？trend_mature, vol_expansion
5. 持仓周期3-6 bar

与CrowdingFade的区别:
- CrowdingFade: OI动量 × funding方向 × 量排名
- 本因子: OI动量 × 价格方向 × 量排名
- 价格方向可能比funding方向更直接、噪音更少
"""
from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OICrowdingExhaustionFade(FactorRegistry):
    factor_name = "OICrowdingExhaustionFade"
    parameters = {"oi_lag": 12, "price_lag": 6, "oi_window": 48, "vol_window": 24}
    inputs = ["open_interest", "close", "volume"]
    timeframes = ["1h", "4h"]
    rationale = (
        "When OI is growing strongly, price has moved in one direction, and volume "
        "is elevated — the market is crowded. The crowd's positioning is fragile. "
        "Fade the price direction. Three-way AND condition reduces false signals."
    )
    mathematical_formula = (
        "-1 * zscore(delta(open_interest, 12), 48) "
        "* sign(pct_change(close, 6)) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
