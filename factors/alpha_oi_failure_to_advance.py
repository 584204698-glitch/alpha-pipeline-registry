"""
OIFailureToAdvance — OI accumulation failure fade.

五问:
1. 赚谁的钱？追OI趋势但忽略价格效率的交易者
2. 对手方为何持续犯错？多数人只看OI方向不看OI效率
3. 为何不被套利？OI-价格效率是二阶信号，需要同时监控两个维度
4. 哪些regime有效？trend_mature, range_bound
5. 持仓周期覆盖成本？事件触发+低换手，3-6 bar

Formula: sign(-ΔOI) * (1 - |z(price)|) * |z(ΔOI)|
当OI显著变化但价格平坦 → 强信号。OI方向被吸收 → fade OI方向。
"""
from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OIFailureToAdvance(FactorRegistry):
    factor_name = "OIFailureToAdvance"
    parameters = {"oi_lookback": 9, "price_lookback": 3, "z_window": 48}
    inputs = ["open_interest", "close"]
    timeframes = ["1h", "4h"]
    rationale = (
        "OI accumulation failure: when OI moves significantly but price is flat, "
        "the OI direction is being absorbed. Fade the OI direction, weighted by "
        "OI change magnitude and price flatness."
    )
    mathematical_formula = (
        "-1 * sign(delta(open_interest, 9)) "
        "* (1 - abs(zscore(pct_change(close, 3), 48))) "
        "* abs(zscore(delta(open_interest, 9), 48))"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
