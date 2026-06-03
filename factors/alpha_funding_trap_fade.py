"""
FundingTrapFade — funding extreme without price follow-through.

五问:
1. 赚谁的钱？支付高funding但价格不配合的拥挤方
2. 对手方为何持续犯错？funding是显性成本，拥挤方承受时间衰减却不看价格确认
3. 为何不被套利？需要同时监控funding极端+价格停滞两个条件
4. 哪些regime有效？funding_extreme, range
5. 持仓周期3-6 bar(1h)，只在funding极端+价格停滞时触发

Formula: sign(-funding) * |z(funding)| * (1 - |price·sign(funding)|) * |robust_z(ΔOI)|
funding极端但价格不朝funding方向移动 → funding方向被套 → fade funding方向
"""
from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class FundingTrapFade(FactorRegistry):
    factor_name = "FundingTrapFade"
    parameters = {
        "funding_window": 24,
        "price_lookback": 3,
        "oi_lookback": 6,
        "oi_window": 24,
    }
    inputs = ["funding_rate", "close", "open_interest"]
    timeframes = ["1h", "4h"]
    rationale = (
        "Funding trap: when funding is extremely positive but price is not going up "
        "(or going down), the longs are trapped — fade long. Symmetric for negative funding. "
        "Strengthened by OI change confirmation."
    )
    mathematical_formula = (
        "-1 * sign(funding_rate) "
        "* abs(zscore(funding_rate, 24)) "
        "* (1 - clip(abs(pct_change(close, 3) * sign(funding_rate)), 0, 1)) "
        "* abs(robust_zscore(delta(open_interest, 6), 24))"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
