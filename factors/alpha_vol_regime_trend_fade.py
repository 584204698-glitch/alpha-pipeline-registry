"""
VolRegimeTrendFade — Volatility regime shift + trend direction oscillator.

五问:
1. 赚谁的钱？在波动率上升时追趋势、在波动率下降时做反转的交易者
2. 对手方为何犯错？波动率上升时趋势可能加速(应跟随)，波动率下降时趋势衰竭(应反转)
3. 为何不被套利？长窗口波动结构变化是慢信号，短窗口交易者难以捕捉
4. 哪些regime有效？vol_expansion（跟随）, vol_contraction（反转）
5. 持仓周期4-8 bar

核心创新:
- 不用固定方向(fade) — 波动率加速时跟随趋势，减速时反转
- 超长窗口(48/24/96)捕捉宏观波动结构
- 这是第一个"自适应方向"因子，不是纯fade
"""
from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class VolRegimeTrendFade(FactorRegistry):
    factor_name = "VolRegimeTrendFade"
    parameters = {"mad_window": 48, "accel_lag": 24, "z_window": 96, "trend_lag": 12}
    inputs = ["close"]
    timeframes = ["4h"]
    rationale = (
        "Volatility regime adaptive oscillator. When vol is accelerating "
        "(robust_zscore positive), follow the trend — momentum may continue in "
        "chaotic regimes. When vol is decelerating, fade the trend — mean reversion "
        "dominates in calming markets. Uses MAD and robust_zscore throughout."
    )
    mathematical_formula = (
        "robust_zscore(delta(rolling_mad(pct_change(close, 1), 48), 24), 96) "
        "* robust_zscore(pct_change(close, 12), 96)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
