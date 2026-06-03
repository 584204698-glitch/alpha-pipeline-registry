"""
MADVolStructureFade — Robust volatility structure change + OI direction fade.

五问:
1. 赚谁的钱？在波动结构变化时仍按旧regime方向交易的人
2. 对手方为何犯错？标准差对极端值敏感，MAD能更早更干净地捕捉真实波动结构变化
3. 为何不被套利？MAD+sign(OI)组合需要同时监控稳健波动二阶导和离散OI状态
4. 哪些regime有效？vol_expansion, trend_mature, pre_panic
5. 持仓周期3-6 bar，阈值入场降换手

与VolAccelFade的关键区别:
- rolling_mad替代rolling_std → 对wick/插针/异常成交稳健
- sign(ΔOI)替代zscore(price_momentum) → 保留离散方向信息不被平滑
- robust_zscore替代zscore → median/MAD统计量比mean/std稳健
- 更长窗口(24/12/48 vs 12/6/24)
"""
from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolStructureFade(FactorRegistry):
    factor_name = "MADVolStructureFade"
    parameters = {"mad_window": 24, "accel_lag": 12, "z_window": 48, "oi_lag": 12}
    inputs = ["close", "open_interest"]
    timeframes = ["1h", "4h"]
    rationale = (
        "MAD-based vol acceleration captures genuine volatility structure changes "
        "while ignoring wick/outlier noise. sign(ΔOI) provides clean directional "
        "signal (preserving binary state information). robust_zscore uses median/MAD "
        "for fat-tailed crypto returns. This is a structurally more robust version "
        "of the vol_accel_fade family."
    )
    mathematical_formula = (
        "-1 * robust_zscore(delta(rolling_mad(pct_change(close, 1), 24), 12), 48) "
        "* sign(delta(open_interest, 12)) "
        "* abs(robust_zscore(delta(open_interest, 12), 48))"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
