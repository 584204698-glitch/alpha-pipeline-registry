"""
DeleveragingSnapback — event-triggered deleveraging reversal.

五问:
1. 赚谁的钱？被强制平仓的杠杆交易者（清算后踏空反弹）
2. 对手方为何持续犯错？强制平仓是机械性事件，清算完成时往往超调
3. 为何不被套利？事件型信号频率低(每币每月几次)，不适合高频套利
4. 哪些regime有效？panic_down, squeeze_up
5. 持仓周期2-4 bar(1h)，事件触发故换手极低

Formula: 事件触发 → OI骤降+放量+急跌/急涨 → 反转
+1: 急跌+OI降+放量 → long (清算潮结束)
-1: 急涨+OI降+放量 → short (获利了结潮)
"""
from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class DeleveragingSnapback(FactorRegistry):
    factor_name = "DeleveragingSnapback"
    parameters = {
        "price_window": 48,
        "oi_window": 48,
        "vol_window": 48,
        "price_mult": 3.0,
        "oi_mult": 2.0,
        "vol_mult": 2.0,
        "smooth": 4,
    }
    inputs = ["open_interest", "close", "volume"]
    timeframes = ["1h"]
    rationale = (
        "Deleveraging snapback: crash+OI crash+vol spike → forced liquidation "
        "exhaustion → bounce. Pump+OI crash+vol spike → profit-taking exhaustion → fade."
    )
    mathematical_formula = (
        "rolling_mean("
        "indicator(pct_change(close, 1) < -3 * rolling_std(pct_change(close, 1), 48)) "
        "* indicator(pct_change(open_interest, 1) < -2 * rolling_std(pct_change(open_interest, 1), 48)) "
        "* indicator(volume > 2 * rolling_mean(volume, 48)) "
        "- "
        "indicator(pct_change(close, 1) > 3 * rolling_std(pct_change(close, 1), 48)) "
        "* indicator(pct_change(open_interest, 1) < -2 * rolling_std(pct_change(open_interest, 1), 48)) "
        "* indicator(volume > 2 * rolling_mean(volume, 48))"
        ", 4)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
