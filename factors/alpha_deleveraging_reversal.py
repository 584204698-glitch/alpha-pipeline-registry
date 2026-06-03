from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class DeleveragingReversal(FactorRegistry):
    """去杠杆后反转：OI急降+大振幅+放量+反向=强平/清仓后反弹。"""
    factor_name = 'DeleveragingReversal'
    parameters = {'oi_window': 6, 'ret_window': 6, 'vol_window': 12, 'zscore_window': 24}
    inputs = ['close', 'open_interest', 'volume']
    timeframes = ['1h', '4h']
    rationale = 'When OI collapses sharply (forced deleveraging), returns are large in magnitude, and volume spikes — this is capitulation or short squeeze liquidation, not normal trading. After forced liquidation exhausts, price tends to reverse. The sign(-return) term makes it bidirectional: big drop + OI collapse = long signal (bounce), big rally + OI collapse = short signal (fade).'
    mathematical_formula = 'zscore(-pct_change(open_interest, 6), 24) * zscore(abs(pct_change(close, 1)), 24) * zscore(volume, 24) * sign(-pct_change(close, 1))'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
