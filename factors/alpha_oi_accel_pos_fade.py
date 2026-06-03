from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class OIAccelPosFade(FactorRegistry):
    """OI波动加速×价格位置反转：OI-vol-accel timing + price POSITION方向。
    OI变化率波动加速(持仓异动)+价格高位=分歧→做空；价格低位=恐慌→做多。
    价格位置替代价格动量，大幅降低与VolAccelFade的相关性。"""
    factor_name = 'OIAccelPosFade'
    parameters = {'oi_vol_window': 12, 'accel_lag': 6, 'zscore_window': 24, 'pos_window': 48}
    inputs = ['close', 'open_interest']
    timeframes = ['1h']
    rationale = (
        'OI volatility acceleration timing (position churn chaos) with PRICE POSITION '
        'directional anchor. OI-vol-accel captures when positions are being rapidly '
        'opened AND closed — deep market division where directional bets are fragile. '
        'Price position (rank_pct of close) determines fade direction: near highs → '
        'crowded position building → SHORT; near lows → capitulation → LONG. '
        'Price position (48-bar rank) has lower correlation with price momentum '
        '(6-bar zscore) used in VolAccelFade, reducing factor overlap. '
        'OI-vol-accel spine achieved 3/3 positive regime IC in price-momentum variant.'
    )
    mathematical_formula = (
        '-1 * zscore(delta(rolling_std(delta(open_interest, 1), 12), 6), 24) '
        '* (rank_pct(close, 48) - 0.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
