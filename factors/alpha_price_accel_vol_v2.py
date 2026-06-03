from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class PriceAccelVolFadeV2(FactorRegistry):
    """价格加速度反转V2：用rank_pct替代zscore强制对称(50% long/short)。
    abs(vol_zscore)仍作conviction乘数。V1有OOS=0.216但long=20%→V2修复对称。"""
    factor_name = 'PriceAccelVolFadeV2'
    parameters = {'accel_lag': 6, 'rank_window': 48, 'vol_mean_window': 48, 'zscore_window': 24}
    inputs = ['close', 'volume']
    timeframes = ['1h']
    rationale = (
        'Price acceleration reversal with FORCED symmetry via rank_pct (centered at 0.5). '
        'V1 used zscore for price accel direction, which was biased SHORT (20% long) '
        'due to slight positive drift in crypto returns. V2 replaces zscore with '
        '(rank_pct - 0.5) which is perfectly symmetric by construction (~50% long/short). '
        'Volume anomaly (abs zscore of vol/MA) provides conviction scaling. '
        'OOS ICIR=0.216 in V1 proves the signal works; V2 fixes the symmetry gate.'
    )
    mathematical_formula = (
        '-1 * (rank_pct(delta(pct_change(close, 1), 6), 48) - 0.5) '
        '* abs(zscore(volume / rolling_mean(volume, 48), 24))'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
