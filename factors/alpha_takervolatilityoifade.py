from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class Takervolatilityoifade(FactorRegistry):
    factor_name = 'TakerVolatilityOIFade'
    parameters = {'taker_ratio_zscore_window': 12, 'vol_std_window': 24, 'vol_zscore_window': 48, 'oi_delta_lag': 6}
    inputs = ['taker_volume', 'volume', 'close', 'open_interest']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Fades when taker volume dominance and volatility are extreme, while OI trend confirms exhaustion.'
    mathematical_formula = 'zscore(taker_volume / volume, 12) * zscore(rolling_std(pct_change(close,1), 24), 48) * sign(delta(open_interest, 6)) * -1'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
