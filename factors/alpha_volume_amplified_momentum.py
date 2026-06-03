from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class VolumeAmplifiedMomentum(FactorRegistry):
    factor_name = 'VolumeAmplifiedMomentum'
    parameters = {'window': 20, 'lag': 6, 'vol_window': 24}
    inputs = ['close', 'volume']
    timeframes = ['15m', '1h', '4h']
    rationale = 'Price momentum scaled by relative volume rank: big moves on high volume persist, small moves on low volume are noise. Captures genuine directional conviction vs passive drift.'
    mathematical_formula = 'zscore(pct_change(close, 6) * rank_pct(volume, 24), 20)'

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
