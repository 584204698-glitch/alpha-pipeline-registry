from __future__ import annotations

import numpy as np
import pandas as pd
from factors.base import FactorRegistry


class FPR_CSNeutral(FactorRegistry):
    """Formula A: 横截面 sign 中性化。
    z(Funding) × z(Return) × [sign(-ΔOI) - cross_sectional_mean(sign(-ΔOI))]
    保留 sign() 的离散状态切换，减去每期截面均值消除方向偏置。
    """
    factor_name = 'FPR_CSNeutral'
    parameters = {'funding_window': 24, 'oi_window': 6, 'ret_window': 24}
    inputs = ['funding_rate', 'open_interest', 'close']
    timeframes = ['1h']
    rationale = (
        'CS-neutral sign: preserve binary OI regime signal, '
        'subtract cross-sectional mean to remove systematic directional bias.'
    )
    mathematical_formula = (
        'zscore(funding_rate,24) * zscore(-pct_change(close,1),24) '
        '* (sign(-delta(open_interest,6)) - cs_mean(sign(-delta_OI)))'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        idx = data.index
        w = self.parameters['funding_window']

        def _zs(ser, win):
            m = ser.groupby(level='symbol').transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level='symbol').transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, np.nan)
            return ((ser - m) / s).fillna(0.0)

        fz = _zs(data['funding_rate'], w)
        neg_ret = -data.groupby(level='symbol')['close'].transform(lambda s: s.pct_change())
        rz = _zs(neg_ret, w)

        oi_delta = data.groupby(level='symbol')['open_interest'].transform(lambda s: s.diff(6))
        s_oi = pd.Series(np.sign(-oi_delta.values).astype(float), index=idx)
        cs_mean = s_oi.groupby(level='timestamp').transform('mean')
        s_oi_neutral = (s_oi - cs_mean).fillna(0.0)

        raw = fz * rz * s_oi_neutral
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return raw.reindex(idx).fillna(0.0)
