from __future__ import annotations

import numpy as np
import pandas as pd
from factors.base import FactorRegistry


class FPR_StateWiseNorm(FactorRegistry):
    """Formula B: State-wise 标准化。
    A = z(Funding) × z(Return)
    Signal = sign(-ΔOI) × z_i(A | sign(-ΔOI))
    分别在 OI 上升/下降 regime 内各自标准化 A。
    """
    factor_name = 'FPR_StateWiseNorm'
    parameters = {'funding_window': 24, 'oi_window': 6, 'ret_window': 24, 'min_regime_samples': 30}
    inputs = ['funding_rate', 'open_interest', 'close']
    timeframes = ['1h']
    rationale = 'State-wise normalization separates OI-rising and OI-falling regimes before ranking.'
    mathematical_formula = (
        'A = zscore(funding_rate,24) * zscore(-pct_change(close,1),24); '
        'signal = sign(-delta_OI) * z_i(A | sign(-delta_OI))'
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
        A_vals = (fz * rz).fillna(0.0).values

        oi_delta = data.groupby(level='symbol')['open_interest'].transform(lambda s: s.diff(6))
        S_vals = np.sign(-oi_delta.values).astype(float)

        min_n = self.parameters['min_regime_samples']
        result = np.zeros(len(idx))

        ts_level = data.index.get_level_values('timestamp')
        unique_ts = ts_level.unique()
        ts_codes = ts_level.map({v: i for i, v in enumerate(unique_ts)}).values

        for ti in range(len(unique_ts)):
            mask = ts_codes == ti
            a_t = A_vals[mask]
            s_t = S_vals[mask]
            pos = s_t > 0
            neg = s_t < 0
            if pos.sum() >= min_n and neg.sum() >= min_n:
                p_mean, p_std = a_t[pos].mean(), a_t[pos].std() or 1.0
                n_mean, n_std = a_t[neg].mean(), a_t[neg].std() or 1.0
                result[mask] = np.where(pos,
                    s_t * (a_t - p_mean) / p_std,
                    s_t * (a_t - n_mean) / n_std)
            else:
                cs_mean, cs_std = a_t.mean(), a_t.std() or 1.0
                result[mask] = s_t * (a_t - cs_mean) / cs_std

        signal = pd.Series(result, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.reindex(idx).fillna(0.0)
