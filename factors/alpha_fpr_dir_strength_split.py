from __future__ import annotations

import numpy as np
import pandas as pd
from factors.base import FactorRegistry


class FPR_DirStrengthSplit(FactorRegistry):
    """Formula C: 方向强度拆分。
    Direction = sign(-ΔOI) - cs_mean
    Strength = cs_rank(|z(Funding) × z(Return)|)
    Signal = Direction × Strength
    """
    factor_name = 'FPR_DirStrengthSplit'
    parameters = {'funding_window': 24, 'oi_window': 6, 'ret_window': 24}
    inputs = ['funding_rate', 'open_interest', 'close']
    timeframes = ['1h']
    rationale = 'Split OI direction from funding-return conviction to decouple direction/magnitude.'
    mathematical_formula = (
        'Direction = sign(-delta_OI,6) - cs_mean; '
        'Strength = cs_rank(|zscore(funding,24) * zscore(-ret,24)|); '
        'Signal = Direction * Strength'
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
        strength_raw = (fz * rz).abs().fillna(0.0).values

        oi_delta = data.groupby(level='symbol')['open_interest'].transform(lambda s: s.diff(6))
        dir_raw = np.sign(-oi_delta.values).astype(float)

        ts_level = data.index.get_level_values('timestamp')
        unique_ts = ts_level.unique()
        ts_codes = ts_level.map({v: i for i, v in enumerate(unique_ts)}).values

        result = np.zeros(len(idx))
        for ti in range(len(unique_ts)):
            mask = ts_codes == ti
            n = mask.sum()
            if n <= 1:
                continue
            # Direction: sign neutralized
            d_t = dir_raw[mask]
            d_neutral = d_t - d_t.mean()
            # Strength: cross-sectional rank
            s_t = strength_raw[mask]
            s_rank = s_t.argsort().argsort() / (n - 1)
            result[mask] = d_neutral * s_rank

        signal = pd.Series(result, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.reindex(idx).fillna(0.0)
