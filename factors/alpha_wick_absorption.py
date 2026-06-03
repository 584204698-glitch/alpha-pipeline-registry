from __future__ import annotations

import numpy as np
import pandas as pd
from factors.base import FactorRegistry


class WickAbsorption(FactorRegistry):
    """Formula 3: 双边 wick absorption。
    ShortAbs = z(Taker/Vol) × max(z(Δlog_OI),0) × z(LowerWick×DownAttempt)
    LongAbs  = z(Taker/Vol) × max(z(Δlog_OI),0) × z(UpperWick×UpAttempt)
    Signal = cs_rank(ShortAbs) - cs_rank(LongAbs)
    """
    factor_name = 'WickAbsorption'
    parameters = {'vol_window': 24, 'oi_window': 12}
    inputs = ['open', 'high', 'low', 'close', 'volume', 'taker_volume', 'open_interest']
    timeframes = ['1h']
    rationale = (
        'Upper wick + taker surge + OI build = longs absorbed by overhead → bearish. '
        'Lower wick + taker surge + OI build = shorts absorbed by demand → bullish. '
        'Rank-difference ensures bilateral cross-sectional balance.'
    )
    mathematical_formula = (
        'rank(short_absorbed) - rank(long_absorbed) where '
        'short_absorbed = z(taker/vol) * max(z(delta_log_OI),0) * z(lower_wick * down_attempt)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        idx = data.index
        eps = 1e-8
        w = self.parameters['vol_window']

        hl = (data['high'] - data['low']).clip(lower=eps)
        upper_wick = (data['high'] - data['close']) / hl
        lower_wick = (data['close'] - data['low']) / hl
        up_attempt = (data['high'] - data['open']) / (data['open'].abs() + eps)
        down_attempt = (data['open'] - data['low']) / (data['open'].abs() + eps)
        taker_ratio = data['taker_volume'] / (data['volume'].clip(lower=eps))
        oi_log = data.groupby(level='symbol')['open_interest'].transform(
            lambda s: np.log(s + eps).diff(12)
        )

        def _zs(ser, win):
            m = ser.groupby(level='symbol').transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level='symbol').transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, np.nan)
            return ((ser - m) / s).fillna(0.0)

        z_taker = _zs(taker_ratio, w)
        z_oi = _zs(oi_log, w)
        oi_gate = np.maximum(z_oi.values, 0.0)

        z_long_abs = _zs(upper_wick * up_attempt, w)
        z_short_abs = _zs(lower_wick * down_attempt, w)

        long_score = pd.Series(z_taker.values * oi_gate * z_long_abs.values, index=idx).fillna(0.0)
        short_score = pd.Series(z_taker.values * oi_gate * z_short_abs.values, index=idx).fillna(0.0)

        # Cross-sectional rank per timestamp
        ts_values = data.index.get_level_values('timestamp')
        unique_ts = ts_values.unique()
        ts_codes = ts_values.map({v: i for i, v in enumerate(unique_ts)}).values

        result = np.zeros(len(idx))
        long_v = long_score.values
        short_v = short_score.values

        for ti in range(len(unique_ts)):
            mask = ts_codes == ti
            n = mask.sum()
            if n <= 1:
                continue
            rl = long_v[mask].argsort().argsort() / (n - 1)
            rs = short_v[mask].argsort().argsort() / (n - 1)
            result[mask] = rs - rl

        signal = pd.Series(result, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.reindex(idx).fillna(0.0)
