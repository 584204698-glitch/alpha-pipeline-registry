from __future__ import annotations

import numpy as np
import pandas as pd
from factors.base import FactorRegistry


class FundingGatedAbsorption(FactorRegistry):
    """Formula 4: Funding-gated crowded absorption。
    LongCrowdedAbs  = max(+z(Funding),0) × z(Taker/Vol) × max(z(Δlog_OI),0) × z(UpperWick×UpAttempt)
    ShortCrowdedAbs = max(-z(Funding),0) × z(Taker/Vol) × max(z(Δlog_OI),0) × z(LowerWick×DownAttempt)
    Signal = cs_rank(ShortCrowdedAbs) - cs_rank(LongCrowdedAbs)
    """
    factor_name = 'FundingGatedAbsorption'
    parameters = {'funding_window': 24, 'vol_window': 24, 'oi_window': 12}
    inputs = ['open', 'high', 'low', 'close', 'volume', 'taker_volume', 'open_interest', 'funding_rate']
    timeframes = ['1h']
    rationale = (
        'Funding acts as crowding prior: positive funding → longs crowded, upper wick failure → bearish. '
        'Negative funding → shorts crowded, lower wick failure → bullish.'
    )
    mathematical_formula = (
        'rank(max(-z(funding),0)*z(taker/vol)*max(z(delta_log_OI),0)*z(lower_rejection)) '
        '- rank(max(+z(funding),0)*z(taker/vol)*max(z(delta_log_OI),0)*z(upper_rejection))'
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

        z_funding = _zs(data['funding_rate'], self.parameters['funding_window'])
        z_taker = _zs(taker_ratio, w)
        z_oi = _zs(oi_log, w)
        oi_gate = np.maximum(z_oi.values, 0.0)

        z_long_abs = _zs(upper_wick * up_attempt, w)
        z_short_abs = _zs(lower_wick * down_attempt, w)

        funding_pos = np.maximum(z_funding.values, 0.0)
        funding_neg = np.maximum(-z_funding.values, 0.0)

        long_crowded = pd.Series(funding_pos * z_taker.values * oi_gate * z_long_abs.values, index=idx).fillna(0.0)
        short_crowded = pd.Series(funding_neg * z_taker.values * oi_gate * z_short_abs.values, index=idx).fillna(0.0)

        ts_values = data.index.get_level_values('timestamp')
        unique_ts = ts_values.unique()
        ts_codes = ts_values.map({v: i for i, v in enumerate(unique_ts)}).values

        result = np.zeros(len(idx))
        lv = long_crowded.values
        sv = short_crowded.values

        for ti in range(len(unique_ts)):
            mask = ts_codes == ti
            n = mask.sum()
            if n <= 1:
                continue
            rl = lv[mask].argsort().argsort() / (n - 1)
            rs = sv[mask].argsort().argsort() / (n - 1)
            result[mask] = rs - rl

        signal = pd.Series(result, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.reindex(idx).fillna(0.0)
