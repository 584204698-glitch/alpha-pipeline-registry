from __future__ import annotations

import numpy as np
import pandas as pd
from factors.base import FactorRegistry


class TrendTrapAbsorption(FactorRegistry):
    """Formula 5: Prior-trend trap absorption。
    LongTrap  = max(+z(Trend_prev),0) × z(Taker/Vol) × max(z(Δlog_OI),0) × max(-z(Return),0)
    ShortTrap = max(-z(Trend_prev),0) × z(Taker/Vol) × max(z(Δlog_OI),0) × max(+z(Return),0)
    Signal = cs_rank(ShortTrap) - cs_rank(LongTrap)
    """
    factor_name = 'TrendTrapAbsorption'
    parameters = {'trend_window': 6, 'vol_window': 24, 'oi_window': 12}
    inputs = ['close', 'volume', 'taker_volume', 'open_interest']
    timeframes = ['1h']
    rationale = (
        'Prior uptrend → longs chasing → if current return reverses down with taker+OI surge → longs trapped → bearish. '
        'Prior downtrend → shorts chasing → if current return reverses up with taker+OI surge → shorts trapped → bullish.'
    )
    mathematical_formula = (
        'rank(max(-z(trend_prev),0)*z(taker/vol)*max(z(delta_log_OI),0)*max(z(ret),0)) '
        '- rank(max(+z(trend_prev),0)*z(taker/vol)*max(z(delta_log_OI),0)*max(-z(ret),0))'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        idx = data.index
        eps = 1e-8
        k = self.parameters['trend_window']
        w = self.parameters['vol_window']

        ret = data.groupby(level='symbol')['close'].transform(lambda s: s.pct_change())
        trend_prev = data.groupby(level='symbol')['close'].transform(lambda s: s.pct_change(k))
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

        z_trend = _zs(trend_prev, w)
        z_taker = _zs(taker_ratio, w)
        z_oi = _zs(oi_log, w)
        z_ret = _zs(ret, w)

        oi_gate = np.maximum(z_oi.values, 0.0)
        trend_up = np.maximum(z_trend.values, 0.0)
        trend_down = np.maximum(-z_trend.values, 0.0)
        ret_up = np.maximum(z_ret.values, 0.0)
        ret_down = np.maximum(-z_ret.values, 0.0)

        long_trap = pd.Series(trend_up * z_taker.values * oi_gate * ret_down, index=idx).fillna(0.0)
        short_trap = pd.Series(trend_down * z_taker.values * oi_gate * ret_up, index=idx).fillna(0.0)

        ts_values = data.index.get_level_values('timestamp')
        unique_ts = ts_values.unique()
        ts_codes = ts_values.map({v: i for i, v in enumerate(unique_ts)}).values

        result = np.zeros(len(idx))
        lv = long_trap.values
        sv = short_trap.values

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
