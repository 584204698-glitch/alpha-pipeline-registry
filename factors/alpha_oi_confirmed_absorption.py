from __future__ import annotations

import numpy as np
import pandas as pd
from factors.base import FactorRegistry


class OIConfirmedAbsorption(FactorRegistry):
    """OI 确认版 taker 吸收。
    只在 OI 堆积时信号有效——新资金入场被吸收比存量换手更有意义。

    NetTaker = (taker_buy - taker_sell) / volume
    PriceMove = sign(close - open)  (price direction)
    Absorption = indicator(NetTaker × PriceMove < 0)  → taker pushes one way, price moves opposite

    OI Gate = max(z(Δlog_OI), 0)  → only activate when new positions being built

    Signal = z(NetTaker,24) × Absorption_sign × OI_Gate
    其中 Absorption_sign = indicator(NetTaker < 0) - indicator(NetTaker > 0)
    """
    factor_name = 'OIConfirmedAbsorption'
    parameters = {'z_window': 24, 'oi_window': 12}
    inputs = ['close', 'open', 'volume', 'taker_buy_volume', 'taker_sell_volume', 'open_interest']
    timeframes = ['1h']
    rationale = (
        'Absorption is most meaningful when new positions are being built (OI rising). '
        'If OI is flat or falling, aggressive flow could just be position unwinding, not absorption. '
        'OI gate filters for genuine liquidity absorption events.'
    )
    mathematical_formula = (
        'NetTaker = (taker_buy - taker_sell) / volume; '
        'Absorption = indicator(NetTaker * sign(close-open) < 0); '
        'CS neutral zscore of NetTaker × Absorption_sign, gated by max(z(delta_log_OI),0)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        idx = data.index
        w = self.parameters['z_window']
        eps = 1e-8

        net_taker = (data['taker_buy_volume'] - data['taker_sell_volume']) / (data['volume'] + eps)
        price_sign = np.sign(data['close'].values - data['open'].values).astype(float)

        # Absorption flag: taker pushes opposite to price
        net_taker_sign = np.sign(net_taker.values).astype(float)
        absorbed = (net_taker_sign * price_sign < 0).astype(float)

        # OI accumulation gate
        oi_log_delta = data.groupby(level='symbol')['open_interest'].transform(
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

        z_oi = _zs(oi_log_delta, w)
        oi_gate = np.maximum(z_oi.values, 0.0)

        # Net taker zscore, but neutralized within absorption flag
        z_net = _zs(net_taker, w)

        # Direction: when absorbed + taker net sell → bullish (sellers absorbed)
        #             when absorbed + taker net buy  → bearish (buyers absorbed)
        signal_raw = -net_taker_sign * absorbed * z_net.values * oi_gate

        signal = pd.Series(signal_raw, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.reindex(idx).fillna(0.0)
