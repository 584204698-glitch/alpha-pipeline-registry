from __future__ import annotations

import numpy as np
import pandas as pd
from factors.base import FactorRegistry


class TakerAbsorption(FactorRegistry):
    """真实 taker buy/sell 吸收因子。
    买方吸收: taker_buy > taker_sell 但价格下跌 → 卖家吸收了买方攻击 → 偏空
    卖方吸收: taker_sell > taker_buy 但价格上涨 → 买家吸收了卖方攻击 → 偏多

    BuyAbsorb  = z(taker_buy/taker_sell) × indicator(close < open) × z(vol)
    SellAbsorb = z(taker_sell/taker_buy) × indicator(close > open) × z(vol)
    Signal = cs_rank(SellAbsorb) - cs_rank(BuyAbsorb)
    """
    factor_name = 'TakerAbsorption'
    parameters = {'z_window': 24}
    inputs = ['close', 'open', 'volume', 'taker_buy_volume', 'taker_sell_volume']
    timeframes = ['1h']
    rationale = (
        'When aggressive buyers dominate (taker_buy >> taker_sell) but price falls, '
        'limit-order sellers absorbed all the buying pressure → bearish signal. '
        'When aggressive sellers dominate but price rises, limit-order buyers absorbed '
        'the selling pressure → bullish signal. Uses REAL taker buy/sell split data.'
    )
    mathematical_formula = (
        'rank(z(taker_sell/taker_buy) * indicator(close>open) * z(vol)) '
        '- rank(z(taker_buy/taker_sell) * indicator(close<open) * z(vol))'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        idx = data.index
        w = self.parameters['z_window']
        eps = 1e-8

        buy = data['taker_buy_volume']
        sell = data['taker_sell_volume']

        # Taker imbalance ratios
        buy_ratio = buy / (sell + eps)
        sell_ratio = sell / (buy + eps)

        # Price direction
        price_up = (data['close'] > data['open']).astype(float)
        price_down = (data['close'] < data['open']).astype(float)

        def _zs(ser, win):
            m = ser.groupby(level='symbol').transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level='symbol').transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, np.nan)
            return ((ser - m) / s).fillna(0.0)

        z_buy_ratio = _zs(buy_ratio, w)
        z_sell_ratio = _zs(sell_ratio, w)
        z_vol = _zs(data['volume'], w)

        # Buy absorption: aggressive buyers but price falling
        buy_absorb = z_buy_ratio * price_down * z_vol
        # Sell absorption: aggressive sellers but price rising
        sell_absorb = z_sell_ratio * price_up * z_vol

        buy_series = pd.Series(buy_absorb.values, index=idx).fillna(0.0)
        sell_series = pd.Series(sell_absorb.values, index=idx).fillna(0.0)

        # Cross-sectional rank
        ts_values = data.index.get_level_values('timestamp')
        unique_ts = ts_values.unique()
        ts_codes = ts_values.map({v: i for i, v in enumerate(unique_ts)}).values

        result = np.zeros(len(idx))
        bv = buy_series.values
        sv = sell_series.values

        for ti in range(len(unique_ts)):
            mask = ts_codes == ti
            n = mask.sum()
            if n <= 1:
                continue
            rb = bv[mask].argsort().argsort() / (n - 1)
            rs = sv[mask].argsort().argsort() / (n - 1)
            result[mask] = rs - rb

        signal = pd.Series(result, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.reindex(idx).fillna(0.0)
