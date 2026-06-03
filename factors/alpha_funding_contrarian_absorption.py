from __future__ import annotations

import numpy as np
import pandas as pd
from factors.base import FactorRegistry


class FundingContrarianAbsorption(FactorRegistry):
    """Funding-taker 背离吸收因子。
    正 funding → 多头拥挤 → 若 taker net sell + 价涨 → 多头被吸收 → 加倍偏空
    负 funding → 空头拥挤 → 若 taker net buy  + 价跌 → 空头被吸收 → 加倍偏多

    CrowdDirection = sign(funding_rate)  → which side is crowded
    TakerPressure   = sign(taker_buy - taker_sell) → which side is aggressive
    ContrarianSignal = indicator(CrowdDirection ≠ TakerPressure) → crowd losing to aggressive flow

    Signal = z(taker_imbalance_abs, 24) × ContrarianSignal × sign(funding × taker_net > 0 ? -1 : +1)
    """
    factor_name = 'FundingContrarianAbsorption'
    parameters = {'z_window': 24}
    inputs = ['close', 'open', 'volume', 'taker_buy_volume', 'taker_sell_volume', 'funding_rate']
    timeframes = ['1h']
    rationale = (
        'When funding is positive (crowded longs) but taker net sellers dominate and price rises, '
        'crowded longs are being absorbed → stronger bearish reversal signal. '
        'The funding rate provides the "prior" on which side has the most to lose.'
    )
    mathematical_formula = (
        'CrowdDirection = sign(funding); TakerPressure = sign(taker_buy - taker_sell); '
        'Contrarian = indicator(CrowdDirection ≠ TakerPressure); '
        'Signal = z(|taker_net|/vol,24) × Contrarian × (-CrowdDirection)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        idx = data.index
        w = self.parameters['z_window']
        eps = 1e-8

        # Crowd direction from funding
        crowd_dir = np.sign(data['funding_rate'].values).astype(float)

        # Taker pressure direction
        taker_net = (data['taker_buy_volume'] - data['taker_sell_volume']).values
        taker_dir = np.sign(taker_net).astype(float)

        # Contrarian: crowd and taker pushing opposite directions
        # crowd_dir > 0 = longs crowded, taker_dir < 0 = sellers aggressive → contrarian
        # crowd_dir < 0 = shorts crowded, taker_dir > 0 = buyers aggressive → contrarian
        contrarian = (crowd_dir * taker_dir < 0).astype(float)

        # Taker intensity
        taker_intensity = np.abs(taker_net) / (data['volume'].values + eps)

        def _zs(ser, win):
            m = ser.groupby(level='symbol').transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).mean()
            )
            s = ser.groupby(level='symbol').transform(
                lambda s: s.rolling(win, min_periods=max(2, win // 4)).std()
            ).replace(0, np.nan)
            return ((ser - m) / s).fillna(0.0)

        z_intensity = _zs(pd.Series(taker_intensity, index=idx), w)

        # Direction: when contrarian firing:
        #   crowded longs + aggressive sellers → bearish = -1
        #   crowded shorts + aggressive buyers → bullish = +1
        # Simplification: -crowd_dir when contrarian, 0 otherwise
        direction = -crowd_dir * contrarian

        signal_raw = z_intensity.values * direction
        signal = pd.Series(signal_raw, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.reindex(idx).fillna(0.0)
