from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class ForcedLiquidationCascadeV3(FactorRegistry):
    """V3: rank_pct替代zscore处理OI delta（抗极端尾部），更长窗口提高稳定性。
    当价格暴跌(robust_zscore<0)+OI骤降(rank_pct<0.5)→product正→LONG超卖反弹。"""
    factor_name = 'ForcedLiquidationCascadeV3'
    parameters = {'price_window': 24, 'oi_delta': 12, 'oi_rank_window': 48}
    inputs = ['close', 'open_interest']
    timeframes = ['1h']
    rationale = (
        'Price×OI liquidation detection with rank-based OI for robustness. '
        'When price collapses (robust_zscore negative) and OI drops sharply '
        '(OI delta rank below median), leveraged positions are being force-closed. '
        'The selling cascade exhausts as positions clear → oversold bounce. '
        'Uses rank_pct for OI delta (outlier-resistant) instead of zscore, '
        'and robust_zscore for price to handle fat tails. Longer windows for stability.'
    )
    mathematical_formula = (
        'robust_zscore(pct_change(close, 1), 24) '
        '* (rank_pct(delta(open_interest, 12), 48) - 0.5)'
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
