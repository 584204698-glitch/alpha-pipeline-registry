"""
BodyTakerCSRank — Body Bias CS Rank × Taker Flow CS Rank

Logic (实体主动量截面排名):
Pure double cross-sectional rank approach. At each timestamp, rank both
body_bias and taker_ratio across symbols. When both are extreme AND aligned
→ conviction absorption signal.

Cross-sectional ranking is structurally different from rolling zscore:
- CS rank captures peer-relative extremes (which symbols are hot RIGHT NOW)
- Rolling zscore captures self-relative extremes (is this symbol hot vs its own past)
- CS rank is naturally mean-reverting at the extremes (rank 0/1 bounds)

Formula:
-1 * (cs_rank((2*close-high-low)/(high-low+1e-8)) - 0.5)
    * (cs_rank(taker_volume/(volume+1e-8)) - 0.5)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from factors.base import FactorRegistry


class BodyTakerCSRank(FactorRegistry):
    factor_name = "BodyTakerCSRank"
    parameters = {}
    inputs = ["open", "high", "low", "close", "volume", "taker_volume"]
    timeframes = ["1h"]
    rationale = (
        "Double cross-sectional rank: body_bias × taker_ratio. At each timestamp, "
        "rank both components across symbols. When a symbol has both extreme body "
        "AND extreme taker flow relative to peers → absorption signal → fade. "
        "CS rank is bounded [0,1] → centered [-0.5, +0.5] → naturally symmetric."
    )
    mathematical_formula = (
        "-1 * (cs_rank((2*close-high-low)/(high-low+1e-8)) - 0.5) "
        "* (cs_rank(taker_volume/(volume+1e-8)) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        eps = 1e-8
        idx = data.index

        hl = (data["high"] - data["low"]).clip(lower=eps)
        body_bias = (2 * data["close"] - data["high"] - data["low"]) / hl
        taker_ratio = data["taker_volume"] / (data["volume"].clip(lower=eps))
        taker_ratio[taker_ratio > 2.0] = 2.0

        ts = data.index.get_level_values("timestamp")
        unique_ts = ts.unique()
        ts_map = {v: i for i, v in enumerate(unique_ts)}
        ts_codes = np.array([ts_map[t] for t in ts])

        n_total = len(idx)
        body_rank = np.zeros(n_total)
        taker_rank = np.zeros(n_total)

        for ti in range(len(unique_ts)):
            mask = ts_codes == ti
            n = mask.sum()
            if n <= 1:
                continue
            bv = body_bias.values[mask]
            tv = taker_ratio.values[mask]
            body_rank[mask] = bv.argsort().argsort() / (n - 1)
            taker_rank[mask] = tv.argsort().argsort() / (n - 1)

        body_rank_c = pd.Series(body_rank - 0.5, index=idx).fillna(0.0)
        taker_rank_c = pd.Series(taker_rank - 0.5, index=idx).fillna(0.0)

        signal = pd.Series(
            -1.0 * body_rank_c.values * taker_rank_c.values, index=idx
        )
        return signal.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
