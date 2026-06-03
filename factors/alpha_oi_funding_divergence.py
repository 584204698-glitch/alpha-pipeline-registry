from __future__ import annotations

from factors.base import FactorRegistry, evaluate_formula


class OiFundingDivergence(FactorRegistry):
    factor_name = "OI_Funding_Divergence"
    parameters = {"oi_lookback": 4, "window": 12, "quantile_window": 24, "funding_quantile": 0.2}
    inputs = ["open_interest", "funding_rate", "close"]
    mathematical_formula = (
        "zscore(delta(open_interest, oi_lookback), window) * "
        "indicator(funding_rate < rolling_quantile(funding_rate, quantile_window, funding_quantile))"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data):
        return evaluate_formula(self.mathematical_formula, data, self.parameters)
