"""Multiple testing correction for factor mining.

Provides Bonferroni, Holm, and Benjamini-Hochberg FDR corrections.
Designed for the factor discovery workflow where many factors are tested.
"""

from __future__ import annotations

import numpy as np


def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> dict:
    """Bonferroni correction — most conservative."""
    p = np.asarray(p_values, dtype=float)
    p = p[~np.isnan(p)]
    m = len(p)
    if m == 0:
        return {"method": "bonferroni", "alpha": alpha, "n_tests": 0, "threshold": np.nan, "significant": []}
    threshold = alpha / m
    significant = [bool(pv <= threshold) for pv in p]
    return {
        "method": "bonferroni",
        "alpha": alpha,
        "n_tests": m,
        "threshold": float(threshold),
        "significant": significant,
        "n_significant": int(sum(significant)),
    }


def holm(p_values: np.ndarray, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down — less conservative than Bonferroni."""
    p = np.asarray(p_values, dtype=float)
    nan_mask = np.isnan(p)
    valid_idx = np.where(~nan_mask)[0]
    p_valid = p[valid_idx]
    m = len(p_valid)
    if m == 0:
        return {"method": "holm", "alpha": alpha, "n_tests": 0, "thresholds": [], "significant": []}

    order = np.argsort(p_valid)
    significant = np.zeros(m, dtype=bool)

    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_valid[idx] <= threshold:
            significant[idx] = True
        else:
            break  # step-down: stop at first non-significant

    # Map back to original positions
    full_significant = np.zeros(len(p), dtype=bool)
    full_significant[valid_idx] = significant

    return {
        "method": "holm",
        "alpha": alpha,
        "n_tests": m,
        "n_significant": int(significant.sum()),
        "significant": [bool(s) for s in full_significant],
        "adjusted_p": _holm_adjusted(p_valid, order),
    }


def _holm_adjusted(p_valid: np.ndarray, order: np.ndarray) -> list[float]:
    """Compute Holm-adjusted p-values."""
    m = len(p_valid)
    adjusted = np.zeros(m)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(p_valid[idx] * (m - rank), 1.0)
    # Ensure monotonicity
    for rank in range(m - 1):
        i1 = order[rank]
        i2 = order[rank + 1]
        adjusted[i2] = max(adjusted[i2], adjusted[i1])
    return [float(v) for v in adjusted]


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR control — recommended for exploratory factor mining."""
    p = np.asarray(p_values, dtype=float)
    nan_mask = np.isnan(p)
    valid_idx = np.where(~nan_mask)[0]
    p_valid = p[valid_idx]
    m = len(p_valid)
    if m == 0:
        return {"method": "bh_fdr", "alpha": alpha, "n_tests": 0, "q_values": [], "significant": []}

    order = np.argsort(p_valid)
    ranks = np.arange(1, m + 1)
    thresholds = ranks * alpha / m

    # Find the largest k where p(k) <= k*alpha/m
    significant = np.zeros(m, dtype=bool)
    k_max = -1
    for i, idx in enumerate(order):
        if p_valid[idx] <= thresholds[i]:
            k_max = i
        else:
            break

    if k_max >= 0:
        for i in range(k_max + 1):
            significant[order[i]] = True

    # BH q-values
    q_values = _bh_qvalues(p_valid, order)

    full_significant = np.zeros(len(p), dtype=bool)
    full_significant[valid_idx] = significant

    return {
        "method": "bh_fdr",
        "alpha": alpha,
        "n_tests": m,
        "n_significant": int(significant.sum()),
        "significant": [bool(s) for s in full_significant],
        "q_values": q_values,
    }


def _bh_qvalues(p_valid: np.ndarray, order: np.ndarray) -> list[float]:
    """Compute Benjamini-Hochberg q-values."""
    m = len(p_valid)
    q = np.ones(m)
    for i, idx in enumerate(order):
        q[idx] = min(p_valid[idx] * m / (i + 1), 1.0)
    # Ensure monotonicity (descending)
    for i in range(m - 2, -1, -1):
        idx_curr = order[i]
        idx_next = order[i + 1]
        q[idx_curr] = min(q[idx_curr], q[idx_next])
    return [float(v) for v in q]


def factor_pvalue_from_ic(ic_series: np.ndarray, two_sided: bool = True) -> float:
    """Convert IC series to an approximate p-value via t-test on mean.

    Uses standard t-test (not NW) for the p-value estimate.
    For NW-corrected, use newey_west_tstat from statistics.py.
    """
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[~np.isnan(ic)]
    n = len(ic)
    if n < 2:
        return 1.0
    mean = ic.mean()
    se = ic.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if mean != 0 else 1.0
    from scipy import stats

    t_stat = mean / se
    if two_sided:
        p = 2.0 * stats.t.sf(abs(t_stat), df=n - 1)
    else:
        p = stats.t.sf(t_stat, df=n - 1)
    return float(p)


def correct_factor_batch(
    factor_names: list[str],
    ic_series_list: list[np.ndarray],
    alpha: float = 0.05,
    method: str = "bh_fdr",
) -> dict:
    """Apply multiple testing correction to a batch of factors.

    Returns dict with per-factor q-values and which are significant.
    """
    p_values = np.array([factor_pvalue_from_ic(ic) for ic in ic_series_list])

    if method == "bonferroni":
        result = bonferroni(p_values, alpha)
    elif method == "holm":
        result = holm(p_values, alpha)
    elif method == "bh_fdr":
        result = benjamini_hochberg(p_values, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    per_factor = []
    for i, name in enumerate(factor_names):
        per_factor.append({
            "factor_name": name,
            "p_value": float(p_values[i]) if i < len(p_values) else np.nan,
            "q_value": result.get("q_values", [np.nan] * len(factor_names))[i]
            if "q_values" in result and i < len(result["q_values"])
            else np.nan,
            "significant": result["significant"][i] if i < len(result["significant"]) else False,
        })

    return {"method": method, "alpha": alpha, "n_factors": len(factor_names), "factors": per_factor}
