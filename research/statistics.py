"""Small-sample statistical inference for alpha factor IC series.

Provides:
- Newey-West standard error with automatic lag selection
- Bootstrap confidence intervals for mean IC
- Positive IC ratio
- IC drawdown (removing top contributors)

All functions handle NaN gracefully and return dicts ready for serialization.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------
# Newey-West
# ---------------------------------------------------------------

def newey_west_se(x: np.ndarray, lag: int | None = None) -> float:
    """Newey-West HAC standard error of the mean.

    Auto-selects lag via Newey-West (1994) rule: floor(4*(T/100)^(2/9))
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 3:
        return float(np.std(x, ddof=1) / np.sqrt(max(T, 1))) if T > 0 else np.nan

    if lag is None:
        lag = int(4 * (T / 100.0) ** (2.0 / 9.0))
    lag = min(lag, T - 1)

    x_dm = x - x.mean()
    gamma0 = np.dot(x_dm, x_dm) / T
    var = gamma0
    for L in range(1, lag + 1):
        weight = 1.0 - L / (lag + 1.0)
        gamma = np.dot(x_dm[L:], x_dm[:-L]) / T
        var += 2.0 * weight * gamma
    return float(np.sqrt(max(var, 0.0) / T))


def newey_west_tstat(x: np.ndarray, lag: int | None = None) -> dict[str, float]:
    """Return dict with mean, ICIR (naive), and NW t-stat."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return {"mean": 0.0, "std": 0.0, "icir": 0.0, "nw_se": np.nan, "nw_tstat": np.nan}
    mu = float(x.mean())
    sigma = float(x.std(ddof=1))
    icir = mu / sigma if sigma > 0 else 0.0
    se = newey_west_se(x, lag)
    tstat = mu / se if se > 0 else 0.0
    return {"mean": mu, "std": sigma, "icir": icir, "nw_se": se, "nw_tstat": tstat}


# ---------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------

def bootstrap_ci(
    x: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap percentile confidence interval for mean IC."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 5:
        return {"ci_lower": float(np.nan), "ci_upper": float(np.nan), "n_boot": n_boot, "alpha": alpha}

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(x, size=T, replace=True)
        boot_means[i] = sample.mean()

    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return {"ci_lower": lo, "ci_upper": hi, "n_boot": n_boot, "alpha": alpha}


def bootstrap_ci_multi(
    x: np.ndarray,
    alphas: tuple[float, ...] = (0.20, 0.10, 0.05),
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Return bootstrap CIs at multiple alpha levels (80%, 90%, 95%)."""
    result: dict[str, dict[str, float]] = {}
    for a in alphas:
        key = f"ci_{int((1 - a) * 100)}"
        result[key] = bootstrap_ci(x, n_boot=n_boot, alpha=a, seed=seed + int(a * 1000))
    return result


# ---------------------------------------------------------------
# IC quality metrics
# ---------------------------------------------------------------

def positive_ic_ratio(x: np.ndarray) -> float:
    """Fraction of IC values that are strictly positive."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0
    return float((x > 0).mean())


def ic_drawdown(x: np.ndarray, drop_frac: float = 0.05, drop_n: int | None = None) -> dict[str, float]:
    """Compute mean IC after removing the top-contributing bars.

    drop_frac: fraction of top IC bars to remove (default 5%)
    drop_n:   exact number to remove (overrides drop_frac)
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return {"original_mean_ic": 0.0, "trimmed_mean_ic": 0.0, "dropped": 0}
    if drop_n is None:
        drop_n = max(1, int(len(x) * drop_frac))
    drop_n = min(drop_n, len(x) - 1)
    sorted_x = np.sort(x)
    trimmed = sorted_x[:-drop_n]  # remove the largest
    return {
        "original_mean_ic": float(x.mean()),
        "trimmed_mean_ic": float(trimmed.mean()),
        "dropped": drop_n,
    }


# ---------------------------------------------------------------
# Comprehensive factor statistics report
# ---------------------------------------------------------------

def factor_statistical_report(
    ic_series: np.ndarray,
    factor_name: str = "unknown",
    n_boot: int = 5000,
) -> dict:
    """Generate a complete statistical report dict for a factor's IC series."""
    ic = np.asarray(ic_series, dtype=float)
    ic_clean = ic[~np.isnan(ic)]

    nw = newey_west_tstat(ic_clean)
    ci = bootstrap_ci_multi(ic_clean, n_boot=n_boot)
    pos_ratio = positive_ic_ratio(ic_clean)
    dd = ic_drawdown(ic_clean)

    # Decision rules
    decision = "FAIL"
    reasons: list[str] = []

    mean_ic = nw["mean"]
    nw_t = nw["nw_tstat"]
    ci_80_lo = ci.get("ci_80", {}).get("ci_lower", np.nan)
    ci_90_lo = ci.get("ci_90", {}).get("ci_lower", np.nan)
    ci_95_lo = ci.get("ci_95", {}).get("ci_lower", np.nan)

    # Research PASS
    if mean_ic > 0 and (not np.isnan(nw_t) and nw_t > 1.3) and (not np.isnan(ci_80_lo) and ci_80_lo > 0) and pos_ratio > 0.52:
        decision = "RESEARCH_PASS"
    else:
        if mean_ic <= 0:
            reasons.append("mean_ic <= 0")
        if np.isnan(nw_t) or nw_t <= 1.3:
            reasons.append(f"NW_t={nw_t:.2f} <= 1.3")
        if np.isnan(ci_80_lo) or ci_80_lo <= 0:
            reasons.append(f"CI80_lower={ci_80_lo:.4f} <= 0")
        if pos_ratio <= 0.52:
            reasons.append(f"pos_ratio={pos_ratio:.2f} <= 0.52")

    # Paper PASS (stronger)
    if decision == "RESEARCH_PASS":
        if nw_t > 1.8 and (not np.isnan(ci_90_lo) and ci_90_lo > 0) and dd["trimmed_mean_ic"] > 0:
            decision = "PAPER_CANDIDATE"

    return {
        "factor_name": factor_name,
        "n_observations": int(len(ic_clean)),
        "mean_ic": float(mean_ic),
        "std_ic": float(nw["std"]),
        "icir": float(nw["icir"]),
        "naive_tstat": float(mean_ic / (nw["std"] / np.sqrt(max(len(ic_clean), 1))) if nw["std"] > 0 else 0.0),
        "newey_west_se": float(nw["nw_se"]),
        "newey_west_tstat": float(nw_t),
        "bootstrap_ci": ci,
        "positive_ic_ratio": float(pos_ratio),
        "ic_drawdown": dd,
        "decision": decision,
        "failure_reasons": reasons if decision == "FAIL" else [],
    }
