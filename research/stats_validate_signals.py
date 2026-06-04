"""
Statistical Validation of Historical Signals
=============================================
Computes IC/ICIR, score distribution, and multiple hypothesis correction
for each alpha's historical signals.

Output: research/stats/signal_validation.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
OUT = ROOT / "research" / "stats"
OUT.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────

def _load_signals(path: Path) -> list[dict]:
    """Load JSONL signals file."""
    signals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("status") in ("SHADOW_SIGNAL", "CONFIRMED", "confirmed", ""):
                    signals.append(d)
    return signals


def _compute_ic(signals: list[dict], data: pd.DataFrame, hold_bars: int = 2) -> dict:
    """Compute IC (rank correlation between event_score and forward return)."""
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}

    scores = []
    fwd_rets = []
    by_alpha: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for s in signals:
        ts = pd.Timestamp(s["timestamp"])
        sym = s["symbol"]
        alpha = s.get("event_type", "?")
        score = s.get("event_score", 0)

        idx = ts_to_idx.get(ts)
        if idx is None:
            continue
        exit_idx = idx + hold_bars
        if exit_idx >= len(ts_list):
            continue
        exit_ts = ts_list[exit_idx]

        try:
            entry_px = float(data.loc[(ts, sym), "close"])
            exit_px = float(data.loc[(exit_ts, sym), "close"])
        except (KeyError, TypeError):
            continue

        direction = s.get("direction", "long")
        if direction == "short":
            ret = (entry_px / exit_px - 1) * 10000
        else:
            ret = (exit_px / entry_px - 1) * 10000

        scores.append(score)
        fwd_rets.append(ret)
        by_alpha[alpha].append((score, ret))

    results = {
        "n_valid": len(scores),
        "overall": {},
        "by_alpha": {},
    }

    if len(scores) >= 10:
        s_arr = np.array(scores)
        r_arr = np.array(fwd_rets)
        ic = np.corrcoef(s_arr, r_arr)[0, 1] if len(s_arr) > 1 else 0
        ic_rank = np.corrcoef(
            pd.Series(s_arr).rank(), pd.Series(r_arr).rank()
        )[0, 1] if len(s_arr) > 1 else 0
        results["overall"] = {
            "ic_pearson": round(ic, 4),
            "ic_spearman": round(ic_rank, 4),
            "mean_score": round(s_arr.mean(), 4),
            "std_score": round(s_arr.std(), 4),
            "mean_fwd_bps": round(r_arr.mean(), 1),
            "std_fwd_bps": round(r_arr.std(), 1),
        }

    for alpha, pairs in sorted(by_alpha.items()):
        if len(pairs) < 5:
            continue
        s_arr = np.array([p[0] for p in pairs])
        r_arr = np.array([p[1] for p in pairs])
        ic = np.corrcoef(s_arr, r_arr)[0, 1]
        ic_rank = np.corrcoef(
            pd.Series(s_arr).rank(), pd.Series(r_arr).rank()
        )[0, 1]
        t_stat = ic * np.sqrt(len(pairs) - 2) / np.sqrt(1 - ic**2) if abs(ic) < 1 else 0
        results["by_alpha"][alpha] = {
            "n": len(pairs),
            "ic_pearson": round(ic, 4),
            "ic_spearman": round(ic_rank, 4),
            "t_statistic": round(t_stat, 3),
            "mean_score": round(s_arr.mean(), 4),
            "mean_fwd_bps": round(r_arr.mean(), 1),
            "hit_rate": round((r_arr > 0).mean(), 3),
            "score_percentiles": {
                "p10": round(np.percentile(s_arr, 10), 4),
                "p25": round(np.percentile(s_arr, 25), 4),
                "p50": round(np.percentile(s_arr, 50), 4),
                "p75": round(np.percentile(s_arr, 75), 4),
                "p90": round(np.percentile(s_arr, 90), 4),
            },
        }

    return results


def _multiple_testing_correction(alpha_results: dict) -> dict:
    """Apply Benjamini-Hochberg FDR correction to IC p-values."""
    from scipy import stats as scipy_stats

    entries = []
    for alpha, info in alpha_results.items():
        ic = info.get("ic_pearson", 0)
        n = info.get("n", 0)
        if n < 5:
            continue
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2) if abs(ic) < 1 else 0
        p_val = 2 * scipy_stats.t.sf(abs(t_stat), df=n - 2)
        entries.append({
            "alpha": alpha,
            "ic": ic,
            "n": n,
            "t_stat": t_stat,
            "p_value": p_val,
        })

    if not entries:
        return {"error": "No valid alphas"}

    df = pd.DataFrame(entries)
    df = df.sort_values("p_value")

    # BH-FDR
    m = len(df)
    df["rank"] = range(1, m + 1)
    df["bh_critical"] = df["rank"] / m * 0.05
    df["significant_bh05"] = df["p_value"] <= df["bh_critical"]

    # Bonferroni
    df["significant_bonf"] = df["p_value"] <= 0.05 / m

    results = {}
    for _, row in df.iterrows():
        results[row["alpha"]] = {
            "ic": round(row["ic"], 4),
            "n": int(row["n"]),
            "t_statistic": round(row["t_stat"], 3),
            "p_value": float(f"{row['p_value']:.2e}"),
            "significant_bh05": bool(row["significant_bh05"]),
            "significant_bonferroni": bool(row["significant_bonf"]),
        }

    return results


def _signal_density(signals: list[dict]) -> dict:
    """Compute signal density per day and per symbol."""
    if not signals:
        return {"error": "No signals"}

    by_day: dict[str, int] = defaultdict(int)
    by_symbol: dict[str, int] = defaultdict(int)
    by_alpha_day: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for s in signals:
        ts = pd.Timestamp(s["timestamp"])
        day = ts.strftime("%Y-%m-%d")
        sym = s["symbol"]
        alpha = s.get("event_type", "?")

        by_day[day] += 1
        by_symbol[sym] += 1
        by_alpha_day[alpha][day] += 1

    days = sorted(by_day.keys())
    n_days = len(days)
    per_day = [by_day[d] for d in days]

    symbols = sorted(by_symbol.keys())
    n_symbols = len(symbols)
    per_symbol = [by_symbol[s] for s in symbols]

    results = {
        "total_signals": len(signals),
        "n_days": n_days,
        "n_symbols": n_symbols,
        "mean_per_day": round(np.mean(per_day), 1),
        "median_per_day": round(np.median(per_day), 1),
        "max_per_day": int(np.max(per_day)),
        "mean_per_symbol": round(np.mean(per_symbol), 1),
        "median_per_symbol": round(np.median(per_symbol), 1),
        "max_per_symbol": int(np.max(per_symbol)),
        "by_alpha_per_day": {},
    }

    for alpha, day_counts in sorted(by_alpha_day.items()):
        vals = [day_counts.get(d, 0) for d in days]
        results["by_alpha_per_day"][alpha] = {
            "mean": round(np.mean(vals), 1),
            "median": round(np.median(vals), 1),
            "max": int(np.max(vals)),
        }

    return results


# ── Main ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Signal Statistical Validation")
    print("=" * 60)

    # Load data
    t0 = pd.Timestamp.now()
    print("\n[1/4] Loading data...")
    data = pd.read_parquet(ROOT / "data" / "data_storage_1h.parquet")
    print(f"  {len(data):,} rows, {data.index.get_level_values('symbol').nunique()} symbols")

    # Load signals
    print("\n[2/4] Loading signals...")
    signal_path = ROOT / "logs" / "shadow" / "historical_signals.jsonl"
    if not signal_path.exists():
        # Fall back to the simulated orders (only filled ones)
        orders_path = ROOT / "logs" / "shadow" / "simulated_orders.jsonl"
        if orders_path.exists():
            print(f"  Using simulated orders: {orders_path}")
            signals = _load_signals(orders_path)
        else:
            print("  No signal file found!")
            return
    else:
        signals = _load_signals(signal_path)
    print(f"  {len(signals)} signals loaded")

    if len(signals) == 0:
        print("  No signals to analyze!")
        return

    # IC computation
    print("\n[3/4] Computing IC...")
    ic_results = _compute_ic(signals, data)
    print(f"  Valid signals with fwd returns: {ic_results['n_valid']}")
    if ic_results["overall"]:
        o = ic_results["overall"]
        print(f"  Overall: IC_pearson={o['ic_pearson']:.4f} IC_spearman={o['ic_spearman']:.4f}")

    # Multiple testing correction
    print("\n[4/4] Multiple hypothesis testing...")
    mt_results = _multiple_testing_correction(ic_results["by_alpha"])
    for alpha, info in mt_results.items():
        sig = "✅" if info.get("significant_bh05") else "❌"
        print(f"  {sig} {alpha}: IC={info['ic']:.4f} p={info['p_value']} BH-FDR: {info['significant_bh05']}")

    # Signal density
    density = _signal_density(signals)

    # Assemble final report
    report = {
        "version": "v1.0",
        "generated": str(pd.Timestamp.now(tz="UTC")),
        "data_range": f"{data.index.get_level_values('timestamp').min()} → {data.index.get_level_values('timestamp').max()}",
        "total_signals_loaded": len(signals),
        "ic_analysis": ic_results,
        "multiple_testing": mt_results,
        "signal_density": density,
        "gate_calibration_suggestions": _suggest_gates(ic_results, density),
    }

    path = OUT / "signal_validation.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {path}")
    print(f"Done in {(pd.Timestamp.now() - t0).total_seconds():.0f}s")


def _suggest_gates(ic_results: dict, density: dict) -> dict:
    """Suggest gate calibration based on stats."""
    suggestions = {}

    # Score threshold suggestion
    by_alpha = ic_results.get("by_alpha", {})
    for alpha, info in by_alpha.items():
        pcts = info.get("score_percentiles", {})
        n = info.get("n", 0)
        ic = info.get("ic_pearson", 0)

        # If IC is significant, suggest score threshold at p25
        if ic > 0.02 and n > 20:
            suggestions[alpha] = {
                "status": "KEEP",
                "min_score_threshold": round(pcts.get("p25", 0.1), 3),
                "rationale": f"IC={ic:.4f}, use p25 as min score to filter noise",
            }
        elif ic < 0:
            suggestions[alpha] = {
                "status": "REVIEW",
                "rationale": f"Negative IC ({ic:.4f}) — event_score inversely predicts returns, check scoring direction",
            }
        elif n < 10:
            suggestions[alpha] = {
                "status": "TOO_FEW",
                "rationale": f"Only {n} valid events, insufficient for gate calibration",
            }
        else:
            suggestions[alpha] = {
                "status": "MARGINAL",
                "min_score_threshold": round(pcts.get("p50", 0.5), 3),
                "rationale": f"IC={ic:.4f} marginal, use p50 as threshold",
            }

    # Density suggestion
    mean_per_day = density.get("mean_per_day", 0)
    if mean_per_day > 20:
        suggestions["density"] = {
            "status": "TOO_DENSE",
            "action": "Raise score thresholds or add per-day cap",
            "current_mean_per_day": mean_per_day,
        }
    elif mean_per_day < 1:
        suggestions["density"] = {
            "status": "SPARSE",
            "action": "Consider relaxing some gates if signals are high quality",
            "current_mean_per_day": mean_per_day,
        }
    else:
        suggestions["density"] = {
            "status": "REASONABLE",
            "current_mean_per_day": mean_per_day,
        }

    return suggestions


if __name__ == "__main__":
    main()
