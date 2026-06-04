"""
CVD Data Audit — mandatory before any CVD-based alpha research.
Output: research/cvd/data_audit_cvd.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
OUT = ROOT / "research" / "cvd"
OUT.mkdir(parents=True, exist_ok=True)


def audit(data: pd.DataFrame) -> dict:
    results = {
        "version": "v1.0",
        "timestamp": str(pd.Timestamp.now(tz="UTC")),
        "data_shape": list(data.shape),
        "columns": list(data.columns),
        "checks": {},
    }

    # ── Check 1: Column existence ──
    required = ["taker_buy_volume", "taker_sell_volume"]
    for col in required:
        results["checks"][f"column_{col}"] = {
            "exists": col in data.columns,
            "dtype": str(data[col].dtype) if col in data.columns else None,
        }

    if not all(col in data.columns for col in required):
        results["checks"]["verdict"] = "DATA_FAIL"
        results["checks"]["reason"] = f"Missing columns: {[c for c in required if c not in data.columns]}"
        return results

    buy = data["taker_buy_volume"]
    sell = data["taker_sell_volume"]

    # ── Check 2: Missing rate ──
    buy_missing = buy.isna().mean()
    sell_missing = sell.isna().mean()
    results["checks"]["missing_rate"] = {
        "taker_buy": round(buy_missing, 4),
        "taker_sell": round(sell_missing, 4),
    }

    # ── Check 3: All-zero / duplicate rate ──
    buy_zero = (buy.fillna(0) == 0).mean()
    sell_zero = (sell.fillna(0) == 0).mean()
    results["checks"]["zero_rate"] = {
        "taker_buy": round(buy_zero, 4),
        "taker_sell": round(sell_zero, 4),
    }

    # ── Check 4: Buy + sell vs total volume ──
    total_taker = buy.fillna(0) + sell.fillna(0)
    total_volume = data["volume"].fillna(0)
    ratio = (total_taker / total_volume.replace(0, np.nan)).dropna()
    ratio = ratio.clip(0, 100)  # remove absurd outliers
    results["checks"]["taker_to_volume_ratio"] = {
        "mean": round(ratio.mean(), 3),
        "median": round(ratio.median(), 3),
        "p10": round(ratio.quantile(0.10), 3),
        "p90": round(ratio.quantile(0.90), 3),
        "pct_in_0_2": round(((ratio >= 0) & (ratio <= 2)).mean(), 4),
        "pct_0": round((ratio == 0).mean(), 4),
    }

    # ── Check 5: buy_ratio in [0, 1] ──
    buy_ratio = (buy.fillna(0) / total_taker.replace(0, np.nan)).dropna()
    buy_ratio = buy_ratio.clip(0, 2)  # allow small float errors near boundaries
    results["checks"]["buy_ratio"] = {
        "mean": round(buy_ratio.mean(), 4),
        "pct_in_0_1": round(((buy_ratio >= 0) & (buy_ratio <= 1)).mean(), 4),
        "pct_eq_0": round((buy_ratio == 0).mean(), 4),
        "pct_eq_1": round((buy_ratio == 1).mean(), 4),
        "pct_gt_1": round((buy_ratio > 1).mean(), 4),
    }

    # ── Check 6: CVD vs same-period return correlation ──
    data_sorted = data.sort_index()
    cvd = buy.fillna(0) - sell.fillna(0)
    ret = data_sorted["close"].groupby(level="symbol").transform(
        lambda s: s.pct_change(1)
    )

    # Per-symbol CVD-return corr
    symbols = data_sorted.index.get_level_values("symbol").unique()
    corrs = []
    for sym in symbols[:200]:  # sample for speed
        try:
            sym_mask = data_sorted.index.get_level_values("symbol") == sym
            cvd_sym = cvd[sym_mask].values
            ret_sym = ret[sym_mask].values
            valid = ~(np.isnan(cvd_sym) | np.isnan(ret_sym) | (cvd_sym == 0))
            if valid.sum() < 10:
                continue
            c = np.corrcoef(cvd_sym[valid], ret_sym[valid])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
        except Exception:
            continue

    if corrs:
        corr_arr = np.array(corrs)
        results["checks"]["cvd_return_correlation"] = {
            "n_symbols": len(corrs),
            "mean": round(corr_arr.mean(), 4),
            "median": round(np.median(corr_arr), 4),
            "std": round(corr_arr.std(), 4),
            "pct_significant_positive": round((corr_arr > 0.1).mean(), 4),
            "pct_significant_negative": round((corr_arr < -0.1).mean(), 4),
        }
    else:
        results["checks"]["cvd_return_correlation"] = {"error": "No valid symbols"}

    # ── Check 7: Symbol synchronization anomaly ──
    # Check if CVD across symbols shows suspiciously high cross-correlation
    cvd_z_scores = []
    ts_unique = sorted(data_sorted.index.get_level_values("timestamp").unique())[-200:]
    for ts in ts_unique:
        mask = data_sorted.index.get_level_values("timestamp") == ts
        syms = data_sorted.index.get_level_values("symbol")[mask]
        cvd_vals = cvd[mask].fillna(0).values
        if len(cvd_vals) > 20:
            z = (cvd_vals - cvd_vals.mean()) / (cvd_vals.std() + 1e-8)
            cvd_z_scores.append(z[:20])  # first 20 for consistency

    if len(cvd_z_scores) > 50:
        cross_corr = np.corrcoef(np.array(cvd_z_scores).T)
        upper_tri = cross_corr[np.triu_indices_from(cross_corr, k=1)]
        results["checks"]["symbol_cvd_synchronization"] = {
            "mean_cross_corr": round(upper_tri.mean(), 4),
            "max_cross_corr": round(upper_tri.max(), 4),
            "pct_gt_0_8": round((abs(upper_tri) > 0.8).mean(), 4),
            "interpretation": "OK" if upper_tri.mean() < 0.3 else "WARNING: high synchronization — CVD may not be independent across symbols",
        }

    # ── Final verdict ──
    checks = results["checks"]
    failures = []

    if buy_missing > 0.3 or sell_missing > 0.3:
        failures.append(f"Missing rate too high: buy={buy_missing:.1%} sell={sell_missing:.1%}")
    if buy_zero > 0.5 or sell_zero > 0.5:
        failures.append(f"Zero rate too high: buy={buy_zero:.1%} sell={sell_zero:.1%}")
    if checks["taker_to_volume_ratio"]["pct_in_0_2"] < 0.5:
        failures.append("Taker/volume ratio outside [0,2] for >50% of rows")
    if checks["buy_ratio"]["pct_in_0_1"] < 0.90:
        failures.append(f"Buy ratio outside [0,1] for {1-checks['buy_ratio']['pct_in_0_1']:.1%} of rows")

    if failures:
        results["checks"]["verdict"] = "DATA_FAIL"
        results["checks"]["failures"] = failures
    else:
        results["checks"]["verdict"] = "DATA_PASS"

    return results


def main():
    print("=" * 60)
    print("CVD Data Audit")
    print("=" * 60)

    parquet_path = ROOT / "data" / "data_storage_1h.parquet"
    print(f"\nLoading: {parquet_path}")
    data = pd.read_parquet(parquet_path)
    print(f"  {len(data):,} rows, {data.index.get_level_values('symbol').nunique()} symbols")

    results = audit(data)

    # Print summary
    verdict = results["checks"].get("verdict", "?")
    print(f"\nVerdict: {verdict}")
    for check_name, check_data in results["checks"].items():
        if isinstance(check_data, dict) and "exists" in check_data:
            status = "✅" if check_data.get("exists") else "❌"
            print(f"  {status} {check_name}: {check_data}")
        elif isinstance(check_data, dict) and "mean" in check_data:
            print(f"  {check_name}: mean={check_data.get('mean', '?')}")
        elif isinstance(check_data, dict) and "verdict" in check_data:
            pass  # already printed above
        elif check_name == "failures":
            for f in check_data:
                print(f"  ❌ {f}")

    path = OUT / "data_audit_cvd.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {path}")

    return 0 if verdict == "DATA_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
