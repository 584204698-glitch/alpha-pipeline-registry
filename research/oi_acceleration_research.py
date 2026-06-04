"""
OI Acceleration Research — 幻方量化视角
=========================================
Theory: OI first derivative (velocity) is standard.
Second derivative (acceleration) reveals inflection points:
  - OI accelerating down + price dropping = forced selling INTENSIFYING → stronger reversal
  - OI decelerating down + price dropping = forced selling EXHAUSTING → weaker reversal

Use case: enhance Deleveraging confidence without touching scanner gates.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "oi_acceleration"
OUT.mkdir(parents=True, exist_ok=True)


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "data_storage_1h.parquet")
    df = df.sort_index()

    gb = df.groupby("symbol")

    # OI delta (velocity) — existing
    df["oi_delta"] = gb["open_interest"].transform(lambda x: x.diff())

    # OI acceleration (2nd derivative)
    df["oi_accel"] = gb["oi_delta"].transform(lambda x: x.diff())

    # Rolling z-scores
    df["oi_delta_z"] = gb["oi_delta"].transform(
        lambda x: (x - x.rolling(48, min_periods=16).mean())
        / x.rolling(48, min_periods=16).std().clip(1e-8)
    )
    df["oi_accel_z"] = gb["oi_accel"].transform(
        lambda x: (x - x.rolling(48, min_periods=16).mean())
        / x.rolling(48, min_periods=16).std().clip(1e-8)
    )

    # Price features
    df["ret_1h"] = gb["close"].transform(lambda x: x.pct_change())
    df["ret_z"] = gb["ret_1h"].transform(
        lambda x: (x - x.rolling(48, min_periods=16).mean())
        / x.rolling(48, min_periods=16).std().clip(1e-8)
    )
    df["close_loc"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).clip(1e-8)

    # Forward returns
    for h in [1, 2, 3, 4, 6, 8]:
        df[f"fwd_{h}h"] = gb["close"].transform(lambda x: x.shift(-h) / x - 1.0)

    return df


def evaluate_quadrant(
    df: pd.DataFrame,
    oi_delta_sign: str,   # "negative" or "positive"
    oi_accel_sign: str,    # "negative" or "positive"
    price_condition: str,  # "down" or "up"
    direction: str,         # "long" or "short"
    hold: int,
    label: str,
) -> dict:
    """Evaluate a specific OI-price quadrant for predictive power."""

    # Build mask
    mask = pd.Series(True, index=df.index)

    # OI delta
    if oi_delta_sign == "negative":
        mask &= df["oi_delta_z"] < -1.5
    else:
        mask &= df["oi_delta_z"] > 1.5

    # OI accel
    if oi_accel_sign == "negative":
        mask &= df["oi_accel_z"] < -1.0
    else:
        mask &= df["oi_accel_z"] > 1.0

    # Price
    if price_condition == "down":
        mask &= df["ret_z"] < -1.5
    elif price_condition == "up":
        mask &= df["ret_z"] > 1.5

    sig = df[mask].copy()
    if len(sig) < 30:
        return {"label": label, "n": len(sig), "error": "too few signals"}

    fwd_col = f"fwd_{hold}h"
    sig["signal_ret"] = sig[fwd_col]
    if direction == "short":
        sig["signal_ret"] = -sig["signal_ret"]

    returns = sig["signal_ret"].dropna() * 10000
    if len(returns) < 30:
        return {"label": label, "n": len(returns), "error": "too few valid returns"}

    gross = returns.sum()
    wins = (returns > 0).sum()
    losses = (returns <= 0).sum()
    pf = abs(returns[returns > 0].sum() / returns[returns <= 0].sum()) if losses > 0 and returns[returns <= 0].sum() != 0 else 999

    # IC
    ic, ic_p = stats.spearmanr(
        df.loc[mask, "ret_z"].fillna(0) if "ret_z" in df.columns else mask.astype(int),
        df.loc[mask, fwd_col].fillna(0),
    )

    # Concentration
    top3_n = returns.nlargest(3).sum()
    top3_pct = top3_n / abs(max(returns.sum(), 1)) * 100

    return {
        "label": label,
        "n": len(returns),
        "gross_bps": round(gross, 1),
        "net_9bps": round(gross - len(returns) * 9, 1),
        "pf": round(pf, 2),
        "hit_rate": round(wins / len(returns), 3),
        "mean_bps": round(returns.mean(), 1),
        "median_bps": round(returns.median(), 1),
        "top3_contribution_pct": round(top3_pct, 0),
        "ic": round(ic, 4) if not np.isnan(ic) else 0,
    }


def run_all() -> list[dict]:
    print("=" * 60)
    print("OI Acceleration Research")
    print("=" * 60)

    t0 = time.time()
    print("\n[1/3] Loading & computing features...")
    df = load_and_prepare()
    print(f"  {len(df):,} rows, features computed")
    n_bars = df.index.get_level_values("timestamp").nunique()
    print(f"  {n_bars} bars")

    # Regime distribution of OI accel
    accel_extreme = (df["oi_accel_z"].abs() > 2.0).sum()
    print(f"  OI accel extreme (±2σ): {accel_extreme} ({accel_extreme/len(df)*100:.1f}%)")

    print("\n[2/3] Testing quadrants...")
    results = []

    configs = [
        # Classic Deleveraging: OI down + price down
        ("negative", "negative", "down", "long", 2, "D1: OI↓加速 + 价跌 → long"),
        ("negative", "positive", "down", "long", 2, "D2: OI↓减速 + 价跌 → long"),
        # Compare: which OI accel state predicts better reversal?
        ("positive", "negative", "up", "long", 2, "U1: OI↑加速 + 价涨 → long"),
        ("positive", "positive", "up", "long", 2, "U2: OI↑减速 + 价涨 → long"),
        # Short scenarios
        ("positive", "negative", "up", "short", 2, "S1: OI↑加速 + 价涨 → short (fade)"),
        ("negative", "positive", "up", "short", 2, "S2: OI↓减速 + 价涨 → short"),
        # Different holds
        ("negative", "negative", "down", "long", 4, "D1 hold=4h"),
        ("negative", "positive", "down", "long", 4, "D2 hold=4h"),
    ]

    for oid, oia, price, direc, hold, label in configs:
        r = evaluate_quadrant(df, oid, oia, price, direc, hold, label)
        results.append(r)
        status = "✅" if r.get("net_9bps", 0) > 0 else "❌"
        if "error" in r:
            print(f"  {status} {label}: {r['error']}")
        else:
            print(f"  {status} {label}: n={r['n']:5d} Net9={r['net_9bps']:8.1f} PF={r['pf']:5.2f} Hit={r['hit_rate']:.1%} Top3={r['top3_contribution_pct']:.0f}%")

    # Cross-sectional: within signals, compare accel states
    print("\n[3/3] Accel vs decel comparison...")
    cs_results = compare_accel_states(df)
    results.extend(cs_results)

    # Save
    output = {
        "version": "v1.0",
        "generated": str(pd.Timestamp.now(tz="UTC")),
        "n_bars": n_bars,
        "n_signals": len(df),
        "results": results,
    }
    path = OUT / "oi_acceleration_results.json"
    path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved: {path}")
    print(f"Total: {time.time() - t0:.0f}s")
    return results


def compare_accel_states(df: pd.DataFrame) -> list[dict]:
    """Within Deleveraging-like conditions, compare accelerating vs decelerating OI."""
    results = []

    # Classic Deleveraging condition
    base = (
        (df["oi_delta_z"] < -1.5) &
        (df["ret_z"] < -1.5)
    )

    # Split by accel
    accel_down = base & (df["oi_accel_z"] < -0.5)
    accel_up = base & (df["oi_accel_z"] > 0.5)

    for name, mask, hold in [
        ("Delev + ACCEL_DOWN", accel_down, 2),
        ("Delev + ACCEL_DOWN", accel_down, 4),
        ("Delev + ACCEL_UP (decel)", accel_up, 2),
        ("Delev + ACCEL_UP (decel)", accel_up, 4),
    ]:
        sig = df[mask].copy()
        if len(sig) < 20:
            results.append({"label": f"{name} h={hold}", "error": f"n={len(sig)}"})
            continue

        returns = sig[f"fwd_{hold}h"].dropna() * 10000
        if len(returns) < 20:
            continue

        gross = returns.sum()
        net9 = gross - len(returns) * 9
        wins = (returns > 0).sum()
        losses = (returns <= 0).sum()
        pf = abs(returns[returns > 0].sum() / returns[returns <= 0].sum()) if losses > 0 else 999

        results.append({
            "label": f"{name} h={hold}h",
            "type": "delev_accel_comparison",
            "n": len(returns),
            "gross_bps": round(gross, 1),
            "net_9bps": round(net9, 1),
            "pf": round(pf, 2),
            "hit_rate": round(wins / len(returns), 3),
            "mean_bps": round(returns.mean(), 1),
            "median_bps": round(returns.median(), 1),
        })

    return results


if __name__ == "__main__":
    run_all()
