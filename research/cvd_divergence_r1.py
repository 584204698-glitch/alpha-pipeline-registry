"""
CVD Divergence Research — Round 1 (VECTORIZED)
===============================================
Fully vectorized using groupby shift for forward returns.
Output: research/cvd/cvd_divergence_research_report.json
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
OUT = ROOT / "research" / "cvd"
OUT.mkdir(parents=True, exist_ok=True)

HOLD_BARS = [1, 2, 3, 6]
COST_BPS = [9, 12, 15]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    gb = df.groupby("symbol")
    buy = df["taker_buy_volume"].fillna(0)
    sell = df["taker_sell_volume"].fillna(0)
    cvd_raw = buy - sell
    df["cvd_cum"] = gb["taker_buy_volume"].transform(lambda s: (s.fillna(0) - df.loc[s.index, "taker_sell_volume"].fillna(0)).cumsum())

    for w in [6, 12]:
        df[f"cvd_delta_{w}h"] = gb["cvd_cum"].transform(lambda s: s.diff(w))
        df[f"ret_{w}h"] = gb["close"].transform(lambda s: s.pct_change(w))

    df["cvd_z"] = gb["cvd_delta_12h"].transform(
        lambda s: (s - s.rolling(24, min_periods=8).mean())
        / s.rolling(24, min_periods=8).std().clip(1e-8)
    ).fillna(0)

    df["price_ret_z"] = gb["ret_6h"].transform(
        lambda s: (s - s.rolling(24, min_periods=8).mean())
        / s.rolling(24, min_periods=8).std().clip(1e-8)
    ).fillna(0)

    df["absorption_ratio"] = abs(df["cvd_z"]) / (abs(df["price_ret_z"]) + 0.1)
    df["cvd_price_divergence"] = -np.sign(df["cvd_delta_6h"]) * df["ret_6h"]

    df["vol_z"] = gb["volume"].transform(
        lambda s: (s - s.rolling(12, min_periods=4).mean())
        / s.rolling(12, min_periods=4).std().clip(1e-8)
    ).fillna(0)

    hl_range = (df["high"] - df["low"]).clip(lower=1e-8)
    df["close_loc"] = (df["close"] - df["low"]) / hl_range
    df["ret_6h_rank"] = df.groupby(level="timestamp")["ret_6h"].transform(lambda s: s.rank(pct=True))
    df["absorption_rank"] = df.groupby(level="timestamp")["absorption_ratio"].transform(lambda s: s.rank(pct=True))
    return df


def evaluate(fwd: np.ndarray) -> dict:
    n = len(fwd)
    if n < 5:
        return {"n": n, "error": "too few"}
    results = {}
    for cost in COST_BPS:
        net = fwd - cost
        pos = net[net > 0].sum()
        neg = abs(net[net < 0].sum())
        results[f"{cost}bps"] = {
            "n": n, "net_bps": round(net.sum(), 1),
            "pf": round(pos / neg, 3) if neg > 0 else 999,
            "hit_rate": round((net > 0).mean(), 3),
        }
    top3 = np.sort(fwd)[-max(1, n // 20):]
    return {
        "n": n, "gross": round(fwd.sum(), 1),
        "top3_contrib": round(top3.sum(), 1),
        "net_wo_top3_9bps": round(fwd.sum() - top3.sum() - (n - len(top3)) * 9, 1),
        "by_cost": results,
    }


def main():
    print("=" * 60)
    print("CVD Divergence — Round 1 (vectorized)")
    print("=" * 60)
    t0 = pd.Timestamp.now()

    data = pd.read_parquet(ROOT / "data" / "data_storage_1h.parquet")
    print(f"[1/3] Data: {len(data):,} rows")
    df = build_features(data)
    print(f"[2/3] Features: {len(df.columns)}")

    # Pre-compute all forward returns (vectorized)
    gb = df.groupby("symbol")
    fwd_returns = {}
    for hold in HOLD_BARS:
        fwd_close = gb["close"].shift(-hold)
        fwd_ret = (fwd_close / df["close"] - 1) * 10000
        fwd_returns[hold] = fwd_ret

    # Event definitions
    events = {
        "cvd_z_lt_neg2_ret_gt_neg1pct": (df["cvd_z"] < -2.0) & (df["ret_6h"] > -0.01),
        "cvd_z_lt_neg2_close_loc_gt_05": (df["cvd_z"] < -2.0) & (df["close_loc"] > 0.5),
        "cvd_z_lt_neg2_ret_rank_gt_40pct": (df["cvd_z"] < -2.0) & (df["ret_6h_rank"] > 0.4),
        "absorption_ratio_top5": df["absorption_ratio"] >= df["absorption_ratio"].quantile(0.95),
        "cvd_divergence_vol_z_gt_1": (df["cvd_price_divergence"] > df["cvd_price_divergence"].quantile(0.90)) & (df["vol_z"] > 1.0),
        "cvd_divergence_all": df["cvd_price_divergence"] > df["cvd_price_divergence"].quantile(0.90),
    }

    print(f"[3/3] Testing {len(events)} definitions × {len(HOLD_BARS)} holds × 2 directions")
    all_results = {}

    for name, mask in events.items():
        n_events = mask.sum()
        if n_events < 10:
            all_results[name] = {"n_events": int(n_events), "error": "too few"}
            continue

        ev_results = {}
        best_net = -999999
        best_key = None

        for direction in ["long", "short"]:
            sign = -1 if direction == "short" else 1
            for hold in HOLD_BARS:
                fwd_raw = fwd_returns[hold] * sign
                fwd_vals = fwd_raw[mask].dropna().values
                r = evaluate(fwd_vals)
                key = f"{direction}_h{hold}"
                ev_results[key] = r
                net12 = r.get("by_cost", {}).get("12bps", {}).get("net_bps", -999999)
                if net12 > best_net and r.get("n", 0) >= 10:
                    best_net = net12
                    best_key = key

        ev_results["best"] = {"key": best_key, "net_12bps": best_net}
        all_results[name] = ev_results
        c12 = ev_results.get(best_key, {}).get("by_cost", {}).get("12bps", {}) if best_key else {}
        status = "✅" if best_net > 0 and c12.get("pf", 0) >= 1.15 else "❌"
        print(f"  {status} {name}: n={c12.get('n',0)} Net12={best_net:.0f} PF={c12.get('pf',0):.2f}")

    report = {
        "version": "v1.0_round1",
        "timestamp": str(pd.Timestamp.now(tz="UTC")),
        "results": all_results,
    }
    path = OUT / "cvd_divergence_research_report.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {path}")
    print(f"Time: {(pd.Timestamp.now() - t0).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
