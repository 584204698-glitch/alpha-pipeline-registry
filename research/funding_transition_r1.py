"""
Funding Regime Transition Research — Round 1
=============================================
Settlement-aware funding research. Studies:
1. Funding Extreme Persistence
2. Funding Mean-Reversion (extreme → neutral)
3. Pre/Post Settlement Effect
4. Funding + CVD/OI Confirmation

Strictly no future funding data in return calculations.
Output: research/funding_transition/funding_transition_r1.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
OUT = ROOT / "research" / "funding_transition"
OUT.mkdir(parents=True, exist_ok=True)

HOLD_BARS = [1, 2, 3, 6]
COST_BPS = [9, 12, 15]
FUNDING_PERIODS = 8  # typical funding settlement every 8h


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy().sort_index()
    gb = df.groupby("symbol")

    df["funding_z"] = gb["funding_rate"].transform(
        lambda s: (s - s.rolling(24, min_periods=8).mean())
        / s.rolling(24, min_periods=8).std().clip(1e-8)
    ).fillna(0)

    # Funding delta (change over N bars)
    for w in [4, 8, 12]:
        df[f"funding_delta_{w}h"] = gb["funding_rate"].transform(lambda s: s.diff(w))

    # Persistence: how many consecutive bars with extreme funding
    for threshold in [-1.5, -2.0, -2.5]:
        extreme = df["funding_z"] < threshold
        df[f"funding_persistence_{abs(threshold)}"] = (
            extreme.groupby("symbol").transform(
                lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
            ) * extreme.astype(int)
        )

    # Settlement hour detection (funding typically settles at 00, 08, 16 UTC)
    df["hour"] = df.index.get_level_values("timestamp").hour
    settlement_hours = [0, 8, 16]
    df["near_settlement"] = df["hour"].isin(
        [(h - 1) % 24 for h in settlement_hours] + settlement_hours + [(h + 1) % 24 for h in settlement_hours]
    )
    df["pre_settlement"] = df["hour"].isin([(h - 2) % 24 for h in settlement_hours] + [(h - 1) % 24 for h in settlement_hours])
    df["post_settlement"] = df["hour"].isin([(h + 1) % 24 for h in settlement_hours] + [(h + 2) % 24 for h in settlement_hours])

    # Standard features
    df["ret_1h"] = gb["close"].transform(lambda s: s.pct_change(1))
    df["ret_6h"] = gb["close"].transform(lambda s: s.pct_change(6))
    df["vol_z"] = gb["volume"].transform(
        lambda s: (s - s.rolling(12, min_periods=4).mean())
        / s.rolling(12, min_periods=4).std().clip(1e-8)
    ).fillna(0)

    # OI delta
    df["oi_delta_z"] = gb["open_interest"].transform(
        lambda s: (s.diff(2) - s.diff(2).rolling(12, min_periods=4).mean())
        / s.diff(2).rolling(12, min_periods=4).std().clip(1e-8)
    ).fillna(0)

    # CVD features
    cvd = df["taker_buy_volume"].fillna(0) - df["taker_sell_volume"].fillna(0)
    df["cvd_z"] = gb.apply(lambda g: (
        (cvd.loc[g.index] - cvd.loc[g.index].rolling(12, min_periods=4).mean())
        / cvd.loc[g.index].rolling(12, min_periods=4).std().clip(1e-8)
    ).fillna(0)).reset_index(level=0, drop=True)

    return df


def compute_fwd_returns(df, event_mask, hold, direction) -> np.ndarray:
    ts_list = sorted(df.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    event_bars = df[event_mask]
    fwd = []
    for (ts, sym), _ in event_bars.iterrows():
        idx = ts_to_idx.get(ts)
        if idx is None:
            continue
        exit_idx = idx + hold
        if exit_idx >= len(ts_list):
            continue
        exit_ts = ts_list[exit_idx]
        try:
            ep = float(df.loc[(ts, sym), "close"])
            xp = float(df.loc[(exit_ts, sym), "close"])
        except KeyError:
            continue
        ret = (xp / ep - 1) * 10000
        if direction == "short":
            ret = -ret
        fwd.append(ret)
    return np.array(fwd)


def evaluate(fwd: np.ndarray) -> dict:
    n = len(fwd)
    if n < 5:
        return {"n": n, "error": "too few"}
    results = {}
    for cost in COST_BPS:
        net = fwd - cost
        pos = net[net > 0].sum()
        neg = abs(net[net < 0].sum())
        results[f"cost_{cost}bps"] = {
            "n": n, "net_bps": round(net.sum(), 1),
            "pf": round(pos / neg, 3) if neg > 0 else 999,
            "hit_rate": round((net > 0).mean(), 3),
        }
    top3 = np.sort(fwd)[-max(1, n // 20):]
    return {
        "n": n, "gross_sum": round(fwd.sum(), 1),
        "top3_contrib": round(top3.sum(), 1),
        "net_wo_top3_9bps": round(fwd.sum() - top3.sum() - (n - len(top3)) * 9, 1),
        "by_cost": results,
    }


def main():
    print("=" * 60)
    print("Funding Regime Transition — Round 1")
    print("=" * 60)

    t0 = pd.Timestamp.now()
    data = pd.read_parquet(ROOT / "data" / "data_storage_1h.parquet")
    print(f"\n[1/3] Data: {len(data):,} rows")

    print("[2/3] Building features...")
    df = build_features(data)

    print("[3/3] Testing 4 funding mechanisms...")
    all_results = {}

    # ── 1. Funding Extreme Persistence ──
    print("\n  [1] Funding Extreme Persistence...")
    fp_results = {}
    for threshold in [1.5, 2.0, 2.5]:
        for dur in [3, 5, 8]:
            mask = df[f"funding_persistence_{threshold}"] >= dur
            n = mask.sum()
            if n < 10:
                continue
            for direction in ["long"]:  # long on extremely negative funding
                for hold in HOLD_BARS:
                    fwd = compute_fwd_returns(df, mask, hold, direction)
                    r = evaluate(fwd)
                    key = f"persist_z{threshold}_d{dur}_{direction}_h{hold}"
                    r["event_type"] = "extreme_persistence"
                    r["threshold"] = threshold
                    r["min_duration"] = dur
                    fp_results[key] = r
    all_results["extreme_persistence"] = fp_results

    # ── 2. Funding Mean-Reversion ──
    print("  [2] Funding Mean-Reversion...")
    mr_results = {}
    for extreme_thresh in [2.0, 2.5]:
        extreme_now = df["funding_z"] < -extreme_thresh
        reverting = df["funding_delta_8h"] > 0.5  # 8h improvement
        for price_ok in [True, False]:
            mask = extreme_now & reverting
            if price_ok:
                mask = mask & (df["ret_6h"] > -0.03)  # price not crashing
            n = mask.sum()
            if n < 10:
                continue
            for direction in ["long"]:
                for hold in HOLD_BARS:
                    fwd = compute_fwd_returns(df, mask, hold, direction)
                    r = evaluate(fwd)
                    key = f"mr_z{extreme_thresh}_priceOk{price_ok}_{direction}_h{hold}"
                    r["event_type"] = "mean_reversion"
                    r["extreme_threshold"] = extreme_thresh
                    r["price_filter"] = price_ok
                    mr_results[key] = r
    all_results["mean_reversion"] = mr_results

    # ── 3. Pre/Post Settlement ──
    print("  [3] Pre/Post Settlement Effect...")
    ss_results = {}
    # Pre-settlement: extreme funding + approaching settlement
    for period in ["pre", "post"]:
        period_mask = df[f"{period}_settlement"]
        funding_mask = df["funding_z"] < -2.0
        mask = period_mask & funding_mask
        n = mask.sum()
        if n < 10:
            continue
        for direction in ["long"]:
            for hold in [1, 2, 3]:
                fwd = compute_fwd_returns(df, mask, hold, direction)
                r = evaluate(fwd)
                if abs(r.get("n", 0)) < 5:
                    continue
                key = f"{period}_settle_fz2_{direction}_h{hold}"
                r["event_type"] = f"{period}_settlement"
                # Split price vs funding PnL (simplified: all is price PnL since we don't have settlement-level funding data)
                r["funding_pnl_bps"] = 0  # Cannot compute without actual settlement data
                r["price_pnl_bps"] = r.get("gross_sum", 0)
                ss_results[key] = r
    all_results["settlement_effect"] = ss_results

    # ── 4. Funding + CVD/OI Confirmation ──
    print("  [4] Funding + CVD/OI Confirmation...")
    fc_results = {}
    # Extreme funding + CVD absorption (sell pressure absorbed)
    for fz_thresh in [2.0, 2.5]:
        f_mask = df["funding_z"] < -fz_thresh
        c_mask = df["cvd_z"] < -1.5  # CVD sell pressure
        # Price not falling despite CVD sell + extreme funding = absorption
        mask = f_mask & c_mask & (df["ret_6h"] > -0.02)
        n = mask.sum()
        if n < 10:
            continue
        for direction in ["long"]:
            for hold in HOLD_BARS:
                fwd = compute_fwd_returns(df, mask, hold, direction)
                r = evaluate(fwd)
                key = f"fz{fz_thresh}_cvd15_priceOk_{direction}_h{hold}"
                r["event_type"] = "funding_cvd_confirmation"
                fc_results[key] = r

    # OI confirmation: extreme funding + OI not collapsing
    for fz_thresh in [2.0, 2.5]:
        f_mask = df["funding_z"] < -fz_thresh
        oi_mask = df["oi_delta_z"] > -1.0  # OI not collapsing
        mask = f_mask & oi_mask
        n = mask.sum()
        if n < 10:
            continue
        for direction in ["long"]:
            for hold in HOLD_BARS:
                fwd = compute_fwd_returns(df, mask, hold, direction)
                r = evaluate(fwd)
                key = f"fz{fz_thresh}_oiOk_{direction}_h{hold}"
                r["event_type"] = "funding_oi_confirmation"
                fc_results[key] = r
    all_results["funding_cvd_oi_confirmation"] = fc_results

    # ── Summary ──
    print("\n  === Round 1 Summary ===")
    for category, results in all_results.items():
        if not results:
            print(f"  {category}: no valid results")
            continue
        best = max(results.values(), key=lambda r: r.get("by_cost", {}).get("cost_12bps", {}).get("net_bps", -999999))
        c12 = best.get("by_cost", {}).get("cost_12bps", {})
        c9 = best.get("by_cost", {}).get("cost_9bps", {})
        status = "✅" if c12.get("net_bps", 0) > 0 and c12.get("pf", 0) >= 1.15 else "❌"
        print(f"  {status} {category}: n={c9.get('n',0)} Net9={c9.get('net_bps',0):.0f} Net12={c12.get('net_bps',0):.0f} PF={c12.get('pf',0):.2f}")

    # ── Save ──
    report = {
        "version": "v1.0_round1",
        "timestamp": str(pd.Timestamp.now(tz="UTC")),
        "categories": all_results,
    }
    path = OUT / "funding_transition_r1.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {path}")
    print(f"Done in {(pd.Timestamp.now() - t0).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
