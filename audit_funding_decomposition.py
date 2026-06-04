"""Funding Carry EU Audit 1: funding_pnl vs price_pnl decomposition.

Critical question: Is the alpha from funding carry or from session-based mean-reversion?
"""
from __future__ import annotations

import json, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
sys.path.insert(0, str(ROOT))
from research.regime_detector import detect_regime_fast


def main():
    data = pd.read_parquet(ROOT / "data" / "data_storage.parquet")
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))

    # Universe
    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    universe = set(vol_rank[(vol_rank >= 20) & (vol_rank <= 100)].index)
    universe &= set(data.index.get_level_values("symbol").unique())

    # Precompute funding_z
    fr = data["funding_rate"]
    gf = fr.groupby(level="symbol")
    fm = gf.transform(lambda s: s.rolling(24, min_periods=8).mean())
    fs = gf.transform(lambda s: s.rolling(24, min_periods=8).std()).replace(0, np.nan)
    funding_z = ((fr - fm) / fs).fillna(0.0)

    # Precompute oi_delta_z
    oi_d = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(6))
    go = oi_d.groupby(level="symbol")
    om = go.transform(lambda s: s.rolling(48, min_periods=8).mean())
    os_ = go.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
    oi_delta_z = ((oi_d - om) / os_).fillna(0.0)

    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
    hold_bars = 12

    trades = []
    for i in range(0, len(ts_list) - hold_bars - 1, 4):
        ts = ts_list[i]
        # EU session: 08-16 UTC
        if ts.hour < 8 or ts.hour >= 16:
            continue
        regime = regime_map.get(ts, "unknown")
        if regime == "panic_down":
            continue

        mask = data.index.get_level_values("timestamp") == ts
        syms = set(data.index[mask].get_level_values("symbol")) & universe
        for sym in syms:
            try:
                fz = float(funding_z.loc[(ts, sym)])
                oi_z = float(oi_delta_z.loc[(ts, sym)])
            except (KeyError, TypeError):
                continue
            if np.isnan(fz) or fz > -2.5:
                continue
            if oi_z < -2.0:
                continue  # ongoing liquidation

            idx = ts_to_idx.get(ts)
            if idx is None or idx + hold_bars >= len(ts_list):
                continue
            exit_ts = ts_list[idx + hold_bars]
            try:
                entry_px = float(data.loc[(ts, sym), "close"])
                exit_px = float(data.loc[(exit_ts, sym), "close"])
            except KeyError:
                continue

            # Price PnL
            price_bps = (exit_px / entry_px - 1.0) * 10000

            # Funding PnL: sum funding_rate over hold, annualized correctly
            fund_sum = 0.0
            valid = 0
            for j in range(hold_bars):
                try:
                    fund_sum += float(data.loc[(ts_list[idx + j], sym), "funding_rate"])
                    valid += 1
                except:
                    pass
            # funding_rate < 0 → shorts pay longs. For a LONG position, earned = -rate
            fund_bps = -(fund_sum / max(valid, 1)) * 10000 * (hold_bars * 0.25 / 8)

            total_bps = price_bps + fund_bps

            trades.append({
                "symbol": sym,
                "entry_ts": str(ts),
                "price_bps": round(price_bps, 2),
                "funding_bps": round(fund_bps, 2),
                "total_bps": round(total_bps, 2),
                "fz": round(fz, 2),
                "regime": regime,
            })

    if not trades:
        print("No trades")
        return

    price_arr = np.array([t["price_bps"] for t in trades])
    fund_arr = np.array([t["funding_bps"] for t in trades])
    total_arr = np.array([t["total_bps"] for t in trades])

    print(f"Total trades: {len(trades)}")
    print()

    # Separate by sign
    price_pos = price_arr[price_arr > 0]
    price_neg = price_arr[price_arr < 0]
    fund_pos = fund_arr[fund_arr > 0]
    fund_neg = fund_arr[fund_arr < 0]

    print("=== PnL Decomposition ===")
    print(f"{'Component':<20} {'Sum':>10} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    for name, arr in [("price_pnl", price_arr), ("funding_pnl", fund_arr), ("total_pnl", total_arr)]:
        print(f"{name:<20} {arr.sum():+10.1f} {arr.mean():+10.2f} {np.median(arr):+10.2f} {arr.std():>10.2f} {arr.min():+10.1f} {arr.max():+10.1f}")

    # Contribution breakdown
    print(f"\n=== Contribution ===")
    total_sum = total_arr.sum()
    print(f"Price contributes: {price_arr.sum() / total_sum * 100:.1f}% of total PnL")
    print(f"Funding contributes: {fund_arr.sum() / total_sum * 100:.1f}% of total PnL")

    # Correlation
    corr = np.corrcoef(price_arr, fund_arr)[0, 1]
    print(f"\nPrice-Funding correlation: {corr:.3f}")

    # By funding bucket
    print(f"\n=== By Funding Z-Score Bucket ===")
    for lo, hi in [(-5, -3.5), (-3.5, -3.0), (-3.0, -2.75), (-2.75, -2.5)]:
        mask = (np.array([t["fz"] for t in trades]) >= lo) & (np.array([t["fz"] for t in trades]) < hi)
        if mask.sum() == 0:
            continue
        p = price_arr[mask]
        f = fund_arr[mask]
        t = total_arr[mask]
        print(f"  fz [{lo:.1f}, {hi:.1f}): n={mask.sum():3d} "
              f"price={p.sum():+8.0f} fund={f.sum():+6.0f} total={t.sum():+8.0f} "
              f"pf={(t[t>0].sum()/abs(t[t<0].sum()) if (t<0).any() else 999):.2f}")

    # Net after cost
    print(f"\n=== Net After Cost ===")
    for cost in [9, 12, 15]:
        net = total_arr - 2 * cost
        pos = net[net > 0].sum()
        neg = abs(net[net < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        hit = (net > 0).mean()
        top3 = sum(sorted(net, reverse=True)[:3])
        print(f"  cost_{cost}bps: net={net.sum():+.0f} pf={pf:.2f} hit={hit:.1%} "
              f"top3={top3:.0f} net_no_top3={net.sum()-top3:+.0f}")

    # Save
    out = ROOT / "audit_funding_pnl_decomposition.json"
    summary = {
        "n_trades": len(trades),
        "price_sum": round(price_arr.sum(), 1),
        "funding_sum": round(fund_arr.sum(), 1),
        "total_sum": round(total_arr.sum(), 1),
        "price_mean": round(price_arr.mean(), 2),
        "funding_mean": round(fund_arr.mean(), 2),
        "price_contribution_pct": round(price_arr.sum() / total_sum * 100, 1),
        "funding_contribution_pct": round(fund_arr.sum() / total_sum * 100, 1),
        "price_funding_corr": round(corr, 3),
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
