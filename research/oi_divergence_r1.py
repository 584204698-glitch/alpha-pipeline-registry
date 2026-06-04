"""
OI-Price Multi-Bar Divergence Research — Round 1
=================================================
Studies 4 OI×Price quadrant patterns across multiple windows and thresholds.
Tests overlap with old OI Shock and Deleveraging signals.

Output: research/oi_divergence/oi_price_divergence_r1.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/mnt/e/alpha_pipeline")
OUT = ROOT / "research" / "oi_divergence"
OUT.mkdir(parents=True, exist_ok=True)

HOLD_BARS = [1, 2, 3, 6]
COST_BPS = [9, 12, 15]
WINDOWS = [6, 12, 24]
SPEARMAN_THRESHOLDS = [-0.3, -0.5, -0.7]
MIN_DURATION = [2, 3, 4]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy().sort_index()
    gb = df.groupby("symbol")

    # Price returns
    df["ret_1h"] = gb["close"].transform(lambda s: s.pct_change(1))

    # OI change
    df["oi_delta_1h"] = gb["open_interest"].transform(lambda s: s.diff(1))

    # OI and price rolling slopes (per symbol)
    for w in WINDOWS:
        df[f"oi_slope_{w}h"] = gb["open_interest"].transform(
            lambda s: s.rolling(w, min_periods=4).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 4 else 0, raw=True
            )
        )
        df[f"price_slope_{w}h"] = gb["close"].transform(
            lambda s: s.rolling(w, min_periods=4).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 4 else 0, raw=True
            )
        )

    # OI-Price Spearman correlation (rolling)
    for w in WINDOWS:
        df[f"oi_price_spearman_{w}h"] = gb.apply(
            lambda g: g[["open_interest", "close"]].rolling(w, min_periods=8).apply(
                lambda x: spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0] if len(x) >= 8 else 0, raw=False
            )["close"]
        ).reset_index(level=0, drop=True)

    # Divergence duration (consecutive bars with negative spearman)
    for threshold in SPEARMAN_THRESHOLDS:
        neg_mask = df[f"oi_price_spearman_{12}h"] < threshold
        df[f"div_duration_{threshold}"] = (
            neg_mask.groupby("symbol").transform(
                lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
            ) * neg_mask.astype(int)
        )

    # Divergence acceleration (change in spearman)
    df["div_accel"] = df[f"oi_price_spearman_{12}h"].groupby("symbol").transform(
        lambda s: s.diff(1)
    )

    # Standard features
    df["vol_z"] = gb["volume"].transform(
        lambda s: (s - s.rolling(12, min_periods=4).mean())
        / s.rolling(12, min_periods=4).std().clip(1e-8)
    ).fillna(0)

    hl_range = (df["high"] - df["low"]).clip(lower=1e-8)
    df["close_loc"] = (df["close"] - df["low"]) / hl_range

    df["funding_z"] = gb["funding_rate"].transform(
        lambda s: (s - s.rolling(6, min_periods=4).mean())
        / s.rolling(6, min_periods=4).std().clip(1e-8)
    ).fillna(0)

    df["ret_6h"] = gb["close"].transform(lambda s: s.pct_change(6))

    return df


def quadrant_mask(df: pd.DataFrame, quad: str, window: int) -> pd.Series:
    """Return mask for one of 4 OI×Price quadrants."""
    oi_slope = df[f"oi_slope_{window}h"]
    price_slope = df[f"price_slope_{window}h"]
    oi_delta = df["oi_delta_1h"]

    if quad == "oi_up_price_down":
        return (oi_slope > 0) & (price_slope < 0) & (oi_delta.notna())
    elif quad == "oi_up_price_flat":
        return (oi_slope > 0) & (price_slope.abs() < price_slope.abs().quantile(0.3)) & (oi_delta.notna())
    elif quad == "oi_down_price_up":
        return (oi_slope < 0) & (price_slope > 0) & (oi_delta.notna())
    elif quad == "oi_down_price_flat":
        return (oi_slope < 0) & (price_slope.abs() < price_slope.abs().quantile(0.3)) & (oi_delta.notna())

    return pd.Series(False, index=df.index)


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
            "n": n,
            "net_bps": round(net.sum(), 1),
            "pf": round(pos / neg, 3) if neg > 0 else 999,
            "hit_rate": round((net > 0).mean(), 3),
            "mean_win": round(net[net > 0].mean(), 1) if (net > 0).any() else 0,
            "mean_loss": round(net[net < 0].mean(), 1) if (net < 0).any() else 0,
        }
    top3 = np.sort(fwd)[-max(1, n // 20):]
    return {
        "n": n,
        "gross_sum": round(fwd.sum(), 1),
        "top3_contrib": round(top3.sum(), 1),
        "net_wo_top3_9bps": round(fwd.sum() - top3.sum() - (n - len(top3)) * 9, 1),
        "by_cost": results,
    }


def load_old_signals() -> set:
    """Load old OI Shock + Deleveraging signal (timestamp, symbol) pairs for overlap check."""
    old_signals = set()
    signal_path = ROOT / "logs" / "shadow" / "historical_signals.jsonl"
    if signal_path.exists():
        with open(signal_path) as f:
            for line in f:
                d = json.loads(line.strip()) if line.strip() else {}
                et = d.get("event_type", "")
                if et in ("OIShockAbsorption", "DeleveragingReversal"):
                    old_signals.add((d.get("timestamp", ""), d.get("symbol", "")))
    return old_signals


def main():
    print("=" * 60)
    print("OI-Price Multi-Bar Divergence — Round 1")
    print("=" * 60)

    t0 = pd.Timestamp.now()
    data = pd.read_parquet(ROOT / "data" / "data_storage_1h.parquet")
    print(f"\n[1/4] Data: {len(data):,} rows")

    print("[2/4] Building features (this takes ~30s)...")
    df = build_features(data)
    print(f"  {len(df.columns)} features computed")

    print("[3/4] Testing 4 quadrants × 3 windows × 3 spearman × 3 durations × 4 holds × 2 directions...")
    old_signals = load_old_signals()
    print(f"  Old signals for overlap: {len(old_signals)}")

    QUADRANTS = ["oi_up_price_down", "oi_up_price_flat", "oi_down_price_up", "oi_down_price_flat"]
    QUAD_DIRECTION = {
        "oi_up_price_down": "long",
        "oi_up_price_flat": "long",
        "oi_down_price_up": "short",
        "oi_down_price_flat": "long",
    }
    QUAD_LABEL = {
        "oi_up_price_down": "OI↑ Price↓ (squeeze candidate)",
        "oi_up_price_flat": "OI↑ Price→ (absorption)",
        "oi_down_price_up": "OI↓ Price↑ (distribution)",
        "oi_down_price_flat": "OI↓ Price→ (deleveraging done)",
    }

    all_results = {}
    best_overall = {"net": -999999, "key": None}

    for quad in QUADRANTS:
        print(f"\n  {QUAD_LABEL[quad]}...")
        quad_results = {}

        for window in WINDOWS:
            base_mask = quadrant_mask(df, quad, window)
            for sp_thresh in SPEARMAN_THRESHOLDS:
                for dur in MIN_DURATION:
                    # Combined mask: quadrant + spearman + duration
                    sp_mask = df[f"oi_price_spearman_{window}h"] < sp_thresh
                    dur_mask = df[f"div_duration_{sp_thresh}"] >= dur
                    combined = base_mask & sp_mask & dur_mask
                    n_events = combined.sum()

                    if n_events < 10:
                        continue

                    for direction in [QUAD_DIRECTION[quad], "short"] if quad != "oi_down_price_up" else ["short"]:
                        for hold in HOLD_BARS:
                            fwd = compute_fwd_returns(df, combined, hold, direction)
                            r = evaluate(fwd)
                            key = f"w{window}_sp{abs(sp_thresh)}_d{dur}_{direction}_h{hold}"
                            r["quadrant"] = quad
                            r["window"] = window
                            r["spearman_threshold"] = sp_thresh
                            r["min_duration"] = dur
                            r["direction"] = direction
                            r["hold"] = hold
                            r["n_events"] = int(n_events)

                            # Overlap check
                            event_pairs = set()
                            for (ts, sym), _ in df[combined].iterrows():
                                event_pairs.add((str(ts), sym))
                            overlap = event_pairs & old_signals
                            r["old_alpha_overlap_pct"] = round(len(overlap) / max(len(event_pairs), 1), 3)
                            r["old_alpha_duplicate_risk"] = r["old_alpha_overlap_pct"] > 0.30

                            quad_results[key] = r

                            c12 = r.get("by_cost", {}).get("cost_12bps", {})
                            net12 = c12.get("net_bps", -999999)
                            if net12 > best_overall["net"]:
                                best_overall["net"] = net12
                                best_overall["key"] = key

        all_results[quad] = quad_results

    # ── Summary ──
    print(f"\n[4/4] Best per quadrant:")
    for quad, results in all_results.items():
        if not results:
            print(f"  {QUAD_LABEL[quad]}: no valid results")
            continue
        best = max(results.values(), key=lambda r: r.get("by_cost", {}).get("cost_12bps", {}).get("net_bps", -999999))
        c12 = best.get("by_cost", {}).get("cost_12bps", {})
        c9 = best.get("by_cost", {}).get("cost_9bps", {})
        dup = "⚠️ DUPLICATE" if best.get("old_alpha_duplicate_risk") else ""
        print(f"  {QUAD_LABEL[quad]}: n={c9.get('n',0)} Net9={c9.get('net_bps',0):.0f} Net12={c12.get('net_bps',0):.0f} PF={c12.get('pf',0):.2f} overlap={best.get('old_alpha_overlap_pct',0):.1%} {dup}")

    print(f"\n  Best overall: {best_overall['key']} → Net12={best_overall['net']:.0f}")

    # ── Save ──
    report = {
        "version": "v1.0_round1",
        "timestamp": str(pd.Timestamp.now(tz="UTC")),
        "best_overall": {"key": best_overall["key"], "net_12bps": best_overall["net"]},
        "quadrants": all_results,
    }

    path = OUT / "oi_price_divergence_r1.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {path}")
    print(f"Done in {(pd.Timestamp.now() - t0).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
