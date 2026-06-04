"""
Taker Flow Microstructure Research
===================================
幻方量化视角: Predict when other traders are forced to exit.
Extreme taker buy/sell + OI change → identify trapped counterparties.

Four regimes:
  AGGRESSIVE_LONG:  taker buy >> sell + OI up → fresh longs (bearish fade?)
  SHORT_COVERING:   taker buy >> sell + OI down → shorts trapped (bullish)
  AGGRESSIVE_SHORT: taker sell >> buy + OI up → fresh shorts (bullish fade?)
  LONG_LIQUIDATION: taker sell >> buy + OI down → longs trapped (bearish)

Methodology v2: default guilty, 5-layer audit required.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent  # project root
OUT_DIR = ROOT / "research" / "taker_flow"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load 1h data with taker volume."""
    df = pd.read_parquet(ROOT / "data" / "data_storage_1h.parquet")
    required = ["close", "volume", "open_interest", "taker_buy_volume", "taker_sell_volume"]
    for col in required:
        if col not in df.columns:
            print(f"WARNING: {col} not in data")
    return df


def compute_taker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute taker flow features on full dataset.

    Returns DataFrame with added columns:
      - taker_buy_ratio: buy / (buy + sell)
      - taker_net_volume: buy - sell
      - taker_net_z: z-score of net volume (per symbol)
      - oi_delta: OI change
      - oi_delta_z: z-score of OI change (per symbol)
      - ret_1h: 1h return
      - future_ret_1h/2h/4h: forward returns for signal testing
    """
    result = df.copy()
    result = result.sort_index()

    # Per-symbol computations
    for col in result.columns:
        if result[col].dtype == object:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # Taker ratio and net
    buy = result["taker_buy_volume"].fillna(0)
    sell = result["taker_sell_volume"].fillna(0)
    total = buy + sell
    result["taker_buy_ratio"] = (buy / total.replace(0, np.nan)).clip(0, 1)
    result["taker_net_volume"] = buy - sell

    # Per-symbol rolling z-scores (24-bar lookback)
    grouped = result.groupby("symbol")

    result["taker_net_z"] = grouped["taker_net_volume"].transform(
        lambda x: (x - x.rolling(24, min_periods=8).mean()) / x.rolling(24, min_periods=8).std().clip(1e-8)
    )

    oi = result["open_interest"]
    result["oi_delta"] = grouped["open_interest"].transform(lambda x: x.diff())
    result["oi_delta_z"] = grouped["oi_delta"].transform(
        lambda x: (x - x.rolling(24, min_periods=8).mean()) / x.rolling(24, min_periods=8).std().clip(1e-8)
    )

    result["ret_1h"] = grouped["close"].transform(lambda x: x.pct_change())
    result["ret_vol_z"] = grouped["volume"].transform(
        lambda x: (x - x.rolling(24, min_periods=8).mean()) / x.rolling(24, min_periods=8).std().clip(1e-8)
    )

    # Forward returns for signal evaluation
    for h in [1, 2, 4, 8]:
        result[f"fwd_ret_{h}h"] = grouped["close"].transform(lambda x: x.shift(-h) / x - 1.0)

    return result


def identify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each bar into taker flow regimes."""
    df = df.copy()

    # Thresholds
    TAKER_Z_LONG = 2.0    # extreme taker buying
    TAKER_Z_SHORT = -2.0   # extreme taker selling
    OI_Z_UP = 2.0
    OI_Z_DOWN = -2.0

    tnz = df["taker_net_z"].fillna(0)
    oiz = df["oi_delta_z"].fillna(0)

    conditions = [
        (tnz > TAKER_Z_LONG) & (oiz > 0.5),
        (tnz > TAKER_Z_SHORT) & (oiz < -0.5),
        (tnz < TAKER_Z_SHORT) & (oiz > 0.5),
        (tnz < TAKER_Z_SHORT) & (oiz < -0.5),
    ]
    choices = ["AGGRESSIVE_LONG", "SHORT_COVERING", "AGGRESSIVE_SHORT", "LONG_LIQUIDATION"]
    df["taker_regime"] = np.select(conditions, choices, default="NEUTRAL")

    return df


def evaluate_signals(
    df: pd.DataFrame,
    regime: str,
    direction: str,
    hold_bars: int = 2,
    cost_bps: float = 9.0,
    min_symbols: int = 20,
) -> dict[str, Any]:
    """Evaluate if a taker flow regime predicts forward returns.

    direction: 'long' or 'short'
    """
    mask = df["taker_regime"] == regime
    sig_df = df[mask].copy()

    # Filter to bars with enough symbols
    ts_counts = sig_df.groupby(level="timestamp").size()
    dense_ts = ts_counts[ts_counts >= min_symbols].index
    sig_df = sig_df[sig_df.index.get_level_values("timestamp").isin(dense_ts)]

    if len(sig_df) < 30:
        return {"regime": regime, "error": f"Only {len(sig_df)} signals"}

    # Forward return
    fwd_col = f"fwd_ret_{hold_bars}h"
    sig_df["signal_ret"] = sig_df[fwd_col]
    if direction == "short":
        sig_df["signal_ret"] = -sig_df["signal_ret"]

    returns = sig_df["signal_ret"].dropna() * 10000  # bps
    if len(returns) < 30:
        return {"regime": regime, "error": "Too few valid returns"}

    gross = returns.sum()
    net_9 = gross - len(returns) * cost_bps
    net_12 = gross - len(returns) * 12
    net_15 = gross - len(returns) * 15

    # IC (rank correlation)
    ic, ic_pval = stats.spearmanr(returns, sig_df["signal_ret"].dropna().rank())

    wins = (returns > 0).sum()
    losses = (returns <= 0).sum()
    pf = abs(returns[returns > 0].sum() / returns[returns <= 0].sum()) if losses > 0 and returns[returns <= 0].sum() != 0 else 999

    # Concentration
    top3 = returns.nlargest(3).sum()
    net_wo_top3 = net_9 - top3

    # By symbol
    by_sym = sig_df.groupby("symbol")["signal_ret"].agg(["count", "sum", "mean"])
    sym_positive = (by_sym["sum"] > 0).sum()
    sym_total = len(by_sym)

    return {
        "regime": regime,
        "direction": direction,
        "hold_bars": hold_bars,
        "n_signals": len(returns),
        "n_symbols": sig_df.index.get_level_values("symbol").nunique(),
        "gross_bps": round(gross, 1),
        "net_9bps": round(net_9, 1),
        "net_12bps": round(net_12, 1),
        "net_15bps": round(net_15, 1),
        "pf": round(pf, 2),
        "hit_rate": round(wins / max(len(returns), 1), 3),
        "median_bps": round(returns.median(), 1),
        "mean_bps": round(returns.mean(), 1),
        "std_bps": round(returns.std(), 1),
        "ic": round(ic, 4),
        "ic_pval": round(ic_pval, 4),
        "top3_contribution": round(top3, 1),
        "net_without_top3": round(net_wo_top3, 1),
        "sym_positive_ratio": round(sym_positive / max(sym_total, 1), 3),
        "cost_to_gross_pct": round(cost_bps * len(returns) / abs(max(gross, 1)) * 100, 1),
    }


def run_full_research() -> dict:
    """Run complete taker flow research pipeline."""
    print("=" * 60)
    print("Taker Flow Microstructure Research")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    t0 = time.time()
    df = load_data()
    print(f"  {len(df):,} rows, {df.index.get_level_values('symbol').nunique()} symbols")

    print("\n[2/4] Computing taker features...")
    df = compute_taker_features(df)
    df = identify_regimes(df)

    # Regime distribution
    regime_counts = df["taker_regime"].value_counts()
    print(f"  Regimes: {dict(regime_counts)}")

    print("\n[3/4] Evaluating signals...")
    results = []

    test_configs = [
        # (regime, direction, hold_bars, hypothesis)
        ("SHORT_COVERING", "long", 2, "Shorts trapped → long squeeze"),
        ("SHORT_COVERING", "long", 4, "Shorts trapped → longer squeeze"),
        ("LONG_LIQUIDATION", "short", 2, "Longs trapped → continued sell-off"),
        ("LONG_LIQUIDATION", "short", 4, "Longs trapped → extended decline"),
        ("AGGRESSIVE_LONG", "short", 2, "Extreme buying → mean-revert down"),
        ("AGGRESSIVE_SHORT", "long", 2, "Extreme selling → mean-revert up"),
    ]

    for regime, direction, hold, hypothesis in test_configs:
        r = evaluate_signals(df, regime, direction, hold_bars=hold)
        r["hypothesis"] = hypothesis
        results.append(r)
        print(f"\n  {regime:20s} {direction:5s} hold={hold}h:")
        if "error" in r:
            print(f"    ERROR: {r['error']}")
        else:
            print(f"    n={r['n_signals']:5d}  Net9={r['net_9bps']:8.1f}  PF={r['pf']:5.2f}  Hit={r['hit_rate']:.2%}  IC={r['ic']:.4f}  Top3%={r['top3_contribution']/max(abs(r['net_9bps']),1)*100:.0f}%")

    # Cross-sectional: within each bar, rank by taker_net_z, go long top, short bottom
    print("\n[4/4] Cross-sectional test...")
    cs_results = evaluate_cross_sectional(df)
    results.append(cs_results)

    # Save
    output = {
        "version": "v1.0",
        "generated": str(pd.Timestamp.now(tz="UTC")),
        "regime_distribution": {str(k): int(v) for k, v in regime_counts.items()},
        "results": results,
        "cross_sectional": cs_results,
    }

    out_path = OUT_DIR / "taker_flow_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved: {out_path}")

    elapsed = time.time() - t0
    print(f"Total: {elapsed:.0f}s")
    return output


def evaluate_cross_sectional(df: pd.DataFrame) -> dict:
    """Cross-sectional: long top taker_net_z decile, short bottom decile."""
    fwd_col = "fwd_ret_2h"
    results = []

    for ts, bar in df.groupby(level="timestamp"):
        if bar["taker_net_z"].count() < 20:
            continue
        bar = bar.copy()
        bar["decile"] = pd.qcut(bar["taker_net_z"].rank(pct=True), 10, labels=False, duplicates="drop")
        top = bar[bar["decile"] >= 8]["fwd_ret_2h"].mean()
        bot = bar[bar["decile"] <= 1]["fwd_ret_2h"].mean()
        if pd.notna(top) and pd.notna(bot):
            results.append({"ts": ts, "top_decile_ret": top, "bot_decile_ret": bot, "spread": top - bot})

    if not results:
        return {"error": "No cross-sectional data"}

    res_df = pd.DataFrame(results)
    spreads = res_df["spread"] * 10000
    gross = spreads.sum()
    net = gross - len(spreads) * 9

    return {
        "type": "cross_sectional",
        "n_bars": len(spreads),
        "mean_spread_bps": round(spreads.mean(), 2),
        "gross_bps": round(gross, 1),
        "net_9bps": round(net, 1),
        "hit_rate": round((spreads > 0).mean(), 3),
        "t_stat": round(spreads.mean() / max(spreads.std(), 1e-8) * np.sqrt(len(spreads)), 2),
    }


if __name__ == "__main__":
    run_full_research()
