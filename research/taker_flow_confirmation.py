"""
Taker Flow Confirmation Layer
==============================
On existing shadow signals, check taker flow for confirmation.
Enhances confidence without touching scanner gates.

Rules:
  - Deleveraging (long): taker_sell extreme → forced selling confirmed → HIGH confidence
  - OI Shock (long): taker_buy extreme → squeeze building → HIGH confidence
  - RS Shock (long): taker_buy supports momentum → HIGH confidence
  - RS Shock (short): taker_sell supports momentum → HIGH confidence
  - Funding Carry (long): low taker activity → stable carry → HIGH confidence

Output: Confirmation profile per symbol/alpha.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "taker_flow"
OUT.mkdir(parents=True, exist_ok=True)


def build_confirmation_layer(data: pd.DataFrame) -> dict:
    """For each bar, compute confirmation scores for each alpha direction."""
    df = data.copy().sort_index()

    # Taker features
    buy = df["taker_buy_volume"].fillna(0)
    sell = df["taker_sell_volume"].fillna(0)
    total = buy + sell
    df["taker_buy_ratio"] = (buy / total.replace(0, np.nan)).clip(0, 1)

    gb = df.groupby("symbol")
    df["taker_net_z"] = gb["taker_buy_ratio"].transform(
        lambda x: (x - x.rolling(24, min_periods=8).mean())
        / x.rolling(24, min_periods=8).std().clip(1e-8)
    )

    # Classify each bar's taker state
    tz = df["taker_net_z"].fillna(0)
    conds = [
        tz > 2.0,
        tz < -2.0,
        (tz >= -1.0) & (tz <= 1.0),
    ]
    choices = ["TAKER_BUY_EXTREME", "TAKER_SELL_EXTREME", "TAKER_NEUTRAL"]
    df["taker_state"] = np.select(conds, choices, default="TAKER_MODERATE")

    # Confirmation scores per alpha
    confirmation = {}

    # For each bar, assess confirmation
    for ts, bar in df.groupby(level="timestamp"):
        ts_str = str(ts)
        for sym in bar.index:
            state = bar.loc[sym, "taker_state"] if "taker_state" in bar.columns else "UNKNOWN"
            taker_z = bar.loc[sym, "taker_net_z"] if "taker_net_z" in bar.columns else 0

            scores = {
                "DeleveragingReversal": 0.0,
                "OIShockAbsorption": 0.0,
                "RelativeStrengthShock_long": 0.0,
                "RelativeStrengthShock_short": 0.0,
                "FundingCarryEU": 0.0,
            }

            # Deleveraging: wants taker sell extreme → forced selling
            if state == "TAKER_SELL_EXTREME":
                scores["DeleveragingReversal"] = min(1.0, abs(taker_z) / 4.0)
            elif state == "TAKER_MODERATE":
                scores["DeleveragingReversal"] = 0.3

            # OI Shock: wants taker buy extreme → squeeze building
            if state == "TAKER_BUY_EXTREME":
                scores["OIShockAbsorption"] = min(1.0, abs(taker_z) / 4.0)

            # RS Shock long: taker buy supports
            if taker_z > 0.5:
                scores["RelativeStrengthShock_long"] = min(1.0, taker_z / 3.0)

            # RS Shock short: taker sell supports
            if taker_z < -0.5:
                scores["RelativeStrengthShock_short"] = min(1.0, abs(taker_z) / 3.0)

            # Funding Carry: wants low taker activity → stable
            if state == "TAKER_NEUTRAL":
                scores["FundingCarryEU"] = 0.8
            elif state == "TAKER_MODERATE":
                scores["FundingCarryEU"] = 0.5

            key = f"{ts_str}|{sym}"
            confirmation[key] = scores

    return confirmation


def evaluate_confirmation(
    data: pd.DataFrame,
    confirmation: dict,
    signal_file: Path,
) -> dict:
    """Evaluate if confirmation improves signal quality."""
    if not signal_file.exists():
        return {"error": "No signal file"}

    signals = []
    with open(signal_file) as f:
        for line in f:
            d = json.loads(line)
            if d.get("status") == "SHADOW_SIGNAL":
                signals.append(d)

    if not signals:
        return {"error": "No shadow signals found"}

    results = {"total_signals": len(signals), "by_alpha": {}}

    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}

    for alpha in ["DeleveragingReversal", "OIShockAbsorption", "RelativeStrengthShock", "FundingCarryEU"]:
        alpha_sigs = [s for s in signals if s["event_type"] == alpha]
        if not alpha_sigs:
            continue

        # For each signal, get confirmation score and forward return
        conf_scores = []
        fwd_rets = []
        for s in alpha_sigs:
            ts = pd.Timestamp(s["timestamp"])
            sym = s["symbol"]
            key = f"{str(ts)}|{sym}"
            conf = confirmation.get(key, {})
            direction = s.get("direction", "long")
            alpha_key = f"{alpha}" if alpha not in ["RelativeStrengthShock"] else f"{alpha}_{direction}"
            score = conf.get(alpha_key, conf.get(alpha, 0))

            # Forward return
            hold = s.get("hold_bars", 2)
            try:
                idx = ts_to_idx[ts]
                exit_idx = idx + hold
                if exit_idx < len(ts_list):
                    exit_ts = ts_list[exit_idx]
                    entry_px = float(data.loc[(ts, sym), "close"])
                    exit_px = float(data.loc[(exit_ts, sym), "close"])
                    if direction == "short":
                        ret = (entry_px / exit_px - 1) * 10000
                    else:
                        ret = (exit_px / entry_px - 1) * 10000
                else:
                    continue
            except (KeyError, ValueError):
                continue

            conf_scores.append(score)
            fwd_rets.append(ret)

        if len(conf_scores) < 10:
            results["by_alpha"][alpha] = {"n": len(conf_scores), "error": "too few"}
            continue

        conf_arr = np.array(conf_scores)
        ret_arr = np.array(fwd_rets)

        # Split by confirmation tertile
        low_mask = conf_arr <= np.percentile(conf_arr, 33)
        mid_mask = (conf_arr > np.percentile(conf_arr, 33)) & (conf_arr <= np.percentile(conf_arr, 67))
        high_mask = conf_arr > np.percentile(conf_arr, 67)

        tiers = {"low_conf": low_mask, "mid_conf": mid_mask, "high_conf": high_mask}
        tier_results = {}
        for tier_name, mask in tiers.items():
            tier_ret = ret_arr[mask]
            if len(tier_ret) < 5:
                continue
            gross = tier_ret.sum()
            net9 = gross - len(tier_ret) * 9
            wins = (tier_ret > 0).sum()
            losses = (tier_ret <= 0).sum()
            pf = abs(tier_ret[tier_ret > 0].sum() / tier_ret[tier_ret <= 0].sum()) if losses > 0 and tier_ret[tier_ret <= 0].sum() != 0 else 999 if wins > 0 else 0
            tier_results[tier_name] = {
                "n": len(tier_ret),
                "gross_bps": round(gross, 1),
                "net_9bps": round(net9, 1),
                "pf": round(pf, 2),
                "hit_rate": round(wins / len(tier_ret), 3),
                "mean_bps": round(tier_ret.mean(), 1),
                "avg_conf_score": round(conf_arr[mask].mean(), 3),
            }

        results["by_alpha"][alpha] = tier_results

    return results


def main():
    print("=" * 60)
    print("Taker Flow Confirmation Layer")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/3] Loading data...")
    data = pd.read_parquet(ROOT / "data" / "data_storage_1h.parquet")
    print(f"  {len(data):,} rows")

    print("\n[2/3] Building confirmation scores...")
    confirmation = build_confirmation_layer(data)
    print(f"  {len(confirmation):,} bar-symbol pairs scored")

    print("\n[3/3] Evaluating on shadow signals...")
    signal_file = ROOT / "logs" / "shadow" / "portfolio_signals_20260604.jsonl"
    results = evaluate_confirmation(data, confirmation, signal_file)

    # Print
    for alpha, tr in results.get("by_alpha", {}).items():
        print(f"\n  {alpha}:")
        if "error" in tr:
            print(f"    {tr['error']}")
            continue
        for tier, r in sorted(tr.items()):
            net = r.get("net_9bps", 0)
            pf = r.get("pf", 0)
            status = "✅" if net > 0 and pf >= 1.1 else "❌"
            print(f"    {status} {tier:12s}: n={r['n']:4d} Net9={net:8.1f} PF={pf:5.2f} Hit={r['hit_rate']:.1%} conf={r['avg_conf_score']:.3f}")

    output = {
        "version": "v1.0",
        "generated": str(pd.Timestamp.now(tz="UTC")),
        "method": "At signal bar, check taker flow state for directional confirmation",
        "results": results,
    }
    path = OUT / "taker_flow_confirmation.json"
    path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved: {path}")
    print(f"Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
