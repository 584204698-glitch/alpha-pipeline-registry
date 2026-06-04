import json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/mnt/e/alpha_pipeline")
OUT = ROOT / "research" / "cvd"
OUT.mkdir(parents=True, exist_ok=True)

data = pd.read_parquet(ROOT / "data" / "data_storage_1h.parquet")
gb = data.groupby("symbol")

# Features
buy = data["taker_buy_volume"].fillna(0); sell = data["taker_sell_volume"].fillna(0)
data["cvd_cum"] = (buy-sell).groupby(data.index.get_level_values("symbol")).cumsum()
for w in [6,12]:
    data[f"cvd_d{w}h"] = gb["cvd_cum"].transform(lambda s: s.diff(w))
    data[f"ret_{w}h"] = gb["close"].transform(lambda s: s.pct_change(w))
data["cvd_z"] = gb["cvd_d12h"].transform(lambda s: (s-s.rolling(12,min_periods=4).mean())/s.rolling(12,min_periods=4).std().clip(1e-8)).fillna(0)
data["vol_z"] = gb["volume"].transform(lambda s: (s-s.rolling(12,min_periods=4).mean())/s.rolling(12,min_periods=4).std().clip(1e-8)).fillna(0)
hl_range = (data["high"] - data["low"]).clip(lower=1e-8)
data["close_loc"] = (data["close"] - data["low"]) / hl_range

# Forward returns
for w in [1,2,3,6]:
    data[f"fwd_{w}h"] = (gb["close"].shift(-w) / data["close"] - 1) * 10000

# BTC regime (simple: 1h drop > 3% = panic)
btc_mask = data.index.get_level_values("symbol") == "Binance:BTCUSDT"
btc_ret = data.loc[btc_mask, "close"].droplevel("symbol").pct_change()
data["btc_panic"] = False
panic_ts = btc_ret[btc_ret < -0.03].index
data.loc[data.index.get_level_values("timestamp").isin(panic_ts), "btc_panic"] = True

def analyze(label, mask, hold, direction="long"):
    sign = -1 if direction == "short" else 1
    fwd = data[f"fwd_{hold}h"][mask].dropna().values * sign
    n = len(fwd)
    if n < 10:
        return {"n": n, "error": "too few"}

    results = {"n": n, "label": label, "hold": hold}

    # By cost
    for cost in [4, 6, 9, 12, 15]:
        net = fwd - cost
        pos = net[net>0].sum(); neg = abs(net[net<0].sum())
        results[f"cost{cost}bps"] = {
            "net": round(net.sum(),1), "pf": round(pos/neg,3) if neg>0 else 999,
            "hit": round((net>0).mean(),3),
            "mean_win": round(net[net>0].mean(),1) if (net>0).any() else 0,
            "mean_loss": round(net[net<0].mean(),1) if (net<0).any() else 0,
        }

    # Top3 + net_wo_top3
    top_n = max(1, n // 20)
    sorted_fwd = np.sort(fwd)[::-1]
    top3_sum = sorted_fwd[:top_n].sum()
    net_wo_top3 = fwd.sum() - top3_sum - (n - top_n) * 9
    results["top3_contrib"] = round(top3_sum, 1)
    results["top3_pct"] = round(top3_sum / abs(fwd.sum()) * 100, 1) if fwd.sum() != 0 else 0
    results["net_wo_top3_9bps"] = round(net_wo_top3, 1)

    # Reverse signal test
    rev_fwd = -fwd
    results["reverse_net12"] = round((rev_fwd - 12).sum(), 1)
    results["reverse_warning"] = (rev_fwd - 12).sum() > 0

    # Symbol concentration
    event_syms = data[mask].index.get_level_values("symbol")
    top_sym_counts = pd.Series(event_syms).value_counts().head(5)
    results["top_symbols"] = {str(k): int(v) for k, v in top_sym_counts.items()}

    # Regime split
    for reg_label, reg_mask in [
        ("no_btc_panic", ~data.loc[mask, "btc_panic"] if "btc_panic" in data.columns else pd.Series(True, index=data[mask].index)),
    ]:
        pass

    # BTC panic split
    if "btc_panic" in data.columns:
        panic_fwd = data[f"fwd_{hold}h"][mask & data["btc_panic"]].dropna().values * sign
        normal_fwd = data[f"fwd_{hold}h"][mask & ~data["btc_panic"]].dropna().values * sign
        results["regime_split"] = {
            "no_panic": {"n": len(normal_fwd), "net12": round((normal_fwd-12).sum(),1) if len(normal_fwd)>5 else 0},
            "btc_panic": {"n": len(panic_fwd), "net12": round((panic_fwd-12).sum(),1) if len(panic_fwd)>5 else 0},
        }

    return results

print("="*60)
print("CVD Round 3 — Parameter Neighborhood")
print("="*60)

all_r3 = {}

# ── V3 family: cvd_z threshold sweep + close_loc ──
print("\n[V3] cvd_z threshold + close_loc sweep:")
for z_thresh in [-2.3, -2.5, -2.8, -3.0]:
    for cl_thresh in [0.4, 0.5, 0.6]:
        mask = (data["cvd_z"] < z_thresh) & (data["close_loc"] > cl_thresh)
        n = mask.sum()
        if n < 20:
            continue
        for h in [3, 6]:
            r = analyze(f"V3_z{z_thresh}_cl{cl_thresh}_h{h}", mask, h)
            c12 = r.get("cost12bps", {})
            status = "✅" if c12.get("net",0) > 0 and c12.get("pf",0)>=1.15 and r.get("net_wo_top3_9bps",0)>=0 else "❌"
            print(f"  {status} z<{z_thresh} cl>{cl_thresh} h{h}: n={n} Net12={c12.get('net',0):.0f} PF={c12.get('pf',0):.2f} Hit={c12.get('hit',0):.1%} wo_top3={r.get('net_wo_top3_9bps',0):.0f}")
            all_r3[f"V3_z{z_thresh}_cl{cl_thresh}_h{h}"] = r

# ── V5 family: extreme CVD threshold sweep ──
print("\n[V5] Extreme CVD threshold sweep:")
for z_thresh in [-2.8, -3.0, -3.5, -4.0]:
    mask = data["cvd_z"] < z_thresh
    n = mask.sum()
    if n < 10:
        continue
    for h in [2, 3, 6]:
        r = analyze(f"V5_z{z_thresh}_h{h}", mask, h)
        c12 = r.get("cost12bps", {})
        status = "✅" if c12.get("net",0) > 0 and c12.get("pf",0)>=1.15 and r.get("net_wo_top3_9bps",0)>=0 else "❌"
        print(f"  {status} z<{z_thresh} h{h}: n={n} Net12={c12.get('net',0):.0f} PF={c12.get('pf',0):.2f} Hit={c12.get('hit',0):.1%} wo_top3={r.get('net_wo_top3_9bps',0):.0f}")
        all_r3[f"V5_z{z_thresh}_h{h}"] = r

# ── Best overall ──
best_key = max(all_r3, key=lambda k: all_r3[k].get("cost12bps",{}).get("net", -999999))
best = all_r3[best_key]
print(f"\n🏆 Best: {best_key}")
print(f"  n={best['n']} Net12={best['cost12bps']['net']:.0f} PF={best['cost12bps']['pf']:.2f}")
print(f"  Hit={best['cost12bps']['hit']:.1%} wo_top3={best['net_wo_top3_9bps']:.0f} top3_pct={best['top3_pct']:.1f}%")
print(f"  Reverse Net12={best['reverse_net12']:.0f} (warning={best['reverse_warning']})")
print(f"  Top symbols: {best['top_symbols']}")
print(f"  Regime: {best.get('regime_split',{})}")

report = {"version": "v1.0_round3", "timestamp": str(pd.Timestamp.now(tz="UTC")), "best_key": best_key, "results": all_r3}
json.dump(report, open(OUT/"cvd_divergence_r3.json","w"), indent=2, default=str)
print(f"\nSaved: {OUT/'cvd_divergence_r3.json'}")
