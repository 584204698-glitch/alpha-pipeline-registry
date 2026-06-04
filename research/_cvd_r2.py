import json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/mnt/e/alpha_pipeline")
OUT = ROOT / "research" / "cvd"
OUT.mkdir(parents=True, exist_ok=True)
HOLD_BARS = [1, 2, 3, 6]
COST_BPS = [9, 12, 15]

data = pd.read_parquet(ROOT / "data" / "data_storage_1h.parquet")
print(f"Data: {len(data):,} rows")

gb = data.groupby("symbol")

# Features (same as R1)
buy = data["taker_buy_volume"].fillna(0); sell = data["taker_sell_volume"].fillna(0)
data["cvd_cum"] = (buy-sell).groupby(data.index.get_level_values("symbol")).cumsum()
for w in [6,12]:
    data[f"cvd_d{w}h"] = gb["cvd_cum"].transform(lambda s: s.diff(w))
    data[f"ret_{w}h"] = gb["close"].transform(lambda s: s.pct_change(w))

data["cvd_z"] = gb["cvd_d12h"].transform(lambda s: (s-s.rolling(12,min_periods=4).mean())/s.rolling(12,min_periods=4).std().clip(1e-8)).fillna(0)
data["price_ret_z"] = gb["ret_6h"].transform(lambda s: (s-s.rolling(12,min_periods=4).mean())/s.rolling(12,min_periods=4).std().clip(1e-8)).fillna(0)
data["absorption_ratio"] = abs(data["cvd_z"]) / (abs(data["price_ret_z"]) + 0.1)
data["vol_z"] = gb["volume"].transform(lambda s: (s-s.rolling(12,min_periods=4).mean())/s.rolling(12,min_periods=4).std().clip(1e-8)).fillna(0)
hl_range = (data["high"] - data["low"]).clip(lower=1e-8)
data["close_loc"] = (data["close"] - data["low"]) / hl_range

for w in HOLD_BARS:
    data[f"fwd_{w}h"] = (gb["close"].shift(-w) / data["close"] - 1) * 10000

print(f"Features built")

# Round 2 variants (tighter thresholds)
variants = {
    "V1_cvd_z_lt_neg25": (data["cvd_z"] < -2.5),
    "V2_cvd_z_lt_neg25_ret_gt_neg1pct": (data["cvd_z"] < -2.5) & (data["ret_6h"] > -0.01),
    "V3_cvd_z_lt_neg25_close_loc_gt_05": (data["cvd_z"] < -2.5) & (data["close_loc"] > 0.5),
    "V4_cvd_z_lt_neg25_vol_z_gt_1": (data["cvd_z"] < -2.5) & (data["vol_z"] > 1.0),
    "V5_cvd_z_lt_neg30": (data["cvd_z"] < -3.0),
    "V6_absorption_top1pct": data["absorption_ratio"] >= data["absorption_ratio"].quantile(0.99),
}

all_results = {}
for vname, mask in variants.items():
    n = mask.sum()
    if n < 10:
        all_results[vname] = {"n_events": int(n), "error": "too few"}
        continue

    best_net = -999999
    best_key = None
    best_r = None

    for direction in ["long"]:
        sign = -1 if direction == "short" else 1
        for hold in HOLD_BARS:
            fwd = data[f"fwd_{hold}h"][mask].dropna().values * sign
            if len(fwd) < 10:
                continue
            net12 = (fwd - 12).sum()
            if net12 > best_net:
                best_net = net12
                best_key = f"{direction}_h{hold}"
                pf = fwd[fwd>0].sum() / abs(fwd[fwd<0].sum()) if (fwd<0).any() else 999
                wo_top3_n = max(1, len(fwd)//20)
                top3 = np.sort(fwd)[-wo_top3_n:].sum()
                wo3 = fwd.sum() - top3 - (len(fwd)-wo_top3_n)*9
                best_r = {
                    "n": len(fwd), "net12": round(net12,1), "pf": round(pf,3),
                    "hit": round((fwd>0).mean(),3), "top3_contrib": round(top3,1),
                    "net_wo_top3_9bps": round(wo3,1),
                    "net9": round((fwd-9).sum(),1), "net15": round((fwd-15).sum(),1),
                }

    status = "✅ PASS" if best_net > 0 and best_r and best_r["pf"] >= 1.15 else "❌ FAIL"
    print(f"{status} {vname}: n={n} best={best_key} Net12={best_net:.0f} PF={best_r['pf'] if best_r else 0:.2f}")
    all_results[vname] = {"n_events": int(n), "best": best_r, "best_key": best_key}

report = {"version": "v1.0_round2", "timestamp": str(pd.Timestamp.now(tz="UTC")), "results": all_results}
json.dump(report, open(OUT/"cvd_divergence_r2.json","w"), indent=2, default=str)
print(f"\nSaved: {OUT/'cvd_divergence_r2.json'}")
