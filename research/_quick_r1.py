import json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/mnt/e/alpha_pipeline")
COST_BPS = [9, 12, 15]
HOLD_BARS = [1, 2, 3, 6]

data = pd.read_parquet(ROOT / "data" / "data_storage_1h.parquet")
print(f"Data: {len(data):,} rows")

gb = data.groupby("symbol")
buy = data["taker_buy_volume"].fillna(0); sell = data["taker_sell_volume"].fillna(0)
data["cvd_cum"] = (buy-sell).groupby(data.index.get_level_values("symbol")).cumsum()
for w in [6,12]:
    data[f"cvd_d{w}h"] = gb["cvd_cum"].transform(lambda s: s.diff(w))
data["cvd_z"] = gb["cvd_d12h"].transform(lambda s: (s-s.rolling(12,min_periods=4).mean())/s.rolling(12,min_periods=4).std().clip(1e-8)).fillna(0)
data["oi_z"] = gb["open_interest"].transform(lambda s: (s.diff(2)-s.diff(2).rolling(12,min_periods=4).mean())/s.diff(2).rolling(12,min_periods=4).std().clip(1e-8)).fillna(0)
for w in HOLD_BARS:
    data[f"fwd_{w}h"] = (gb["close"].shift(-w) / data["close"] - 1) * 10000

print("Features built")

# OI-Price
results_oi = {}
for quad_name, oi_sign, price_sign in [
    ("OI_up_Price_down", 1, -1),
    ("OI_up_Price_flat", 1, 0),
    ("OI_down_Price_up", -1, 1),
    ("OI_down_Price_flat", -1, 0),
]:
    oi_slope = gb["open_interest"].transform(lambda s: s.diff(6))
    price_slope = gb["close"].transform(lambda s: s.pct_change(6))
    if price_sign == 0:
        mask = (oi_slope * oi_sign > 0) & (price_slope.abs() < price_slope.abs().quantile(0.3))
    else:
        mask = (oi_slope * oi_sign > 0) & (price_slope * price_sign > 0)
    n = mask.sum()
    if n < 10:
        continue
    direction = "long" if quad_name != "OI_down_Price_up" else "short"
    sign_fwd = -1 if direction == "short" else 1
    best_net = -999999
    for h in HOLD_BARS:
        fwd = data[f"fwd_{h}h"][mask].dropna().values * sign_fwd
        if len(fwd) < 10:
            continue
        net12 = (fwd - 12).sum()
        if net12 > best_net:
            best_net = net12
    print(f"  OI {quad_name}: n={n} best_Net12={best_net:.0f}")
    results_oi[quad_name] = {"n": int(n), "best_net12": round(best_net, 1)}

# Funding
data["funding_z"] = gb["funding_rate"].transform(lambda s: (s-s.rolling(24,min_periods=8).mean())/s.rolling(24,min_periods=8).std().clip(1e-8)).fillna(0)
data["funding_d8h"] = gb["funding_rate"].transform(lambda s: s.diff(8))
results_fc = {}
for thr in [1.5, 2.0, 2.5]:
    mask = data["funding_z"] < -thr
    fwd_vals = data["fwd_2h"][mask].dropna().values
    net12 = (fwd_vals-12).sum() if len(fwd_vals) > 5 else 0
    results_fc[f"persist_z{thr}"] = {"n": int(mask.sum()), "net12_2h": round(net12,1)}
    print(f"  FC persist_z{thr}: n={mask.sum()} Net12_2h={net12:.0f}")

for thr in [2.0, 2.5]:
    mask = (data["funding_z"] < -thr) & (data["funding_d8h"] > 0.5)
    fwd_vals = data["fwd_2h"][mask].dropna().values
    net12 = (fwd_vals-12).sum() if len(fwd_vals) > 5 else 0
    results_fc[f"mr_z{thr}"] = {"n": int(mask.sum()), "net12_2h": round(net12,1)}
    print(f"  FC mr_z{thr}: n={mask.sum()} Net12_2h={net12:.0f}")

mask = (data["funding_z"] < -2.0) & (data["cvd_z"] < -1.5)
fwd_vals = data["fwd_2h"][mask].dropna().values
net12 = (fwd_vals-12).sum() if len(fwd_vals) > 5 else 0
results_fc["fz_cvd"] = {"n": int(mask.sum()), "net12_2h": round(net12,1)}
print(f"  FC fz_cvd: n={mask.sum()} Net12_2h={net12:.0f}")

(ROOT/"research/oi_divergence").mkdir(parents=True, exist_ok=True)
(ROOT/"research/funding_transition").mkdir(parents=True, exist_ok=True)
json.dump({"oi": results_oi}, open(ROOT/"research/oi_divergence/oi_r1_quick.json","w"), indent=2)
json.dump({"fc": results_fc}, open(ROOT/"research/funding_transition/fc_r1_quick.json","w"), indent=2)
print("\nDone")
