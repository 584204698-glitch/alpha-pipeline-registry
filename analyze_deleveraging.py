"""Analyze Deleveraging Reversal trades in detail."""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from pathlib import Path
from backtest_engine import BacktestEngine

ROOT = Path("/mnt/e/alpha_pipeline")
engine = BacktestEngine(ROOT)
data = engine._load_data()

# Small-cap universe
avg_vol = data["volume"].groupby(level="symbol").mean()
vol_rank = avg_vol.rank(ascending=False)
small_syms = set(vol_rank[(vol_rank >= 20) & (vol_rank <= 100)].index)

ret_1h = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(4))
oi_delta = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(6))

def roll_z(series, window=48):
    g = series.groupby(level="symbol")
    mean = g.transform(lambda s: s.rolling(window, min_periods=8).mean())
    std = g.transform(lambda s: s.rolling(window, min_periods=8).std()).replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)

vol_z = roll_z(data["volume"], 48)
oi_z = roll_z(oi_delta, 48)
ts_list = sorted(data.index.get_level_values("timestamp").unique())
ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}

trades = []
for ts in ts_list:
    mask = data.index.get_level_values("timestamp") == ts
    syms = data.index.get_level_values("symbol")[mask]
    for sym in syms:
        if sym not in small_syms:
            continue
        try:
            oz = float(oi_z.loc[(ts, sym)])
            vz = float(vol_z.loc[(ts, sym)])
            r1h = float(ret_1h.loc[(ts, sym)])
        except:
            continue
        if oz > -1.5 or vz < 1.0 or abs(r1h) < 0.025:
            continue
        
        direction = "long" if r1h < -0.025 else "short"
        entry_idx = ts_to_idx[ts]
        try:
            entry_px = data.loc[(ts, sym), "close"]
        except:
            continue
        
        for h in range(1, 4):
            if entry_idx + h >= len(ts_list):
                break
            exit_ts = ts_list[entry_idx + h]
            try:
                exit_px = data.loc[(exit_ts, sym), "close"]
            except:
                continue
            ret = (exit_px / entry_px) - 1.0
            if direction == "short":
                ret = -ret
            gbp = ret * 10000
            nbp = gbp - 6
            hit = ((direction == "long" and ret > 0) or (direction == "short" and ret < 0))
            trades.append({
                "symbol": sym, "direction": direction,
                "gross_bps": gbp, "net_bps": nbp, "hit": hit,
                "hold_bar": h, "entry_ts": ts,
                "oi_z": oz, "vol_z": vz, "ret_1h": r1h,
            })

df = pd.DataFrame(trades)

print("Deleveraging Reversal — h=3 cost=6bps")
print("=" * 80)
print("Total trades: %d" % len(df))

print("\nHold breakdown:")
for h, g in df.groupby("hold_bar"):
    hit_pct = g["hit"].mean() * 100
    print("  h=%d: %d trades, hit=%.1f%%, gross=%.0f, net=%.0f" % (h, len(g), hit_pct, g["gross_bps"].sum(), g["net_bps"].sum()))

print("\nDirection breakdown:")
for d, g in df.groupby("direction"):
    hit_pct = g["hit"].mean() * 100
    n_events = len(g) / 3  # 3 hold bars per event
    print("  %s: %d trades (%d events), hit=%.1f%%, gross=%.0f, net=%.0f" % (d, len(g), n_events, hit_pct, g["gross_bps"].sum(), g["net_bps"].sum()))

print("\nReturn distribution (net_bps):")
for pct in [10, 25, 50, 75, 90]:
    val = df["net_bps"].quantile(pct / 100)
    print("  %dth: %.0f bps" % (pct, val))

win_trades = df[df["net_bps"] > 0]
loss_trades = df[df["net_bps"] < 0]
print("\nWin/loss breakdown:")
print("  Wins: %d (%.1f%%), avg +%.0f bps, total +%.0f" % (len(win_trades), len(win_trades)/len(df)*100, win_trades["net_bps"].mean(), win_trades["net_bps"].sum()))
print("  Losses: %d (%.1f%%), avg %.0f bps, total %.0f" % (len(loss_trades), len(loss_trades)/len(df)*100, loss_trades["net_bps"].mean(), loss_trades["net_bps"].sum()))

print("\nTop 10 winners:")
for _, t in df.nlargest(10, "net_bps").iterrows():
    print("  %-30s %6s h=%d gross=%8.0f net=%8.0f oi_z=%.1f ret_1h=%.1f%%" % (t["symbol"], t["direction"], t["hold_bar"], t["gross_bps"], t["net_bps"], t["oi_z"], t["ret_1h"]*100))

print("\nWorst 10 losers:")
for _, t in df.nsmallest(10, "net_bps").iterrows():
    print("  %-30s %6s h=%d gross=%8.0f net=%8.0f oi_z=%.1f ret_1h=%.1f%%" % (t["symbol"], t["direction"], t["hold_bar"], t["gross_bps"], t["net_bps"], t["oi_z"], t["ret_1h"]*100))

print("\nHit rate by OI severity:")
df["oi_bin"] = pd.cut(df["oi_z"], [-5, -3, -2.5, -2, -1.5, 0])
for b, g in df.groupby("oi_bin", observed=False):
    if len(g) > 0:
        hit_pct = g["hit"].mean() * 100
        print("  OI z in %s: %d trades, hit=%.1f%%, net=%.0f" % (str(b), len(g), hit_pct, g["net_bps"].sum()))

# Unique events (not trades)
events_df = df.drop_duplicates(subset=["symbol", "entry_ts"])
print("\nUnique events: %d" % len(events_df))
print("Event frequency: %.1f per bar" % (len(events_df) / len(ts_list)))
