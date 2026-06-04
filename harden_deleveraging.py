"""
Deleveraging Reversal V1.1 — add Funding Extreme filter + regime hardening.
Tests:
  A: Baseline (current scanner logic)
  B: Reject if funding_z > 1.5 (longs still paying → not done liquidating)
  C: Reject if abs(funding_z) > 2.0 (extreme funding = unstable)
  D: B + C combined
  E: Market-wide filter (>5 coins triggering simultaneously → systemic, not single-coin)
"""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from pathlib import Path
from backtest_engine import BacktestEngine
from research.regime_detector import detect_regimes

ROOT = Path(".")
engine = BacktestEngine(ROOT)
data = engine._load_data()
regimes = detect_regimes(data)
regime_map = dict(zip(regimes.index, regimes))

ts_list = sorted(data.index.get_level_values("timestamp").unique())
ts_to_idx = {ts: i for i, ts in enumerate(ts_list)}
cost = 6
round_trip = cost / 10000.0

avg_vol = data["volume"].groupby(level="symbol").mean()
vol_rank = avg_vol.rank(ascending=False)
small_syms = set(vol_rank[(vol_rank >= 20) & (vol_rank <= 100)].index)

ret_1h = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change(4))
oi_delta = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(6))

def roll_z(series, window=48):
    g = series.groupby(level="symbol")
    m = g.transform(lambda s: s.rolling(window, min_periods=8).mean())
    s = g.transform(lambda s: s.rolling(window, min_periods=8).std()).replace(0, np.nan)
    return ((series - m) / s).fillna(0.0)

vol_z = roll_z(data["volume"], 48)
oi_z = roll_z(oi_delta, 48)
funding = data["funding_rate"]
funding_z = roll_z(funding, 24)
hl_range = (data["high"] - data["low"]).clip(lower=1e-8)
close_loc = (data["close"] - data["low"]) / hl_range

def run_filter(label, extra_reject_fn=None, market_wide_max=999):
    trades = []
    cooldowns = {}
    for ts in ts_list:
        mask = data.index.get_level_values("timestamp") == ts
        syms = set(data.index.get_level_values("symbol")[mask])
        btc_reg = regime_map.get(ts, "unknown")

        # Market-wide OI collapse check
        oi_collapse_count = 0
        if market_wide_max < 999:
            for sym in small_syms & syms:
                try:
                    if float(oi_z.loc[(ts, sym)]) < -1.5:
                        oi_collapse_count += 1
                except:
                    pass

        for sym in small_syms & syms:
            if sym in cooldowns:
                continue
            try:
                oz = float(oi_z.loc[(ts, sym)])
                vz = float(vol_z.loc[(ts, sym)])
                r1h = float(ret_1h.loc[(ts, sym)])
                cl = float(close_loc.loc[(ts, sym)])
                fz = float(funding_z.loc[(ts, sym)])
            except:
                continue

            # Base gates
            if oz > -1.5 or oz < -5.0:
                continue
            if vz < 1.0 or vz > 8.0:
                continue
            if r1h > -0.025 or r1h < -0.12:
                continue
            if cl < 0.35:
                continue
            if btc_reg == "panic_down":
                continue

            # Market-wide gate
            if oi_collapse_count > market_wide_max:
                continue

            # Extra filter
            if extra_reject_fn and extra_reject_fn(fz, oz, r1h, btc_reg):
                continue

            entry_idx = ts_to_idx[ts]
            try:
                entry_px = data.loc[(ts, sym), "close"]
            except:
                continue
            for h in range(1, 3):
                if entry_idx + h >= len(ts_list):
                    break
                exit_ts = ts_list[entry_idx + h]
                try:
                    exit_px = data.loc[(exit_ts, sym), "close"]
                except:
                    continue
                ret = (exit_px / entry_px) - 1.0
                nbp = ret * 10000 - round_trip * 10000
                trades.append(nbp)
            cooldowns[sym] = 16

    if not trades:
        return {"label": label, "n": 0, "net": 0, "pf": 0, "hit": 0, "med": 0, "max_loss": 0}
    a = np.array(trades)
    pf = a[a > 0].sum() / abs(a[a < 0].sum()) if (a < 0).any() else 99.0
    return {
        "label": label, "n": len(a),
        "net": round(a.sum(), 0), "pf": round(pf, 2),
        "hit": round(np.mean(a > 0) * 100, 0),
        "med": round(np.median(a), 0),
        "max_loss": round(a.min(), 0),
    }

filters = [
    ("A: Baseline (no funding filter)", None, 999),
    ("B: Reject funding_z > 1.5", lambda fz, oz, r1h, reg: fz > 1.5, 999),
    ("C: Reject |funding_z| > 2.0", lambda fz, oz, r1h, reg: abs(fz) > 2.0, 999),
    ("D: B + C combined", lambda fz, oz, r1h, reg: fz > 1.5 or abs(fz) > 2.0, 999),
    ("E: Market-wide >5 OI collapses", None, 5),
    ("F: D + E (funding filter + market-wide gate)", lambda fz, oz, r1h, reg: fz > 1.5 or abs(fz) > 2.0, 5),
]

print(f"{'Filter':<50} {'n':>5} {'Net':>8} {'PF':>6} {'Hit':>5} {'Med':>5} {'MaxLoss':>7}")
print("-" * 95)

results = []
for label, fn, mw in filters:
    r = run_filter(label, fn, mw)
    results.append(r)
    print(f"{label:<50} {r['n']:>5} {r['net']:>8.0f} {r['pf']:>6.2f} {r['hit']:>4.0f}% {r['med']:>5.0f} {r['max_loss']:>7.0f}")

# Best combination
print("\n--- Verdict ---")
baseline = results[0]
for r in results[1:]:
    delta_pf = r["pf"] - baseline["pf"]
    delta_net = r["net"] - baseline["net"]
    delta_n = r["n"] - baseline["n"]
    print(f"{r['label']}: ΔPF={delta_pf:+.2f} ΔNet={delta_net:+.0f} Δn={delta_n:+d}")
