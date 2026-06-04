"""Trade rule optimization matrix — full grid for CrowdingFade + VolAccelFade."""
import json, sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research.paper_trading import PaperTradingSimulator, PaperConfig
from factors import load_factor_from_path
from backtest_engine import BacktestEngine

ROOT = Path("/mnt/e/alpha_pipeline")
THRESHOLDS = [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
HOLDS = [1, 2, 3, 4, 6, 8]
COSTS = [9, 6, 4]
FACTORS = ["alpha_crowding_fade", "alpha_vol_accel_fade"]

SEP = "=" * 80

engine = BacktestEngine(ROOT)
data = engine._load_data()
sim = PaperTradingSimulator(ROOT)

all_results = {}

for fstem in FACTORS:
    print("\n=== " + fstem + " ===")
    fpath = ROOT / "factors" / (fstem + ".py")
    loaded = load_factor_from_path(fpath)
    fname = loaded.factor.factor_name
    
    t0 = time.time()
    signal = loaded.factor.compute(data).reindex(data.index).fillna(0.0)
    print("  Signal computed in %.1fs" % (time.time() - t0))

    results = []
    total = len(THRESHOLDS) * len(HOLDS) * len(COSTS)
    n = 0

    for z in THRESHOLDS:
        for hold in HOLDS:
            for cost in COSTS:
                n += 1
                cfg = PaperConfig(
                    top_frac=0.10, bottom_frac=0.10,
                    hold_bars=hold, cost_bps=float(cost),
                    execution_mode="threshold_entry",
                    signal_threshold=z, exit_threshold=z * 0.33,
                )
                r = sim._simulate_single(signal, data, cfg)

                results.append({
                    "factor": fname,
                    "threshold_z": z,
                    "hold_bars": hold,
                    "cost_bps": cost,
                    "trade_count": r.n_trades,
                    "turnover": round(r.turnover, 1),
                    "gross_pnl": round(r.gross_pnl, 4),
                    "net_pnl": round(r.net_pnl, 4),
                    "cost_to_gross": round(r.cost_gross_ratio * 100, 1),
                    "pf": round(r.pf, 3),
                    "long_net": round(r.long_pnl, 4),
                    "short_net": round(r.short_pnl, 4),
                    "classification": r.classification,
                    "regime_pnl": r.regime_pnl,
                })

                cg = r.cost_gross_ratio * 100
                print("  [%d/%d] z=%.2f hold=%d cost=%d | Net=%.2f Gross=%.2f C/G=%.0f%% Turn=%.0f PF=%.2f %s" % (
                    n, total, z, hold, cost, r.net_pnl, r.gross_pnl, cg, r.turnover, r.pf, r.classification))
    
    all_results[fname] = results

# Save results
out_path = ROOT / "logs" / "optimization_matrix.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(all_results, indent=2, default=str))
print("\nResults saved to " + str(out_path))

# ── Print heatmap tables ──
for fname, results in all_results.items():
    df = pd.DataFrame(results)
    
    for cost_label, cost_val in [("9bps", 9), ("6bps", 6), ("4bps", 4)]:
        subset = df[df["cost_bps"] == cost_val]
        
        def print_heatmap(title, value_key, fmt):
            print("\n" + SEP)
            print("  %s — %s (cost=%s)" % (fname, title, cost_label))
            print(SEP)
            hdr = "%-10s" % "z/hold"
            for h in HOLDS:
                hdr += "%10d" % h
            print(hdr)
            print("-" * (10 + 10 * len(HOLDS)))
            for z in THRESHOLDS:
                row_str = "%-10s" % ("z=%.2f" % z)
                for h in HOLDS:
                    r = subset[(subset["threshold_z"] == z) & (subset["hold_bars"] == h)]
                    if len(r) > 0:
                        val = r.iloc[0][value_key]
                        row_str += fmt % val
                    else:
                        row_str += "%10s" % "N/A"
                print(row_str)
        
        print_heatmap("Net PnL", "net_pnl", "%10.2f")
        print_heatmap("Cost/Gross %%", "cost_to_gross", "%9.0f%%")
        print_heatmap("Trade Count", "trade_count", "%10d")
        print_heatmap("Turnover", "turnover", "%10.1f")
        print_heatmap("PF", "pf", "%10.2f")
        print_heatmap("Classification", "classification", "%10s")

# ── Final verdict per factor ──
print("\n" + SEP)
print("  FINAL VERDICT")
print(SEP)

for fname, results in all_results.items():
    df = pd.DataFrame(results)
    
    # Baseline: full_rebalance hold=1 cost=9
    baseline_net = -90.87 if "Crowding" in fname else -90.49
    
    # Best net across all configs
    best = df.loc[df["net_pnl"].idxmax()]
    improvement = (baseline_net - best["net_pnl"]) / abs(baseline_net) * 100
    
    # Count configs with net > 0
    positive = df[df["net_pnl"] > 0]
    
    # Check for parameter plateau
    # Look at z=2.0-2.5, hold=3-6, cost=6-9 region
    plateau = df[(df["threshold_z"] >= 2.0) & (df["threshold_z"] <= 2.5) &
                 (df["hold_bars"] >= 3) & (df["hold_bars"] <= 6) &
                 (df["cost_bps"] >= 6) & (df["cost_bps"] <= 9)]
    
    plateau_positive = plateau[plateau["net_pnl"] > 0]
    
    print("\n" + fname + ":")
    print("  Baseline Net PnL: %.2f" % baseline_net)
    print("  Best Net PnL: %.2f (z=%.2f hold=%d cost=%d, improvement=%.0f%%)" % (
        best["net_pnl"], best["threshold_z"], best["hold_bars"], best["cost_bps"], improvement))
    print("  Configs with Net>0: %d/%d" % (len(positive), len(df)))
    print("  Plateau region (z=2.0-2.5, hold=3-6, cost=6-9): %d configs" % len(plateau))
    print("  Plateau with Net>0: %d/%d" % (len(plateau_positive), len(plateau)))
    
    if len(plateau_positive) > 0:
        print("  VERDICT: PAPER_REPAIR_PASS — net positive in plateau region, trade rule repair successful")
    elif improvement > 70:
        print("  VERDICT: TRADE_RULE_REPAIR — major improvement but net still negative")
    else:
        print("  VERDICT: KILL — cannot be salvaged by trade rule optimization")

print("\n=== DONE ===")
