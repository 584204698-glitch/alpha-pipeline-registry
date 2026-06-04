"""Quick comparison: 4 execution modes × 5 factors on current 1h data."""
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research.paper_trading import (
    PaperTradingSimulator, PaperConfig, ExecutionMode, PaperResult
)
from factors import load_factor_from_path

ROOT = Path("/mnt/e/alpha_pipeline")

# ── Configs: one representative per mode ──
CONFIGS = [
    PaperConfig(top_frac=0.10, bottom_frac=0.10, hold_bars=1, cost_bps=9,
                execution_mode="full_rebalance"),
    PaperConfig(top_frac=0.10, bottom_frac=0.10, hold_bars=2, cost_bps=9,
                execution_mode="direction_trigger"),
    PaperConfig(top_frac=0.10, bottom_frac=0.10, hold_bars=2, cost_bps=9,
                execution_mode="threshold_entry",
                signal_threshold=1.5, exit_threshold=0.5),
    PaperConfig(top_frac=0.10, bottom_frac=0.10, hold_bars=1, cost_bps=9,
                execution_mode="position_smoothing",
                smoothing_rate=0.33),
]

# ── Find factor files ──
factor_map = {}
for f in (ROOT / "factors").glob("alpha_*.py"):
    name = f.stem.replace("alpha_", "")
    factor_map[name] = f

sim = PaperTradingSimulator(ROOT)
data = sim.engine._load_data()
print(f"Data: {len(data)} rows, {data.index.get_level_values('symbol').nunique()} symbols")

results = []
for fname, fpath in sorted(factor_map.items()):
    try:
        loaded = load_factor_from_path(fpath)
        signal = loaded.factor.compute(data).reindex(data.index).fillna(0.0)
    except Exception as e:
        print(f"  SKIP {fname}: {e}")
        continue

    for cfg in CONFIGS:
        try:
            r = sim._simulate_single(signal, data, cfg)
            r.factor_name = fname
            results.append(r)
        except Exception as e:
            print(f"  FAIL {fname} {cfg.execution_mode}: {e}")

# ── Print comparison table ──
print(f"\n{'Factor':<28} {'Mode':<22} {'Net PnL':>10} {'Gross':>10} {'Cost':>10} {'C/G%':>8} {'Turnover':>9} {'PF':>6} {'Class':>20}")
print("-" * 140)

# Group by factor
by_factor: dict[str, list] = {}
for r in results:
    by_factor.setdefault(r.factor_name, []).append(r)

for fname in sorted(by_factor.keys()):
    for i, r in enumerate(by_factor[fname]):
        prefix = fname if i == 0 else ""
        print(f"{prefix:<28} {r.config.execution_mode:<22} {r.net_pnl:>10.4f} {r.gross_pnl:>10.4f} {r.cost_total:>10.4f} {r.cost_gross_ratio*100:>7.1f}% {r.turnover:>9.1f} {r.pf:>6.2f} {r.classification:<20}")
    print()
