"""Test regime-gated scanner backtest: A/B comparison."""
import sys
sys.path.insert(0, '.')
from backtest_engine import BacktestEngine
from research.regime_detector import detect_regime_fast
from event_scanner import backtest_scanner

engine = BacktestEngine('.')
data = engine._load_data()
regimes = detect_regime_fast(data)
regime_map = dict(zip(regimes.index, regimes))

print("=== REGIME DISTRIBUTION ===")
for reg, cnt in regimes.value_counts().items():
    print(f"  {reg}: {cnt} ({cnt/len(regimes)*100:.1f}%)")

print()
print("=== A: WITHOUT regime rules (baseline) ===")
r_no = backtest_scanner(data, regime_map, hold_bars=2, entry_mode='current_close', use_regime_rules=False)
for k, v in sorted(r_no.items()):
    if k.startswith('cost_'):
        print(f"  {k}: n={v['n_trades']:4d} Net={v['net_bps']:+6.0f} PF={v['pf']:.2f} C/G={v['cost_to_gross_pct']:3.0f}% Hit={v['hit_rate']*100:.1f}% Med={v['median_bps']:+5.0f}")
    if k == 'regime_breakdown':
        print(f"  Regime breakdown:")
        for reg, info in v.items():
            print(f"    {reg:12s}: n={info['n_trades']:3d} Net={info['net_9bps']:+6.0f} Hit={info['hit_rate']*100:.0f}%")

print()
print("=== B: WITH regime rules (position sizing + thresholds) ===")
r_yes = backtest_scanner(data, regime_map, hold_bars=2, entry_mode='current_close', use_regime_rules=True)
for k, v in sorted(r_yes.items()):
    if k.startswith('cost_'):
        print(f"  {k}: n={v['n_trades']:4d} Net={v['net_bps']:+6.0f} PF={v['pf']:.2f} C/G={v['cost_to_gross_pct']:3.0f}% Hit={v['hit_rate']*100:.1f}% Med={v['median_bps']:+5.0f}")
    if k == 'regime_breakdown':
        print(f"  Regime breakdown:")
        for reg, info in v.items():
            print(f"    {reg:12s}: n={info['n_trades']:3d} Net={info['net_9bps']:+6.0f} Hit={info['hit_rate']*100:.0f}%")

print()
print("=== COMPARISON ===")
a9 = r_no['cost_9bps']
b9 = r_yes['cost_9bps']
print(f"  {'Metric':20s} {'Baseline':>10s} {'Regime-gated':>12s} {'Delta':>10s}")
print(f"  {'n_trades':20s} {str(a9['n_trades']):>10s} {str(b9['n_trades']):>12s} {b9['n_trades']-a9['n_trades']:>+10d}")
print(f"  {'Net (9bps)':20s} {a9['net_bps']:>+10.0f} {b9['net_bps']:>+12.0f} {b9['net_bps']-a9['net_bps']:>+10.0f}")
print(f"  {'PF':20s} {a9['pf']:>10.2f} {b9['pf']:>12.2f} {b9['pf']-a9['pf']:>+10.2f}")
print(f"  {'Hit Rate':20s} {a9['hit_rate']*100:>9.1f}% {b9['hit_rate']*100:>11.1f}% {b9['hit_rate']*100-a9['hit_rate']*100:>+10.1f}%")
print(f"  {'Median bps':20s} {a9['median_bps']:>+10.0f} {b9['median_bps']:>+12.0f} {b9['median_bps']-a9['median_bps']:>+10.0f}")
print(f"  {'Avg Win':20s} {a9['avg_win_bps']:>+10.0f} {b9['avg_win_bps']:>+12.0f} {b9['avg_win_bps']-a9['avg_win_bps']:>+10.0f}")
print(f"  {'Avg Loss':20s} {a9['avg_loss_bps']:>+10.0f} {b9['avg_loss_bps']:>+12.0f} {b9['avg_loss_bps']-a9['avg_loss_bps']:>+10.0f}")
