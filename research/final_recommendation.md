# Final Recommendation — New Alpha Family Research

**Date:** 2026-06-04  
**Researcher:** Hermes Agent (autonomous)  
**Status:** Research complete, 1 family advances, 2 families killed

---

## Executive Summary

Three new alpha families were researched across 5 total rounds. One family (CVD Divergence) achieved PAPER_CANDIDATE status with all criteria met. Two families (OI-Price Divergence, Funding Transition) were killed after Round 1 due to unambiguous failure.

---

## ✅ PAPER_CANDIDATE: CVD Divergence (Extreme Absorption)

### Signal Definition

```
CVD_z < -3.0 AND close_loc > 0.5 → LONG, hold 3 bars (3h)
```

**Financial logic:** When cumulative volume delta (taker buys minus taker sells) shows extreme selling pressure (z < -3.0), but price closes in the upper half of the bar (close_loc > 0.5), the selling is being **absorbed** by passive buyers. This absorption precedes a reversal: sellers exhaust, price recovers.

### Performance (42 days, 500 symbols, 1h bars)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Events | 119 | — | Sparse but sufficient |
| Net @ 9bps | +10,939 | > 0 | ✅ |
| Net @ 12bps | +8,136 | > 0 | ✅ |
| Net @ 15bps | +5,022 | — | Survives |
| PF | 2.02 | ≥ 1.15 | ✅ |
| Hit Rate | 54.6% | — | Good |
| net_wo_top3 @ 9bps | +1,671 | ≥ 0 | ✅ |
| Top3 contribution | 22.5% of gross | — | Acceptable |
| Reverse signal Net12 | Negative | Not profitable | ✅ |
| BTC panic exposure | 0 events | — | Clean |

### Variant Stability

| close_loc threshold | n | Net12 | PF | net_wo_top3 |
|---------------------|---|-------|-----|-------------|
| > 0.5 (winner) | 119 | +8,136 | 2.02 | +1,671 |
| > 0.6 | 79 | +5,531 | 2.04 | +1,861 |

Both variants pass all criteria. The >0.5 variant is recommended for larger sample size; >0.6 is a viable tighter alternative.

### Risks

1. **Sparsity:** 2.8 events/day across 500 symbols. May require universe expansion or patience.
2. **Out-of-sample:** 42 days is short. Needs forward walk-forward validation.
3. **Parameter sensitivity:** Only tested on 1 bar frequency. 4h/15m results may differ.
4. **CVD data quality:** Audit passed, but taker volume data is Coinglass-derived, not exchange-native.

### Recommendation

**Enter Paper Order Simulator.** Write a dedicated CVD Paper Order Simulator that tests:
- Entry delay (next bar open vs current close)
- Slippage models (bar_close, HH:15, +5/10/15 bps)
- Per-symbol spread costs
- Position sizing (equal weight, volatility-weighted)

Do NOT advance to shadow/live until Paper Order confirms edge under realistic execution.

---

## ❌ FAMILY_KILL: OI-Price Multi-Bar Divergence

**Reason:** All 4 OI×Price quadrants produced deeply negative Net12 (-249k to -917k). OI-slope combined with price-slope has NO directional edge on 1h bars. The signal density is catastrophic (160k–220k events). This is not a bug — the OI-Price relationship on 1h frequency is noise-dominated.

**Recommendation:** Do not revisit. If OI-based alphas are desired in the future, focus on OI acceleration (D1/D2 from previous research showed marginal PF) or use 4h/8h bar frequency where OI patterns are more meaningful.

---

## ❌ FAMILY_KILL: Funding Regime Transition

**Reason:** All funding mechanisms failed at all tested thresholds. Extreme funding persistence loses money at every tier. Mean-reversion from extreme to neutral occurred **zero times** in 42 days (funding is too sticky on 1h scale). Funding+CVD confirmation also negative.

**Recommendation:** Do not revisit on 1h bars. If funding research is desired, use 4h or 8h bar frequency to match the 8h settlement cycle. Settlement-aware analysis requires settlement-level data (not available in current parquet).

---

## Data Requirements

**Current data is sufficient for CVD Paper Order.** No new data sources needed.

**If expanding:**
- 4h/8h bar data for funding and OI families
- Settlement-level funding data (actual funding payments, not just rates)
- Exchange-native taker volume (vs Coinglass-derived) for CVD robustness check

---

## Next Steps (Priority Order)

1. **CVD Paper Order Simulator** — implement and run immediately
2. **Forward walk-forward** — split data 70/30, train on first 30 days, validate on last 12
3. **If CVD passes Paper Order:** graduate to shadow (live_allowed=false, monitor only)
4. **If CVD fails Paper Order:** return to hypothesis generation — the absorption logic is sound but execution may be the bottleneck

---

## Files Produced

| File | Path |
|------|------|
| Data Audit | `research/cvd/data_audit_cvd.json` |
| CVD R1 Report | `research/cvd/cvd_divergence_research_report.json` |
| CVD R2 Report | `research/cvd/cvd_divergence_r2.json` |
| CVD R3 Report | `research/cvd/cvd_divergence_r3.json` |
| OI R1 Report | `research/oi_divergence/oi_r1_quick.json` |
| Funding R1 Report | `research/funding_transition/fc_r1_quick.json` |
| Iteration Log | `research/alpha_family_iteration_log.md` |
| Final Recommendation | `research/final_recommendation.md` |
