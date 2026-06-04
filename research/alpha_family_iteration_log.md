# Alpha Family Iteration Log

## 2026-06-04 — New Alpha Family Research

**Context:** All 4 old alpha families (DeleveragingReversal, OIShockAbsorption, RelativeStrengthShock, FundingCarryEU) failed statistical validation and Paper Order simulation after frequency fixes. Mandate: find new alpha families.

---

## Family 1: CVD Divergence

### Round 1 — 6 Event Definitions

| Event | n | Net12 | PF | Decision |
|-------|---|-------|-----|----------|
| cvd_z < -2 & ret_6h > -1% | 11,452 | -80,022 | 0.94 | FAIL |
| cvd_z < -2 & close_loc > 0.5 | 10,531 | -68,385 | 0.94 | FAIL |
| cvd_z < -2 & ret_rank > 40% | 11,881 | -113,504 | 0.92 | FAIL |
| absorption_ratio top 5% | 23,511 | -106,354 | 0.94 | FAIL |
| cvd_divergence & vol_z > 1 | 10,922 | **+13,625** | **1.01** | FAIL (PF < 1.15) |
| cvd_divergence (all) | 46,537 | -430,463 | 0.94 | FAIL |

**Failure attribution:** CVD signal density too high (10k–46k events → ~250–1100 per day). Weak edge diluted by noise. The one positive variant (vol_z filter) suggests tighter thresholds + confirmation filters may help.

### Round 2 — 6 Tightened Variants

| Variant | n | Net12 | PF | Decision |
|---------|---|-------|-----|----------|
| V1: z < -2.5 | 4,342 | -45,787 | 1.02 | FAIL |
| V2: z < -2.5 & ret > -1% | 2,110 | -27,954 | 0.98 | FAIL |
| V3: z < -2.5 & close_loc > 0.5 | 1,315 | **+6,259** | **1.16** | ✅ PASS |
| V4: z < -2.5 & vol_z > 1 | 2,788 | -8,001 | 1.11 | FAIL |
| V5: z < -3.0 | 347 | **+10,246** | **1.34** | ✅ PASS |
| V6: absorption top 1% | 4,716 | -7,544 | 1.11 | FAIL |

**Failure attribution on V5:** Top3 contribution 215% of gross — net_wo_top3 = -19,568. Signal relies too heavily on a few outliers.

### Round 3 — Parameter Neighborhood

| Variant | n | Net12 | PF | Hit | net_wo_top3 | Decision |
|---------|---|-------|-----|-----|-------------|----------|
| z<-3.0 & cl>0.5 & h3 | 119 | **+8,136** | **2.02** | 54.6% | **+1,671** | ✅ **PAPER_CANDIDATE** |
| z<-3.0 & cl>0.6 & h3 | 79 | **+5,531** | **2.04** | 53.2% | **+1,861** | ✅ **PAPER_CANDIDATE** |
| z<-3.0 & h3 (V5) | 347 | +10,246 | 1.23 | 50.9% | **-19,568** | FAIL (top3) |
| z<-2.8 & cl>0.5 & h3 | 378 | +8,577 | 1.28 | 49.5% | **-10,913** | FAIL (top3) |

**Final Verdict: PAPER_CANDIDATE**

Winner: `CVD_z < -3.0 AND close_loc > 0.5, hold=3h`
- 119 events across 42 days (2.8/day) — sparse, high quality
- PF=2.02, Hit=54.6%
- net_wo_top3 = +1,671 (positive after removing top 5%)
- Reverse signal: Net12 = negative (not profitable in reverse)
- No BTC panic exposure
- Symbol distribution: well-diversified (max 7 per symbol)
- Survives 15bps cost: Net15 = +5,022

**Risk:** Very sparse (2.8 events/day across 500 symbols). May need universe expansion to be practically tradeable. Parameter sensitivity: close_loc threshold 0.5→0.6 drops n from 119→79 (33% reduction).

---

## Family 2: OI-Price Multi-Bar Divergence

### Round 1 — 4 Quadrants

| Quadrant | n | best Net12 | Decision |
|----------|---|-----------|----------|
| OI↑ Price↓ | 39,651 | -519,713 | FAIL |
| OI↑ Price→ | 63,816 | -832,100 | FAIL |
| OI↓ Price↑ | 46,736 | -249,612 | FAIL |
| OI↓ Price→ | 76,756 | -917,786 | FAIL |

**Failure attribution:** All quadrants deeply negative. OI+Price slope-based classification without additional filters generates massive noise (160k–220k events across 42 days). The OI-Price relationship on 1h bars has no directional edge — both up and down OI movements produce negative forward returns. This is consistent with findings from old OIShockAbsorption (IC=-0.158).

**Final Verdict: FAMILY_KILL**

Reason: OI-slope × price-slope quadrants have zero predictive power on 1h bars. Moving to multi-bar doesn't help without a directional thesis beyond "divergence."

---

## Family 3: Funding Regime Transition

### Round 1 — 4 Mechanisms

| Mechanism | n | best Net12 | Decision |
|-----------|-----|-----------|----------|
| Persistence z<-1.5 | 34,604 | -231,572 | FAIL |
| Persistence z<-2.0 | 22,284 | -104,456 | FAIL |
| Persistence z<-2.5 | 14,355 | -61,149 | FAIL |
| Mean-Reversion z<-2.0 | **0** | — | FAIL (no events) |
| Mean-Reversion z<-2.5 | **0** | — | FAIL (no events) |
| Funding+CVD confirmation | 4,462 | -9,564 | FAIL |

**Failure attribution:** Extreme funding persistence is a trap on 1h bars — every tier loses money (the more extreme, the less it loses, but still negative). Mean-reversion from extreme to neutral literally never happens in 42 days of data (funding is sticky on 1h scale). Funding+CVD confirmation also loses.

**Final Verdict: FAMILY_KILL**

Reason: Funding rate on 1h bars has no standalone alpha. The signal is too slow-moving and settlement mechanics (8h period) make 1h bar research fundamentally misaligned. Settlement-aware analysis was attempted but 0 mean-reversion events occurred. Would need 4h/8h bar frequency to properly capture funding cycles — out of scope for current data.

---

## Summary

| Family | Rounds | Final Verdict | Key Metric |
|--------|--------|---------------|------------|
| CVD Divergence | 3 | **PAPER_CANDIDATE** | Net12=+8,136 PF=2.02 net_wo_top3≥0 |
| OI-Price Divergence | 1 | **FAMILY_KILL** | All quadrants Net12 < -249k |
| Funding Transition | 1 | **FAMILY_KILL** | All mechanisms negative; MR had 0 events |
