"""Shadow Monitor — mandatory metrics, daily reports, promotion criteria.

Tracks 5 mandatory indicators during shadow observation:
1. Signal fidelity (real-time scan vs paper conditions)
2. Top contribution real-time monitoring
3. Slippage pressure (3 cost tiers)
4. Signal delay measurement
5. Rejected-event post-hoc performance

Tiny_live promotion criteria are hard-coded — no subjective judgment.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


# ── Hard-Coded Promotion Criteria ─────────────────

# These are NOT configurable — changing them requires a git commit + audit.
TINY_LIVE_CRITERIA = {
    "min_shadow_trades": 30,
    "min_net_12bps": 0,              # net must be positive at 12bps
    "min_pf": 1.15,
    "require_net_without_top3_positive": True,  # net_no_top3 >= 0
    "max_single_loss_tolerance": "subjective",   # human review required
    "regime_slice_no_runaway_loss": True,        # no regime slice wildly negative
    "no_manual_rule_changes": True,              # rules frozen during shadow
}

# Deleveraging-specific: stricter due to extreme concentration
DELEVERAGING_TINY_LIVE_CRITERIA = {
    **TINY_LIVE_CRITERIA,
    "require_net_without_top3_positive": True,   # MANDATORY — non-negotiable
    "min_shadow_trades": 50,                      # higher bar for low-frequency event
}


# ── Data Structures ──────────────────────────────

@dataclass
class ShadowSignal:
    """A single shadow trade record with all mandatory fields."""
    symbol: str
    event_type: str
    trigger_time: str          # ISO timestamp
    bar_close_time: str        # ISO timestamp
    simulated_entry_time: str  # ISO timestamp
    entry_delay_seconds: float
    direction: str
    regime: str
    regime_confidence: int     # 0=low, 1=medium, 2=high
    event_score: float
    entry_price_sim: float
    exit_price_sim: float | None = None
    hold_bars: int = 2
    cost_assumption_bps: float = 9.0
    gross_bps: float | None = None
    net_bps_9: float | None = None
    net_bps_12: float | None = None
    net_bps_15: float | None = None
    trigger_conditions: dict = field(default_factory=dict)
    reject_reason: str = ""


@dataclass
class RejectedSignal:
    """A rejected signal for post-hoc analysis."""
    symbol: str
    event_type: str
    trigger_time: str
    reject_reason: str
    reject_category: str       # regime, confidence, systemic_oi, spread, data_quality, score
    regime: str = "unknown"
    event_score: float = 0.0
    would_have_gross_bps: float | None = None    # filled later
    would_have_net_bps: float | None = None       # filled later


# ── Shadow Monitor ───────────────────────────────

class ShadowMonitor:
    """Tracks all shadow signals and generates daily reports.

    Usage:
        monitor = ShadowMonitor(event_name="DeleveragingReversal", output_dir="logs/shadow/")
        # On each signal:
        monitor.record_signal(ShadowSignal(...))
        monitor.record_rejected(RejectedSignal(...))
        # Daily:
        report = monitor.daily_report()
    """

    def __init__(self, event_name: str, output_dir: str = "logs/shadow/"):
        self.event_name = event_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.signals: list[ShadowSignal] = []
        self.rejected: list[RejectedSignal] = []

    # ── Recording ────────────────────────────────

    def record_signal(self, sig: ShadowSignal) -> None:
        """Record an executed shadow signal."""
        self.signals.append(sig)

    def record_rejected(self, rej: RejectedSignal) -> None:
        """Record a rejected signal for post-hoc analysis."""
        self.rejected.append(rej)

    def backfill_rejected_returns(self, future_returns: dict[str, float]) -> None:
        """Fill would-have returns for rejected signals.

        Args:
            future_returns: dict mapping (symbol, trigger_time) → gross_bps
        """
        for rej in self.rejected:
            key = f"{rej.symbol}|{rej.trigger_time}"
            if key in future_returns:
                gbp = future_returns[key]
                rej.would_have_gross_bps = gbp
                rej.would_have_net_bps = gbp - (9.0 * (-1 if "short" in str(rej.event_type).lower() else 1))

    # ── Metrics ──────────────────────────────────

    def metric_1_signal_fidelity(self) -> dict:
        """Check that shadow signals match paper conditions."""
        n = len(self.signals)
        if n == 0:
            return {"status": "no_signals", "n": 0}

        # Count by event_type and direction
        by_type = {}
        for s in self.signals:
            key = f"{s.event_type}|{s.direction}"
            by_type[key] = by_type.get(key, 0) + 1

        return {
            "n_total": n,
            "by_event_direction": by_type,
            "unique_symbols": len(set(s.symbol for s in self.signals)),
            "regime_distribution": self._count_regimes(),
            "note": "Verify these match paper expectations. Deviations → investigate.",
        }

    def metric_2_top_contribution(self) -> dict:
        """Real-time top contribution monitoring."""
        arr = np.array([s.net_bps_9 for s in self.signals if s.net_bps_9 is not None])
        n = len(arr)
        if n < 3:
            return {"status": "insufficient_data", "n": n}

        sorted_arr = np.sort(arr)[::-1]
        total = sorted_arr.sum()
        top1 = sorted_arr[0]
        top3 = sorted_arr[:3].sum()
        top5pct_n = max(1, int(n * 0.05))
        top5pct = sorted_arr[:top5pct_n].sum()

        net_no_top3 = sorted_arr[3:].sum()
        net_no_top5pct = sorted_arr[top5pct_n:].sum()

        verdict = "PASS"
        if top3 / total > 0.5 and total > 0:
            verdict = "WARN"
        if net_no_top3 < 0 and total > 0:
            verdict = "FAIL"

        return {
            "n_trades": n,
            "total_net_9bps": round(total, 0),
            "top1_net": round(top1, 0),
            "top1_pct": round(top1 / total * 100, 0) if total != 0 else 0,
            "top3_net": round(top3, 0),
            "top3_pct": round(top3 / total * 100, 0) if total != 0 else 0,
            "top5pct_net": round(top5pct, 0),
            "net_no_top3": round(net_no_top3, 0),
            "net_no_top3_pct": round(net_no_top3 / total * 100, 0) if total != 0 else 0,
            "net_no_top5pct": round(net_no_top5pct, 0),
            "verdict": verdict,
            "tiny_live_blocked": verdict == "FAIL" or net_no_top3 < 0,
        }

    def metric_3_slippage_pressure(self) -> dict:
        """Three cost tiers."""
        result = {}
        for cost, field in [(9, "net_bps_9"), (12, "net_bps_12"), (15, "net_bps_15")]:
            arr = np.array([getattr(s, field, None) for s in self.signals if getattr(s, field, None) is not None])
            if len(arr) == 0:
                result[f"cost_{cost}bps"] = {"n": 0, "net": 0, "alive": False}
                continue
            pos = arr[arr > 0].sum()
            neg = abs(arr[arr < 0].sum())
            pf = pos / neg if neg > 0 else float("inf")
            result[f"cost_{cost}bps"] = {
                "n": len(arr),
                "net": round(arr.sum(), 0),
                "pf": round(pf, 2),
                "hit": round((arr > 0).mean() * 100, 1),
                "alive": arr.sum() > 0,
            }

        alive_12 = result.get("cost_12bps", {}).get("alive", False)
        result["verdict"] = "PASS" if alive_12 else "WARN"
        result["note"] = "Must survive 12bps. If 9bps alive + 12bps dead → margin too thin for live."
        return result

    def metric_4_signal_delay(self) -> dict:
        """Signal delay statistics."""
        delays = [s.entry_delay_seconds for s in self.signals if s.entry_delay_seconds >= 0]
        if not delays:
            return {"n": 0, "status": "no_data"}

        arr = np.array(delays)
        p50 = np.percentile(arr, 50)
        p95 = np.percentile(arr, 95)
        p99 = np.percentile(arr, 99)

        return {
            "n": len(arr),
            "mean_seconds": round(arr.mean(), 1),
            "median_seconds": round(p50, 1),
            "p95_seconds": round(p95, 1),
            "p99_seconds": round(p99, 1),
            "max_seconds": round(arr.max(), 1),
            "verdict": "PASS" if p95 < 60 else "WARN",
            "note": "Event-type strategies die if p95 delay > 60s. Bounce may be gone.",
        }

    def metric_5_rejected_analysis(self) -> dict:
        """Post-hoc: would rejected signals have made money?"""
        by_category: dict[str, dict] = {}
        for rej in self.rejected:
            cat = rej.reject_category
            if cat not in by_category:
                by_category[cat] = {"n": 0, "would_have_net_sum": 0.0, "would_have_net_pos": 0, "samples": []}
            by_category[cat]["n"] += 1
            if rej.would_have_net_bps is not None:
                by_category[cat]["would_have_net_sum"] += rej.would_have_net_bps
                if rej.would_have_net_bps > 0:
                    by_category[cat]["would_have_net_pos"] += 1
                by_category[cat]["samples"].append(rej.would_have_net_bps)

        results = {}
        for cat, data in by_category.items():
            n = data["n"]
            samps = data["samples"]
            results[cat] = {
                "n_rejected": n,
                "n_with_return": len(samps),
                "would_have_total_net": round(data["would_have_net_sum"], 0),
                "would_have_hit_rate": round(data["would_have_net_pos"] / len(samps) * 100, 0) if samps else 0,
                "avg_would_have_net": round(np.mean(samps), 1) if samps else 0,
            }

            # Warning: rejected signals that would have been profitable
            if samps and np.mean(samps) > 0:
                results[cat]["warning"] = f"Rejected by {cat} but would have been profitable — filter may be too strict"

        results["verdict"] = "PASS" if not any(
            v.get("warning") for v in results.values() if isinstance(v, dict)
        ) else "WARN_REVIEW"
        return results

    # ── Daily Report ─────────────────────────────

    def daily_report(self) -> dict:
        """Generate full daily shadow report."""
        return {
            "event_name": self.event_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "shadow_active_days": self._active_days(),
            "1_signal_fidelity": self.metric_1_signal_fidelity(),
            "2_top_contribution": self.metric_2_top_contribution(),
            "3_slippage_pressure": self.metric_3_slippage_pressure(),
            "4_signal_delay": self.metric_4_signal_delay(),
            "5_rejected_analysis": self.metric_5_rejected_analysis(),
            "tiny_live_checklist": self._tiny_live_check(),
        }

    def _tiny_live_check(self) -> dict:
        """Check tiny_live promotion criteria."""
        m2 = self.metric_2_top_contribution()
        m3 = self.metric_3_slippage_pressure()

        criteria = DELEVERAGING_TINY_LIVE_CRITERIA if self.event_name == "DeleveragingReversal" else TINY_LIVE_CRITERIA

        checks = {
            "min_shadow_trades": {
                "required": criteria["min_shadow_trades"],
                "actual": len([s for s in self.signals if s.net_bps_9 is not None]),
                "pass": len([s for s in self.signals if s.net_bps_9 is not None]) >= criteria["min_shadow_trades"],
            },
            "net_positive_at_12bps": {
                "required": "> 0",
                "actual": m3.get("cost_12bps", {}).get("net", 0),
                "pass": m3.get("cost_12bps", {}).get("net", 0) > 0,
            },
            "pf_min_1_15": {
                "required": ">= 1.15",
                "actual": m3.get("cost_12bps", {}).get("pf", 0),
                "pass": m3.get("cost_12bps", {}).get("pf", 0) >= 1.15,
            },
            "net_without_top3_positive": {
                "required": ">= 0",
                "actual": m2.get("net_no_top3", 0),
                "pass": m2.get("net_no_top3", 0) >= 0,
            },
        }

        all_pass = all(c["pass"] for c in checks.values())
        return {
            "checks": checks,
            "all_pass": all_pass,
            "verdict": "READY_FOR_TINY_LIVE_REVIEW" if all_pass else "CONTINUE_SHADOW",
        }

    # ── Helpers ──────────────────────────────────

    def _active_days(self) -> int:
        if not self.signals:
            return 0
        dates = set()
        for s in self.signals:
            try:
                dates.add(s.trigger_time[:10])
            except (TypeError, IndexError):
                pass
        return len(dates)

    def _count_regimes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.signals:
            counts[s.regime] = counts.get(s.regime, 0) + 1
        return counts

    def save_report(self) -> Path:
        """Save daily report to JSON."""
        report = self.daily_report()
        fname = f"{self.event_name}_shadow_{datetime.now().strftime('%Y%m%d')}.json"
        path = self.output_dir / fname
        path.write_text(json.dumps(report, indent=2, default=str))
        return path


# ── Shadow config builder for trading Hermes ─────

def build_shadow_config() -> dict:
    """Generate shadow_config.json for trading Hermes.

    This config MUST be loaded by the trading Hermes before any shadow run.
    It enforces: shadow-only, no orders, no AI override.
    """
    return {
        "version": "1.0",
        "mode": "shadow_only",
        "live_execution": False,
        "order_placement": False,
        "ai_override_forbidden": True,
        "events": {
            "SmallCapDeleveragingReversalV1": {
                "status": "shadow_candidate",
                "tiny_live_allowed": False,
                "priority": 1,
                "min_regime_confidence": 0,
                "enabled_gates": [
                    "return_1h_filter",
                    "oi_bell_score",
                    "volume_z_filter",
                    "panic_chop_filter",
                    "close_location_filter",
                    "funding_z_filter",
                    "systemic_deleveraging_gate",
                    "regime_score_thresholds",
                ],
            },
            "OIShockAbsorptionV1": {
                "status": "shadow_candidate",
                "tiny_live_allowed": False,
                "priority": 2,
                "min_regime_confidence": 0,
                "symmetric_trading_forbidden": True,
                "allow_short_side": False,
                "allow_long_side": True,
                "enabled_gates": [
                    "oi_delta_z_filter",
                    "volume_z_filter",
                    "close_loc_filter",
                    "price_failure_filter",
                    "panic_chop_filter",
                ],
            },
            "RelativeStrengthShockV1": {
                "status": "shadow_candidate",
                "tiny_live_allowed": False,
                "priority": 3,
                "requires_regime_confidence": True,
                "min_regime_confidence": 1,
                "enabled_gates": [
                    "rs_threshold_filter",
                    "volume_z_filter",
                    "oi_delta_z_filter",
                    "close_location_filter",
                    "funding_z_filter",
                    "event_score_filter",
                    "btc_ret_floor_filter",
                    "regime_skip_filter",
                ],
                "range_regime_long_restriction": "shadow_only — requires volume+OI+BTC confirmation",
            },
        },
        "conflict_resolution": {
            "priority_order": [
                "DeleveragingReversal",
                "OIShockAbsorption",
                "RelativeStrengthShock",
            ],
            "direction_conflict_rule": "SKIP_ALL",
            "same_direction_rule": "accept_highest_priority",
        },
        "mandatory_shadow_fields": [
            "symbol", "event_type", "trigger_time", "bar_close_time",
            "simulated_entry_time", "entry_delay_seconds",
            "direction", "regime", "regime_confidence", "event_score",
            "entry_price_sim", "exit_price_sim",
            "hold_bars", "cost_assumption_bps",
            "trigger_conditions", "reject_reason",
            "gross_bps", "net_bps_9", "net_bps_12", "net_bps_15",
        ],
        "daily_report_metrics": [
            "signal_fidelity",
            "top_contribution",
            "slippage_pressure",
            "signal_delay",
            "rejected_analysis",
        ],
        "tiny_live_promotion_criteria": {
            "general": TINY_LIVE_CRITERIA,
            "DeleveragingReversal_special": DELEVERAGING_TINY_LIVE_CRITERIA,
        },
        "forbidden_actions": [
            "Do NOT place orders",
            "Do NOT modify event rules during shadow",
            "Do NOT skip gates without audit",
            "Do NOT promote to tiny_live without all criteria met",
            "Do NOT re-enable symmetric trading for OI Absorption",
        ],
    }


# ── CLI ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "gen-config":
        config = build_shadow_config()
        path = Path("shadow_config.json")
        path.write_text(json.dumps(config, indent=2))
        print(f"Shadow config written to {path}")
        print(f"\nKey settings:")
        print(f"  mode: {config['mode']}")
        print(f"  live_execution: {config['live_execution']}")
        print(f"  order_placement: {config['order_placement']}")
        print(f"  ai_override_forbidden: {config['ai_override_forbidden']}")
        print(f"  # events in shadow: {len(config['events'])}")
        for name, cfg in config['events'].items():
            print(f"    {name}: status={cfg['status']}, tiny_live={cfg['tiny_live_allowed']}")
    else:
        print("Shadow Monitor v1.0")
        print("Usage: python shadow_monitor.py gen-config")
        print()
        print("Tiny Live Promotion Criteria:")
        print(json.dumps(TINY_LIVE_CRITERIA, indent=2))
        print()
        print("Deleveraging Special Criteria:")
        print(json.dumps(DELEVERAGING_TINY_LIVE_CRITERIA, indent=2))
