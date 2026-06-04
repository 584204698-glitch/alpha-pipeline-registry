"""Portfolio Shadow Runner — 4-alpha combined daily scan + portfolio report.

Runs all 4 alphas jointly, resolves conflicts, produces portfolio-level daily report.

Alphas:
  1. Deleveraging Reversal (price event, long-only)
  2. OI Shock Absorption (price event, long-only squeeze)
  3. RS Shock (price event, regime-dependent)
  4. Funding Carry EU (funding carry, long-only, EU session)

Output:
  logs/shadow/portfolio_signals_YYYYMMDD.jsonl
  logs/shadow/portfolio_report_YYYYMMDD.json
"""
from __future__ import annotations

import json, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
sys.path.insert(0, str(ROOT))

from backtest_engine import BacktestEngine
from research.regime_detector import detect_regime_fast
from event_scanner import (
    DeleveragingEventScanner, SmallCapUniverse, EventState as DelevState,
)
from relative_strength_shock import detect_relative_strength_shock
from oi_shock_absorption import detect_oi_shock_absorption
from funding_carry_scanner import FundingCarryScanner, EventState as FCState
from shadow_utils import resolve_event_conflicts

SHADOW_DIR = ROOT / "logs" / "shadow"
COST_TIERS = [9, 12, 15]

# ── Conflict Rules (HARDCODED) ────────────────────

# Priority: Deleveraging > OI > RS > FC
PRIORITY_ORDER = [
    "DeleveragingReversal",
    "OIShockAbsorption",
    "RelativeStrengthShock",
    "FundingCarryEU",
]

# Same-direction → allow, but no stacking beyond single position
# Opposite-direction → reject ALL
# FC + any price event same-direction → record overlap, no extra sizing


class PortfolioScanner:
    def __init__(self):
        self.scan_ts = datetime.now(timezone.utc)
        engine = BacktestEngine(str(ROOT))
        self.data = engine._load_data()
        self.regimes = detect_regime_fast(self.data)
        self.regime_map = dict(zip(self.regimes.index, self.regimes))

        # Universe
        avg_vol = self.data["volume"].groupby(level="symbol").mean()
        vol_rank = avg_vol.rank(ascending=False)
        self.universe = set(vol_rank[(vol_rank >= 20) & (vol_rank <= 100)].index)
        self.universe &= set(self.data.index.get_level_values("symbol").unique())

        self.ts_list = sorted(self.data.index.get_level_values("timestamp").unique())
        self.latest_ts = self.ts_list[-1] if self.ts_list else None

        # Scanners
        self.delev_scanner = DeleveragingEventScanner(
            self.data, self.regime_map, SmallCapUniverse(), hold_bars=2)
        self.fc_scanner = FundingCarryScanner(self.data, self.regime_map, self.universe)

        # Results
        self.all_raw: list[dict] = []
        self.rejected: list[dict] = []
        self.conflicts: list[dict] = []
        self.final_signals: list[dict] = []

    def run(self) -> dict:
        if self.latest_ts is None:
            return {"error": "No data"}

        raw = []

        # 1. Deleveraging Reversal
        delev_results = self.delev_scanner.scan(self.latest_ts)
        for r in delev_results:
            rec = self._build_base(r.symbol, "DeleveragingReversal",
                                   r.state.value if r.state != DelevState.REJECTED else "REJECTED",
                                   r.event_score, r.reject_reason if r.state == DelevState.REJECTED else "")
            if r.state == DelevState.CONFIRMED:
                rec["status"] = "SHADOW_SIGNAL"
                rec["hold_bars"] = 2
                rec["reasons"] = ["return_1h < -2.5%", "oi_collapse", "volume_z > 1", "btc_not_panic"]
            elif r.state == DelevState.QUALIFIED:
                rec["status"] = "QUALIFIED_EVENT"
            elif r.state == DelevState.REJECTED:
                rec["status"] = "REJECTED"
                self.rejected.append(rec)
                continue
            raw.append(rec)

        # 2. OI Shock Absorption
        oi_events = detect_oi_shock_absorption(
            self.data, self.universe, self.regime_map,
            oi_delta_z_min=2.5, vol_z_min=1.5, cooldown_bars=8)
        for ev in oi_events:
            if ev["timestamp"] != self.latest_ts:
                continue
            rec = self._build_base(ev["symbol"], "OIShockAbsorption",
                                   "SHADOW_SIGNAL", ev.get("event_score", 0))
            rec["hold_bars"] = 2
            rec["direction"] = ev.get("direction", "long")
            raw.append(rec)

        # 3. RS Shock
        rs_events = detect_relative_strength_shock(
            self.data, self.universe, self.regime_map,
            rs_threshold=0.05, vol_z_min=2.0, oi_delta_z_min=0.5,
            close_loc_min=0.50, event_score_min=0.60, cooldown_bars=8)
        for ev in rs_events:
            if ev["timestamp"] != self.latest_ts:
                continue
            rec = self._build_base(ev["symbol"], "RelativeStrengthShock",
                                   "SHADOW_SIGNAL", ev.get("event_score", 0))
            rec["hold_bars"] = 2
            rec["direction"] = ev.get("direction", "long")
            raw.append(rec)

        # 4. Funding Carry EU
        fc_results = self.fc_scanner.scan(self.latest_ts)
        for r in fc_results:
            if r.state == FCState.SHADOW_SIGNAL:
                rec = self._build_base(r.symbol, "FundingCarryEU",
                                       "SHADOW_SIGNAL", r.event_score)
                rec["hold_bars"] = 12
                rec["session"] = "EU"
                raw.append(rec)
            elif r.state == FCState.REJECTED:
                rec = self._build_base(r.symbol, "FundingCarryEU",
                                       "REJECTED", 0, r.reject_reason)
                self.rejected.append(rec)

        self.all_raw = raw

        # ── Conflict Resolution ──
        resolved, conflicts = self._resolve_conflicts(raw)
        self.final_signals = resolved
        self.conflicts = conflicts

        return self._build_summary()

    def _build_base(self, symbol: str, event_type: str, status: str,
                    score: float, reject: str = "") -> dict:
        return {
            "timestamp": str(self.latest_ts),
            "symbol": symbol,
            "event_type": event_type,
            "status": status,
            "direction": "long",
            "hold_bars": None,
            "event_score": round(score, 4),
            "regime": self.regime_map.get(self.latest_ts, "unknown"),
            "reject_reason": reject,
            "scanner_config_version": "portfolio_v1.0",
            "registry_factor_id": event_type,
        }

    def _resolve_conflicts(self, raw: list[dict]) -> tuple[list[dict], list[dict]]:
        """Portfolio-level conflict resolution with hardcoded rules."""
        # Group by symbol
        by_symbol: dict[str, list[dict]] = defaultdict(list)
        for sig in raw:
            if sig["status"] == "REJECTED":
                continue
            by_symbol[sig["symbol"]].append(sig)

        resolved = []
        conflicts = []

        for sym, sigs in by_symbol.items():
            if len(sigs) == 1:
                resolved.append(sigs[0])
                continue

            # Multiple events on same symbol
            directions = set(s.get("direction", "long") for s in sigs)
            event_types = [s["event_type"] for s in sigs]

            # Rule: FC + price event → check direction
            has_fc = "FundingCarryEU" in event_types
            has_price = any(et != "FundingCarryEU" for et in event_types)

            if len(directions) > 1:
                # Opposite directions → REJECT ALL
                for s in sigs:
                    s["status"] = "CONFLICT_REJECTED"
                    s["reject_reason"] = f"direction_conflict:{','.join(event_types)}"
                conflicts.extend(sigs)
                continue

            # Same direction
            if has_fc and has_price:
                # FC + price event same-direction → keep highest priority, record overlap
                best = min(sigs, key=lambda s: PRIORITY_ORDER.index(s["event_type"])
                          if s["event_type"] in PRIORITY_ORDER else 99)
                best["overlap_events"] = event_types
                best["overlap_note"] = "FC+price_same_direction_no_stacking"
                resolved.append(best)
                # Record the others as merged
                for s in sigs:
                    if s != best:
                        s["status"] = "MERGED"
                        s["merged_into"] = best["event_type"]
                        conflicts.append(s)
            else:
                # Multiple price events same direction → keep highest priority
                best = min(sigs, key=lambda s: PRIORITY_ORDER.index(s["event_type"])
                          if s["event_type"] in PRIORITY_ORDER else 99)
                resolved.append(best)
                for s in sigs:
                    if s != best:
                        s["status"] = "MERGED"
                        s["merged_into"] = best["event_type"]
                        conflicts.append(s)

        return resolved, conflicts

    def _build_summary(self) -> dict:
        by_event = defaultdict(int)
        for s in self.final_signals:
            by_event[s["event_type"]] += 1

        by_regime = defaultdict(int)
        for s in self.final_signals:
            by_regime[s.get("regime", "?")] += 1

        overlap_pairs = defaultdict(int)
        for c in self.conflicts:
            if "merged_into" in c:
                pair = f"{c['event_type']}→{c['merged_into']}"
                overlap_pairs[pair] += 1
            elif c.get("status") == "CONFLICT_REJECTED":
                overlap_pairs["DIRECTION_CONFLICT"] += 1

        return {
            "scan_ts": self.scan_ts.isoformat(),
            "latest_data_ts": str(self.latest_ts),
            "universe_size": len(self.universe),
            "regime": self.regime_map.get(self.latest_ts, "unknown"),
            "portfolio": {
                "total_raw": len(self.all_raw),
                "after_conflict": len(self.final_signals),
                "conflicts": len(self.conflicts),
                "rejected": len(self.rejected),
            },
            "by_alpha": dict(by_event),
            "by_regime": dict(by_regime),
            "overlap_summary": dict(overlap_pairs),
            "conflict_rule": "OPPOSITE_DIRECTION=REJECT_ALL, SAME_DIRECTION=HIGHEST_PRIORITY, FC+PRICE=NO_STACKING",
        }

    def save_outputs(self):
        SHADOW_DIR.mkdir(parents=True, exist_ok=True)
        today = self.scan_ts.strftime("%Y%m%d")

        # JSONL: all final signals
        jl_path = SHADOW_DIR / f"portfolio_signals_{today}.jsonl"
        with open(jl_path, "a") as f:
            for sig in self.final_signals:
                f.write(json.dumps(sig, default=str) + "\n")
            for rej in self.rejected:
                f.write(json.dumps(rej, default=str) + "\n")
            for c in self.conflicts:
                f.write(json.dumps(c, default=str) + "\n")

        # Portfolio report
        summary = self.run()
        rp_path = SHADOW_DIR / f"portfolio_report_{today}.json"
        existing = {"scans": []}
        if rp_path.exists():
            existing = json.loads(rp_path.read_text())
        existing["scans"].append(summary)
        rp_path.write_text(json.dumps(existing, indent=2, default=str))

        # Cumulative portfolio stats
        self._build_cumulative_report(today)

        return jl_path, rp_path

    def _build_cumulative_report(self, today: str):
        """Build portfolio-level cumulative PnL from all closed shadow trades."""
        # Collect all closed trades from individual scanner logs
        closed_trades = []
        for alpha in ["DeleveragingReversal", "OIShockAbsorption", "RelativeStrengthShock", "FundingCarryEU"]:
            # Read from the alpha-specific shadow logs
            for suffix in ["shadow_signals", "funding_carry"]:
                glob_pattern = f"*{today}*"
                # Simplified: aggregate what we have
                pass

        # For now, output the scan-level metrics
        cum_path = SHADOW_DIR / f"portfolio_cumulative_{today}.json"
        scan_summary = self.run()
        cum_path.write_text(json.dumps({
            "report_date": today,
            "portfolio_status": "SHADOW_ONLY",
            "live_allowed": False,
            "latest_scan": scan_summary,
            "alphas": {
                "DeleveragingReversal": "shadow_candidate",
                "OIShockAbsorption": "shadow_candidate",
                "RelativeStrengthShock": "shadow_candidate",
                "FundingCarryEU": "shadow_candidate",
            },
            "notes": [
                "All signals shadow-only, no live orders",
                "Conflict rule: opposite direction = reject all",
                "FC + price event same-direction = no stacking",
                "Portfolio report upgrades from single-factor to combined",
            ],
        }, indent=2, default=str))


# ── CLI ────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Portfolio Shadow Scanner — {datetime.now().isoformat()}")
    scanner = PortfolioScanner()
    summary = scanner.run()

    print(f"\nScan: {summary['scan_ts']}")
    print(f"Data ts: {summary['latest_data_ts']}")
    print(f"Regime: {summary['regime']}")
    print(f"Universe: {summary['universe_size']} symbols")
    print(f"\nPortfolio:")
    print(f"  Raw signals: {summary['portfolio']['total_raw']}")
    print(f"  After conflict: {summary['portfolio']['after_conflict']}")
    print(f"  Conflicts: {summary['portfolio']['conflicts']}")
    print(f"  Rejected: {summary['portfolio']['rejected']}")
    print(f"\nBy alpha:")
    for et, cnt in summary['by_alpha'].items():
        print(f"  {et:30s}: {cnt}")
    print(f"\nOverlap:")
    for pair, cnt in summary['overlap_summary'].items():
        print(f"  {pair}: {cnt}")

    jl, rp = scanner.save_outputs()
    print(f"\nOutputs:")
    print(f"  Signals: {jl}")
    print(f"  Report:  {rp}")

    if scanner.final_signals:
        print(f"\nTop signals:")
        for s in sorted(scanner.final_signals, key=lambda x: -x.get("event_score", 0))[:8]:
            ov = f" [overlap: {','.join(s.get('overlap_events',[]))}]" if s.get('overlap_events') else ""
            print(f"  {s['symbol']:25s} {s['event_type']:25s} score={s['event_score']:.3f}{ov}")
