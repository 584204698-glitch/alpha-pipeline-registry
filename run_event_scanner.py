"""SmallCap Event Scanner Runner — automated 15m scan → shadow record.

Usage:
    python run_event_scanner.py              # single scan
    python run_event_scanner.py --daemon     # run once, for cron

Output:
    logs/shadow/shadow_signals_YYYYMMDD.jsonl   # all signals (accepted + rejected)
    logs/shadow/shadow_summary_YYYYMMDD.json    # daily summary

This is designed to be called every 15 minutes by cron.
"""
from __future__ import annotations

import json
import sys
import time
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
    DeleveragingEventScanner,
    SmallCapUniverse,
    EventState,
    REGIME_RULES,
    get_regime_rule,
)
from relative_strength_shock import detect_relative_strength_shock
from oi_shock_absorption import detect_oi_shock_absorption
from shadow_utils import compute_regime_confidence, resolve_event_conflicts


# ── Constants ──────────────────────────────────────

EVENT_PRIORITY = ["DeleveragingReversal", "OIShockAbsorption", "RelativeStrengthShock"]
SHADOW_DIR = ROOT / "logs" / "shadow"
HOLD_BARS = 2
COST_TIERS = [9, 12, 15]


# ── Scanner Runner ─────────────────────────────────

class EventScannerRunner:
    """Loads data, runs all 3 event scanners, records shadow signals."""

    def __init__(self):
        self.scan_ts = datetime.now(timezone.utc)

        # Load data
        self.engine = BacktestEngine(str(ROOT))
        self.data = self.engine._load_data()
        self.regimes = detect_regime_fast(self.data)
        self.regime_map = dict(zip(self.regimes.index, self.regimes))
        self.regime_conf = compute_regime_confidence(self.regimes)
        self.regime_conf_map = dict(zip(self.regime_conf.index, self.regime_conf))

        # Small-cap universe
        avg_vol = self.data["volume"].groupby(level="symbol").mean()
        vol_rank = avg_vol.rank(ascending=False)
        self.small_syms = set(vol_rank[(vol_rank >= 20) & (vol_rank <= 100)].index)

        # Latest timestamp
        self.ts_list = sorted(self.data.index.get_level_values("timestamp").unique())
        self.latest_ts = self.ts_list[-1] if self.ts_list else None

        # Results
        self.all_signals: list[dict] = []
        self.rejected: list[dict] = []

    def run(self) -> dict:
        """Execute all scans and return summary."""
        # 1. Deleveraging Reversal (scanner-based)
        delev_signals, delev_rejects = self._scan_deleveraging()

        # 2. OI Shock Absorption (event-based)
        oi_signals, oi_rejects = self._scan_oi_absorption()

        # 3. RS Shock (event-based)
        rs_signals, rs_rejects = self._scan_rs_shock()

        all_raw = delev_signals + oi_signals + rs_signals
        all_rejects = delev_rejects + oi_rejects + rs_rejects

        # Conflict resolution
        resolved = resolve_event_conflicts(all_raw, EVENT_PRIORITY)
        for sig in all_raw:
            if sig not in resolved:
                sig["status"] = "REJECTED"
                sig["reject_reason"] = "conflict_resolution"
                all_rejects.append(sig)

        self.all_signals = resolved
        self.rejected = all_rejects

        return {
            "scan_ts": self.scan_ts.isoformat(),
            "latest_data_ts": str(self.latest_ts),
            "small_cap_pool_size": len(self.small_syms),
            "signals": {
                "total_raw": len(all_raw),
                "after_conflict_resolution": len(resolved),
                "by_event": self._count_by_event(resolved),
                "by_regime": self._count_by_regime(resolved),
            },
            "rejected": {
                "total": len(all_rejects),
                "by_reason": self._count_by_reason(all_rejects),
            },
        }

    # ── Per-Event Scanners ──────────────────────────

    def _scan_deleveraging(self) -> tuple[list[dict], list[dict]]:
        """Scan Deleveraging Reversal using the event_scanner."""
        scanner = DeleveragingEventScanner(self.data, self.regime_map, hold_bars=HOLD_BARS)
        results = scanner.scan(self.latest_ts) if self.latest_ts else []

        signals = []
        rejects = []
        for r in results:
            record = self._build_base_record(r.symbol, "DeleveragingReversal", "long", r.event_score)
            record["metrics"]["oi_z"] = round(r.metrics.get("oi_z", 0), 2)
            record["metrics"]["ret_1h"] = round(r.metrics.get("ret_1h", 0), 4)
            record["metrics"]["vol_z"] = round(r.metrics.get("vol_z", 0), 2)
            record["metrics"]["close_loc"] = round(r.metrics.get("close_loc", 0), 3)

            if r.state == EventState.REJECTED:
                record["status"] = "REJECTED"
                record["reject_reason"] = r.reject_reason
                if "funding_z" in str(r.reject_reason).lower():
                    record["reject_category"] = "funding_z"
                elif "systemic" in str(r.reject_reason).lower():
                    record["reject_category"] = "systemic_oi"
                elif "panic" in str(r.reject_reason).lower():
                    record["reject_category"] = "regime"
                elif "return" in str(r.reject_reason).lower():
                    record["reject_category"] = "return"
                elif "oi_z" in str(r.reject_reason).lower():
                    record["reject_category"] = "oi_z"
                elif "volume" in str(r.reject_reason).lower():
                    record["reject_category"] = "volume_z"
                else:
                    record["reject_category"] = "other"
                rejects.append(record)
            elif r.state in (EventState.CONFIRMED, EventState.QUALIFIED):
                record["status"] = "SHADOW_SIGNAL" if r.state == EventState.CONFIRMED else "QUALIFIED_EVENT"
                record["hold_bars"] = HOLD_BARS
                record["entry_price_sim"] = self._get_price(r.symbol)
                record["direction"] = "long"
                record["expected_cost_bps"] = 9
                record["reasons"] = [
                    "return_1h < -2.5%",
                    "-3 < oi_delta_z <= -2",
                    "volume_z > 1.0",
                    "btc_not_panic_down",
                    "close_location > 0.35",
                    "|funding_z| < 2.0",
                    "systemic_collapse_filter_pass",
                ]
                signals.append(record)
            else:
                record["status"] = "WATCH"
                signals.append(record)

        return signals, rejects

    def _scan_oi_absorption(self) -> tuple[list[dict], list[dict]]:
        """Scan OI Shock Absorption events."""
        events = detect_oi_shock_absorption(
            self.data, self.small_syms, self.regime_map,
            oi_delta_z_min=2.5, vol_z_min=1.5, cooldown_bars=8,
        )
        # Filter to only latest timestamp events
        latest_events = [e for e in events if e["timestamp"] == self.latest_ts] if self.latest_ts else []

        signals = []
        rejects = []
        for ev in latest_events:
            record = self._build_base_record(ev["symbol"], "OIShockAbsorption", ev["direction"], ev["event_score"])
            record["metrics"]["oi_z"] = ev.get("oi_z", 0)
            record["metrics"]["vol_z"] = ev.get("vol_z", 0)
            record["metrics"]["close_loc"] = ev.get("close_loc", 0)
            record["metrics"]["ret_15m_pct"] = ev.get("ret_15m_pct", 0)
            record["metrics"]["ret_1h_pct"] = ev.get("ret_1h_pct", 0)
            record["status"] = "SHADOW_SIGNAL"
            record["hold_bars"] = HOLD_BARS
            record["entry_price_sim"] = self._get_price(ev["symbol"])
            record["expected_cost_bps"] = 9
            record["reasons"] = [
                "oi_delta_z > 2.5",
                "volume_z > 1.5",
                "close_loc < 0.4 (trapped shorts)",
                "price_failure_confirmed",
            ]
            signals.append(record)

        return signals, rejects

    def _scan_rs_shock(self) -> tuple[list[dict], list[dict]]:
        """Scan Relative Strength Shock events."""
        events = detect_relative_strength_shock(
            self.data, self.small_syms, self.regime_map,
            rs_threshold=0.05, vol_z_min=2.0, oi_delta_z_min=0.5,
            close_loc_min=0.50, event_score_min=0.60, cooldown_bars=8,
        )
        latest_events = [e for e in events if e["timestamp"] == self.latest_ts] if self.latest_ts else []

        signals = []
        rejects = []
        for ev in latest_events:
            record = self._build_base_record(ev["symbol"], "RelativeStrengthShock", ev["direction"], ev["event_score"])
            record["metrics"]["rs_pct"] = ev.get("rs_pct", 0)
            record["metrics"]["ret_1h_pct"] = ev.get("ret_1h_pct", 0)
            record["metrics"]["btc_ret_1h_pct"] = ev.get("btc_ret_1h_pct", 0)
            record["metrics"]["vol_z"] = ev.get("vol_z", 0)
            record["metrics"]["oi_delta_z"] = ev.get("oi_delta_z", 0)
            record["metrics"]["close_loc"] = ev.get("close_loc", 0)

            rg_conf = self.regime_conf_map.get(ev.get("regime", "unknown"), 0)

            # RS Shock requires regime_confidence >= 1 for range trades
            if ev.get("regime") == "range" and rg_conf < 1:
                record["status"] = "REJECTED"
                record["reject_reason"] = "regime_confidence_low"
                record["reject_category"] = "regime_confidence"
                rejects.append(record)
                continue

            record["status"] = "SHADOW_SIGNAL"
            record["hold_bars"] = HOLD_BARS
            record["entry_price_sim"] = self._get_price(ev["symbol"])
            record["expected_cost_bps"] = 9
            record["reasons"] = [
                "RS > 5%",
                "volume_z > 2.0",
                "oi_delta_z > 0.5",
                "close_loc > 0.50",
                "|funding_z| < 2.5",
                "event_score > 0.60",
                f"regime={ev.get('regime')}",
            ]
            signals.append(record)

        return signals, rejects

    # ── Helpers ─────────────────────────────────────

    def _build_base_record(self, symbol: str, event_type: str, direction: str, score: float) -> dict:
        """Build base shadow record with all mandatory fields."""
        regime = self.regime_map.get(self.latest_ts, "unknown") if self.latest_ts else "unknown"
        rg_conf = int(self.regime_conf_map.get(self.latest_ts, 0)) if self.latest_ts else 0

        return {
            "timestamp": str(self.latest_ts) if self.latest_ts else str(self.scan_ts),
            "scan_time": self.scan_ts.isoformat(),
            "symbol": symbol,
            "event_type": event_type,
            "status": "UNKNOWN",
            "direction": direction,
            "event_score": round(score, 4),
            "regime": regime,
            "regime_confidence": rg_conf,
            "hold_bars": None,
            "entry_price_sim": None,
            "expected_cost_bps": None,
            "reject_reason": "",
            "reject_category": "",
            "reasons": [],
            "metrics": {},
        }

    def _get_price(self, symbol: str) -> float | None:
        """Get latest close price for a symbol."""
        if self.latest_ts is None:
            return None
        try:
            return round(float(self.data.loc[(self.latest_ts, symbol), "close"]), 8)
        except KeyError:
            return None

    def _count_by_event(self, signals: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in signals:
            counts[s["event_type"]] = counts.get(s["event_type"], 0) + 1
        return counts

    def _count_by_regime(self, signals: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in signals:
            counts[s.get("regime", "unknown")] = counts.get(s.get("regime", "unknown"), 0) + 1
        return counts

    def _count_by_reason(self, rejects: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in rejects:
            cat = r.get("reject_category", r.get("reject_reason", "unknown"))
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def save_outputs(self) -> tuple[Path, Path]:
        """Save signals JSONL and summary JSON."""
        SHADOW_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")

        # JSONL: all signals + rejected
        jsonl_path = SHADOW_DIR / f"shadow_signals_{today}.jsonl"
        with open(jsonl_path, "a") as f:
            for sig in self.all_signals:
                f.write(json.dumps(sig, default=str) + "\n")
            for rej in self.rejected:
                f.write(json.dumps(rej, default=str) + "\n")

        # Summary
        summary_path = SHADOW_DIR / f"shadow_summary_{today}.json"
        summary = self.run()
        if summary_path.exists():
            existing = json.loads(summary_path.read_text())
            existing["scans"].append(summary)
        else:
            existing = {"event_name": "SmallCapEventScanner", "scans": [summary]}
        summary_path.write_text(json.dumps(existing, indent=2, default=str))

        return jsonl_path, summary_path


# ── CLI ────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Event Scanner Runner — {datetime.now().isoformat()}")
    runner = EventScannerRunner()
    summary = runner.run()

    print(f"\nScan: {summary['scan_ts']}")
    print(f"Data timestamp: {summary['latest_data_ts']}")
    print(f"Small-cap pool: {summary['small_cap_pool_size']} symbols")
    print(f"\nSignals:")
    print(f"  Raw: {summary['signals']['total_raw']}")
    print(f"  After conflict resolution: {summary['signals']['after_conflict_resolution']}")
    print(f"  By event: {summary['signals']['by_event']}")
    print(f"  By regime: {summary['signals']['by_regime']}")
    print(f"\nRejected:")
    print(f"  Total: {summary['rejected']['total']}")
    print(f"  By reason: {summary['rejected']['by_reason']}")

    jsonl_p, sum_p = runner.save_outputs()
    print(f"\nOutputs:")
    print(f"  Signals: {jsonl_p}")
    print(f"  Summary: {sum_p}")

    # Show top signals
    if runner.all_signals:
        print(f"\nTop shadow signals:")
        for sig in sorted(runner.all_signals, key=lambda s: -s.get("event_score", 0))[:5]:
            print(f"  {sig['symbol']:20s} {sig['event_type']:25s} "
                  f"score={sig['event_score']:.3f} dir={sig['direction']:5s} "
                  f"regime={sig.get('regime','?')}")
