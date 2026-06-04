"""Funding Carry EU Scanner — real-time shadow signal generator.

Event: Extreme negative funding (< -2.5σ) during EU session (07-17 UTC)
→ earn funding carry from short crowding.

Output: Shadow JSONL + daily summary. Shadow-only, no live orders.
"""
from __future__ import annotations

import json, sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/alpha_pipeline")
sys.path.insert(0, str(ROOT))
from research.regime_detector import detect_regime_fast
from shadow_utils import build_shadow_record


# ── Constants ──────────────────────────────────────

HOLD_BARS = 12
COOLDOWN_BARS = 16
COST_TIERS = [9, 12, 15]
SESSION_START = 7   # 07 UTC
SESSION_END = 17     # 17 UTC
FUNDING_THRESHOLD = -2.5
SHADOW_DIR = ROOT / "logs" / "shadow"


class EventState(Enum):
    WATCH = "watch"
    QUALIFIED = "qualified"
    SHADOW_SIGNAL = "shadow_signal"
    ACTIVE = "active"
    EXITED = "exited"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ScanResult:
    symbol: str
    timestamp: pd.Timestamp
    state: EventState
    event_score: float = 0.0
    direction: str = "long"
    reject_reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ActivePosition:
    symbol: str
    entry_ts: pd.Timestamp
    entry_price: float
    bars_held: int = 0
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None


# ── Funding Carry Scanner ──────────────────────────

class FundingCarryScanner:
    """Real-time scanner for funding carry events in EU session."""

    def __init__(self, data: pd.DataFrame, regime_map: dict, universe: set[str]):
        self.data = data
        self.regime_map = regime_map
        self.universe = universe
        self.ts_list = sorted(data.index.get_level_values("timestamp").unique())
        self.ts_to_idx = {ts: i for i, ts in enumerate(self.ts_list)}
        self.active: dict[str, ActivePosition] = {}
        self.cooldowns: dict[str, int] = {}
        self.signals: list[dict] = []
        self.rejects: list[dict] = []
        self.closed: list[dict] = []

        # Precompute funding_z
        fr = data["funding_rate"]
        gf = fr.groupby(level="symbol")
        fm = gf.transform(lambda s: s.rolling(24, min_periods=8).mean())
        fs = gf.transform(lambda s: s.rolling(24, min_periods=8).std()).replace(0, np.nan)
        self.funding_z = ((fr - fm) / fs).fillna(0.0)

        # Precompute oi_delta_z
        oi_d = data["open_interest"].groupby(level="symbol").transform(lambda s: s.diff(6))
        go = oi_d.groupby(level="symbol")
        om = go.transform(lambda s: s.rolling(48, min_periods=8).mean())
        os_ = go.transform(lambda s: s.rolling(48, min_periods=8).std()).replace(0, np.nan)
        self.oi_delta_z = ((oi_d - om) / os_).fillna(0.0)

    def scan(self, ts: pd.Timestamp) -> list[ScanResult]:
        """Run one scan tick."""
        results: list[ScanResult] = []

        # Session gate
        if ts.hour < SESSION_START or ts.hour >= SESSION_END:
            return results

        # Regime gate
        regime = self.regime_map.get(ts, "unknown")
        if regime == "panic_down":
            return results

        # Decrement cooldowns
        for sym in list(self.cooldowns):
            self.cooldowns[sym] -= 1
            if self.cooldowns[sym] <= 0:
                del self.cooldowns[sym]

        # Check exits
        self._check_exits(ts)

        # Scan symbols at this timestamp
        mask = self.data.index.get_level_values("timestamp") == ts
        syms_at_ts = set(self.data.index[mask].get_level_values("symbol")) & self.universe

        for sym in syms_at_ts:
            if sym in self.cooldowns or sym in self.active:
                continue

            try:
                fz = float(self.funding_z.loc[(ts, sym)])
                oi_z = float(self.oi_delta_z.loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            if np.isnan(fz):
                continue

            # ── Gates ──
            reject = None
            if fz > FUNDING_THRESHOLD:
                reject = f"funding_z={fz:.1f} > {FUNDING_THRESHOLD}"
            elif oi_z < -2.0:
                reject = f"oi_collapse (oi_z={oi_z:.1f})"

            if reject:
                results.append(ScanResult(
                    symbol=sym, timestamp=ts, state=EventState.REJECTED,
                    reject_reason=reject,
                    metrics={"funding_z": fz, "oi_delta_z": oi_z}))
                self.rejects.append(self._to_record(sym, ts, "REJECTED", reject, fz, oi_z))
                continue

            # ── Qualified → Shadow Signal ──
            score = min(1.0, abs(fz - FUNDING_THRESHOLD) / 1.5)
            results.append(ScanResult(
                symbol=sym, timestamp=ts, state=EventState.SHADOW_SIGNAL,
                event_score=round(score, 3),
                metrics={"funding_z": fz, "oi_delta_z": oi_z}))

            self.signals.append(self._to_record(sym, ts, "SHADOW_SIGNAL", "", fz, oi_z, score))

            # Enter position (shadow only)
            self._enter(sym, ts)

        return results

    def _enter(self, sym: str, ts: pd.Timestamp):
        try:
            px = float(self.data.loc[(ts, sym), "close"])
        except KeyError:
            return
        self.active[sym] = ActivePosition(sym, ts, px)

    def _check_exits(self, current_ts: pd.Timestamp):
        current_idx = self.ts_to_idx.get(current_ts)
        if current_idx is None:
            return
        for sym in list(self.active):
            pos = self.active[sym]
            entry_idx = self.ts_to_idx.get(pos.entry_ts)
            if entry_idx is None:
                continue
            bars = current_idx - entry_idx
            if bars >= HOLD_BARS:
                self._exit(sym, current_ts)
            elif self.regime_map.get(current_ts, "") == "panic_down":
                self._exit(sym, current_ts, reason="panic_down")

    def _exit(self, sym: str, ts: pd.Timestamp, reason: str = "hold_complete"):
        pos = self.active.pop(sym, None)
        if pos is None:
            return
        try:
            exit_px = float(self.data.loc[(ts, sym), "close"])
        except KeyError:
            exit_px = pos.entry_price
        gross = (exit_px / pos.entry_price - 1.0) * 10000
        self.closed.append({
            "symbol": sym, "entry_ts": str(pos.entry_ts), "exit_ts": str(ts),
            "gross_bps": round(gross, 1), "reason": reason,
        })
        self.cooldowns[sym] = COOLDOWN_BARS

    def _to_record(self, sym: str, ts: pd.Timestamp, status: str, reject: str,
                   fz: float, oi_z: float, score: float = 0) -> dict:
        return {
            "timestamp": str(ts),
            "symbol": sym,
            "event_type": "FundingCarryEU",
            "status": status,
            "direction": "long",
            "hold_bars": HOLD_BARS,
            "event_score": round(score, 4),
            "regime": self.regime_map.get(ts, "unknown"),
            "funding_z": round(fz, 2),
            "oi_delta_z": round(oi_z, 2),
            "session": "EU" if SESSION_START <= ts.hour < SESSION_END else "OUTSIDE",
            "reject_reason": reject,
            "scanner_config_version": "funding_carry_v1.0",
            "registry_factor_id": "FundingCarryEUV1.0",
        }

    def save_outputs(self, scan_ts: datetime):
        SHADOW_DIR.mkdir(parents=True, exist_ok=True)
        today = scan_ts.strftime("%Y%m%d")

        # JSONL
        jsonl_path = SHADOW_DIR / f"funding_carry_{today}.jsonl"
        with open(jsonl_path, "a") as f:
            for sig in self.signals:
                f.write(json.dumps(sig, default=str) + "\n")
            for rej in self.rejects:
                f.write(json.dumps(rej, default=str) + "\n")

        # Daily summary
        summary = {
            "event_type": "FundingCarryEU",
            "scan_ts": scan_ts.isoformat(),
            "signals_today": len(self.signals),
            "rejected_today": len(self.rejects),
            "active_positions": len(self.active),
            "closed_today": len(self.closed),
            "total_signals_cumulative": len(self.signals),
            "total_closed_cumulative": len(self.closed),
        }
        if self.closed:
            gross_arr = np.array([t["gross_bps"] for t in self.closed])
            for cost in COST_TIERS:
                net_arr = gross_arr - 2 * cost
                pos = net_arr[net_arr > 0].sum()
                neg = abs(net_arr[net_arr < 0].sum())
                summary[f"cumulative_cost_{cost}bps"] = {
                    "n": len(self.closed),
                    "net_bps": round(net_arr.sum(), 0),
                    "pf": round(pos / neg, 3) if neg > 0 else None,
                    "hit_rate": round((net_arr > 0).mean(), 3),
                }

        summary_path = SHADOW_DIR / f"funding_carry_summary_{today}.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        return jsonl_path, summary_path


# ── CLI ────────────────────────────────────────────

if __name__ == "__main__":
    from backtest_engine import BacktestEngine

    print(f"Funding Carry EU Scanner — {datetime.now().isoformat()}")
    engine = BacktestEngine(str(ROOT))
    data = engine._load_data()
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))

    # Universe: rank 20-100
    avg_vol = data["volume"].groupby(level="symbol").mean()
    vol_rank = avg_vol.rank(ascending=False)
    universe = set(vol_rank[(vol_rank >= 20) & (vol_rank <= 100)].index)
    universe &= set(data.index.get_level_values("symbol").unique())
    print(f"Universe: {len(universe)} symbols")

    scanner = FundingCarryScanner(data, regime_map, universe)
    ts_list = sorted(data.index.get_level_values("timestamp").unique())
    latest_ts = ts_list[-1] if ts_list else None
    print(f"Latest timestamp: {latest_ts}")

    if latest_ts:
        results = scanner.scan(latest_ts)
        signals = [r for r in results if r.state == EventState.SHADOW_SIGNAL]
        rejects = [r for r in results if r.state == EventState.REJECTED]
        print(f"\nScan results:")
        print(f"  Signals: {len(signals)}")
        print(f"  Rejected: {len(rejects)}")

        for s in signals[:5]:
            print(f"  {s.symbol:25s} score={s.event_score:.3f} fz={s.metrics.get('funding_z',0):.2f}")

        scanner.save_outputs(datetime.now(timezone.utc))
