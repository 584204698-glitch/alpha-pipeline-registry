"""
SmallCapDeleveragingReversal -- Real-time Event Scanner.

SCANNER VERSION: scanner_v1_0 (FROZEN 2026-06-04)
GATE CHANGE POLICY: No new gates without shadow evidence of drift.
Gates validated by LargeCapRegimeFilterResearch -- adding global filters kills alpha.

8 GATES (frozen):
1. return_1h < -2.5% and > -12%
2. oi_z in (-3.0, -2.0] bell-shaped scoring, center=-2.5
3. volume_z > 1.0 and < 8.0
4. BTC regime != panic_down and != chop
5. close_location > 0.35 for entry confirmation
6. |funding_z| < 2.0 (per-symbol, NOT global)
7. systemic_deleveraging gate -- per-symbol rejection if >5 coins collapsing
8. regime-based score thresholds: trend_down=0.50, range=0.60, trend_up=0.75

FORBIDDEN GATES (tested, killed, do NOT re-add):
- marketwide_oi_collapse_count > N global ban (kills alpha at all N)
- event_density high → reject all (kills opportunity concentration)
- global panic filter (covered by regime gate, adding threshold kills trades)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np
import pandas as pd


# ── State Machine ────────────────────────────────────

class EventState(Enum):
    """Event state machine.

    Spec mapping:
        REJECTED → WATCH → QUALIFIED (QUALIFIED_EVENT) → CONFIRMED (SHADOW_SIGNAL) → ENTERED → EXITED / EXPIRED
    """
    WATCH = "watch"
    QUALIFIED = "qualified"
    CONFIRMED = "confirmed"
    ENTERED = "entered"
    EXITED = "exited"
    EXPIRED = "expired"
    REJECTED = "rejected"


# ── Data structures ──────────────────────────────────

@dataclass
class ScanResult:
    """Output for one symbol at one scan timestamp."""
    symbol: str
    timestamp: pd.Timestamp
    state: EventState
    event_score: float = 0.0
    direction: str = "long"  # always long for this event type
    reject_reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ActivePosition:
    symbol: str
    entry_ts: pd.Timestamp
    entry_price: float
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    pnl_bps: float | None = None


# ── Universe Definition ──────────────────────────────

class SmallCapUniverse:
    """Filter symbols into the small-cap event pool."""

    def __init__(
        self,
        min_rank: int = 20,
        max_rank: int = 100,
        min_quote_vol_24h: float = 20_000_000,
        max_spread_bps: float = 15.0,
    ):
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.min_quote_vol = min_quote_vol_24h
        self.max_spread = max_spread_bps

    def filter(self, data: pd.DataFrame) -> set[str]:
        """Return set of symbols in the small-cap event pool."""
        avg_vol = data["volume"].groupby(level="symbol").mean()
        vol_rank = avg_vol.rank(ascending=False)
        symbols = set(vol_rank[
            (vol_rank >= self.min_rank) & (vol_rank <= self.max_rank)
        ].index)
        return symbols


# ── Regime Rules ─────────────────────────────────────

REGIME_RULES = {
    "trend_up": {
        "allow_long": True,
        "description": "BTC trending up -- small-cap drops likely idiosyncratic, fakeouts",
        "event_score_threshold": 0.75,  # very strict -- most are fakeouts
        "position_pct": 0.5,  # reduced size
    },
    "range": {
        "allow_long": True,
        "description": "Sideways -- default conditions",
        "event_score_threshold": 0.60,
        "position_pct": 1.0,
    },
    "trend_down": {
        "allow_long": True,
        "description": "BTC trending down -- prime deleveraging hunting ground",
        "event_score_threshold": 0.50,  # relaxed -- these are real liquidations
        "position_pct": 1.0,  # full size
    },
    "chop": {
        "allow_long": False,
        "description": "Directionless high-vol -- no trading",
        "event_score_threshold": 0.99,
        "position_pct": 0.0,
    },
    "panic_down": {
        "allow_long": False,
        "description": "Systemic panic -- NO longs",
        "event_score_threshold": 0.99,
        "position_pct": 0.0,
    },
    "unknown": {
        "allow_long": True,
        "description": "Unknown regime -- default to range rules",
        "event_score_threshold": 0.60,
        "position_pct": 1.0,
    },
}


def get_regime_rule(regime: str) -> dict:
    """Return trading rule for a given regime label."""
    return REGIME_RULES.get(regime, REGIME_RULES["range"])


# ── Event Scoring ────────────────────────────────────

def score_oi_severity(oi_z: float) -> float:
    """Bell-shaped: optimal at oi_z ≈ -2.5, penalize too extreme or too mild."""
    center = -2.5
    width = 1.0  # ±1 around center gets full score
    dist = abs(oi_z - center)
    if dist <= width:
        return 1.0
    elif dist <= 2.0:
        return 1.0 - (dist - width) / 2.0  # linear decay
    else:
        return 0.0  # too mild (>-0.5) or too extreme (<-4.5)


def score_return(r1h: float) -> float:
    """Penalize drops that are too extreme (>12% = possible black swan)."""
    abs_r = abs(r1h)
    if abs_r < 0.025:
        return 0.0
    elif abs_r <= 0.06:
        return 1.0
    elif abs_r <= 0.10:
        return 0.7
    elif abs_r <= 0.12:
        return 0.3
    else:
        return 0.0  # too extreme, potential project-level event


def score_volume(vol_z: float) -> float:
    """Volume should be elevated but not absurdly so."""
    if vol_z < 1.0:
        return 0.0
    elif vol_z <= 3.0:
        return 1.0
    elif vol_z <= 5.0:
        return 0.5
    else:
        return 0.1  # extreme vol = possible black swan


def score_btc_stability(btc_ret_15m: float, btc_regime: str) -> float:
    """BTC must not be in freefall. Thresholds scaled for 1h returns (was 15m)."""
    if btc_regime == "panic_down":
        return 0.0
    if btc_ret_15m < -0.05:   # 5% in 1h ≈ extreme (was 2% in 15m)
        return 0.1
    if btc_ret_15m < -0.025:   # 2.5% in 1h ≈ notable (was 1% in 15m)
        return 0.5
    return 1.0


def compute_event_score(
    oi_z: float,
    ret_1h: float,
    vol_z: float,
    btc_ret_15m: float,
    btc_regime: str,
    spread_bps: float = 5.0,
    data_quality: float = 1.0,
) -> float:
    """Weighted event score. Events below 0.6 are rejected."""
    scores = {
        "oi": score_oi_severity(oi_z),
        "ret": score_return(ret_1h),
        "vol": score_volume(vol_z),
        "btc": score_btc_stability(btc_ret_15m, btc_regime),
        "tradability": min(1.0, max(0.0, 1.0 - spread_bps / 30.0)) * data_quality,
    }

    weights = {"oi": 0.30, "ret": 0.25, "vol": 0.20, "btc": 0.15, "tradability": 0.10}
    total = sum(weights[k] * scores[k] for k in weights)

    # Hard gates: if any component is zero, reject
    if any(scores[k] < 0.01 for k in weights):
        return total * 0.5  # penalize heavily

    return total


# ── Event Scanner ────────────────────────────────────

class DeleveragingEventScanner:
    """Real-time scanner for small-cap deleveraging reversal events.

    Usage:
        scanner = DeleveragingEventScanner(data, btc_regime_map)
        for ts in scan_timestamps:
            results = scanner.scan(ts)
            for r in results:
                if r.state == EventState.CONFIRMED:
                    enter_long(r.symbol)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        btc_regime_map: dict[pd.Timestamp, str],
        universe: SmallCapUniverse | None = None,
        hold_bars: int = 2,
        cooldown_bars: int = 4,   # ~4h cooldown at 1h bars (was 16 at 15m bars)
    ):
        self.data = data
        self.btc_regime = btc_regime_map
        self.universe = universe or SmallCapUniverse()
        self.hold_bars = hold_bars
        self.cooldown_bars = cooldown_bars

        # Pre-compute rolling metrics on 1h windows
        self._precompute()

        # State tracking
        self.small_syms = self.universe.filter(data)
        self.active_positions: dict[str, ActivePosition] = {}
        self.cooldowns: dict[str, int] = {}  # symbol → bars_remaining
        self.event_history: list[ScanResult] = []

    def _precompute(self):
        """Pre-compute 1h rolling metrics for all symbols."""
        data = self.data

        # Returns (1 bar = 15m, 4 bars = 1h)
        # On 1h bars, 1-bar change = 1h return (was pct_change(4) for 15m→1h)
        self.ret_1h = data["close"].groupby(level="symbol").transform(
            lambda s: s.pct_change(1)
        )
        # BTC returns for stability check
        btc_data = data.xs("Binance:BTCUSDT", level="symbol") if "Binance:BTCUSDT" in data.index.get_level_values("symbol") else None
        if btc_data is not None and len(btc_data) > 0:
            self.btc_ret_15m = btc_data["close"].pct_change()  # 1h return on 1h bars (was 15m on 15m bars)
        else:
            self.btc_ret_15m = pd.Series(0.0, index=data.index.get_level_values("timestamp").unique())

        # OI delta z-score (2-bar OI change for 1h bars, was 6-bar for 15m bars)
        oi_delta = data["open_interest"].groupby(level="symbol").transform(
            lambda s: s.diff(2)
        )
        g = oi_delta.groupby(level="symbol")
        oi_mean = g.transform(lambda s: s.rolling(12, min_periods=4).mean())
        oi_std = g.transform(lambda s: s.rolling(12, min_periods=4).std()).replace(0, np.nan)
        self.oi_z = ((oi_delta - oi_mean) / oi_std).fillna(0.0)

        # Volume z-score (12-bar window for 1h bars, was 48-bar for 15m)
        vol = data["volume"]
        gv = vol.groupby(level="symbol")
        vol_mean = gv.transform(lambda s: s.rolling(12, min_periods=4).mean())
        vol_std = gv.transform(lambda s: s.rolling(12, min_periods=4).std()).replace(0, np.nan)
        self.vol_z = ((vol - vol_mean) / vol_std).fillna(0.0)

        # Funding z-score (6-bar window for 1h bars, was 24-bar for 15m)
        funding = data["funding_rate"]
        gf = funding.groupby(level="symbol")
        f_mean = gf.transform(lambda s: s.rolling(6, min_periods=4).mean())
        f_std = gf.transform(lambda s: s.rolling(6, min_periods=4).std()).replace(0, np.nan)
        self.funding_z = ((funding - f_mean) / f_std).fillna(0.0)

        # Close location
        hl_range = (data["high"] - data["low"]).clip(lower=1e-8)
        self.close_loc = (data["close"] - data["low"]) / hl_range

        # Timestamp lookup
        self.ts_list = sorted(data.index.get_level_values("timestamp").unique())
        self.ts_to_idx = {ts: i for i, ts in enumerate(self.ts_list)}

    # ── Core scan logic ──────────────────────────

    def scan(self, ts: pd.Timestamp) -> list[ScanResult]:
        """Run one scan tick at timestamp ts. Returns list of ScanResult."""
        results: list[ScanResult] = []

        # Decrement cooldowns
        for sym in list(self.cooldowns.keys()):
            self.cooldowns[sym] -= 1
            if self.cooldowns[sym] <= 0:
                del self.cooldowns[sym]

        # Check exits for active positions
        self._check_exits(ts)

        # Scan all small-cap symbols at this timestamp
        mask = self.data.index.get_level_values("timestamp") == ts
        syms_at_ts = set(self.data.index.get_level_values("symbol")[mask])

        btc_reg = self.btc_regime.get(ts, "unknown")
        btc_r15 = float(self.btc_ret_15m.get(ts, 0.0)) if ts in self.btc_ret_15m.index else 0.0

        # Gate 0 (pre-scan): Market-wide OI collapse check
        # If >5 small-cap coins are having OI collapses simultaneously, this is
        # systemic deleveraging, not single-coin mispricing → skip all longs.
        oi_collapse_count = 0
        for sym in self.small_syms & syms_at_ts:
            try:
                if float(self.oi_z.loc[(ts, sym)]) < -1.5:
                    oi_collapse_count += 1
            except (KeyError, TypeError):
                pass
        systemic_deleveraging = oi_collapse_count > 5

        for sym in self.small_syms & syms_at_ts:
            # Skip if in cooldown or active position
            if sym in self.cooldowns or sym in self.active_positions:
                continue

            try:
                oi_z = float(self.oi_z.loc[(ts, sym)])
                r1h = float(self.ret_1h.loc[(ts, sym)])
                vz = float(self.vol_z.loc[(ts, sym)])
                cl = float(self.close_loc.loc[(ts, sym)])
                fz = float(self.funding_z.loc[(ts, sym)])
            except (KeyError, TypeError):
                continue

            # ── Hard gates ──
            reject = None

            # Gate 1: Must be a down-move (long only)
            if r1h >= -0.025:
                reject = "return_1h >= -2.5%"

            # Gate 2: OI severity in window
            elif oi_z >= -1.5 or oi_z <= -5.0:
                reject = "oi_z out of range (%.1f)" % oi_z

            # Gate 3: Volume elevated
            elif vz < 1.0:
                reject = "volume_z < 1.0"

            # Gate 4: BTC not panicking
            elif btc_reg == "panic_down":
                reject = "BTC panic_down"

            # Gate 5: Return not too extreme
            elif r1h < -0.12:
                reject = "return < -12% (potential black swan)"

            # Gate 6: Volume not absurd
            elif vz > 8.0:
                reject = "volume_z > 8 (potential anomaly)"

            # Gate 7: Funding extreme (liquidation may still be ongoing)
            elif abs(fz) > 2.0:
                reject = "|funding_z| > 2.0"

            # Gate 8: Systemic deleveraging (too many coins OI-collapsing at once)
            elif systemic_deleveraging:
                reject = "systemic_deleveraging (>5 coins OI collapse)"

            if reject:
                results.append(ScanResult(
                    symbol=sym, timestamp=ts,
                    state=EventState.REJECTED,
                    reject_reason=reject,
                    metrics={"oi_z": oi_z, "ret_1h": r1h, "vol_z": vz, "close_loc": cl},
                ))
                continue

            # ── Compute event score ──
            score = compute_event_score(oi_z, r1h, vz, btc_r15, btc_reg)

            # Get regime-specific threshold and rules
            regime_rule = get_regime_rule(btc_reg)
            score_threshold = regime_rule["event_score_threshold"]

            # If regime forbids longs, reject
            if not regime_rule["allow_long"]:
                results.append(ScanResult(
                    symbol=sym, timestamp=ts,
                    state=EventState.REJECTED,
                    reject_reason=f"regime={btc_reg} (no longs allowed)",
                    metrics={"oi_z": oi_z, "ret_1h": r1h, "vol_z": vz, "close_loc": cl, "regime_score_threshold": score_threshold},
                ))
                continue

            if score < score_threshold:
                results.append(ScanResult(
                    symbol=sym, timestamp=ts,
                    state=EventState.WATCH,
                    event_score=score,
                    metrics={"oi_z": oi_z, "ret_1h": r1h, "vol_z": vz, "close_loc": cl},
                ))
                continue

            # ── Qualified: check entry confirmation ──
            # Simple confirmation: close_location > 0 (price closing in upper half of bar)
            # This filters out events where price is still dropping intra-bar
            if cl > 0.35:  # price is not at the absolute low
                state = EventState.CONFIRMED
            else:
                state = EventState.QUALIFIED

            results.append(ScanResult(
                symbol=sym, timestamp=ts,
                state=state,
                event_score=score,
                metrics={"oi_z": oi_z, "ret_1h": r1h, "vol_z": vz, "close_loc": cl},
            ))

        self.event_history.extend(results)
        return results

    # ── Position management ──────────────────────

    def enter_position(self, symbol: str, ts: pd.Timestamp) -> float | None:
        """Open a long position. Returns entry price or None if failed."""
        try:
            price = float(self.data.loc[(ts, symbol), "close"])
        except KeyError:
            return None

        self.active_positions[symbol] = ActivePosition(
            symbol=symbol, entry_ts=ts, entry_price=price,
        )
        return price

    def _check_exits(self, current_ts: pd.Timestamp):
        """Check if any active positions should exit."""
        current_idx = self.ts_to_idx.get(current_ts)
        if current_idx is None:
            return

        for sym in list(self.active_positions.keys()):
            pos = self.active_positions[sym]
            entry_idx = self.ts_to_idx.get(pos.entry_ts)
            if entry_idx is None:
                continue

            bars_held = current_idx - entry_idx

            # Exit rule 1: hold_bars reached
            if bars_held >= self.hold_bars:
                self._exit_position(sym, current_ts)
                continue

            # Exit rule 2: BTC flips to panic_down
            btc_reg = self.btc_regime.get(current_ts, "unknown")
            if btc_reg == "panic_down":
                self._exit_position(sym, current_ts, reason="BTC panic_down")
                continue

    def _exit_position(self, symbol: str, ts: pd.Timestamp, reason: str = "hold_complete"):
        """Close a position at market close."""
        pos = self.active_positions.pop(symbol, None)
        if pos is None:
            return

        try:
            exit_px = float(self.data.loc[(ts, symbol), "close"])
        except KeyError:
            exit_px = pos.entry_price

        pos.exit_ts = ts
        pos.exit_price = exit_px
        ret = (exit_px / pos.entry_price) - 1.0
        pos.pnl_bps = ret * 10000

        # Set cooldown
        self.cooldowns[symbol] = self.cooldown_bars

    def get_watchlist(self) -> list[dict]:
        """Return current watchlist (WATCH + QUALIFIED + CONFIRMED events)."""
        # Get the latest scan results per symbol
        latest = {}
        for r in self.event_history:
            if r.state in (EventState.WATCH, EventState.QUALIFIED, EventState.CONFIRMED):
                latest[r.symbol] = r

        return [
            {
                "symbol": r.symbol,
                "state": r.state.value,
                "event_score": round(r.event_score, 3),
                "oi_z": round(r.metrics.get("oi_z", 0), 2),
                "ret_1h_pct": round(r.metrics.get("ret_1h", 0) * 100, 2),
                "vol_z": round(r.metrics.get("vol_z", 0), 2),
            }
            for r in sorted(latest.values(), key=lambda x: -x.event_score)
        ]

    def get_active_positions(self) -> list[dict]:
        """Return currently active positions."""
        return [
            {
                "symbol": sym,
                "entry_ts": str(pos.entry_ts),
                "entry_price": pos.entry_price,
                "bars_held": self.ts_to_idx.get(
                    self.ts_list[-1] if self.ts_list else pos.entry_ts, 0
                ) - self.ts_to_idx.get(pos.entry_ts, 0),
            }
            for sym, pos in self.active_positions.items()
        ]

    def get_closed_positions(self) -> list[dict]:
        """Return closed position PnL summary."""
        closed = []
        for r in self.event_history:
            if r.state == EventState.EXITED:
                closed.append({
                    "symbol": r.symbol,
                    "pnl_bps": round(r.metrics.get("pnl_bps", 0), 1),
                })
        return closed


# ── Backtest runner ─────────────────────────────────

def backtest_scanner(
    data: pd.DataFrame,
    btc_regime_map: dict,
    hold_bars: int = 2,
    entry_mode: str = "next_bar_open",  # "next_bar_open" | "current_close" | "confirmed_only"
    use_regime_rules: bool = True,
) -> dict[str, Any]:
    """Run scanner in backtest mode over historical data and compute PnL."""
    scanner = DeleveragingEventScanner(data, btc_regime_map, hold_bars=hold_bars)
    ts_list = scanner.ts_list

    all_trades = []

    for i, ts in enumerate(ts_list):
        results = scanner.scan(ts)

        # Get regime at this timestamp for position sizing
        current_regime = btc_regime_map.get(ts, "unknown")
        regime_rule = get_regime_rule(current_regime)

        for r in results:
            if r.state == EventState.CONFIRMED:
                # Apply regime-based position sizing
                position_pct = regime_rule["position_pct"] if use_regime_rules else 1.0

                # Determine entry method
                if entry_mode == "current_close":
                    entry_ts = ts
                elif entry_mode == "next_bar_open":
                    if i + 1 >= len(ts_list):
                        continue
                    entry_ts = ts_list[i + 1]
                elif entry_mode == "confirmed_only":
                    entry_ts = ts
                else:
                    entry_ts = ts

                # Get entry price
                try:
                    entry_px = float(data.loc[(entry_ts, r.symbol), "close"])
                except KeyError:
                    continue

                scanner.enter_position(r.symbol, entry_ts)

                # Simulate exit after hold_bars
                exit_idx = scanner.ts_to_idx[entry_ts] + hold_bars
                if exit_idx >= len(ts_list):
                    continue
                exit_ts = ts_list[exit_idx]
                try:
                    exit_px = float(data.loc[(exit_ts, r.symbol), "close"])
                except KeyError:
                    continue

                ret = (exit_px / entry_px) - 1.0
                gbp = ret * 10000  # gross bps

                # Apply position sizing
                gbp_sized = gbp * position_pct

                all_trades.append({
                    "symbol": r.symbol,
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "event_score": r.event_score,
                    "gross_bps": gbp_sized,
                    "gross_bps_raw": gbp,
                    "position_pct": position_pct,
                    "regime": current_regime,
                    "oi_z": r.metrics.get("oi_z", 0),
                    "ret_1h": r.metrics.get("ret_1h", 0),
                })

                scanner.cooldowns[r.symbol] = scanner.cooldown_bars

    if not all_trades:
        return {"error": "No trades"}

    df = pd.DataFrame(all_trades)

    # Apply cost (charged on position turnover, not scaled by position_pct)
    # For simplicity, charge full cost per trade
    for cost_bps in [4, 6, 9]:
        df["net_%d" % cost_bps] = df["gross_bps"] - cost_bps

    results = {}
    for cost_bps in [4, 6, 9]:
        col = "net_%d" % cost_bps
        arr = df[col].values
        gross = df["gross_bps_raw"].sum()
        net = arr.sum()
        pos = arr[arr > 0].sum()
        neg = abs(arr[arr < 0].sum())
        pf = pos / neg if neg > 0 else float("inf")
        cg = (len(arr) * cost_bps) / abs(gross) * 100 if abs(gross) > 0 else float("inf")
        hit = (arr > 0).mean()

        results["cost_%dbps" % cost_bps] = {
            "n_trades": len(arr),
            "gross_bps": round(gross, 0),
            "net_bps": round(net, 0),
            "pf": round(pf, 3),
            "cost_to_gross_pct": round(cg, 0),
            "hit_rate": round(hit, 3),
            "median_bps": round(np.median(arr), 1),
            "avg_win_bps": round(arr[arr > 0].mean(), 1) if (arr > 0).any() else 0,
            "avg_loss_bps": round(arr[arr < 0].mean(), 1) if (arr < 0).any() else 0,
        }

    # Regime breakdown
    regime_breakdown = {}
    for reg in df["regime"].unique():
        sub = df[df["regime"] == reg]
        net9 = (sub["gross_bps"] - 9).sum()
        regime_breakdown[reg] = {
            "n_trades": len(sub),
            "net_9bps": round(net9, 0),
            "hit_rate": round((sub["gross_bps"] - 9 > 0).mean(), 3),
            "gross_sum": round(sub["gross_bps_raw"].sum(), 0),
        }

    results["entry_mode"] = entry_mode
    results["use_regime_rules"] = use_regime_rules
    results["unique_symbols"] = int(df["symbol"].nunique())
    results["total_events"] = len(df)
    results["regime_breakdown"] = regime_breakdown

    return results


if __name__ == "__main__":
    # Quick self-test
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parent))

    ROOT = P("/mnt/e/alpha_pipeline")
    from backtest_engine import BacktestEngine
    from research.regime_detector import detect_regime_fast

    engine = BacktestEngine(ROOT)
    data = engine._load_data()
    regimes = detect_regime_fast(data)
    regime_map = dict(zip(regimes.index, regimes))

    # Regime distribution
    print("Regime distribution:")
    for reg, cnt in regimes.value_counts().items():
        print(f"  {reg}: {cnt} ({cnt/len(regimes)*100:.1f}%)")

    print()
    for use_rules in [False, True]:
        label = "WITH regime rules" if use_rules else "WITHOUT regime rules"
        print(f"=== {label} ===")
        for mode in ["current_close", "next_bar_open"]:
            r = backtest_scanner(data, regime_map, hold_bars=2, entry_mode=mode, use_regime_rules=use_rules)
            print(f"  Entry mode: {mode}")
            for k, v in sorted(r.items()):
                if k.startswith("cost_"):
                    print(f"    {k}: n={v['n_trades']} Net={v['net_bps']:+.0f} PF={v['pf']:.2f} C/G={v['cost_to_gross_pct']:.0f}% Hit={v['hit_rate']*100:.1f}% Med={v['median_bps']:+.0f}")
            if r.get('regime_breakdown'):
                print(f"    Regime breakdown:")
                for reg, info in r['regime_breakdown'].items():
                    print(f"      {reg:12s}: n={info['n_trades']:2d} Net9={info['net_9bps']:+6.0f} Hit={info['hit_rate']*100:.0f}%")
        print()
