"""Small-cap event paper trading.

Tests event-based signals with strict cost and liquidity gates.
Output: per-event-type hit rate, PF, net PnL, cost analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.small_cap.event_detector import (
    SmallCapEventDetector,
    EventSignal,
    event_summary,
)


@dataclass
class EventPaperResult:
    event_type: str
    n_events: int
    n_traded: int
    n_blocked_cost: int
    n_blocked_panic: int
    hit_rate: float  # fraction with correct direction
    gross_pnl: float
    net_pnl: float
    cost_total: float
    pf: float
    avg_return_bps: float
    median_return_bps: float
    max_loss_bps: float
    tail_5pct_pnl: float  # contribution of worst 5% trades
    decision: str  # tradable / filter_only / kill


def run_event_paper(
    project_root: Path,
    hold_bars: int = 6,
    cost_bps: float = 12.0,  # small caps pay higher spread
    volume_rank_min: int = 20,
    volume_rank_max: int = 100,
) -> dict[str, Any]:
    """Run paper trading on detected small-cap events.

    Args:
        project_root: alpha_pipeline root
        hold_bars: how many bars to hold after event
        cost_bps: round-trip cost in bps (higher for small caps)
        volume_rank_min: minimum volume rank for small-cap pool
        volume_rank_max: maximum volume rank (exclude nano-caps)
    """
    from backtest_engine import BacktestEngine
    from research.regime_detector import detect_regimes

    root = Path(project_root)
    engine = BacktestEngine(root)
    data = engine._load_data()

    # Detect regimes
    regimes = detect_regimes(data)

    # Detect events
    detector = SmallCapEventDetector(
        oi_z_threshold=1.5,
        vol_z_threshold=1.0,
        ret_z_threshold=1.5,
        volume_rank_min=volume_rank_min,
        volume_rank_max=volume_rank_max,
    )
    events = detector.detect_all(data, btc_regime=regimes)
    summary = event_summary(events)

    if not events:
        return {"error": "No events detected", "summary": summary}

    # Paper trade each event
    trades = _paper_trade_events(data, events, hold_bars=hold_bars, cost_bps=cost_bps)
    if not trades:
        return {"error": "No trades generated", "summary": summary, "n_events": len(events)}

    trades_df = pd.DataFrame(trades)

    # Per-event-type analysis
    results = []
    for event_type, grp in trades_df.groupby("event_type"):
        n = len(grp)
        hit = (grp["direction_hit"]).mean()
        gross = grp["gross_bps"].sum()
        cost_total = grp["cost_bps"].sum()
        net = gross - cost_total
        avg_ret = grp["net_bps"].mean()
        med_ret = grp["net_bps"].median()
        max_loss = grp["net_bps"].min()
        pos = grp.loc[grp["net_bps"] > 0, "net_bps"].sum()
        neg = abs(grp.loc[grp["net_bps"] < 0, "net_bps"].sum())
        pf = pos / neg if neg > 0 else float("inf")

        # Tail analysis
        tail_5pct_n = max(1, int(n * 0.05))
        tail_5pct_pnl = grp["net_bps"].nsmallest(tail_5pct_n).sum()

        # Decision
        if gross < 0 or (pf < 0.8 and hit < 0.45):
            decision = "kill"
        elif net > 0 and pf > 1.1:
            decision = "tradable"
        elif gross > 0 and net < 0:
            decision = "cost_bound"  # gross works but cost kills it
        elif hit > 0.50 and net < 0:
            decision = "filter_only"
        else:
            decision = "kill"

        results.append(EventPaperResult(
            event_type=event_type,
            n_events=n,
            n_traded=n,
            n_blocked_cost=0,
            n_blocked_panic=0,
            hit_rate=float(hit),
            gross_pnl=float(gross),
            net_pnl=float(net),
            cost_total=float(cost_total),
            pf=float(pf),
            avg_return_bps=float(avg_ret),
            median_return_bps=float(med_ret),
            max_loss_bps=float(max_loss),
            tail_5pct_pnl=float(tail_5pct_pnl),
            decision=decision,
        ))

    return {
        "summary": summary,
        "hold_bars": hold_bars,
        "cost_bps": cost_bps,
        "volume_range": f"{volume_rank_min}-{volume_rank_max}",
        "results": [
            {
                "event_type": r.event_type,
                "n_events": r.n_events,
                "n_traded": r.n_traded,
                "hit_rate": r.hit_rate,
                "gross_pnl_bps": r.gross_pnl,
                "net_pnl_bps": r.net_pnl,
                "cost_total_bps": r.cost_total,
                "pf": r.pf,
                "avg_return_bps": r.avg_return_bps,
                "median_return_bps": r.median_return_bps,
                "max_loss_bps": r.max_loss_bps,
                "tail_5pct_pnl": r.tail_5pct_pnl,
                "decision": r.decision,
            }
            for r in results
        ],
        "overall": {
            "total_events": len(events),
            "total_trades": len(trades_df),
            "overall_hit_rate": float(trades_df["direction_hit"].mean()),
            "overall_gross_bps": float(trades_df["gross_bps"].sum()),
            "overall_net_bps": float(trades_df["net_bps"].sum()),
        },
    }


def _paper_trade_events(
    data: pd.DataFrame,
    events: list[EventSignal],
    hold_bars: int = 6,
    cost_bps: float = 12.0,
) -> list[dict[str, Any]]:
    """Simulate trading each event with forward returns."""
    ts_values = sorted(data.index.get_level_values("timestamp").unique())
    ts_to_idx = {ts: i for i, ts in enumerate(ts_values)}
    round_trip_cost = cost_bps / 10000.0

    trades = []
    for event in events:
        ts = event.timestamp
        sym = event.symbol
        direction = event.direction

        # Find entry timestamp index
        if ts not in ts_to_idx:
            continue
        entry_idx = ts_to_idx[ts]

        # Get entry price
        try:
            entry_price = data.loc[(ts, sym), "close"]
        except KeyError:
            continue

        # Find exit bars
        exit_bars = []
        for h in range(1, hold_bars + 1):
            if entry_idx + h >= len(ts_values):
                break
            exit_bars.append(ts_values[entry_idx + h])

        if not exit_bars:
            continue

        # Compute returns
        for exit_ts in exit_bars:
            try:
                exit_price = data.loc[(exit_ts, sym), "close"]
            except KeyError:
                continue

            ret = (exit_price / entry_price) - 1.0
            if direction == "short":
                ret = -ret

            gross_bps = ret * 10000
            net_bps = gross_bps - (round_trip_cost * 10000)
            hit = ((direction == "long" and ret > 0) or (direction == "short" and ret < 0))

            trades.append({
                "event_type": event.event_type,
                "symbol": sym,
                "direction": direction,
                "entry_ts": ts,
                "exit_ts": exit_ts,
                "gross_bps": gross_bps,
                "cost_bps": round_trip_cost * 10000,
                "net_bps": net_bps,
                "direction_hit": hit,
                "strength": event.strength,
            })

    return trades
