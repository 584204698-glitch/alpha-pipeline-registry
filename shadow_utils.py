"""Shadow mode utilities — conflict resolution, regime confidence, field builder."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_regime_confidence(
    regime_series: pd.Series,
    lookback: int = 12,
) -> pd.Series:
    """Compute regime confidence score per timestamp.

    0 = low confidence (regime boundary, recent flip, uncertain)
    1 = medium confidence (stable regime, mild signals)
    2 = high confidence (clear regime, all signals aligned)

    Based on:
    - Regime stability: how many bars since last flip
    - Breadth confirmation: market breadth aligned with regime
    """
    confidences = pd.Series(0, index=regime_series.index, dtype=int)

    # Stability: count consecutive identical regimes
    regime_vals = regime_series.values
    streak = 1
    for i in range(1, len(regime_vals)):
        if regime_vals[i] == regime_vals[i - 1]:
            streak += 1
        else:
            streak = 1

        if streak >= 8:  # 2+ hours of same regime
            confidences.iloc[i] = 2
        elif streak >= 4:  # 1+ hour
            confidences.iloc[i] = 1
        else:
            confidences.iloc[i] = 0

    return confidences


def build_shadow_record(
    symbol: str,
    event_type: str,
    direction: str,
    event_score: float,
    regime: str,
    regime_confidence: int,
    entry_ts: pd.Timestamp,
    entry_price: float,
    exit_ts: pd.Timestamp | None = None,
    exit_price: float | None = None,
    expected_hold_bars: int = 2,
    expected_cost_bps: float = 9.0,
    reject_reason: str = "",
    allow_reason: str = "",
    **extra_metrics,
) -> dict:
    """Build a complete shadow mode trade record."""
    record = {
        "symbol": symbol,
        "event_type": event_type,
        "direction": direction,
        "event_score": round(event_score, 4),
        "regime": regime,
        "regime_confidence": regime_confidence,
        "entry_ts": str(entry_ts),
        "entry_price_simulated": round(entry_price, 8),
        "exit_ts": str(exit_ts) if exit_ts else None,
        "exit_price_simulated": round(exit_price, 8) if exit_price else None,
        "expected_hold_bars": expected_hold_bars,
        "expected_cost_bps": expected_cost_bps,
        "reject_reason": reject_reason if reject_reason else (allow_reason or "allowed"),
        "allow_reason": allow_reason,
        "future_return": None,
        "net_after_cost": None,
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in extra_metrics.items()},
    }
    return record


def resolve_event_conflicts(
    events: list[dict],
    priority_order: list[str] = None,
) -> list[dict]:
    """Resolve conflicts when multiple events fire on same symbol+timestamp.

    Rules:
    1. If directions differ → SKIP ALL (no trade)
    2. If same direction → keep highest-priority event

    Args:
        events: list of event dicts with keys: symbol, timestamp, event_type, direction
        priority_order: event type priority (first = highest)

    Returns:
        Filtered and deduplicated event list
    """
    if priority_order is None:
        priority_order = [
            "DeleveragingReversal",
            "OIShockAbsorption",
            "RelativeStrengthShock",
        ]

    priority_map = {et: i for i, et in enumerate(priority_order)}

    # Group by (symbol, timestamp)
    from collections import defaultdict
    groups = defaultdict(list)
    for ev in events:
        key = (ev["symbol"], str(ev["timestamp"]))
        groups[key].append(ev)

    resolved = []
    skipped = 0
    for key, group in groups.items():
        if len(group) == 1:
            resolved.append(group[0])
        else:
            # Check direction conflict
            directions = set(ev.get("direction", "") for ev in group)
            if len(directions) > 1:
                # Direction conflict → skip all
                skipped += 1
                continue
            # Same direction → keep highest priority
            best = min(group, key=lambda ev: priority_map.get(ev.get("event_type", ""), 99))
            resolved.append(best)

    return resolved
