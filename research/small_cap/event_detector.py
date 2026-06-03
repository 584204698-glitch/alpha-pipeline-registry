"""Small-cap event detector.

Unlike large-cap continuous cross-sectional ranking, small-cap signals are
event-driven: no event = no signal = no trade.

Three event types implemented:
1. OI_Stack_Failure — OI piles up but price reverses (trapped traders)
2. Deleveraging_Reversal — OI collapse + sharp move (forced liquidation bounce)
3. BTC_Relative_Weakness — small coin underperforms BTC in trend_down

Each event returns a Signal or None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EventSignal:
    """A detected event with direction and metadata."""
    timestamp: pd.Timestamp
    symbol: str
    event_type: str  # OI_Stack_Failure / Deleveraging_Reversal / BTC_Relative_Weakness
    direction: str   # long / short
    strength: float  # 0-1, higher = more extreme
    metrics: dict[str, float] = field(default_factory=dict)


class SmallCapEventDetector:
    """Detect trade-worthy events for mid/small-cap perpetuals."""

    def __init__(
        self,
        oi_z_threshold: float = 1.5,
        vol_z_threshold: float = 1.0,
        ret_z_threshold: float = 1.5,
        volume_rank_min: int = 20,   # minimum volume rank to consider
        volume_rank_max: int = 100,  # maximum volume rank (exclude nano-caps)
    ):
        self.oi_z_threshold = oi_z_threshold
        self.vol_z_threshold = vol_z_threshold
        self.ret_z_threshold = ret_z_threshold
        self.volume_rank_min = volume_rank_min
        self.volume_rank_max = volume_rank_max

    def detect_all(
        self,
        data: pd.DataFrame,
        btc_regime: pd.Series | None = None,
    ) -> list[EventSignal]:
        """Run all three event detectors on the full dataset."""
        events: list[EventSignal] = []

        # Pre-compute per-symbol metrics
        data = data.sort_index()
        eps = 1e-8

        # Returns
        ret = data["close"].groupby(level="symbol").transform(lambda s: s.pct_change())

        # Log returns for zscore
        log_ret = np.log(data["close"]).groupby(level="symbol").transform(lambda s: s.diff())

        # Volume zscore
        vol_mean = data["volume"].groupby(level="symbol").transform(
            lambda s: s.rolling(48, min_periods=8).mean()
        )
        vol_std = data["volume"].groupby(level="symbol").transform(
            lambda s: s.rolling(48, min_periods=8).std()
        ).replace(0, np.nan)
        vol_z = ((data["volume"] - vol_mean) / vol_std).fillna(0.0)

        # OI delta zscore
        oi_delta = data["open_interest"].groupby(level="symbol").transform(
            lambda s: s.diff(6)
        )
        oi_mean = oi_delta.groupby(level="symbol").transform(
            lambda s: s.rolling(48, min_periods=8).mean()
        )
        oi_std = oi_delta.groupby(level="symbol").transform(
            lambda s: s.rolling(48, min_periods=8).std()
        ).replace(0, np.nan)
        oi_z = ((oi_delta - oi_mean) / oi_std).fillna(0.0)

        # Return zscore
        ret_mean = ret.groupby(level="symbol").transform(
            lambda s: s.rolling(48, min_periods=8).mean()
        )
        ret_std = ret.groupby(level="symbol").transform(
            lambda s: s.rolling(48, min_periods=8).std()
        ).replace(0, np.nan)
        ret_z = ((ret - ret_mean) / ret_std).fillna(0.0)

        # Close location: where close sits in high-low range
        hl_range = (data["high"] - data["low"]).clip(lower=eps)
        close_loc = (data["close"] - data["low"]) / hl_range  # 0=at low, 1=at high

        # Volume rank for universe filtering
        avg_vol = data["volume"].groupby(level="symbol").mean()
        vol_rank = avg_vol.rank(ascending=False)

        # Determine small-cap symbols
        small_cap_syms = set(
            vol_rank[
                (vol_rank >= self.volume_rank_min) & (vol_rank <= self.volume_rank_max)
            ].index
        )

        # Detect OI Stack Failure events
        oi_events = self._detect_oi_stack_failure(
            data, oi_z, vol_z, ret_z, close_loc, small_cap_syms, btc_regime
        )
        events.extend(oi_events)

        # Detect Deleveraging Reversal events
        delever_events = self._detect_deleveraging_reversal(
            data, oi_z, vol_z, ret_z, small_cap_syms, btc_regime
        )
        events.extend(delever_events)

        # Detect BTC Relative Weakness events
        btc_rel_events = self._detect_btc_relative_weakness(
            data, ret, vol_z, small_cap_syms, btc_regime
        )
        events.extend(btc_rel_events)

        return events

    def _detect_oi_stack_failure(
        self,
        data: pd.DataFrame,
        oi_z: pd.Series,
        vol_z: pd.Series,
        ret_z: pd.Series,
        close_loc: pd.Series,
        small_cap_syms: set,
        btc_regime: pd.Series | None,
    ) -> list[EventSignal]:
        """OI piles up but price fails → trapped traders."""
        events = []
        ts_values = data.index.get_level_values("timestamp").unique()

        for ts in ts_values:
            mask = data.index.get_level_values("timestamp") == ts
            syms_at_ts = data.index.get_level_values("symbol")[mask]

            for sym in syms_at_ts:
                if sym not in small_cap_syms:
                    continue

                try:
                    oi_z_val = float(oi_z.loc[(ts, sym)])
                    vol_z_val = float(vol_z.loc[(ts, sym)])
                    ret_z_val = float(ret_z.loc[(ts, sym)])
                    cl = float(close_loc.loc[(ts, sym)])
                except (KeyError, TypeError):
                    continue

                # Gate: OI must be surging
                if oi_z_val < self.oi_z_threshold:
                    continue
                # Gate: volume must confirm
                if vol_z_val < self.vol_z_threshold:
                    continue

                # Direction: up-failure (close near low) → short
                #            down-failure (close near high) → long
                if cl < 0.30 and ret_z_val < 0:  # price dropped to low → up-failure
                    # BTC panic check
                    if btc_regime is not None and ts in btc_regime.index:
                        if btc_regime.loc[ts] == "panic_down":
                            continue
                    events.append(EventSignal(
                        timestamp=ts, symbol=sym,
                        event_type="OI_Stack_Failure",
                        direction="short",
                        strength=min(oi_z_val / 3.0, 1.0),
                        metrics={"oi_z": oi_z_val, "vol_z": vol_z_val, "close_loc": cl},
                    ))
                elif cl > 0.70 and ret_z_val > 0:  # price rose to high → down-failure
                    events.append(EventSignal(
                        timestamp=ts, symbol=sym,
                        event_type="OI_Stack_Failure",
                        direction="long",
                        strength=min(oi_z_val / 3.0, 1.0),
                        metrics={"oi_z": oi_z_val, "vol_z": vol_z_val, "close_loc": cl},
                    ))

        return events

    def _detect_deleveraging_reversal(
        self,
        data: pd.DataFrame,
        oi_z: pd.Series,
        vol_z: pd.Series,
        ret_z: pd.Series,
        small_cap_syms: set,
        btc_regime: pd.Series | None,
    ) -> list[EventSignal]:
        """OI collapse + sharp move → forced liquidation potential bounce."""
        events = []
        ts_values = data.index.get_level_values("timestamp").unique()

        for ts in ts_values:
            mask = data.index.get_level_values("timestamp") == ts
            syms_at_ts = data.index.get_level_values("symbol")[mask]

            for sym in syms_at_ts:
                if sym not in small_cap_syms:
                    continue

                try:
                    oi_z_val = float(oi_z.loc[(ts, sym)])
                    vol_z_val = float(vol_z.loc[(ts, sym)])
                    ret_z_val = float(ret_z.loc[(ts, sym)])
                except (KeyError, TypeError):
                    continue

                # OI dropping hard
                if oi_z_val > -self.oi_z_threshold:
                    continue
                # Volume surging
                if vol_z_val < self.vol_z_threshold:
                    continue
                # Sharp move
                if abs(ret_z_val) < self.ret_z_threshold:
                    continue

                # Direction: big drop + OI collapse → long (deleveraging reversal)
                #            big rally + OI collapse → short
                if ret_z_val < -self.ret_z_threshold:
                    # BTC panic check — don't catch falling knife
                    if btc_regime is not None and ts in btc_regime.index:
                        reg = btc_regime.loc[ts]
                        if reg == "panic_down":
                            continue
                    events.append(EventSignal(
                        timestamp=ts, symbol=sym,
                        event_type="Deleveraging_Reversal",
                        direction="long",
                        strength=min(abs(ret_z_val) / 5.0, 1.0),
                        metrics={"oi_z": oi_z_val, "vol_z": vol_z_val, "ret_z": ret_z_val},
                    ))
                elif ret_z_val > self.ret_z_threshold:
                    events.append(EventSignal(
                        timestamp=ts, symbol=sym,
                        event_type="Deleveraging_Reversal",
                        direction="short",
                        strength=min(ret_z_val / 5.0, 1.0),
                        metrics={"oi_z": oi_z_val, "vol_z": vol_z_val, "ret_z": ret_z_val},
                    ))

        return events

    def _detect_btc_relative_weakness(
        self,
        data: pd.DataFrame,
        ret: pd.Series,
        vol_z: pd.Series,
        small_cap_syms: set,
        btc_regime: pd.Series | None,
    ) -> list[EventSignal]:
        """BTC trending down + small coin underperforming → continuation short."""
        if btc_regime is None:
            return []

        events = []
        ts_values = data.index.get_level_values("timestamp").unique()

        # Compute cross-sectional return rank per timestamp
        ret_rank = ret.groupby(level="timestamp").transform(
            lambda x: x.rank(pct=True) if len(x) > 5 else pd.Series(0.5, index=x.index)
        )

        for ts in ts_values:
            # Only in BTC trend_down
            if ts not in btc_regime.index or btc_regime.loc[ts] != "trend_down":
                continue

            mask = data.index.get_level_values("timestamp") == ts
            syms_at_ts = data.index.get_level_values("symbol")[mask]

            for sym in syms_at_ts:
                if sym not in small_cap_syms:
                    continue

                try:
                    r_rank = float(ret_rank.loc[(ts, sym)])
                    vz = float(vol_z.loc[(ts, sym)])
                except (KeyError, TypeError):
                    continue

                # Bottom 10% return rank = severe underperformance
                if r_rank > 0.10:
                    continue
                # Some volume confirmation
                if vz < 0:
                    continue

                events.append(EventSignal(
                    timestamp=ts, symbol=sym,
                    event_type="BTC_Relative_Weakness",
                    direction="short",
                    strength=1.0 - r_rank,  # more extreme = stronger
                    metrics={"ret_rank": r_rank, "vol_z": vz},
                ))

        return events


def event_summary(events: list[EventSignal]) -> dict[str, Any]:
    """Summarize detected events by type and direction."""
    summary: dict[str, dict[str, int]] = {}
    for e in events:
        key = e.event_type
        if key not in summary:
            summary[key] = {"long": 0, "short": 0, "total": 0}
        summary[key][e.direction] += 1
        summary[key]["total"] += 1
    return {
        "n_total_events": len(events),
        "by_type": summary,
        "unique_symbols": len(set(e.symbol for e in events)),
        "unique_timestamps": len(set(e.timestamp for e in events)),
    }
