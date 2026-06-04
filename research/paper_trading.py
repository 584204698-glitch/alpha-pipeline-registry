"""Paper Trading Simulator — the death filter between lab PASS and live trading.

Simulates factor signal → bucket selection → cost-deducted forward returns
over multiple configurations.

Core output per factor × config:
- Gross PnL, Net PnL
- Cost/Gross ratio
- Turnover
- PF (profit factor)
- Long/Short PnL separately
- Regime attribution
- Tail dependence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ExecutionMode(Enum):
    FULL_REBALANCE = "full_rebalance"          # original: cross-sectional sort every bar
    DIRECTION_TRIGGER = "direction_trigger"    # only trade on signal sign flip
    THRESHOLD_ENTRY = "threshold_entry"        # only enter when |z-score| > threshold
    POSITION_SMOOTHING = "position_smoothing"  # gradual adjustment toward target weight

from backtest_engine import BacktestEngine
from factors import load_factor_from_path
from pipeline_utils import safe_float


@dataclass
class PaperConfig:
    """Configuration for a single paper trading run."""
    top_frac: float = 0.10       # fraction of top signals to trade long
    bottom_frac: float = 0.10    # fraction of bottom signals to trade short
    hold_bars: int = 1           # how many bars to hold
    cost_bps: float = 9.0        # cost per side in bps
    slippage_bps: float = 3.0    # additional slippage
    spread_bps: float = 1.0      # bid-ask spread
    min_symbols: int = 50        # minimum symbols in cross-section to trade
    # ── new execution control ──
    execution_mode: str = "full_rebalance"   # full_rebalance | direction_trigger | threshold_entry | position_smoothing
    signal_threshold: float = 1.5            # z-score threshold for threshold_entry mode
    exit_threshold: float = 0.5              # exit when |z| < exit_threshold (threshold_entry / direction_trigger)
    smoothing_rate: float = 0.33             # fraction toward target per bar (position_smoothing)

    @property
    def round_trip_cost_bps(self) -> float:
        """Total round-trip cost: 2 × (fee + slippage + spread)."""
        return 2.0 * (self.cost_bps + self.slippage_bps + self.spread_bps) / 10000.0

    @property
    def per_side_cost(self) -> float:
        return (self.cost_bps + self.slippage_bps + self.spread_bps) / 10000.0


@dataclass
class PaperResult:
    factor_name: str
    config: PaperConfig
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    cost_total: float = 0.0
    cost_gross_ratio: float = 0.0
    turnover: float = 0.0
    pf: float = 0.0
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    n_trades: int = 0
    n_bars_traded: int = 0
    regime_pnl: dict[str, float] = field(default_factory=dict)
    symbol_pnl: dict[str, float] = field(default_factory=dict)
    tail_contribution: float = 0.0  # top 5% bars contribution to total PnL
    classification: str = "unknown"  # trade_signal / filter_only / kill


class PaperTradingSimulator:
    """Simulate factor trading with realistic costs and constraints."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.engine = BacktestEngine(project_root)

    def run(
        self,
        factor_path: Path,
        configs: list[PaperConfig] | None = None,
        regime_labels: pd.Series | None = None,
    ) -> list[PaperResult]:
        """Run paper trading for a factor across multiple configs.

        Args:
            factor_path: Path to the factor .py file
            configs: List of configs to test. Default: 5/10/20% × 1/2/3 bars
            regime_labels: Optional Series with regime labels per timestamp

        Returns:
            List of PaperResult, one per config
        """
        if configs is None:
            configs = self._default_configs()

        data = self.engine._load_data()
        loaded = load_factor_from_path(factor_path)
        signal = loaded.factor.compute(data).reindex(data.index).fillna(0.0)

        results = []
        for cfg in configs:
            result = self._simulate_single(signal, data, cfg, regime_labels)
            result.factor_name = loaded.factor.factor_name
            results.append(result)
        return results

    @staticmethod
    def _default_configs() -> list[PaperConfig]:
        configs = []
        # Original full-rebalance configs (baseline)
        for frac in [0.05, 0.10, 0.20]:
            for hold in [1, 2, 3]:
                configs.append(PaperConfig(top_frac=frac, bottom_frac=frac, hold_bars=hold,
                                           execution_mode="full_rebalance"))
        # Direction-triggered configs (hold longer, different hold_bars)
        for frac in [0.05, 0.10, 0.20]:
            for hold in [1, 2, 4, 6]:
                configs.append(PaperConfig(top_frac=frac, bottom_frac=frac, hold_bars=hold,
                                           execution_mode="direction_trigger"))
        # Threshold entry configs
        for frac in [0.05, 0.10, 0.20]:
            for threshold in [1.0, 1.5, 2.0]:
                configs.append(PaperConfig(top_frac=frac, bottom_frac=frac, hold_bars=3,
                                           execution_mode="threshold_entry",
                                           signal_threshold=threshold, exit_threshold=threshold * 0.33))
        # Position smoothing configs
        for frac in [0.05, 0.10, 0.20]:
            for rate in [0.25, 0.50, 0.75]:
                configs.append(PaperConfig(top_frac=frac, bottom_frac=frac, hold_bars=1,
                                           execution_mode="position_smoothing",
                                           smoothing_rate=rate))
        return configs

    def _simulate_single(
        self,
        signal: pd.Series,
        data: pd.DataFrame,
        cfg: PaperConfig,
        regime_labels: pd.Series | None = None,
    ) -> PaperResult:
        """Dispatch to the appropriate simulation mode."""
        mode = cfg.execution_mode
        if mode == "direction_trigger":
            return self._simulate_direction_trigger(signal, data, cfg, regime_labels)
        elif mode == "threshold_entry":
            return self._simulate_threshold_entry(signal, data, cfg, regime_labels)
        elif mode == "position_smoothing":
            return self._simulate_position_smoothing(signal, data, cfg, regime_labels)
        else:
            return self._simulate_full_rebalance(signal, data, cfg, regime_labels)

    def _simulate_full_rebalance(
        self,
        signal: pd.Series,
        data: pd.DataFrame,
        cfg: PaperConfig,
        regime_labels: pd.Series | None = None,
    ) -> PaperResult:
        """Original: cross-sectional sort every bar, full position turnover."""
        ts_values = signal.index.get_level_values("timestamp").unique().sort_values()
        symbols = signal.index.get_level_values("symbol").unique()

        all_trades: list[dict] = []

        for i, ts in enumerate(ts_values):
            mask = signal.index.get_level_values("timestamp") == ts
            sig_t = signal.loc[mask].dropna()
            n_sym = len(sig_t)

            if n_sym < cfg.min_symbols:
                continue

            # Determine long/short buckets
            n_long = max(1, int(n_sym * cfg.top_frac))
            n_short = max(1, int(n_sym * cfg.bottom_frac))
            sorted_idx = sig_t.argsort()
            long_idx = sorted_idx[-n_long:]
            short_idx = sorted_idx[:n_short]

            # Find forward timestamps
            future_positions = []
            for h in range(1, cfg.hold_bars + 1):
                if i + h >= len(ts_values):
                    break
                future_ts = ts_values[i + h]
                future_positions.append(future_ts)

            if not future_positions:
                continue

            # Process longs
            sym_level = sig_t.index.get_level_values("symbol")
            for li in long_idx:
                sym = sym_level[li]
                try:
                    entry_price = data.loc[(ts, sym), "close"]
                except KeyError:
                    continue
                for ft in future_positions:
                    try:
                        exit_price = data.loc[(ft, sym), "close"]
                    except KeyError:
                        continue
                    ret = (exit_price / entry_price) - 1.0
                    gross = ret
                    cost = cfg.per_side_cost  # entry cost (exit will be added)
                    all_trades.append({
                        "timestamp": ts,
                        "symbol": sym,
                        "direction": "long",
                        "gross": gross,
                        "cost": cost,
                        "hold_bars": h,
                    })
            # Process shorts
            for si in short_idx:
                sym = sym_level[si]
                try:
                    entry_price = data.loc[(ts, sym), "close"]
                except KeyError:
                    continue
                for ft in future_positions:
                    try:
                        exit_price = data.loc[(ft, sym), "close"]
                    except KeyError:
                        continue
                    ret = -((exit_price / entry_price) - 1.0)  # short: invert
                    gross = ret
                    cost = cfg.per_side_cost
                    all_trades.append({
                        "timestamp": ts,
                        "symbol": sym,
                        "direction": "short",
                        "gross": gross,
                        "cost": cost,
                        "hold_bars": h,
                    })

        if not all_trades:
            return PaperResult(factor_name="unknown", config=cfg, classification="kill")

        trades_df = pd.DataFrame(all_trades)
        trades_df["net"] = trades_df["gross"] - trades_df["cost"] * 2  # entry + exit cost
        return self._compute_result_metrics(trades_df, cfg, regime_labels)

    # ── Direction-Triggered Rebalancing ──────────────────────────────

    def _simulate_direction_trigger(
        self,
        signal: pd.Series,
        data: pd.DataFrame,
        cfg: PaperConfig,
        regime_labels: pd.Series | None = None,
    ) -> PaperResult:
        """Only trade when signal sign flips for a symbol.

        Positions are held until sign changes or symbol exits the bucket.
        Direction is determined by sign(signal); strength only sizes the position.
        """
        ts_values = signal.index.get_level_values("timestamp").unique().sort_values()

        trades: list[dict] = []
        # positions: symbol -> {'direction': 'long'|'short', 'entry_ts': ts, 'entry_bar': int}
        positions: dict[str, dict] = {}

        for i, ts in enumerate(ts_values):
            mask = signal.index.get_level_values("timestamp") == ts
            sig_t = signal.loc[mask].dropna()
            n_sym = len(sig_t)
            if n_sym < cfg.min_symbols:
                continue

            # Current bar's long/short candidates (by sign, not just rank)
            n_long = max(1, int(n_sym * cfg.top_frac))
            n_short = max(1, int(n_sym * cfg.bottom_frac))
            sorted_idx = sig_t.argsort()
            long_syms = set(sig_t.index.get_level_values("symbol")[sorted_idx[-n_long:]])
            short_syms = set(sig_t.index.get_level_values("symbol")[sorted_idx[:n_short]])

            # Close positions where direction changed or symbol left the bucket
            to_close = []
            for sym, pos in list(positions.items()):
                # Honour hold_bars — can't close too early
                if i - pos["entry_bar"] < cfg.hold_bars:
                    continue

                new_dir = None
                if sym in long_syms:
                    new_dir = "long"
                elif sym in short_syms:
                    new_dir = "short"

                if new_dir is None:
                    # Symbol left the bucket entirely → close
                    to_close.append(sym)
                elif new_dir != pos["direction"]:
                    # Direction flipped → close
                    to_close.append(sym)
                # else: same direction → HOLD (no trade)

            for sym in to_close:
                pos = positions.pop(sym)
                try:
                    entry_price = data.loc[(pos["entry_ts"], sym), "close"]
                    exit_price = data.loc[(ts, sym), "close"]
                except KeyError:
                    continue
                ret = (exit_price / entry_price) - 1.0
                if pos["direction"] == "short":
                    ret = -ret
                # Entry cost was paid on open; exit cost now
                trades.append({
                    "timestamp": ts, "symbol": sym, "direction": pos["direction"],
                    "gross": ret, "cost": cfg.per_side_cost,  # exit only
                    "hold_bars": i - pos["entry_bar"],
                })

            # Open new positions for symbols not currently held
            for sym in (long_syms | short_syms):
                if sym in positions:
                    continue
                direction = "long" if sym in long_syms else "short"
                positions[sym] = {"direction": direction, "entry_ts": ts, "entry_bar": i}
                # Entry cost (exit will be charged on close)
                trades.append({
                    "timestamp": ts, "symbol": sym, "direction": direction,
                    "gross": 0.0, "cost": cfg.per_side_cost,
                    "hold_bars": 0,
                })

        # Force-close all remaining positions at the last timestamp
        if positions:
            last_ts = ts_values[-1]
            last_i = len(ts_values) - 1
            for sym, pos in positions.items():
                try:
                    entry_price = data.loc[(pos["entry_ts"], sym), "close"]
                    exit_price = data.loc[(last_ts, sym), "close"]
                except KeyError:
                    continue
                ret = (exit_price / entry_price) - 1.0
                if pos["direction"] == "short":
                    ret = -ret
                trades.append({
                    "timestamp": last_ts, "symbol": sym, "direction": pos["direction"],
                    "gross": ret, "cost": cfg.per_side_cost,
                    "hold_bars": last_i - pos["entry_bar"],
                })

        if not trades:
            return PaperResult(factor_name="unknown", config=cfg, classification="kill")

        trades_df = pd.DataFrame(trades)
        trades_df["net"] = trades_df["gross"] - trades_df["cost"]

        # Clean: entry-only rows (hold_bars=0) have gross=0, net=-cost
        # exit rows have gross + cost.  Combine them: total net = sum(gross) - sum(cost)
        return self._compute_result_metrics(trades_df, cfg, regime_labels)

    # ── Threshold Entry ──────────────────────────────────────────────

    def _simulate_threshold_entry(
        self,
        signal: pd.Series,
        data: pd.DataFrame,
        cfg: PaperConfig,
        regime_labels: pd.Series | None = None,
    ) -> PaperResult:
        """Only enter when cross-sectional |z-score| > signal_threshold.

        Exit when |z| < exit_threshold or direction flips.
        """
        ts_values = signal.index.get_level_values("timestamp").unique().sort_values()

        trades: list[dict] = []
        positions: dict[str, dict] = {}

        for i, ts in enumerate(ts_values):
            mask = signal.index.get_level_values("timestamp") == ts
            sig_t = signal.loc[mask].dropna()
            n_sym = len(sig_t)
            if n_sym < cfg.min_symbols:
                continue

            # Cross-sectional z-scores
            cs_mean = sig_t.mean()
            cs_std = sig_t.std()
            if cs_std < 1e-12:
                z_scores = pd.Series(0.0, index=sig_t.index)
            else:
                z_scores = (sig_t - cs_mean) / cs_std

            # Candidates: top/bottom by raw signal, filtered by |z| > threshold
            n_long = max(1, int(n_sym * cfg.top_frac))
            n_short = max(1, int(n_sym * cfg.bottom_frac))
            sorted_idx = sig_t.argsort()

            long_candidates = set()
            for idx in sorted_idx[-n_long:]:
                sym = sig_t.index.get_level_values("symbol")[idx]
                z = z_scores.iloc[idx] if hasattr(z_scores, 'iloc') else z_scores[idx]
                if abs(z) > cfg.signal_threshold:
                    long_candidates.add(sym)

            short_candidates = set()
            for idx in sorted_idx[:n_short]:
                sym = sig_t.index.get_level_values("symbol")[idx]
                z = z_scores.iloc[idx] if hasattr(z_scores, 'iloc') else z_scores[idx]
                if abs(z) > cfg.signal_threshold:
                    short_candidates.add(sym)

            # Check existing positions for exit signals
            to_close = []
            for sym, pos in list(positions.items()):
                if i - pos["entry_bar"] < cfg.hold_bars:
                    continue

                # Get current z-score for this symbol
                if sym in z_scores.index:
                    z_now = abs(z_scores.loc[sym])
                else:
                    z_now = 0.0

                new_dir = None
                if sym in long_candidates:
                    new_dir = "long"
                elif sym in short_candidates:
                    new_dir = "short"

                should_exit = False
                if new_dir is None:
                    should_exit = True  # symbol left the bucket
                elif new_dir != pos["direction"]:
                    should_exit = True  # direction flipped
                elif z_now < cfg.exit_threshold:
                    should_exit = True  # |z| below exit threshold

                if should_exit:
                    to_close.append(sym)

            for sym in to_close:
                pos = positions.pop(sym)
                try:
                    entry_price = data.loc[(pos["entry_ts"], sym), "close"]
                    exit_price = data.loc[(ts, sym), "close"]
                except KeyError:
                    continue
                ret = (exit_price / entry_price) - 1.0
                if pos["direction"] == "short":
                    ret = -ret
                trades.append({
                    "timestamp": ts, "symbol": sym, "direction": pos["direction"],
                    "gross": ret, "cost": cfg.per_side_cost,
                    "hold_bars": i - pos["entry_bar"],
                })

            # Open new positions (only if |z| > threshold)
            for sym in (long_candidates | short_candidates):
                if sym in positions:
                    continue
                direction = "long" if sym in long_candidates else "short"
                positions[sym] = {"direction": direction, "entry_ts": ts, "entry_bar": i}
                trades.append({
                    "timestamp": ts, "symbol": sym, "direction": direction,
                    "gross": 0.0, "cost": cfg.per_side_cost,
                    "hold_bars": 0,
                })

        # Force-close remaining
        if positions:
            last_ts = ts_values[-1]
            last_i = len(ts_values) - 1
            for sym, pos in positions.items():
                try:
                    entry_price = data.loc[(pos["entry_ts"], sym), "close"]
                    exit_price = data.loc[(last_ts, sym), "close"]
                except KeyError:
                    continue
                ret = (exit_price / entry_price) - 1.0
                if pos["direction"] == "short":
                    ret = -ret
                trades.append({
                    "timestamp": last_ts, "symbol": sym, "direction": pos["direction"],
                    "gross": ret, "cost": cfg.per_side_cost,
                    "hold_bars": last_i - pos["entry_bar"],
                })

        if not trades:
            return PaperResult(factor_name="unknown", config=cfg, classification="kill")

        trades_df = pd.DataFrame(trades)
        trades_df["net"] = trades_df["gross"] - trades_df["cost"]
        return self._compute_result_metrics(trades_df, cfg, regime_labels)

    # ── Position Smoothing ───────────────────────────────────────────

    def _simulate_position_smoothing(
        self,
        signal: pd.Series,
        data: pd.DataFrame,
        cfg: PaperConfig,
        regime_labels: pd.Series | None = None,
    ) -> PaperResult:
        """Gradually adjust position weights toward target instead of full turnover.

        Each bar: target = equal weight among top N long / bottom N short.
        Actual weight moves smoothing_rate toward target.
        Cost only on the delta, not the full position.
        """
        ts_values = signal.index.get_level_values("timestamp").unique().sort_values()
        sym_list = signal.index.get_level_values("symbol").unique().tolist()

        # Current weights: dict[symbol] = weight (positive=long, negative=short)
        weights: dict[str, float] = {}
        total_gross = 0.0
        total_cost = 0.0
        bar_records: list[dict] = []

        for i, ts in enumerate(ts_values):
            mask = signal.index.get_level_values("timestamp") == ts
            sig_t = signal.loc[mask].dropna()
            n_sym = len(sig_t)
            if n_sym < cfg.min_symbols:
                continue

            # Compute target weights
            n_long = max(1, int(n_sym * cfg.top_frac))
            n_short = max(1, int(n_sym * cfg.bottom_frac))
            sorted_idx = sig_t.argsort()
            long_syms = set(sig_t.index.get_level_values("symbol")[sorted_idx[-n_long:]])
            short_syms = set(sig_t.index.get_level_values("symbol")[sorted_idx[:n_short]])

            target: dict[str, float] = {}
            if long_syms:
                w_long = 1.0 / len(long_syms)
                for s in long_syms:
                    target[s] = w_long
            if short_syms:
                w_short = 1.0 / len(short_syms)
                for s in short_syms:
                    target[s] = -w_short

            # Smooth toward target
            rate = cfg.smoothing_rate
            new_weights: dict[str, float] = {}
            all_syms = set(list(weights.keys()) + list(target.keys()))

            for sym in all_syms:
                old_w = weights.get(sym, 0.0)
                tgt_w = target.get(sym, 0.0)
                new_w = old_w + rate * (tgt_w - old_w)
                # Drop near-zero weights to avoid dust
                if abs(new_w) < 0.0001:
                    continue
                new_weights[sym] = new_w

            # Compute PnL from previous weights
            if i > 0:
                prev_ts = ts_values[i - 1]
                for sym, w in weights.items():
                    try:
                        prev_price = data.loc[(prev_ts, sym), "close"]
                        curr_price = data.loc[(ts, sym), "close"]
                    except KeyError:
                        continue
                    asset_ret = (curr_price / prev_price) - 1.0
                    gross = w * asset_ret  # positive w = long, negative w = short
                    total_gross += gross

            # Compute cost from weight changes
            for sym in new_weights:
                old_w = weights.get(sym, 0.0)
                delta = abs(new_weights[sym] - old_w)
                if delta > 1e-8:
                    total_cost += delta * cfg.per_side_cost

            # Record bar-level stats
            bar_records.append({
                "timestamp": ts,
                "n_positions": len(new_weights),
                "gross_exposure": sum(abs(w) for w in new_weights.values()),
                "turnover_pct": sum(abs(new_weights.get(s, 0.0) - weights.get(s, 0.0))
                                    for s in set(list(weights.keys()) + list(new_weights.keys()))),
            })

            weights = new_weights

        if not bar_records:
            return PaperResult(factor_name="unknown", config=cfg, classification="kill")

        bar_df = pd.DataFrame(bar_records)
        net_pnl = total_gross - total_cost
        n_bars_traded = len(bar_df)

        # Cost/Gross ratio
        cost_gross = total_cost / abs(total_gross) if abs(total_gross) > 0 else float("inf")

        # Turnover: avg fraction adjusted per bar
        avg_turnover = float(bar_df["turnover_pct"].mean()) if len(bar_df) > 0 else 0.0

        # PF approximation (bar-level)
        bar_df["bar_pnl"] = 0.0
        # We can't easily reconstruct bar PnL since pnl is accumulated, but for PF we use net
        positive = max(total_gross - total_cost, 0.0)
        negative = abs(min(total_gross - total_cost, 0.0))
        pf = positive / negative if negative > 0 else float("inf")

        classification = self._classify(net_pnl, pf, cost_gross, 0.0, 0.0, 0.0, total_gross)

        return PaperResult(
            factor_name="unknown",
            config=cfg,
            gross_pnl=float(total_gross),
            net_pnl=float(net_pnl),
            cost_total=float(total_cost),
            cost_gross_ratio=float(cost_gross),
            turnover=float(avg_turnover),
            pf=float(pf),
            long_pnl=0.0,
            short_pnl=0.0,
            n_trades=int(bar_df["n_positions"].sum()),
            n_bars_traded=n_bars_traded,
            classification=classification,
        )

    # ── Shared result computation ────────────────────────────────────

    @staticmethod
    def _compute_result_metrics(
        trades_df: pd.DataFrame,
        cfg: PaperConfig,
        regime_labels: pd.Series | None = None,
    ) -> PaperResult:
        """Compute PaperResult from trade records (used by full_rebalance, direction_trigger, threshold_entry)."""
        if trades_df.empty:
            return PaperResult(factor_name="unknown", config=cfg, classification="kill")

        # Aggregate metrics
        total_gross = trades_df["gross"].sum()
        total_cost = trades_df["cost"].sum()
        net_pnl = trades_df["net"].sum()
        n_trades = len(trades_df)

        # Long vs short PnL (exclude entry-only rows with hold_bars=0 for direction modes)
        trades_with_pnl = trades_df[trades_df["hold_bars"] > 0] if "hold_bars" in trades_df.columns else trades_df
        if len(trades_with_pnl) == 0:
            trades_with_pnl = trades_df

        long_mask = trades_with_pnl["direction"] == "long"
        short_mask = trades_with_pnl["direction"] == "short"
        long_pnl = trades_with_pnl.loc[long_mask, "net"].sum() if long_mask.any() else 0.0
        short_pnl = trades_with_pnl.loc[short_mask, "net"].sum() if short_mask.any() else 0.0

        # PF
        positive = trades_df.loc[trades_df["net"] > 0, "net"].sum()
        negative = abs(trades_df.loc[trades_df["net"] < 0, "net"].sum())
        pf = positive / negative if negative > 0 else float("inf")

        # Cost/Gross ratio
        cost_gross = total_cost / abs(total_gross) if abs(total_gross) > 0 else float("inf")

        # Turnover
        bars_traded = trades_df["timestamp"].nunique()
        turnover = n_trades / max(bars_traded, 1)

        # Regime attribution
        regime_pnl: dict[str, float] = {}
        if regime_labels is not None:
            trades_df_copy = trades_df.copy()
            ts_map = {ts: regime_labels.get(ts, "unknown") for ts in trades_df_copy["timestamp"].unique()}
            trades_df_copy["regime"] = trades_df_copy["timestamp"].map(ts_map)
            regime_pnl = trades_df_copy.groupby("regime")["net"].sum().to_dict()

        # Symbol attribution
        symbol_contrib = trades_df.groupby("symbol")["net"].sum().sort_values()
        symbol_pnl: dict = {
            "top_5": symbol_contrib.tail(5).to_dict(),
            "bottom_5": symbol_contrib.head(5).to_dict(),
            "concentration": float((symbol_contrib.abs().max() / symbol_contrib.abs().sum())
                                    if symbol_contrib.abs().sum() > 0 else 1.0),
        }

        # Tail dependence
        bar_pnl = trades_df.groupby("timestamp")["net"].sum().sort_values()
        n_bars = len(bar_pnl)
        top_n = max(1, int(n_bars * 0.05))
        tail_contrib = bar_pnl.tail(top_n).sum() / abs(bar_pnl.sum()) if abs(bar_pnl.sum()) > 0 else 1.0

        # Classification
        classification = PaperTradingSimulator._classify(
            net_pnl, pf, cost_gross, long_pnl, short_pnl, tail_contrib, total_gross
        )

        return PaperResult(
            factor_name="unknown",
            config=cfg,
            gross_pnl=float(total_gross),
            net_pnl=float(net_pnl),
            cost_total=float(total_cost),
            cost_gross_ratio=float(cost_gross),
            turnover=float(turnover),
            pf=float(pf),
            long_pnl=float(long_pnl),
            short_pnl=float(short_pnl),
            n_trades=n_trades,
            n_bars_traded=bars_traded,
            regime_pnl=regime_pnl,
            symbol_pnl=symbol_pnl,
            tail_contribution=float(tail_contrib),
            classification=classification,
        )

    @staticmethod
    def _classify(
        net_pnl: float,
        pf: float,
        cost_gross: float,
        long_pnl: float,
        short_pnl: float,
        tail_contrib: float,
        gross_pnl: float,
    ) -> str:
        """Four-category classification: KILL / FILTER_ONLY / TRADE_RULE_REPAIR / RETEST_1000BAR."""
        # True KILL: gross itself is negative
        if gross_pnl < 0:
            return "KILL"
        if pf < 0.8:
            return "KILL"
        if tail_contrib > 0.85:
            return "KILL"

        # TRADE_RULE_REPAIR: gross positive but cost eats everything
        if gross_pnl > 0 and cost_gross > 0.80:
            return "TRADE_RULE_REPAIR"

        # FILTER_ONLY: gross has direction, one side works
        if long_pnl > 0 or short_pnl > 0:
            if net_pnl < 0:
                return "FILTER_ONLY"

        # RETEST_1000BAR: borderline case
        if gross_pnl > 0 and 0.50 <= pf < 1.0:
            return "RETEST_1000BAR"

        # Default
        if net_pnl > 0 and pf >= 1.1:
            return "trade_signal"
        return "KILL"


def stress_test_matrix(factor_path: Path, project_root: Path) -> dict:
    """Trading rule stress test matrix — find each factor's tradable niche.

    Tests all combinations of:
    - Mode: full_rebalance, direction_trigger, threshold_entry, position_smoothing
    - Threshold: 5%, 10%, 20%
    - Hold bars: 1, 2, 3, 6
    - Side: long-only, short-only, long-short
    - Cost: 6bps, 9bps, 12bps per side
    - Universe: top50 by volume, top100, all
    - Signal threshold: 1.0, 1.5, 2.0 (threshold_entry only)
    - Smoothing rate: 0.25, 0.50, 0.75 (position_smoothing only)
    """
    from backtest_engine import BacktestEngine
    root = Path(project_root)
    sim = PaperTradingSimulator(root)
    engine = BacktestEngine(root)
    data = engine._load_data()

    # Compute volume rank for universe filtering
    if "volume" in data.columns:
        avg_vol = data["volume"].groupby(level="symbol").mean()
        vol_rank = avg_vol.rank(ascending=False)
    else:
        vol_rank = pd.Series(1, index=data.index.get_level_values("symbol").unique())

    results = []
    configs = []

    # Base configs shared by full_rebalance and direction_trigger
    for frac in [0.05, 0.10, 0.20]:
        for hold in [1, 2, 3, 6]:
            for cost in [6.0, 9.0, 12.0]:
                configs.append(PaperConfig(
                    top_frac=frac, bottom_frac=frac,
                    hold_bars=hold, cost_bps=cost,
                    execution_mode="full_rebalance",
                ))
                configs.append(PaperConfig(
                    top_frac=frac, bottom_frac=frac,
                    hold_bars=hold, cost_bps=cost,
                    execution_mode="direction_trigger",
                ))

    # Threshold entry configs (fewer combos: 3 frac × 3 threshold × 2 hold × 2 cost)
    for frac in [0.05, 0.10, 0.20]:
        for threshold in [1.0, 1.5, 2.0]:
            for hold in [2, 4]:
                for cost in [6.0, 9.0]:
                    configs.append(PaperConfig(
                        top_frac=frac, bottom_frac=frac,
                        hold_bars=hold, cost_bps=cost,
                        execution_mode="threshold_entry",
                        signal_threshold=threshold,
                        exit_threshold=threshold * 0.33,
                    ))

    # Position smoothing configs
    for frac in [0.05, 0.10, 0.20]:
        for rate in [0.25, 0.50, 0.75]:
            for cost in [6.0, 9.0, 12.0]:
                configs.append(PaperConfig(
                    top_frac=frac, bottom_frac=frac,
                    hold_bars=1, cost_bps=cost,
                    execution_mode="position_smoothing",
                    smoothing_rate=rate,
                ))

    for cfg in configs:
        for universe_name, universe_mask in [
            ("top50", vol_rank <= 50),
            ("top100", vol_rank <= 100),
            ("all", None),
        ]:
            for side_mode in ["long-short", "long-only", "short-only"]:
                # Filter data for universe
                if universe_mask is not None:
                    keep_syms = vol_rank[universe_mask].index
                    data_subset = data.loc[data.index.get_level_values("symbol").isin(keep_syms)]
                else:
                    data_subset = data

                if len(data_subset) < 100:
                    continue

                # Modify config for side
                cfg_side = PaperConfig(
                    top_frac=cfg.top_frac if side_mode != "short-only" else 0.0,
                    bottom_frac=cfg.bottom_frac if side_mode != "long-only" else 0.0,
                    hold_bars=cfg.hold_bars,
                    cost_bps=cfg.cost_bps,
                )

                loaded = load_factor_from_path(factor_path)
                signal = loaded.factor.compute(data_subset).reindex(data_subset.index).fillna(0.0)
                result = sim._simulate_single(signal, data_subset, cfg_side)

                results.append({
                    "factor": loaded.factor.factor_name,
                    "threshold": f"{cfg.top_frac:.0%}",
                    "hold_bars": cfg.hold_bars,
                    "side": side_mode,
                    "cost_bps": cfg.cost_bps,
                    "universe": universe_name,
                    "gross_pnl": result.gross_pnl,
                    "net_pnl": result.net_pnl,
                    "pf": result.pf,
                    "cost_gross_ratio": result.cost_gross_ratio,
                    "turnover": result.turnover,
                    "long_net": result.long_pnl,
                    "short_net": result.short_pnl,
                    "n_trades": result.n_trades,
                    "classification": result.classification,
                })

    # Find best niche per factor
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return {"error": "No valid results", "raw": results}

    niches = {}
    for factor_name, grp in results_df.groupby("factor"):
        # Best by net PnL
        best = grp.loc[grp["net_pnl"].idxmax()]
        # Best by PF among positive net
        positive = grp[grp["net_pnl"] > 0]
        best_pf = positive.loc[positive["pf"].idxmax()] if len(positive) > 0 else None

        niches[factor_name] = {
            "best_net_niche": best.to_dict(),
            "best_pf_niche": best_pf.to_dict() if best_pf is not None else None,
            "any_positive_net": len(positive) > 0,
            "n_configs_tested": len(grp),
            "n_positive": len(positive),
        }

    return {"niches": niches, "raw_results": results}


def run_paper_battery(
    project_root: Path,
    factor_names: list[str] | None = None,
    regime_labels: pd.Series | None = None,
) -> dict[str, list[PaperResult]]:
    """Run paper trading on all or specified factors."""
    root = Path(project_root)
    factors_dir = root / "factors"
    sim = PaperTradingSimulator(root)

    if factor_names is None:
        # Auto-discover from production_factors.json
        import json
        prod = json.loads((root / "config" / "production_factors.json").read_text())
        factor_names = [f["factor_name"] for f in prod.get("factors", [])]

    results: dict[str, list[PaperResult]] = {}
    for fname in factor_names:
        # Find the file
        found = None
        fname_clean = fname.lower().replace("_", "")
        for py_file in factors_dir.glob("alpha_*.py"):
            if fname_clean in py_file.stem.lower().replace("_", ""):
                found = py_file
                break
        if found is None:
            continue
        try:
            res = sim.run(found, regime_labels=regime_labels)
            for r in res:
                r.factor_name = fname
            results[fname] = res
        except Exception as e:
            print(f"Paper trading failed for {fname}: {e}")

    return results
