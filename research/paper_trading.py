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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
        for frac in [0.05, 0.10, 0.20]:
            for hold in [1, 2, 3]:
                configs.append(PaperConfig(top_frac=frac, bottom_frac=frac, hold_bars=hold))
        return configs

    def _simulate_single(
        self,
        signal: pd.Series,
        data: pd.DataFrame,
        cfg: PaperConfig,
        regime_labels: pd.Series | None = None,
    ) -> PaperResult:
        """Run one paper trading config and return detailed metrics."""
        ts_values = signal.index.get_level_values("timestamp").unique().sort_values()
        symbols = signal.index.get_level_values("symbol").unique()

        all_trades: list[dict] = []
        total_gross = 0.0
        total_cost = 0.0
        long_gross = 0.0
        short_gross = 0.0
        n_trades = 0

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

        # Aggregate metrics
        total_gross = trades_df["gross"].sum()
        total_cost = trades_df["cost"].sum() * 2  # round-trip
        net_pnl = trades_df["net"].sum()
        n_trades = len(trades_df)

        long_mask = trades_df["direction"] == "long"
        short_mask = trades_df["direction"] == "short"
        long_pnl = trades_df.loc[long_mask, "net"].sum()
        short_pnl = trades_df.loc[short_mask, "net"].sum()

        # PF
        positive = trades_df.loc[trades_df["net"] > 0, "net"].sum()
        negative = abs(trades_df.loc[trades_df["net"] < 0, "net"].sum())
        pf = positive / negative if negative > 0 else float("inf")

        # Cost/Gross ratio
        cost_gross = total_cost / abs(total_gross) if abs(total_gross) > 0 else float("inf")

        # Turnover (avg trades per bar)
        bars_traded = trades_df["timestamp"].nunique()
        turnover = n_trades / max(bars_traded, 1)

        # Regime attribution
        regime_pnl: dict[str, float] = {}
        if regime_labels is not None:
            trades_df["regime"] = trades_df["timestamp"].map(
                {ts: regime_labels.get(ts, "unknown") for ts in trades_df["timestamp"].unique()}
            )
            regime_pnl = trades_df.groupby("regime")["net"].sum().to_dict()

        # Symbol attribution (top contributors)
        symbol_contrib = trades_df.groupby("symbol")["net"].sum().sort_values()
        symbol_pnl = {
            "top_5": symbol_contrib.tail(5).to_dict(),
            "bottom_5": symbol_contrib.head(5).to_dict(),
            "concentration": float((symbol_contrib.abs().max() / symbol_contrib.abs().sum()) if symbol_contrib.abs().sum() > 0 else 1.0),
        }

        # Tail dependence: top 5% bars contribution
        bar_pnl = trades_df.groupby("timestamp")["net"].sum().sort_values()
        n_bars = len(bar_pnl)
        top_n = max(1, int(n_bars * 0.05))
        tail_contrib = bar_pnl.tail(top_n).sum() / abs(bar_pnl.sum()) if abs(bar_pnl.sum()) > 0 else 1.0

        # Classification
        classification = self._classify(net_pnl, pf, cost_gross, long_pnl, short_pnl, tail_contrib, total_gross)

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
    - Threshold: 5%, 10%, 20%
    - Hold bars: 1, 2, 3, 6
    - Side: long-only, short-only, long-short
    - Cost: 6bps, 9bps maker, 12bps taker per side
    - Universe: top50 by volume, top100, all
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

    for frac in [0.05, 0.10, 0.20]:
        for hold in [1, 2, 3, 6]:
            for cost in [6.0, 9.0, 12.0]:
                configs.append(PaperConfig(
                    top_frac=frac, bottom_frac=frac,
                    hold_bars=hold, cost_bps=cost,
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
