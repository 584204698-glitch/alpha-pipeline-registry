"""Append fresh 1h bars to existing parquet. Uses Coinglass proxy API."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
PROXY = "https://proxy.keystore.com.cn/api/v1/proxy/coinglass"

from runtime_env import load_runtime_env
load_runtime_env(ROOT / ".env.runtime")

API_KEY = os.environ.get("COINGLASS_API_KEY", "")
HEADERS = {"X-Api-Key": API_KEY}

# Coinglass endpoint paths (with hyphens, not underscores)
ENDPOINTS = {
    "price": "/api/futures/price/history",
    "open_interest": "/api/futures/open-interest/history",
    "funding_rate": "/api/futures/funding-rate/history",
    "taker_volume": "/api/futures/v2/taker-buy-sell-volume/history",
}

NEW_BARS = 48  # fetch up to 48 new 1h bars (~2 days)


def fetch(endpoint: str, symbol: str, limit: int) -> list:
    url = f"{PROXY}{endpoint}?symbol={symbol}&exchange=Binance&interval=1h&limit={limit}"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if payload.get("code") not in (None, "0", 0):
        raise ValueError(f"API error: {payload}")
    return payload.get("data", [])


def main() -> None:
    parquet_path = ROOT / "data" / "data_storage_1h.parquet"
    universe_path = ROOT / "registry" / "live" / "event_universe.json"

    print("Loading existing data...")
    existing = pd.read_parquet(parquet_path)
    last_ts = existing.index.get_level_values("timestamp").max()
    print(f"  {len(existing):,} rows, last bar: {last_ts}")

    # Collect universe symbols
    with open(universe_path) as f:
        data = json.load(f)

    def collect_symbols(obj) -> set:
        syms = set()
        if isinstance(obj, list):
            syms.update(s for s in obj if isinstance(s, str))
        elif isinstance(obj, dict):
            for v in obj.values():
                syms.update(collect_symbols(v))
        return syms

    binance_symbols = collect_symbols(data)
    print(f"  Universe symbols: {len(binance_symbols)}")

    # Strip Binance: prefix for API, but keep track for parquet writing
    symbol_map: dict[str, str] = {}  # api_name → parquet_name
    for s in binance_symbols:
        if s.startswith("Binance:"):
            symbol_map[s.replace("Binance:", "")] = s
        else:
            symbol_map[s] = s

    # Add BTC for regime detection
    if "BTCUSDT" not in symbol_map:
        symbol_map["BTCUSDT"] = "BTCUSDT"

    failed: list[str] = []
    all_frames: list[pd.DataFrame] = []

    for i, (api_sym, parquet_sym) in enumerate(sorted(symbol_map.items())):
        if i > 0 and i % 20 == 0:
            print(f"  {i}/{len(symbol_map)} ...")
        time.sleep(5.0)  # 5s between symbols to avoid rate limit

        sym_data: dict[pd.Timestamp, dict] = {}
        has_any = False

        for key, ep_path in ENDPOINTS.items():
            # Retry up to 3 times with backoff
            for attempt in range(3):
                try:
                    records = fetch(ep_path, api_sym, NEW_BARS)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(3.0 * (attempt + 1))
                    else:
                        failed.append(f"{parquet_sym}/{key}: {e}")
                        records = []
                        break

            for rec in records:
                ts = pd.Timestamp(rec["time"], unit="ms", tz="UTC")
                if ts <= last_ts:
                    continue
                if ts not in sym_data:
                    sym_data[ts] = {}
                sym_data[ts][key] = rec
                has_any = True

        if not has_any:
            continue

        rows = []
        for ts in sorted(sym_data):
            row = {"timestamp": ts, "symbol": parquet_sym}
            d = sym_data[ts]

            p = d.get("price", {})
            row["open"] = float(p.get("open", np.nan))
            row["high"] = float(p.get("high", np.nan))
            row["low"] = float(p.get("low", np.nan))
            row["close"] = float(p.get("close", np.nan))
            row["volume"] = float(p.get("volume", np.nan))

            oi = d.get("open_interest", {})
            row["open_interest"] = float(oi.get("open", np.nan))

            fr = d.get("funding_rate", {})
            row["funding_rate"] = float(fr.get("rate", np.nan))

            tv = d.get("taker_volume", {})
            row["taker_buy_volume"] = float(tv.get("buyVolume", np.nan))
            row["taker_sell_volume"] = float(tv.get("sellVolume", np.nan))

            rows.append(row)

        if rows:
            all_frames.append(pd.DataFrame(rows).set_index(["timestamp", "symbol"]))

    if not all_frames:
        print("No new data to append.")
        return

    new_data = pd.concat(all_frames).sort_index()
    print(f"\n  New: {len(new_data):,} rows, {new_data.index.get_level_values('symbol').nunique()} symbols")
    print(f"  Range: {new_data.index.get_level_values('timestamp').min()} → {new_data.index.get_level_values('timestamp').max()}")

    combined = pd.concat([existing, new_data]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.to_parquet(parquet_path)

    print(f"  Saved: {len(combined):,} rows ({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")

    if failed:
        print(f"\n  Failed ({len(failed)}/{len(symbol_map) * len(ENDPOINTS)}):")
        for f in failed[:5]:
            print(f"    {f}")


if __name__ == "__main__":
    main()
