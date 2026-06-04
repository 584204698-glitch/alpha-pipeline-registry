"""Retry failed Coinglass API calls with longer delays."""
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

ENDPOINTS = {
    "price": "/api/futures/price/history",
    "open_interest": "/api/futures/open-interest/history",
    "funding_rate": "/api/futures/funding-rate/history",
    "taker_volume": "/api/futures/v2/taker-buy-sell-volume/history",
}

DELAY_S = 10.0  # 10s per symbol to avoid 429
NEW_BARS = 48


def fetch(endpoint: str, symbol: str, limit: int) -> list:
    url = f"{PROXY}{endpoint}?symbol={symbol}&exchange=Binance&interval=1h&limit={limit}"
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"    429, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            payload = r.json()
            if payload.get("code") not in (None, "0", 0):
                raise ValueError(f"API error: {payload}")
            return payload.get("data", [])
        except Exception as e:
            if attempt < 3:
                time.sleep(8)
            else:
                return []
    return []


def main():
    parquet_path = ROOT / "data" / "data_storage_1h.parquet"
    universe_path = ROOT / "registry" / "live" / "event_universe.json"

    df = pd.read_parquet(parquet_path)
    last_ts = df.index.get_level_values("timestamp").max()
    last_orig = pd.Timestamp("2026-06-04 01:00:00+00:00")

    # Find symbols with no data after original last bar
    after = df[df.index.get_level_values("timestamp") > last_orig]
    have_data = set(after.index.get_level_values("symbol"))

    with open(universe_path) as f:
        data = json.load(f)

    def collect(d):
        syms = set()
        if isinstance(d, list): syms.update(d)
        elif isinstance(d, dict):
            for v in d.values(): syms.update(collect(v))
        return syms

    all_universe = collect(data)
    # Add BTC
    all_universe.add("BTCUSDT")
    # Strip Binance: prefix for API
    missing = []
    for s in all_universe:
        api_sym = s.replace("Binance:", "") if s.startswith("Binance:") else s
        parquet_sym = s if s.startswith("Binance:") or s == "BTCUSDT" else f"Binance:{s}"
        if parquet_sym not in have_data and api_sym:
            missing.append((api_sym, parquet_sym))

    print(f"Missing symbols: {len(missing)}")
    if not missing:
        print("All symbols up to date!")
        return

    new_frames = []
    failed = []

    for i, (api_sym, pq_sym) in enumerate(sorted(missing)):
        print(f"  [{i+1}/{len(missing)}] {pq_sym} ...")
        if i > 0:
            time.sleep(DELAY_S)

        sym_data = {}
        has_any = False
        for key, ep in ENDPOINTS.items():
            try:
                recs = fetch(ep, api_sym, NEW_BARS)
            except Exception as e:
                failed.append(f"{pq_sym}/{key}: {e}")
                continue
            for rec in recs:
                ts = pd.Timestamp(rec["time"], unit="ms", tz="UTC")
                if ts <= last_orig:
                    continue
                if ts not in sym_data:
                    sym_data[ts] = {}
                sym_data[ts][key] = rec
                has_any = True

        if not has_any:
            continue

        rows = []
        for ts in sorted(sym_data):
            d = sym_data[ts]
            row = {"timestamp": ts, "symbol": pq_sym}
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
            new_frames.append(pd.DataFrame(rows).set_index(["timestamp", "symbol"]))

    if new_frames:
        new_data = pd.concat(new_frames).sort_index()
        combined = pd.concat([df, new_data]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(parquet_path)
        print(f"\n  Added: {len(new_data):,} rows for {new_data.index.get_level_values('symbol').nunique()} symbols")
        print(f"  Total: {len(combined):,} rows ({parquet_path.stat().st_size/1024/1024:.1f} MB)")

    if failed:
        print(f"\n  Still failed ({len(failed)}):")
        for f in failed[:5]:
            print(f"    {f}")


if __name__ == "__main__":
    main()
