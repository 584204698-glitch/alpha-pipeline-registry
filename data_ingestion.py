from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from pipeline_utils import get_logger

COINGLASS_BASE_URL = "https://proxy.keystore.com.cn/api/v1/proxy/coinglass"
NON_CRYPTO_SYMBOL_BLACKLIST = {
    "AAPL",
    "AMD",
    "AMZN",
    "ARM",
    "BILL",
    "BRENTOIL",
    "CL",
    "COIN",
    "COPPER",
    "CRCL",
    "DRAM",
    "EWY",
    "GOOGL",
    "GOLD",
    "INTC",
    "META",
    "MRVL",
    "MSFT",
    "MSTR",
    "MU",
    "NVDA",
    "ORCL",
    "PLTR",
    "QQQ",
    "SAHARA",
    "SILVER",
    "SOXL",
    "SP500",
    "SPCX",
    "SPX500",
    "SPY",
    "TSLA",
    "TSM",
    "USOIL",
    "XAG",
    "XAU",
}
SUPPORTED_EXCHANGES = ("Binance", "OKX")
ENDPOINT_ORDER = (
    ("price", "/api/futures/price/history?symbol={instrument_id}&exchange={exchange}&interval={interval}&limit={history_limit}"),
    ("open_interest", "/api/futures/open-interest/history?symbol={instrument_id}&exchange={exchange}&interval={interval}&limit={history_limit}"),
    ("funding_rate", "/api/futures/funding-rate/history?symbol={instrument_id}&exchange={exchange}&interval={interval}&limit={history_limit}"),
    ("taker_volume", "/api/futures/v2/taker-buy-sell-volume/history?symbol={instrument_id}&exchange={exchange}&interval={interval}&limit={history_limit}"),
)


def _headers(api_key: str) -> dict[str, str]:
    return {"X-Api-Key": api_key}


def _request_json(api_key: str, path: str, timeout: int = 120) -> dict[str, Any]:
    response = requests.get(f"{COINGLASS_BASE_URL}{path}", headers=_headers(api_key), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("code") not in (None, "0", 0):
        raise ValueError(f"Coinglass request failed for {path}: {payload}")
    return payload


def _request_json_with_retry(api_key: str, path: str, pause_seconds: float, max_attempts: int = 4, timeout: int = 120) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _request_json(api_key, path, timeout=timeout)
        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            if status == 429 and attempt < max_attempts:
                sleep_for = max(pause_seconds * (attempt + 1), 8.0)
                time.sleep(sleep_for)
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(max(pause_seconds, 2.0))
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Retry loop ended unexpectedly for path={path}")


def _auth_preview(api_key: str) -> str:
    if not api_key:
        return "missing"
    if len(api_key) <= 8:
        return f"len={len(api_key)}"
    return f"len={len(api_key)} prefix={api_key[:6]} suffix={api_key[-4:]}"


def filter_supported_symbols(symbols: list[str], limit: int = 200) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        if symbol in NON_CRYPTO_SYMBOL_BLACKLIST:
            continue
        if any(ch in symbol for ch in (" ", "/", "币")):
            continue
        filtered.append(symbol)
        seen.add(symbol)
        if len(filtered) >= limit:
            break
    return filtered


def fetch_supported_exchange_pairs(api_key: str, exchange: str, pause_seconds: float = 0.0) -> list[dict[str, Any]]:
    payload = _request_json_with_retry(api_key, f"/api/futures/supported-exchange-pairs?exchange={exchange}", pause_seconds=pause_seconds)
    return payload.get("data", {}).get(exchange, [])


def select_live_markets(
    api_key: str,
    exchanges: tuple[str, ...] = SUPPORTED_EXCHANGES,
    symbol_limit: int = 200,
    pause_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for exchange in exchanges:
        pairs = fetch_supported_exchange_pairs(api_key, exchange, pause_seconds=pause_seconds)
        perp_candidates: list[dict[str, Any]] = []
        for item in pairs:
            base_asset = str(item.get("base_asset", "")).upper()
            quote_asset = str(item.get("quote_asset", "")).upper()
            instrument_id = str(item.get("instrument_id", ""))
            settlement_currency = str(item.get("settlement_currency", "")).upper()
            has_funding = item.get("funding_interval") is not None
            has_expiry = item.get("expiry_time") is not None
            if not base_asset or not instrument_id:
                continue
            if base_asset in NON_CRYPTO_SYMBOL_BLACKLIST:
                continue
            if quote_asset not in {"USDT", "USD"}:
                continue
            if has_expiry:
                continue
            if not has_funding and "PERP" not in instrument_id and quote_asset != "USDT":
                continue
            key = (exchange, instrument_id)
            if key in seen_keys:
                continue
            perp_candidates.append(
                {
                    "exchange": exchange,
                    "instrument_id": instrument_id,
                    "base_asset": base_asset,
                    "quote_asset": quote_asset,
                    "settlement_currency": settlement_currency,
                    "funding_interval": item.get("funding_interval"),
                }
            )
        perp_candidates.sort(
            key=lambda item: (
                0 if item["quote_asset"] == "USDT" else 1,
                0 if item.get("funding_interval") is not None else 1,
                0 if item["instrument_id"].endswith("PERP") else 1,
                item["instrument_id"],
            )
        )
        for market in perp_candidates:
            key = (market["exchange"], market["instrument_id"])
            if key in seen_keys:
                continue
            selected.append(market)
            seen_keys.add(key)
            if len(selected) >= symbol_limit:
                return selected
    return selected


def _coerce_ohlc(records: list[dict[str, Any]], value_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", value_name])
    frame = frame.rename(columns={"time": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame[value_name] = pd.to_numeric(frame["close"], errors="coerce")
    return frame[["timestamp", value_name]].dropna()


def _price_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = frame.rename(columns={"time": "timestamp", "volume_usd": "volume"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["timestamp", "open", "high", "low", "close", "volume"]].dropna()


def _taker_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "taker_buy_volume", "taker_sell_volume", "taker_volume"])
    frame = frame.rename(columns={"time": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["taker_buy_volume"] = pd.to_numeric(frame["taker_buy_volume_usd"], errors="coerce")
    frame["taker_sell_volume"] = pd.to_numeric(frame["taker_sell_volume_usd"], errors="coerce")
    frame["taker_volume"] = frame["taker_buy_volume"].fillna(0.0) + frame["taker_sell_volume"].fillna(0.0)
    return frame[["timestamp", "taker_buy_volume", "taker_sell_volume", "taker_volume"]].dropna()


def _parse_endpoint_frame(name: str, payload: dict[str, Any]) -> pd.DataFrame:
    records = payload.get("data", [])
    if name == "price":
        return _price_frame(records)
    if name == "open_interest":
        return _coerce_ohlc(records, "open_interest")
    if name == "funding_rate":
        return _coerce_ohlc(records, "funding_rate")
    if name == "taker_volume":
        return _taker_frame(records)
    raise ValueError(f"Unknown endpoint name [{name}]")


def _market_output_path(project_root: Path, exchange: str, instrument_id: str, interval: str) -> Path:
    safe_exchange = exchange.lower().replace("/", "_")
    safe_instrument = instrument_id.replace("/", "_").replace(":", "_")
    return project_root / "data" / "markets" / interval / safe_exchange / f"{safe_instrument}.parquet"


def fetch_market_history(
    api_key: str,
    exchange: str,
    instrument_id: str,
    interval: str = "1h",
    history_limit: int = 200,
    pause_seconds: float = 4.5,
) -> pd.DataFrame:
    endpoint_frames: list[pd.DataFrame] = []
    for index, (name, template) in enumerate(ENDPOINT_ORDER, start=1):
        path = template.format(
            instrument_id=instrument_id,
            exchange=exchange,
            interval=interval,
            history_limit=history_limit,
        )
        payload = _request_json_with_retry(api_key, path, pause_seconds=pause_seconds)
        endpoint_frames.append(_parse_endpoint_frame(name, payload))
        if index < len(ENDPOINT_ORDER):
            time.sleep(max(pause_seconds, 0.0))
    price, oi, funding, taker = endpoint_frames
    merged = price.merge(oi, on="timestamp", how="inner").merge(funding, on="timestamp", how="inner").merge(taker, on="timestamp", how="inner")
    merged["symbol"] = f"{exchange}:{instrument_id}"
    return merged.sort_values("timestamp")


def build_live_dataset(
    project_root: Path,
    api_key: str,
    exchanges: tuple[str, ...] = SUPPORTED_EXCHANGES,
    symbol_limit: int = 200,
    interval: str = "1h",
    history_limit: int = 200,
    pause_seconds: float = 4.5,
) -> tuple[Path, dict[str, Any]]:
    project_root = Path(project_root)
    logger = get_logger(project_root, "DATA_INGESTION")
    if not api_key:
        raise RuntimeError("Missing Coinglass API key for live dataset build")
    logger.info(
        f"Starting live dataset build. exchanges={list(exchanges)} symbol_limit={symbol_limit} interval={interval} history_limit={history_limit} pause_seconds={pause_seconds} auth={_auth_preview(api_key)}"
    )
    selected = select_live_markets(api_key, exchanges=exchanges, symbol_limit=symbol_limit, pause_seconds=pause_seconds)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    output_root = project_root / "data" / "markets" / interval
    output_root.mkdir(parents=True, exist_ok=True)
    for index, market in enumerate(selected, start=1):
        market_path = _market_output_path(project_root, market["exchange"], market["instrument_id"], interval)
        try:
            frame = fetch_market_history(
                api_key=api_key,
                exchange=market["exchange"],
                instrument_id=market["instrument_id"],
                interval=interval,
                history_limit=history_limit,
                pause_seconds=pause_seconds,
            )
            if not frame.empty:
                market_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(market_path, index=False)
                frames.append(frame)
            logger.info(
                f"Fetched live history [{index}/{len(selected)}] for [{market['exchange']}:{market['instrument_id']}] rows={len(frame)} output={market_path}"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"market": market, "error": str(exc)})
            logger.warning(f"Failed live fetch for [{market['exchange']}:{market['instrument_id']}] - {exc}")
        if pause_seconds > 0 and index < len(selected):
            time.sleep(max(pause_seconds, 0.0))
    if not frames:
        raise RuntimeError(f"No live data fetched successfully. failures={failures[:5]}")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index(["timestamp", "symbol"]).sort_index()
    data_path = project_root / "data" / "data_storage.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(data_path)
    progress_path = project_root / "data" / f"download_progress_{interval}.json"
    progress_payload = {
        "interval": interval,
        "history_limit": history_limit,
        "symbol_limit": symbol_limit,
        "successful_markets": [frame["symbol"].iloc[0] for frame in frames if not frame.empty],
        "failed_markets": failures,
        "market_files": sorted(str(path.relative_to(project_root)) for path in output_root.rglob("*.parquet")),
    }
    progress_path.write_text(json.dumps(progress_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "selected_markets": len(selected),
        "successful_markets": len(frames),
        "failed_markets": len(failures),
        "failures": failures[:10],
        "rows_written": int(len(combined)),
        "symbols_written": int(combined.index.get_level_values("symbol").nunique()),
        "interval": interval,
        "history_limit": history_limit,
        "progress_path": str(progress_path),
    }
    logger.info(f"Wrote live dataset with {summary['rows_written']} rows across {summary['symbols_written']} symbols")
    return data_path, summary
