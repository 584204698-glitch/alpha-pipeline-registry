from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from data_ingestion import build_live_dataset
from runtime_env import load_runtime_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slow Coinglass data download runner")
    parser.add_argument("--symbol-limit", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=120)
    parser.add_argument("--interval", type=str, default=None)
    parser.add_argument("--intervals", nargs="*", default=None)
    parser.add_argument("--pause-seconds", type=float, default=4.5)
    parser.add_argument("--exchanges", nargs="*", default=["Binance", "OKX"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    load_runtime_env(root / ".env.runtime")
    api_key = os.environ.get("COINGLASS_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing COINGLASS_API_KEY")

    if args.intervals:
        intervals = list(args.intervals)
    elif args.interval:
        intervals = [args.interval]
    else:
        intervals = ["15m", "2h", "4h"]

    summaries: list[dict] = []
    final_path = None
    for interval in intervals:
        path, summary = build_live_dataset(
            project_root=root,
            api_key=api_key,
            exchanges=tuple(args.exchanges),
            symbol_limit=args.symbol_limit,
            interval=interval,
            history_limit=args.history_limit,
            pause_seconds=args.pause_seconds,
        )
        final_path = path
        summaries.append(summary)

    print(final_path)
    print(json.dumps({"intervals": intervals, "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
