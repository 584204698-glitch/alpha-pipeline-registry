#!/bin/bash
# Auto-chain: after 1h download finishes, download 4h, then backtest absorption factors
set -e
cd /mnt/e/alpha_pipeline

echo "=== Phase 1: Download 4h data ==="
.venv/bin/python download_data.py --interval 4h --symbol-limit 200 --history-limit 1000 --pause-seconds 4.5
cp data/data_storage.parquet data/data_storage_4h_v2.parquet
echo "4h download complete"

echo "=== Phase 2: Verify data density ==="
.venv/bin/python -c "
import pandas as pd
for f, label in [('data/data_storage_1h_v2.parquet','1h'), ('data/data_storage_4h_v2.parquet','4h')]:
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['timestamp','symbol']).sort_index()
    counts = df.groupby('timestamp').size()
    dense = counts[counts >= 100]
    syms = df.index.get_level_values('symbol').nunique()
    cols = list(df.columns)
    has_buy_sell = 'taker_buy_volume' in cols and 'taker_sell_volume' in cols
    print(f'{label}: {len(dense)} dense sections, {syms} symbols, {len(df)} rows, buy/sell={has_buy_sell}')
"

echo "=== Phase 3: Backtest absorption factors ==="
export ALPHA_PIPELINE_DATA_PATH=data/data_storage_1h_v2.parquet
.venv/bin/python batch_absorption_test.py

echo "=== DONE ==="
