#!/bin/bash
# Post-download pipeline: verify → paper → register → report
set -e
cd /mnt/e/alpha_pipeline

NEW_DATA="data/data_storage_1h_v2.parquet"
if [ ! -f "$NEW_DATA" ]; then
    echo "ERROR: $NEW_DATA not found. Check if 1h download completed."
    exit 1
fi

echo "=== Phase 1: Verify 1h data density ==="
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('$NEW_DATA')
if not isinstance(df.index, pd.MultiIndex):
    df = df.set_index(['timestamp','symbol']).sort_index()
counts = df.groupby('timestamp').size()
dense = counts[counts >= 100]
print(f'Total timestamps: {len(counts)}')
print(f'Dense (>=100 symbols): {len(dense)}')
print(f'Symbols: {df.index.get_level_values(\"symbol\").nunique()}')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
has_buy_sell = 'taker_buy_volume' in df.columns
print(f'Taker buy/sell split: {has_buy_sell}')
"
echo ""
echo "=== Phase 2: Paper Trading Death Filter on 1000-bar data ==="
export ALPHA_PIPELINE_DATA_PATH="$NEW_DATA"
.venv/bin/python paper_battery.py

echo ""
echo "=== Phase 3: Verify registry populated ==="
.venv/bin/python -c "
from research.registry import FactorRegistry
r = FactorRegistry('registry')
print(f'Factors registered: {len(r.list_factors())}')
print(f'Pending reviews: {len(r.get_pending_reviews())}')
live = r.get_live_config()
print(f'Live mode: {live[\"mode\"]}')
print(f'Enabled factors: {live[\"enabled_factors\"]}')
"

echo ""
echo "=== DONE ==="
echo "Registry at: /mnt/e/alpha_pipeline/registry/"
echo "Handoff guide: /mnt/e/alpha_pipeline/registry/HANDOFF.md"
