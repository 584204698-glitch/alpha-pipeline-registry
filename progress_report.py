from __future__ import annotations

import argparse
import json
from pathlib import Path

from main import PipelineRunner
from health_check import run_health_check_once
from pipeline_utils import read_json


def build_progress_report(project_root: Path, batch_size: int = 10) -> str:
    root = Path(project_root)
    runner = PipelineRunner(root)
    production = read_json(root / 'config' / 'production_factors.json', default={'factors': []}) or {'factors': []}
    report = run_health_check_once(root)
    data_files = {
        '15m': root / 'data' / 'download_progress_15m.json',
        '2h': root / 'data' / 'download_progress_2h.json',
        '4h': root / 'data' / 'download_progress_4h.json',
    }
    interval_stats = []
    for interval, path in data_files.items():
        if path.exists():
            payload = json.loads(path.read_text(encoding='utf-8'))
            interval_stats.append(f"- {interval}: success={len(payload.get('successful_markets', []))}, failed={len(payload.get('failed_markets', []))}")
        else:
            interval_stats.append(f"- {interval}: no progress file")
    lines = [
        '# Alpha Pipeline Hourly Progress',
        '',
        '## System status',
        '- Pipeline code can run locally: yes',
        '- Data download status:',
        *interval_stats,
        f"- Production factors deployed: {len(production.get('factors', []))}",
        f"- Default DeepSeek batch size: {runner.main_config.get('default_batch_size', batch_size)}",
        f"- Transaction fee (bps): {runner.main_config.get('transaction_fee_bps', 5.0)}",
        '',
        '## Current focus',
        '- Prefer DeepSeek idea generation and local hard-rule validation',
        '- Data download work is parked unless future patches require refresh/expansion',
        '- Continue supervised stabilization before fully unattended long-run mode',
        '',
        f'## Latest audit report\n- {report.name}',
    ]
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--batch-size', type=int, default=10)
    args = parser.parse_args()
    print(build_progress_report(Path(args.project_root), batch_size=args.batch_size))


if __name__ == '__main__':
    main()
