from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest_engine import BacktestEngine
from factors import load_factor_from_path
from pipeline_utils import get_logger, read_json, safe_float, utc_now_z


class HealthChecker:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.data_path = self.project_root / "data" / "data_storage.parquet"
        self.production_path = self.project_root / "config" / "production_factors.json"
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.project_root, "HEALTH_CHECK")

    def _data_freshness(self) -> dict:
        if not self.data_path.exists():
            return {"exists": False, "last_modified": None, "age_hours": None}
        modified = datetime.fromtimestamp(self.data_path.stat().st_mtime, tz=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - modified).total_seconds() / 3600
        return {"exists": True, "last_modified": modified.isoformat(), "age_hours": round(age_hours, 2)}

    def _resource_audit(self) -> dict:
        meminfo = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meminfo[key.strip()] = value.strip()
        total_kb = float(meminfo.get("MemTotal", "0 kB").split()[0])
        avail_kb = float(meminfo.get("MemAvailable", "0 kB").split()[0])
        used_gb = max(total_kb - avail_kb, 0) / 1024 / 1024
        warn = used_gb >= 100.0
        if warn:
            self.logger.warning(f"Memory usage warning: estimated used memory {used_gb:.2f} GB is near threshold.")
        return {"used_gb_estimate": round(used_gb, 2), "warning": warn}

    def _deployed_last_24h(self) -> int:
        payload = read_json(self.production_path, default={"factors": []}) or {"factors": []}
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        count = 0
        for item in payload.get("factors", []):
            ts = item.get("deployed_at")
            if not ts:
                continue
            try:
                deployed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
            if deployed >= cutoff:
                count += 1
        return count

    def _recent_performance(self) -> list[dict]:
        payload = read_json(self.production_path, default={"factors": []}) or {"factors": []}
        if not self.data_path.exists():
            return []
        data = pd.read_parquet(self.data_path)
        if not isinstance(data.index, pd.MultiIndex):
            data = data.set_index(["timestamp", "symbol"]).sort_index()
        cutoff = sorted(data.index.get_level_values("timestamp").unique())[-24:]
        mask = data.index.get_level_values("timestamp").isin(cutoff)
        recent = data.loc[mask]
        forward = BacktestEngine._forward_returns(recent)
        results: list[dict] = []
        for item in payload.get("factors", []):
            path = self.project_root / item.get("factor_file", "")
            if not path.exists():
                continue
            loaded = load_factor_from_path(path)
            signal = loaded.factor.compute(recent).reindex(recent.index).fillna(0.0)
            pnl = safe_float((signal * forward).mean())
            results.append({"factor_name": item.get("factor_name", path.stem), "recent_mean_signal_pnl": pnl})
        return results

    def run_once(self) -> Path:
        payload = {
            "timestamp": utc_now_z(),
            "data_freshness": self._data_freshness(),
            "resource_audit": self._resource_audit(),
            "deployed_last_24h": self._deployed_last_24h(),
            "recent_performance": self._recent_performance(),
        }
        report_lines = [
            "# Alpha Pipeline Audit Report",
            "",
            f"- Timestamp: {payload['timestamp']}",
            f"- Data freshness: {json.dumps(payload['data_freshness'], ensure_ascii=False)}",
            f"- Resource audit: {json.dumps(payload['resource_audit'], ensure_ascii=False)}",
            f"- Deployed factors in last 24h: {payload['deployed_last_24h']}",
            "- Recent factor performance:",
        ]
        if payload["recent_performance"]:
            report_lines.extend(
                [f"  - {item['factor_name']}: {item['recent_mean_signal_pnl']:.6f}" for item in payload["recent_performance"]]
            )
        else:
            report_lines.append("  - none")

        report_path = self.logs_dir / f"audit_report_{datetime.now().strftime('%Y%m%dT%H%M%S')}.md"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        self.logger.info(f"Wrote audit report [{report_path.name}]")
        return report_path


def run_health_check_once(project_root: Path) -> Path:
    return HealthChecker(project_root).run_once()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    run_health_check_once(root)
