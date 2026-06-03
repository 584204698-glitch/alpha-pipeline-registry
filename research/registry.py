"""AI-Trader Registry — shared truth between research and trading Hermes instances.

Files:
  registry/factor_registry.json   — all factors, their status, paper results
  registry/live_config.json       — what the trading Hermes actually reads
  registry/pending_review.json    — research Hermes proposals waiting for human
  registry/approvals/approvals.log — human approval trail (append-only)
  registry/deprecated/            — killed factors

Read/write permissions:
  - research Hermes: write factor_registry (status=paper_candidate), write pending_review
  - human: write factor_registry (status=shadow/tiny_live/live), write live_config
  - trading Hermes: read live_config ONLY
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_STATUSES = (
    "dev",
    "paper_candidate",
    "paper_pass",
    "paper_fail",
    "shadow",
    "tiny_live",
    "live",
    "killed",
)

STATUS_LADDER = {
    "dev": {"promotable_to": ["paper_candidate"]},
    "paper_candidate": {"promotable_to": ["paper_pass", "paper_fail"]},
    "paper_pass": {"promotable_to": ["shadow"], "requires_approval": True},
    "paper_fail": {"promotable_to": ["killed", "dev"]},
    "shadow": {"promotable_to": ["tiny_live", "killed"], "requires_approval": True},
    "tiny_live": {"promotable_to": ["live", "killed"], "requires_approval": True},
    "live": {"promotable_to": ["killed"]},
    "killed": {"promotable_to": ["dev"]},
}


class FactorRegistry:
    """Manage factor lifecycle from dev → paper → shadow → live → killed."""

    def __init__(self, registry_root: Path):
        self.root = Path(registry_root)
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ["factors", "reports", "approvals", "live", "deprecated"]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

        self.registry_path = self.root / "factor_registry.json"
        self.live_config_path = self.root / "live" / "live_config.json"
        self.pending_path = self.root / "pending_review.json"
        self.approvals_path = self.root / "approvals" / "approvals.log"
        self.trade_feedback_path = self.root / "live" / "trade_feedback.jsonl"

        self._ensure_files()

    def _ensure_files(self):
        if not self.registry_path.exists():
            self.registry_path.write_text('{"factors": {}}')
        if not self.live_config_path.exists():
            self._write_live_config({
                "mode": "disabled",
                "enabled_factors": [],
                "max_risk_per_trade": 0.002,
                "allow_ai_override": False,
                "require_risk_check_id": True,
                "block_unapproved_factors": True,
            })
        if not self.pending_path.exists():
            self.pending_path.write_text('{"pending": []}')
        if not self.approvals_path.exists():
            self.approvals_path.write_text("")

    def _write_live_config(self, config: dict):
        self.live_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.live_config_path.write_text(json.dumps(config, indent=2) + "\n")

    def _read_json(self, path: Path) -> dict:
        return json.loads(path.read_text()) if path.exists() else {}

    def _write_json(self, path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    # ── factor_registry.json ──────────────────────────────────

    def get_factor(self, factor_id: str) -> dict | None:
        reg = self._read_json(self.registry_path)
        return reg.get("factors", {}).get(factor_id)

    def list_factors(self, status: str | None = None) -> list[dict]:
        reg = self._read_json(self.registry_path)
        factors = []
        for fid, fdata in reg.get("factors", {}).items():
            if status is None or fdata.get("status") == status:
                factors.append({"factor_id": fid, **fdata})
        return factors

    def register_factor(
        self,
        factor_id: str,
        factor_file: str,
        parameters: dict,
        inputs: list[str],
        rationale: str = "",
        who: str = "research_hermes",
    ) -> dict:
        """Register a new factor or update an existing one. Only allows status transitions."""
        reg = self._read_json(self.registry_path)
        if "factors" not in reg:
            reg["factors"] = {}

        now = datetime.now(timezone.utc).isoformat()

        if factor_id in reg["factors"]:
            return {"error": f"Factor [{factor_id}] already exists. Use update_status()."}

        reg["factors"][factor_id] = {
            "factor_file": factor_file,
            "parameters": parameters,
            "inputs": inputs,
            "rationale": rationale,
            "status": "dev",
            "role": "alpha_signal",
            "registered_at": now,
            "registered_by": who,
            "last_updated": now,
            "history": [{"action": "registered", "who": who, "at": now}],
        }
        self._write_json(self.registry_path, reg)
        return reg["factors"][factor_id]

    def update_status(
        self,
        factor_id: str,
        new_status: str,
        who: str = "research_hermes",
        metadata: dict | None = None,
    ) -> dict:
        """Move factor to a new status. Validates transition."""
        if new_status not in REGISTRY_STATUSES:
            return {"error": f"Invalid status: {new_status}. Valid: {REGISTRY_STATUSES}"}

        reg = self._read_json(self.registry_path)
        if factor_id not in reg.get("factors", {}):
            return {"error": f"Factor [{factor_id}] not found"}

        factor = reg["factors"][factor_id]
        old_status = factor.get("status", "dev")
        allowed = STATUS_LADDER.get(old_status, {}).get("promotable_to", [])
        if new_status not in allowed:
            return {"error": f"Cannot transition [{factor_id}] from {old_status} to {new_status}. Allowed: {allowed}"}

        if STATUS_LADDER.get(new_status, {}).get("requires_approval") and who not in ("human", "admin"):
            return {"error": f"Status [{new_status}] requires human approval."}

        now = datetime.now(timezone.utc).isoformat()
        factor["status"] = new_status
        factor["last_updated"] = now
        factor.setdefault("history", []).append({
            "action": f"status_change:{old_status}→{new_status}",
            "who": who,
            "at": now,
        })
        if metadata:
            factor.update(metadata)

        self._write_json(self.registry_path, reg)

        # Append to approvals
        with open(self.approvals_path, "a") as f:
            f.write(f"{now} {who}: {factor_id} {old_status} → {new_status}\n")

        return factor

    # ── pending_review.json ───────────────────────────────────

    def propose_for_review(
        self,
        factor_id: str,
        paper_metrics: dict,
        recommendation: str = "promote_to_shadow",
        who: str = "research_hermes",
    ) -> dict:
        """Submit a paper-passed factor for human review."""
        pending = self._read_json(self.pending_path)
        now = datetime.now(timezone.utc).isoformat()

        # Remove existing pendings for same factor
        pending["pending"] = [p for p in pending.get("pending", []) if p.get("factor_id") != factor_id]

        proposal = {
            "factor_id": factor_id,
            "status": "paper_pass",
            "paper_pf": paper_metrics.get("pf"),
            "net_pnl": paper_metrics.get("net_pnl"),
            "cost_gross_ratio": paper_metrics.get("cost_gross_ratio"),
            "classification": paper_metrics.get("classification"),
            "recommendation": recommendation,
            "submitted_by": who,
            "submitted_at": now,
            "reviewed": False,
        }
        pending["pending"].append(proposal)
        self._write_json(self.pending_path, pending)
        return proposal

    def get_pending_reviews(self) -> list[dict]:
        return self._read_json(self.pending_path).get("pending", [])

    def mark_reviewed(self, factor_id: str, approved: bool, who: str = "human") -> dict:
        """Human marks a review as done."""
        pending = self._read_json(self.pending_path)
        now = datetime.now(timezone.utc).isoformat()
        for p in pending.get("pending", []):
            if p.get("factor_id") == factor_id and not p.get("reviewed"):
                p["reviewed"] = True
                p["reviewed_by"] = who
                p["reviewed_at"] = now
                p["approved"] = approved
                self._write_json(self.pending_path, pending)

                if approved:
                    self.update_status(factor_id, "shadow", who="human")
                return p
        return {"error": f"No unreviewed proposal for [{factor_id}]"}

    # ── live_config.json ──────────────────────────────────────

    def get_live_config(self) -> dict:
        return self._read_json(self.live_config_path)

    def update_live_config(self, updates: dict, who: str = "human") -> dict:
        """Human-only: update live config."""
        if who not in ("human", "admin"):
            return {"error": "Only human can update live_config."}
        config = self._read_json(self.live_config_path)
        config.update(updates)
        self._write_live_config(config)
        return config

    def enable_factor_live(self, factor_id: str, who: str = "human") -> dict:
        """Human-only: add factor to enabled_factors."""
        if who not in ("human", "admin"):
            return {"error": "Only human can enable live factors."}
        config = self._read_json(self.live_config_path)
        if factor_id not in config.get("enabled_factors", []):
            config.setdefault("enabled_factors", []).append(factor_id)
            self._write_live_config(config)
            self.update_status(factor_id, "live", who="human")
        return config

    def disable_factor_live(self, factor_id: str, who: str = "human") -> dict:
        """Human-only: remove factor from enabled_factors."""
        if who not in ("human", "admin"):
            return {"error": "Only human can disable live factors."}
        config = self._read_json(self.live_config_path)
        config["enabled_factors"] = [f for f in config.get("enabled_factors", []) if f != factor_id]
        self._write_live_config(config)
        return config

    # ── trade_feedback.jsonl ──────────────────────────────────

    def append_trade_feedback(self, feedback: dict) -> None:
        """Append one trade feedback line (trading Hermes writes this)."""
        line = json.dumps(feedback, default=str)
        with open(self.trade_feedback_path, "a") as f:
            f.write(line + "\n")

    def read_trade_feedback(self, tail_lines: int = 200) -> list[dict]:
        if not self.trade_feedback_path.exists():
            return []
        lines = self.trade_feedback_path.read_text().strip().split("\n")
        recent = lines[-tail_lines:]
        result = []
        for line in recent:
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return result

    # ── safety check (called by trading Hermes before any trade) ──

    def is_approved_for_trade(self, factor_id: str) -> tuple[bool, str]:
        """Hard safety check: only enabled_factors in live_config can trade."""
        config = self._read_json(self.live_config_path)
        if config.get("allow_ai_override", False):
            return False, "UNSAFE_CONFIG:allow_ai_override is True"
        if config.get("block_unapproved_factors") is not True:
            return False, "UNSAFE_CONFIG:block_unapproved_factors is not True"
        if factor_id not in config.get("enabled_factors", []):
            return False, f"FACTOR_NOT_APPROVED:{factor_id}"
        return True, "APPROVED"
