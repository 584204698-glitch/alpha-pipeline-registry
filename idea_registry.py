from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

from pipeline_utils import read_json, write_json

_FUNCTION_NAMES = {
    "delta",
    "pct_change",
    "rolling_mean",
    "rolling_std",
    "rolling_max",
    "rolling_min",
    "rolling_sum",
    "rolling_quantile",
    "zscore",
    "rank_pct",
    "indicator",
    "clip",
    "abs",
    "sign",
}
_SERIES_NAMES = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "taker_volume",
    "funding_rate",
    "open_interest",
}
_COMMUTATIVE_BINOPS = {ast.Mult, ast.Add}
_COMPARATORS = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}


def _canonicalize(node: ast.AST) -> str:
    if isinstance(node, ast.Expression):
        return _canonicalize(node.body)
    if isinstance(node, ast.Name):
        if node.id in _SERIES_NAMES:
            return f"series:{node.id}"
        return "param"
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return "const:num"
        if isinstance(node.value, str):
            return "const:str"
        return f"const:{type(node.value).__name__}"
    if isinstance(node, ast.UnaryOp):
        op = type(node.op).__name__
        return f"{op}({_canonicalize(node.operand)})"
    if isinstance(node, ast.BoolOp):
        op = type(node.op).__name__
        values = sorted(_canonicalize(value) for value in node.values)
        return f"{op}({','.join(values)})"
    if isinstance(node, ast.BinOp):
        op = type(node.op).__name__
        left = _canonicalize(node.left)
        right = _canonicalize(node.right)
        if type(node.op) in _COMMUTATIVE_BINOPS:
            left, right = sorted([left, right])
        return f"{op}({left},{right})"
    if isinstance(node, ast.Compare):
        left = _canonicalize(node.left)
        parts: list[str] = []
        current_left = left
        for op, comparator in zip(node.ops, node.comparators):
            right = _canonicalize(comparator)
            symbol = _COMPARATORS.get(type(op), type(op).__name__)
            parts.append(f"cmp({current_left},{symbol},{right})")
            current_left = right
        return "&".join(parts)
    if isinstance(node, ast.Call):
        func_name = _canonicalize(node.func)
        positional = [_canonicalize(arg) for arg in node.args]
        keyword_pairs = sorted((kw.arg or "", _canonicalize(kw.value)) for kw in node.keywords)
        kwargs = [f"{name}={value}" for name, value in keyword_pairs]
        return f"call({func_name}|{','.join(positional + kwargs)})"
    if isinstance(node, ast.Attribute):
        return f"attr({_canonicalize(node.value)}.{node.attr})"
    if isinstance(node, ast.Subscript):
        return f"sub({_canonicalize(node.value)}[{_canonicalize(node.slice)}])"
    if isinstance(node, ast.Tuple):
        return f"tuple({','.join(_canonicalize(item) for item in node.elts)})"
    if isinstance(node, ast.List):
        return f"list({','.join(_canonicalize(item) for item in node.elts)})"
    return type(node).__name__


def _formula_signature(formula: str) -> str:
    parsed = ast.parse(formula or "0", mode="eval")
    return _canonicalize(parsed)


def idea_family_fingerprint(idea: dict[str, Any]) -> str:
    inputs = sorted(str(item).lower() for item in idea.get("inputs", []))
    timeframes = sorted(str(item).lower() for item in idea.get("timeframes", []))
    signature = _formula_signature(str(idea.get("mathematical_formula", "")))
    fingerprint_material = {
        "inputs": inputs,
        "timeframes": timeframes,
        "signature": signature,
    }
    payload = repr(fingerprint_material).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def registry_path(project_root: Path) -> str:
    return str(Path(project_root) / "logs" / "tested_idea_families.json")


def load_registry(project_root: Path) -> dict[str, Any]:
    return read_json(registry_path(project_root), default={"ideas": []}) or {"ideas": []}


def save_registry(project_root: Path, registry: dict[str, Any]) -> None:
    write_json(registry_path(project_root), registry)


def filter_duplicate_ideas(ideas: list[dict[str, Any]], registry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen = {item.get("family_fingerprint") for item in registry.get("ideas", []) if item.get("family_fingerprint")}
    fresh: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    batch_seen = set(seen)
    for idea in ideas:
        fingerprint = idea_family_fingerprint(idea)
        idea["family_fingerprint"] = fingerprint
        if fingerprint in batch_seen:
            skipped.append(
                {
                    "factor_name": idea.get("factor_name", "unknown"),
                    "reason": "duplicate_family",
                    "family_fingerprint": fingerprint,
                }
            )
            continue
        batch_seen.add(fingerprint)
        fresh.append(idea)
    return fresh, skipped


def append_registry_entries(project_root: Path, ideas: list[dict[str, Any]], status: str, reason: str = "") -> None:
    registry = load_registry(project_root)
    entries = registry.setdefault("ideas", [])
    existing = {item.get("family_fingerprint") for item in entries}
    for idea in ideas:
        fingerprint = idea.get("family_fingerprint") or idea_family_fingerprint(idea)
        if fingerprint in existing:
            continue
        entries.append(
            {
                "factor_name": idea.get("factor_name", "unknown"),
                "family_fingerprint": fingerprint,
                "status": status,
                "reason": reason,
                "formula": idea.get("mathematical_formula", ""),
                "inputs": idea.get("inputs", []),
                "timeframes": idea.get("timeframes", []),
            }
        )
        existing.add(fingerprint)
    save_registry(project_root, registry)
