from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import requests

from pipeline_utils import get_logger
from idea_registry import filter_duplicate_ideas, load_registry

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"


def _summarize_failures(failed_factors_summary: list[dict[str, Any]] | list[Any]) -> str:
    chunks: list[str] = []
    for item in failed_factors_summary[:10]:
        if isinstance(item, dict):
            name = item.get("factor_name", "unknown")
            reason = item.get("failure_reason") or item.get("verdict") or "unspecified"
            checks = item.get("failed_checks") or []
            stats = item.get("metrics_snapshot") or {}
            detail_bits = []
            if checks:
                detail_bits.append("checks=" + "/".join(str(bit) for bit in checks[:4]))
            if stats:
                stat_parts = []
                for key in ["overall_ic", "out_of_sample_icir", "long_ratio"]:
                    if key in stats:
                        stat_parts.append(f"{key}={stats[key]}")
                if stat_parts:
                    detail_bits.append(",".join(stat_parts))
            detail = f" ({'; '.join(detail_bits)})" if detail_bits else ""
            chunks.append(f"{name}:{reason}{detail}")
        else:
            chunks.append(str(item))
    return " | ".join(chunks) if chunks else "none"


def _load_registry_quality_context(project_root: Path, limit: int = 12) -> dict[str, Any]:
    registry = load_registry(project_root)
    ideas = registry.get("ideas", []) if isinstance(registry, dict) else []
    recent = ideas[-limit:]
    status_counts = Counter(str(item.get("status", "unknown")) for item in ideas)
    reason_counts = Counter(str(item.get("reason", "")) for item in ideas if item.get("reason"))
    duplicate_families = [
        {
            "factor_name": item.get("factor_name", "unknown"),
            "formula": item.get("formula", ""),
            "reason": item.get("reason", ""),
        }
        for item in recent
        if item.get("status") in {"failed", "queued", "deployed"}
    ]
    return {
        "registry_entries": len(ideas),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "recent_family_examples": duplicate_families,
    }


def _quality_guardrails_text(project_root: Path) -> str:
    context = _load_registry_quality_context(project_root)
    recent = context.get("recent_family_examples", [])[:8]
    examples = []
    for item in recent:
        name = item.get("factor_name", "unknown")
        formula = item.get("formula", "")
        reason = item.get("reason", "")
        if formula:
            examples.append(f"{name}:{formula} [{reason}]".strip())
        else:
            examples.append(f"{name} [{reason}]".strip())
    examples_text = " | ".join(examples) if examples else "none"
    return (
        f"Historical family registry size={context.get('registry_entries', 0)}; "
        f"status_counts={_safe_repr(context.get('status_counts', {}))}; "
        f"reason_counts={_safe_repr(context.get('reason_counts', {}))}. "
        "Hard originality rule: if a candidate only changes factor_name, lag/window, threshold, sign flip, timeframe order, "
        "or variable ordering while keeping the same interaction structure, it will be rejected as duplicate_family. "
        "Do not resubmit these family examples or near-clones: "
        f"{examples_text}. "
        "Hard quality rule: avoid candidates likely to fail on OOS consistency, parameter stability, regime invariance, correlation limit, or signal symmetry. "
        "Before returning an idea, self-reject it unless the interaction structure is materially different from these prior families and the signal is plausibly two-sided and regime-robust."
    )


def _safe_repr(obj: Any) -> str:
    text = str(obj)
    text = text.replace("{", "{{").replace("}", "}}")
    return text


def _slug_to_class_name(name: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", name)
    return "".join(part.capitalize() for part in parts if part)


def _make_idea(index: int, failure_context: str) -> dict[str, Any]:
    templates = [
        {
            "factor_name": f"OI_Funding_Divergence_{index}",
            "rationale": f"Avoid prior failed motifs ({failure_context}) by combining OI acceleration with funding stress.",
            "timeframes": ["15m", "2h"],
            "inputs": ["open_interest", "funding_rate", "close"],
            "mathematical_formula": "zscore(delta(open_interest, oi_lookback), window) * indicator(funding_rate < rolling_quantile(funding_rate, quantile_window, funding_quantile))",
            "parameters": {"oi_lookback": 4, "window": 12, "quantile_window": 24, "funding_quantile": 0.2},
        },
        {
            "factor_name": f"Taker_Imbalance_Reversion_{index}",
            "rationale": f"Counteracts unstable directional bias seen in ({failure_context}) by fading extreme taker flow imbalances.",
            "timeframes": ["15m", "4h"],
            "inputs": ["taker_volume", "volume", "close"],
            "mathematical_formula": "-1 * zscore((taker_volume / (volume + 1e-9)) - rolling_mean(taker_volume / (volume + 1e-9), window), window)",
            "parameters": {"window": 16},
        },
        {
            "factor_name": f"VolAdj_Momentum_Pressure_{index}",
            "rationale": f"Uses volatility-adjusted momentum to reduce regime fragility observed in ({failure_context}).",
            "timeframes": ["15m", "2h", "4h"],
            "inputs": ["close", "volume", "open_interest"],
            "mathematical_formula": "zscore(pct_change(close, lag), window) * zscore(delta(open_interest, lag), window) - zscore(volume, window)",
            "parameters": {"lag": 6, "window": 18},
        },
    ]
    idea = templates[(index - 1) % len(templates)].copy()
    idea["class_name"] = _slug_to_class_name(idea["factor_name"])
    return idea


def _local_ideas(failure_context: str, batch_size: int) -> list[dict[str, Any]]:
    return [_make_idea(i + 1, failure_context) for i in range(max(1, batch_size))]


def _balanced_json_object(raw: str) -> str:
    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON object start found in DeepSeek response")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw)):
        char = raw[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return raw[start:]


def _coerce_pythonish_json(raw: str) -> str:
    patched = raw
    patched = re.sub(r"\bTrue\b", "true", patched)
    patched = re.sub(r"\bFalse\b", "false", patched)
    patched = re.sub(r"\bNone\b", "null", patched)
    patched = re.sub(r",\s*([}\]])", r"\1", patched)
    return patched


def _close_open_json(raw: str) -> str:
    depth_curly = 0
    depth_square = 0
    in_string = False
    escape = False
    for char in raw:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth_curly += 1
        elif char == "}":
            depth_curly = max(0, depth_curly - 1)
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square = max(0, depth_square - 1)
    return raw + ("]" * depth_square) + ("}" * depth_curly)


def _parse_json_lenient(raw: str) -> dict[str, Any]:
    candidates = [
        raw,
        _coerce_pythonish_json(raw),
        _close_open_json(_coerce_pythonish_json(raw)),
    ]
    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                raise ValueError("DeepSeek response is not a JSON object")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if last_exc:
        raise last_exc
    raise ValueError("Unable to parse DeepSeek payload")


def _extract_json_payload(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S)
        if match:
            raw = match.group(1).strip()

    try:
        return _parse_json_lenient(raw)
    except Exception:
        pass

    balanced = _balanced_json_object(raw)
    try:
        return _parse_json_lenient(balanced)
    except Exception:
        pass

    match = re.search(r"(\{.*)", raw, flags=re.S)
    if match:
        snippet = _close_open_json(_coerce_pythonish_json(match.group(1)))
        try:
            return _parse_json_lenient(snippet)
        except Exception:
            pass

    python_obj_match = re.search(r"(\{.*\})", raw, flags=re.S)
    if python_obj_match:
        try:
            payload = ast.literal_eval(python_obj_match.group(1))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    raise ValueError("DeepSeek response is not recoverable as JSON")


def _extract_field(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return match.group(1).strip().strip('"')
    return ""


def _extract_list_field(name: str, text: str, default: list[str]) -> list[str]:
    bracket_match = re.search(rf'{name}\s*[:=]\s*\[(.*?)\]', text, flags=re.I | re.S)
    if bracket_match:
        items = re.findall(r'"([^"]+)"|\'([^\']+)\'|([A-Za-z0-9_]+)', bracket_match.group(1))
        flattened = [next(part for part in item if part) for item in items if any(item)]
        return flattened or default
    value = _extract_field([rf'{name}\s*[:=]\s*([^\n]+)'], text)
    if value:
        parts = [part.strip().strip('"') for part in re.split(r'[,/|]', value) if part.strip()]
        return parts or default
    return default


def _extract_parameters(text: str) -> dict[str, Any]:
    match = re.search(r'parameters\s*[:=]\s*(\{.*?\})', text, flags=re.I | re.S)
    if match:
        raw = _coerce_pythonish_json(_close_open_json(match.group(1)))
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    params: dict[str, Any] = {}
    for key, value in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(-?\d+(?:\.\d+)?)', text):
        params[key] = float(value) if "." in value else int(value)
    return params or {"lag": 4, "window": 12}


def _extract_formula_from_text(text: str) -> str:
    formula = _extract_field(
        [
            r'mathematical_formula\s*[:=]\s*"([^"]+)"',
            r'mathematical_formula\s*[:=]\s*([^\n]+)',
            r'formula\s*[:=]\s*"([^"]+)"',
            r'formula\s*[:=]\s*([^\n]+)',
        ],
        text,
    )
    return formula or "zscore(pct_change(close, 4), 12)"


def _recover_ideas_from_text(text: str, batch_size: int, failure_context: str) -> list[dict[str, Any]]:
    blocks: list[str] = []
    matches = list(re.finditer(r'(?im)^\s*factor_name\s*[:=]', text))
    if matches:
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            blocks.append(text[start:end].strip())
    else:
        blocks = [text]

    recovered: list[dict[str, Any]] = []
    for index, block in enumerate(blocks[: max(1, batch_size)], start=1):
        factor_name = _extract_field(
            [r'factor_name\s*[:=]\s*"([^"]+)"', r'factor_name\s*[:=]\s*([^\n]+)'],
            block,
        ) or f"RecoveredDeepSeekIdea{index}"
        formula = _extract_formula_from_text(block)
        if not _is_safe_expression(formula):
            continue
        rationale = _extract_field(
            [r'rationale\s*[:=]\s*"([^"]+)"', r'rationale\s*[:=]\s*([^\n]+)'],
            block,
        ) or f"Recovered from semi-structured DeepSeek output with failure context: {failure_context}"
        if factor_name.strip().lower() == 'factorname':
            continue
        idea = {
            "factor_name": factor_name.strip(),
            "rationale": rationale.strip(),
            "timeframes": _extract_list_field("timeframes", block, ["15m", "2h", "4h"]),
            "inputs": _extract_list_field("inputs", block, ["close", "volume", "open_interest"]),
            "mathematical_formula": formula,
            "parameters": _extract_parameters(block),
        }
        idea["class_name"] = _slug_to_class_name(idea["factor_name"])
        recovered.append(idea)
    if not recovered:
        raise ValueError("DeepSeek semi-structured text did not contain recoverable safe ideas")
    return recovered[: max(1, batch_size)]


def _is_safe_expression(formula: str) -> bool:
    stripped = formula.strip()
    if not stripped:
        return False
    if any(token in stripped for token in [";", "\n", "="]):
        return False
    lowered = stripped.lower()
    unsupported_tokens = ["ema(", "sma(", "roc("]
    if any(token in lowered for token in unsupported_tokens):
        return False
    safe_names = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "taker_volume",
        "funding_rate",
        "open_interest",
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
        "np",
    }
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", stripped))
    for name in names:
        if name in {"True", "False"}:
            continue
        if name in safe_names:
            continue
        if name in {"e", "pi"}:
            continue
        return False
    return True


def _normalize_ideas(payload: dict[str, Any], batch_size: int, failure_context: str) -> list[dict[str, Any]]:
    ideas = payload.get("ideas")
    if not isinstance(ideas, list) or not ideas:
        raise ValueError("DeepSeek payload missing non-empty ideas list")
    normalized: list[dict[str, Any]] = []
    for index, idea in enumerate(ideas[: max(1, batch_size)], start=1):
        if not isinstance(idea, dict):
            continue
        factor_name = idea.get("factor_name") or f"DeepSeekIdea{index}"
        if str(factor_name).strip().lower() == "factorname":
            continue
        formula = idea.get("mathematical_formula") or idea.get("formula") or "zscore(pct_change(close, 4), 12)"
        if not _is_safe_expression(str(formula)):
            raise ValueError(f"Unsafe or unsupported DeepSeek formula for [{factor_name}]: {formula}")
        timeframes = idea.get("timeframes") or ["15m", "1h", "4h"]
        if isinstance(timeframes, str):
            timeframes = [part.strip() for part in re.split(r"[,/]", timeframes) if part.strip()]
        inputs = idea.get("inputs") or idea.get("data_sources") or ["close", "volume"]
        if isinstance(inputs, str):
            inputs = [part.strip() for part in re.split(r"[,/]", inputs) if part.strip()]
        normalized_idea = {
            "factor_name": factor_name,
            "rationale": idea.get("rationale") or idea.get("financial_logic") or f"Generated from failure context: {failure_context}",
            "hypothesis": idea.get("hypothesis", "unspecified"),
            "observation": idea.get("observation", "unspecified"),
            "novelty_claim": idea.get("novelty_claim", "unspecified"),
            "risk_notes": idea.get("risk_notes") or idea.get("self_critique", "unspecified"),
            "timeframes": timeframes,
            "inputs": inputs,
            "mathematical_formula": formula,
            "parameters": idea.get("parameters") or {"lag": 4, "window": 12},
            "expected_direction": idea.get("expected_direction", "unspecified"),
            "interaction_terms": idea.get("interaction_terms", "unspecified"),
            "financial_logic": idea.get("financial_logic", "statistical pattern driven; explanation pending"),
            "source_of_inspiration": idea.get("source_of_inspiration", "deepseek_generated"),
            "self_critique": idea.get("self_critique", idea.get("risk_notes", "unspecified")),
        }
        normalized_idea["class_name"] = idea.get("class_name") or _slug_to_class_name(factor_name)
        normalized.append(normalized_idea)
    if not normalized:
        raise ValueError("DeepSeek payload could not be normalized into ideas")
    return normalized


def _deepseek_prompt(failure_context: str, batch_size: int, quality_guardrails: str) -> str:
    schema_example = json.dumps(
        {
            "ideas": [
                {
                    "factor_name": "TakerCrowdingReversal",
                    "rationale": "one sentence",
                    "hypothesis": "NL description of what drives predictability",
                    "observation": "what data pattern motivates this",
                    "novelty_claim": "why different from prior families",
                    "risk_notes": "when it would fail",
                    "timeframes": ["15m", "1h", "4h"],
                    "inputs": ["close", "taker_volume", "open_interest"],
                    "mathematical_formula": "zscore(delta(taker_volume,3),12) * indicator(close > rolling_mean(close,24)) * -1",
                    "parameters": {"lag": 3, "window": 12, "trend_window": 24},
                }
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"PRIMARY CONSTRAINT: {quality_guardrails}\n"
        f"FAILED PATTERNS TO AVOID: {failure_context}\n\n"
        f"TASK: Generate exactly {batch_size} novel crypto factor ideas as one JSON object.\n\n"
        "RULES:\n"
        "- Only use: open,high,low,close,volume,taker_volume,funding_rate,open_interest and their transformations.\n"
        "- Operators: delta,pct_change,rolling_mean/std/max/min/sum/quantile,zscore,rank_pct(w),indicator,clip,abs,sign.\n"
        "- No ema,sma,roc,assignments,semicolons,multiline,negative-shifts,future timestamps,look-ahead logic.\n"
        "- Every formula must be a single expression. rank_pct always needs a window argument.\n"
        "- Use 2-3 nonlinear interactions across different data sources, not single-indicator factors.\n"
        "- Each factor must plausibly produce signals that pass: OOS consistency (IC>0 out-of-sample), parameter stability (±30%), two-sided signal symmetry (30-70% long), regime robustness (IC>0 in ≥2/3 volatility regimes).\n"
        "- Never resubmit a near-clone of a prior failed family. Change the interaction structure, not just parameters.\n\n"
        "RETURN exactly this shape, NO markdown, NO text outside JSON:\n"
        f"{schema_example}"
    )


def _message_text(message: dict[str, Any]) -> str:
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    if content:
        return content
    # DeepSeek sometimes returns empty content with the real JSON in reasoning_content
    if reasoning:
        return reasoning
    return ""


def _fetch_from_deepseek(api_key: str, failure_context: str, quality_guardrails: str, batch_size: int, logger) -> dict[str, Any]:
    payload = {
        "model": DEFAULT_MODEL,
        "temperature": 0.7,
        "max_tokens": 8192,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a quantitative alpha researcher. You generate novel, testable factor ideas for crypto perpetual futures. "
                    "Respond ONLY with a single compact JSON object. No markdown, no explanation outside JSON."
                ),
            },
            {
                "role": "user",
                "content": _deepseek_prompt(failure_context, batch_size, quality_guardrails),
            },
        ],
    }
    response = requests.post(
        DEEPSEEK_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    content = _message_text(message)
    # Log first 2000 chars of DeepSeek raw response for debugging
    logger.info(f"DeepSeek raw response (first 4000 chars): {str(content)[:4000]}")
    try:
        parsed = _extract_json_payload(content)
        ideas = _normalize_ideas(parsed, batch_size=batch_size, failure_context=failure_context)
    except Exception:
        logger.warning(f"DeepSeek JSON parse failed for raw content; falling back to recovery")
        ideas = _recover_ideas_from_text(content, batch_size=batch_size, failure_context=failure_context)
    logger.info(f"Generated {len(ideas)} DeepSeek factor ideas using failed summary context: {failure_context}")
    return {"provider": "deepseek", "used_api_key": True, "ideas": ideas}


def fetch_factor_ideas(api_key: str, failed_factors_summary: list, batch_size: int = 5) -> dict:
    """
    向 DeepSeek Pro 发送请求，输入失败因子特征，
    返回符合指定 JSON Schema 的批量因子创意数据。
    """
    project_root = Path(__file__).resolve().parent
    logger = get_logger(project_root, "IDEA_GENERATOR")
    failure_context = _summarize_failures(failed_factors_summary)
    quality_guardrails = _quality_guardrails_text(project_root)
    if api_key:
        try:
            payload = _fetch_from_deepseek(api_key, failure_context, quality_guardrails, batch_size, logger)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"DeepSeek request failed, falling back to local bootstrap ideas: {exc}")
            payload = {"provider": "local_bootstrap", "used_api_key": bool(api_key), "ideas": _local_ideas(failure_context, batch_size)}
    else:
        payload = {"provider": "local_bootstrap", "used_api_key": bool(api_key), "ideas": _local_ideas(failure_context, batch_size)}

    registry = load_registry(project_root)
    fresh_ideas, skipped_duplicates = filter_duplicate_ideas(list(payload.get("ideas", [])), registry)
    if skipped_duplicates:
        logger.info(
            f"Idea generator skipped {len(skipped_duplicates)} duplicate families before handoff: "
            + ", ".join(item.get("factor_name", "unknown") for item in skipped_duplicates)
        )
    payload["ideas"] = fresh_ideas
    payload["skipped_duplicates"] = skipped_duplicates
    logger.info(f"Generated {len(fresh_ideas)} usable factor ideas using failed summary context: {failure_context}")
    return payload
