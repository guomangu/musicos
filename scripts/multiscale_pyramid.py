#!/usr/bin/env python3
"""
Bottom-Up Deviation Pyramid (`multiscale_pyramid.py`, Phase 6 §2.2 & §4.1).

Aggregates H x V scale scores (`S1_physical` -> `S5_context`), enforces cascade
blocking (if S1 or S2 fails blocking, S4 & S5 remain None), writes full matrices
and raw diagnostics to `<project>/renders/diagnostics/`, and guarantees the
LLM-facing JSON payload never exceeds 1,500 characters (<= 400 tokens).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

SCALE_ORDER: tuple[str, ...] = (
    "S1_physical",
    "S2_acoustic",
    "S3_syntax",
    "S4_form",
    "S5_context",
)
_SCALE_PRIORITY = {s: idx for idx, s in enumerate(SCALE_ORDER)}
MAX_COMPACT_CHARS = 1500

# Phase 8 Q7 — new deterministic codes must be visible in the Top-3 deviation pyramid.
_PHASE8_CODE_PRIORITY: dict[str, float] = {
    "ADJUST_TRACK_FADER": 0.20,
    "FIX_STEREO_BALANCE": 0.18,
    "WIDEN_HARMONY_OR_PERC": 0.14,
    "NARROW_EXCESSIVE_SIDE": 0.10,
}


def _extract_action_code(next_action: str) -> str:
    m = re.match(r"\s*([A-Z][A-Z0-9_]{2,})", str(next_action or ""))
    return m.group(1) if m else ""


def _safe_stage_slug(stage: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(stage or "AUDIT").strip()).strip("_") or "AUDIT"


def _round_score(val: float | None) -> float | None:
    if val is None:
        return None
    return round(max(0.0, min(1.0, float(val))), 2)


def _truncate_str(text: str, max_len: int) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    return s[: max_len - 3].rstrip() + "..."


def _normalize_deviation(raw: dict[str, Any]) -> dict[str, Any]:
    scale = str(raw.get("scale") or "S1_physical")
    if scale not in _SCALE_PRIORITY:
        scale = "S1_physical"
    axis = str(raw.get("axis") or "V").upper()
    if axis not in ("V", "H", "HV", "H*V"):
        axis = "V"
    return {
        "scale": scale,
        "axis": axis,
        "location": _truncate_str(str(raw.get("location") or "project"), 48),
        "metric": _truncate_str(str(raw.get("metric") or ""), 56),
        "diagnosis": _truncate_str(str(raw.get("diagnosis") or ""), 110),
        "next_action": _truncate_str(str(raw.get("next_action") or "REVIEW"), 68),
        "code": _extract_action_code(str(raw.get("next_action") or "")),
        "blocking": bool(raw.get("blocking", True)),
        "severity": float(raw.get("severity", 0.5)),
    }


def _compact_to_budget(payload: dict[str, Any], max_chars: int = MAX_COMPACT_CHARS) -> dict[str, Any]:
    """Deterministically enforce `len(json.dumps(payload)) <= max_chars` (<= 400 tokens)."""
    def _serialized_len(obj: dict[str, Any]) -> int:
        # Check ASCII-escaped json.dumps length (worst case >= ensure_ascii=False)
        return max(
            len(json.dumps(obj)),
            len(json.dumps(obj, ensure_ascii=False)),
        )

    out = json.loads(json.dumps(payload, ensure_ascii=False))
    if _serialized_len(out) <= max_chars:
        return out

    # Pass 1: shorten diagnosis / location / metric
    for dev in out.get("top_deviations", []):
        dev["diagnosis"] = _truncate_str(dev.get("diagnosis", ""), 70)
        dev["location"] = _truncate_str(dev.get("location", ""), 32)
        dev["metric"] = _truncate_str(dev.get("metric", ""), 40)
    if _serialized_len(out) <= max_chars:
        return out

    # Pass 2: cap autofix_applied list
    fixes = out.get("autofix_applied") or []
    if len(fixes) > 2:
        out["autofix_applied"] = fixes[:2] + [f"+{len(fixes) - 2}"]
    if _serialized_len(out) <= max_chars:
        return out

    # Pass 3: tighten diagnosis further or reduce top_deviations count if still over budget
    for max_d_len in (45, 28):
        for dev in out.get("top_deviations", []):
            dev["diagnosis"] = _truncate_str(dev.get("diagnosis", ""), max_d_len)
        if _serialized_len(out) <= max_chars:
            return out

    while len(out.get("top_deviations", [])) > 1 and _serialized_len(out) > max_chars:
        out["top_deviations"].pop()
    if _serialized_len(out) <= max_chars:
        return out

    # Final fallback: shorten path & single deviation strings
    if len(str(out.get("full_report_path", ""))) > 48:
        out["full_report_path"] = "..." + str(out["full_report_path"])[-45:]
    for dev in out.get("top_deviations", []):
        dev["diagnosis"] = _truncate_str(dev.get("diagnosis", ""), 20)
        dev["next_action"] = _truncate_str(dev.get("next_action", ""), 36)
    return out


def build_multiscale_pyramid(
    *,
    stage: str,
    scale_matrix: dict[str, dict[str, float | None]],
    deviations: list[dict[str, Any]] | None = None,
    autofix_applied: list[str] | None = None,
    project_dir: str = "",
    full_details: dict[str, Any] | None = None,
    report_filename: str = "",
    cascade_threshold: float = 0.65,
    max_compact_chars: int = MAX_COMPACT_CHARS,
) -> dict[str, Any]:
    """
    Build the bounded Bottom-Up Deviation Pyramid (`<= 400 tokens / <= 1,500 chars`).

    Cascade blocking rule (Phase 6 §2.2):
      If `S1_physical` or `S2_acoustic` has a blocking failure (score < `cascade_threshold`
      or an explicit blocking deviation), `S4_form` and `S5_context` are forced to
      `{"V": None, "H": None}` and omitted from `top_deviations`.
    """
    norm_matrix: dict[str, dict[str, float | None]] = {}
    for s in SCALE_ORDER:
        entry = (scale_matrix or {}).get(s)
        if isinstance(entry, (int, float)):
            norm_matrix[s] = {"V": _round_score(entry), "H": None}
        elif isinstance(entry, dict):
            norm_matrix[s] = {
                "V": _round_score(entry.get("V")),
                "H": _round_score(entry.get("H")),
            }
        else:
            norm_matrix[s] = {"V": None, "H": None}

    raw_devs = [_normalize_deviation(d) for d in (deviations or [])]

    def _scale_failed_blocking(scale_name: str) -> bool:
        sc = norm_matrix[scale_name]
        v_fail = sc["V"] is not None and sc["V"] < cascade_threshold
        h_fail = sc["H"] is not None and sc["H"] < cascade_threshold
        dev_fail = any(d["scale"] == scale_name and d["blocking"] for d in raw_devs)
        return v_fail or h_fail or dev_fail

    s1_blocked = _scale_failed_blocking("S1_physical")
    s2_blocked = _scale_failed_blocking("S2_acoustic")
    cascade_blocked = s1_blocked or s2_blocked

    if cascade_blocked:
        norm_matrix["S4_form"] = {"V": None, "H": None}
        norm_matrix["S5_context"] = {"V": None, "H": None}
        filtered_devs = [d for d in raw_devs if d["scale"] not in ("S4_form", "S5_context")]
    else:
        filtered_devs = list(raw_devs)

    # Sort deviations: S1 -> S5, blocking first, severity descending
    filtered_devs.sort(
        key=lambda d: (
            _SCALE_PRIORITY.get(d["scale"], 99),
            0 if d["blocking"] else 1,
            -(d["severity"] + _PHASE8_CODE_PRIORITY.get(d.get("code", ""), 0.0)),
        )
    )

    # Determine blocking scale & verdict
    blocking_scale: str | None = None
    for s in SCALE_ORDER:
        if norm_matrix[s]["V"] is None and norm_matrix[s]["H"] is None:
            continue
        if _scale_failed_blocking(s):
            blocking_scale = s
            break

    active_vals = [
        v
        for s in SCALE_ORDER
        for v in (norm_matrix[s]["V"], norm_matrix[s]["H"])
        if v is not None
    ]
    global_score = round(sum(active_vals) / len(active_vals), 2) if active_vals else 1.0

    has_swap = any("SWAP" in d["next_action"].upper() for d in filtered_devs[:3])
    if blocking_scale is None and not any(d["blocking"] for d in filtered_devs):
        verdict = "PASS"
    elif has_swap or global_score < 0.40:
        verdict = "SWAP"
    else:
        verdict = "FIX"

    top_devs: list[dict[str, Any]] = []
    for idx, d in enumerate(filtered_devs[:3], start=1):
        top_devs.append({
            "priority": idx,
            "scale": d["scale"],
            "axis": d["axis"],
            "location": d["location"],
            "metric": d["metric"],
            "diagnosis": d["diagnosis"],
            "next_action": d["next_action"],
            "code": d.get("code") or _extract_action_code(d["next_action"]),
        })

    # Write full diagnostic report to disk
    slug = _safe_stage_slug(stage)
    fname = report_filename or f"multiscale_{slug}.json"
    rel_report_path = os.path.join("renders", "diagnostics", fname)
    full_report_abs = ""
    if project_dir and str(project_dir).strip():
        abs_proj = os.path.abspath(os.path.expanduser(project_dir))
        full_report_abs = os.path.join(abs_proj, rel_report_path)
        os.makedirs(os.path.dirname(full_report_abs), exist_ok=True)
        disk_payload = {
            "stage": stage,
            "verdict": verdict,
            "blocking_scale": blocking_scale,
            "cascade_blocked_upper_scales": cascade_blocked,
            "global_alignment_score": global_score,
            "scale_matrix": norm_matrix,
            "autofix_applied": list(autofix_applied or []),
            "all_deviations": filtered_devs,
            "details": full_details or {},
        }
        with open(full_report_abs, "w", encoding="utf-8") as fh:
            json.dump(disk_payload, fh, indent=2, ensure_ascii=False)
        out_report_ref = rel_report_path
    else:
        out_report_ref = rel_report_path

    compact: dict[str, Any] = {
        "stage": str(stage),
        "verdict": verdict,
        "blocking_scale": blocking_scale,
        "global_alignment_score": global_score,
        "scale_matrix": norm_matrix,
        "autofix_applied": [str(x) for x in (autofix_applied or [])],
        "top_deviations": top_devs,
        "full_report_path": out_report_ref,
    }
    compact = _compact_to_budget(compact, max_chars=max_compact_chars)

    # Mandatory invariant check (Phase 6 §1.2 & §7.1)
    dumped = json.dumps(compact)
    assert len(dumped) <= max_compact_chars, (
        f"Pyramid compact output exceeded {max_compact_chars} chars ({len(dumped)})"
    )
    return compact
