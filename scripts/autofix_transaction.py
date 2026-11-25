#!/usr/bin/env python3
"""
v3.1 SSOT-01 — Autofix transactionnel à 3 paliers avec shadow rollback.

Utilisé par `render_and_audit.py` et `multiscale_autofix.py` : chaque action d'autofix est
validée AVANT acceptation — si un invariant de Palier 1 régresse (ou si le score composite
Palier 2+3 ne progresse pas d'au moins epsilon), la mutation est rollbackée immédiatement et
l'action est blacklistée pour le projet. Chaque tentative est journalisée dans
`<proj>/renders/autofix_history.json` (ACCEPTED / ROLLED_BACK + raison exacte).

Paliers (sur tout rapport d'audit) :
  - Palier 1 (invariants durs, zéro régression tolérée) : clipped==0, true_peak <= ceiling+0.2,
    low_end_corr >= 0.50, musical_safety.errors == 0, foundation_tuning_ct <= 6.0,
    micro_glitches == 0, harmonic_audio_clash_ratio <= 0.08.
  - Palier 2 (équilibre macro) : lufs, low_end_share, low_mids, air, stem_hierarchy, LRA, pink_slope.
  - Palier 3 (finesse) : stereo_width, stereo_balance, sharpness, spectral_corridor_loss, dynamic_arc.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
import sys
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

EPSILON = 0.015

TIER1_KEYS = (
    "clipped", "true_peak_dbtp", "low_end_corr", "musical_safety_errors",
    "foundation_tuning_ct", "micro_glitches", "harmonic_audio_clash_ratio",
)
TIER2_KEYS = ("lufs", "low_end_share", "low_mids_share", "air_share", "stem_hierarchy_ok", "lra_lu", "pink_slope_db_per_oct")
TIER3_KEYS = ("stereo_width_db", "stereo_balance_db", "sharpness", "corridor_score", "dynamic_arc_db")


def _get(report: dict[str, Any], key: str) -> float | None:
    """Read a metric from a flat or nested audit report."""
    if key in report:
        v = report[key]
        return float(v) if isinstance(v, (int, float)) else None
    for sub in ("psychoacoustics", "stereo", "quality_gate", "spectral_bands", "musical_safety"):
        sub_d = report.get(sub)
        if isinstance(sub_d, dict) and key in sub_d:
            v = sub_d[key]
            return float(v) if isinstance(v, (int, float)) else None
    return None


def classify_tiers(report: dict[str, Any], config_path: str | None = None, profile: str = "trap_rap", valence: float = 0.0) -> dict[str, Any]:
    """Extract tier scores from an audit report.

    tier1 = hard invariants (any failure => tier1_pass False).
    tier2/tier3 = fraction of metrics inside their config bounds (from mix_profiles.json),
    so the composite is a conformance ratio in [0, 1] (higher = better), not a raw mean.
    """
    from mix_config import get_invariant, get_profile_config, limiter_invariants
    lim = limiter_invariants(config_path)
    ps = get_profile_config(profile, valence, config_path)
    tp_limit = lim["ceiling_dbtp"] + float(get_invariant("true_peak_margin_db", 0.2, config_path))
    le_min = float(get_invariant("low_end_corr_min", 0.50, config_path))
    tt = get_invariant("tuning_tolerance_ct", {}, config_path) or {}
    found_tt = float(tt.get("foundation_stem_max", 6.0)) if isinstance(tt, dict) else 6.0
    t1 = {
        "clipped": _get(report, "clipped") or _get(report, "clipped_samples") or 0,
        "true_peak_dbtp": _get(report, "true_peak_dbtp"),
        "low_end_corr": _get(report, "low_end_corr"),
        "musical_safety_errors": (report.get("musical_safety") or {}).get("errors", 0) if isinstance(report.get("musical_safety"), dict) else 0,
        "foundation_tuning_ct": _get(report, "foundation_tuning_ct"),
        "micro_glitches": _get(report, "micro_glitches") or 0,
        "harmonic_audio_clash_ratio": _get(report, "harmonic_audio_clash_ratio"),
    }
    tier1_failures = []
    if t1["clipped"] and float(t1["clipped"]) > 0:
        tier1_failures.append("clipped>0")
    if t1["true_peak_dbtp"] is not None and float(t1["true_peak_dbtp"]) > tp_limit:
        tier1_failures.append(f"true_peak>{tp_limit}")
    if t1["low_end_corr"] is not None and float(t1["low_end_corr"]) < le_min:
        tier1_failures.append(f"low_end_corr<{le_min}")
    if t1["musical_safety_errors"] and float(t1["musical_safety_errors"]) > 0:
        tier1_failures.append("musical_safety_errors>0")
    if t1["foundation_tuning_ct"] is not None and abs(float(t1["foundation_tuning_ct"])) > found_tt:
        tier1_failures.append(f"foundation_tuning>{found_tt}ct")
    if t1["micro_glitches"] and float(t1["micro_glitches"]) > 0:
        tier1_failures.append("micro_glitches>0")
    if t1["harmonic_audio_clash_ratio"] is not None and float(t1["harmonic_audio_clash_ratio"]) > 0.08:
        tier1_failures.append("harmonic_audio_clash_ratio>0.08")

    # tier 2: conformance ratio vs profile bounds
    lufs_tol = float(get_invariant("lufs_tolerance_db", 1.2, config_path))
    t2_checks: list[bool] = []
    lufs = _get(report, "lufs")
    tgt = _get(report, "target_lufs") or ps.get("target_lufs")
    if lufs is not None and tgt is not None:
        t2_checks.append(abs(float(lufs) - float(tgt)) <= lufs_tol)
    v = _get(report, "low_end_share")
    if v is not None:
        t2_checks.append(float(v) <= float(ps.get("low_end_max_pct", 42.0)))
    v = _get(report, "low_mids_share")
    if v is not None:
        t2_checks.append(float(v) <= float(ps.get("low_mids_max_pct", 24.0)))
    v = _get(report, "air_share")
    if v is not None:
        t2_checks.append(float(v) >= float(ps.get("air_min_pct", 3.0)))
    v = _get(report, "min_short_term_plr_db")
    if v is not None:
        t2_checks.append(float(v) >= float(ps.get("min_short_term_plr_db", 7.5)))
    v = _get(report, "lra_lu")
    if v is not None:
        t2_checks.append(float(ps.get("lra_min_lu", 2.5)) <= float(v) <= float(ps.get("lra_max_lu", 9.0)))
    v = _get(report, "stem_hierarchy_ok")
    if v is not None:
        t2_checks.append(bool(v))
    v = _get(report, "pink_slope_db_per_oct")
    if v is not None:
        t2_checks.append(-5.5 <= float(v) <= -3.0)
    tier2 = sum(t2_checks) / len(t2_checks) if t2_checks else 1.0

    # tier 3: finesse
    t3_checks: list[bool] = []
    v = _get(report, "sharpness")
    if v is not None:
        t3_checks.append(float(v) <= float(ps.get("sharpness_max", 1.90)))
    v = _get(report, "stereo_balance_db")
    if v is not None:
        t3_checks.append(abs(float(v)) <= float(get_invariant("stereo_balance_max_db", 1.2, config_path)))
    v = _get(report, "corridor_score")
    if v is not None:
        t3_checks.append(float(v) >= 0.70)
    v = _get(report, "dynamic_arc_db")
    if v is not None:
        t3_checks.append(float(v) >= float(ps.get("dynamic_arc_min_db", 2.5)))
    v = _get(report, "stereo_width_db")
    if v is not None:
        t3_checks.append(-14.0 <= float(v) <= -1.5)
    tier3 = sum(t3_checks) / len(t3_checks) if t3_checks else 1.0

    return {
        "tier1_failures": tier1_failures,
        "tier1_pass": not tier1_failures,
        "tier2_score": round(tier2, 4),
        "tier3_score": round(tier3, 4),
        "tier1_metrics": t1,
    }


def evaluate_patch(before: dict[str, Any], after: dict[str, Any], epsilon: float = EPSILON, config_path: str | None = None, require_progress: bool = True) -> dict[str, Any]:
    """Decide ACCEPTED vs ROLLED_BACK between two audit reports (pure, deterministic)."""
    tb = classify_tiers(before, config_path)
    ta = classify_tiers(after, config_path)
    if ta["tier1_failures"] and not tb["tier1_failures"]:
        return {"verdict": "ROLLED_BACK", "reason": f"tier1 regressed: {ta['tier1_failures']}"}
    if tb["tier1_failures"] and ta["tier1_failures"] and len(ta["tier1_failures"]) > len(tb["tier1_failures"]):
        return {"verdict": "ROLLED_BACK", "reason": f"tier1 worsened: {tb['tier1_failures']} -> {ta['tier1_failures']}"}
    composite_before = 0.6 * tb["tier2_score"] + 0.4 * tb["tier3_score"]
    composite_after = 0.6 * ta["tier2_score"] + 0.4 * ta["tier3_score"]
    if composite_after < composite_before - 1e-9:
        return {"verdict": "ROLLED_BACK", "reason": f"composite tier2+3 degraded ({composite_before:.4f} -> {composite_after:.4f})"}
    if require_progress and composite_after - composite_before < epsilon and ta["tier1_pass"] and tb["tier1_pass"]:
        # no tier1 gain and no composite progress -> useless patch (oscillation guard)
        return {"verdict": "ROLLED_BACK", "reason": f"no progress >= epsilon ({composite_after - composite_before:+.4f} < {epsilon}) — oscillation guard"}
    return {"verdict": "ACCEPTED", "reason": f"tier1 stable, composite {composite_before:.4f} -> {composite_after:.4f}",
            "tier1_before": tb["tier1_failures"], "tier1_after": ta["tier1_failures"]}


def history_path(project_dir: str) -> str:
    return os.path.join(os.path.abspath(os.path.expanduser(project_dir)), "renders", "autofix_history.json")


def load_history(project_dir: str) -> dict[str, Any]:
    p = history_path(project_dir)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"attempts": [], "blacklisted_actions": []}


def _save_history(project_dir: str, hist: dict[str, Any]) -> None:
    p = history_path(project_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(hist, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, p)


def apply_with_rollback(
    project_dir: str,
    action: dict[str, Any],
    apply_fn: Callable[[], Any],
    eval_fn: Callable[[], dict[str, Any]],
    report_before: dict[str, Any],
    *,
    snapshot_fn: Callable[[], Any] | None = None,
    restore_fn: Callable[[Any], None] | None = None,
    epsilon: float = EPSILON,
    require_progress: bool = True,
) -> dict[str, Any]:
    """Transactional autofix: snapshot -> apply -> re-evaluate -> accept or rollback + journal.

    `apply_fn()` performs the mutation and returns its result; `eval_fn()` returns the new
    audit report; `snapshot_fn`/`restore_fn` save/restore project state (plan JSON, mixer params).
    """
    code = str(action.get("code") or action.get("rule") or "UNKNOWN")
    target = str(action.get("target") or action.get("track") or "")
    hist = load_history(project_dir)
    if any(b.get("code") == code and b.get("target") == target for b in hist.get("blacklisted_actions", [])):
        return {"verdict": "SKIPPED_BLACKLISTED", "code": code, "target": target}
    snap = snapshot_fn() if snapshot_fn else None
    applied = apply_fn()
    report_after = eval_fn()
    decision = evaluate_patch(report_before, report_after, epsilon=epsilon, require_progress=require_progress)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": code,
        "target": target,
        "verdict": decision["verdict"],
        "reason": decision.get("reason"),
    }
    if decision["verdict"] == "ROLLED_BACK":
        if restore_fn and snap is not None:
            restore_fn(snap)
        hist.setdefault("blacklisted_actions", []).append({"code": code, "target": target, "reason": decision.get("reason")})
    hist.setdefault("attempts", []).append(entry)
    _save_history(project_dir, hist)
    return {"verdict": decision["verdict"], "reason": decision.get("reason"), "code": code, "target": target, "applied": applied}
