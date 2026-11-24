#!/usr/bin/env python3
"""
Loader of the single source of truth for mix constants: `config/mix_profiles.json` (v3.0).

Stdlib-only so every module (studio_lib, gl_dsl, render_and_audit, deep_diagnostics,
gain_staging_lib, blueprint_safety) can read the same invariants without divergence:
`get_invariant(name, default)` for global constants, `profile_settings(profile)` merging
`default_profile` + the genre profile, `source_lpf_floor_hz(role)` for the v3.0
anti-brick-wall source LPF floors, `notch_caps()` for the EQ over-treatment guard.
"""
from __future__ import annotations

import json
import os
from typing import Any

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "config", "mix_profiles.json")

_CACHE: dict[str, Any] = {}


def config_path(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("MIX_PROFILES_PATH")
    return env or DEFAULT_CONFIG_PATH


def load_mix_config(explicit_path: str | None = None, use_cache: bool = True) -> dict[str, Any]:
    """Load `config/mix_profiles.json`; returns a minimal fallback dict if unreadable."""
    path = config_path(explicit_path)
    if use_cache and path in _CACHE:
        return _CACHE[path]
    cfg: dict[str, Any] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        if not isinstance(cfg, dict):
            cfg = {}
    except Exception:
        cfg = {}
    cfg.setdefault("schema_version", "3.0")
    cfg.setdefault("global_invariants", {})
    cfg.setdefault("default_profile", {})
    cfg.setdefault("profiles", {})
    if use_cache:
        _CACHE[path] = cfg
    return cfg


def get_invariant(name: str, default: Any = None, explicit_path: str | None = None) -> Any:
    return load_mix_config(explicit_path).get("global_invariants", {}).get(name, default)


def profile_settings(profile: str, explicit_path: str | None = None) -> dict[str, Any]:
    """Merged settings for a genre profile: `default_profile` overlaid by `profiles[profile]`."""
    cfg = load_mix_config(explicit_path)
    merged: dict[str, Any] = dict(cfg.get("default_profile") or {})
    merged.update(cfg.get("profiles", {}).get(profile) or {})
    return merged


def profile_names(explicit_path: str | None = None) -> list[str]:
    return sorted((load_mix_config(explicit_path).get("profiles") or {}).keys())


def source_lpf_floor_hz(role: str, explicit_path: str | None = None) -> float | None:
    """v3.0 anti brick-wall: minimum source LPF per role (chords/pad >= 6.5 kHz, lead/pluck >= 9 kHz)."""
    floors = get_invariant("source_lpf_floor_hz", {}, explicit_path) or {}
    r = (role or "").lower()
    for key in (r, "chords" if r in ("pad", "chord", "keys") else None, "lead" if r in ("pluck", "melody") else None):
        if key and key in floors:
            return float(floors[key])
    return None


def notch_caps(explicit_path: str | None = None) -> tuple[int, int]:
    """(max notches per track, max notches total) before forcing SWAP_SAMPLE."""
    per_track = int(get_invariant("max_eq_notches_per_track", 2, explicit_path))
    total = int(get_invariant("max_eq_notches_total", 3, explicit_path))
    return per_track, total


def limiter_invariants(explicit_path: str | None = None) -> dict[str, Any]:
    return {
        "ceiling_dbtp": float(get_invariant("master_limiter_ceiling_dbtp", -1.2, explicit_path)),
        "max_gain_db": float(get_invariant("master_limiter_max_gain_db", 5.5, explicit_path)),
        "safety_02db": bool(get_invariant("master_limiter_safety_02db", True, explicit_path)),
    }


def lead_vs_harmony_invariants(explicit_path: str | None = None) -> dict[str, Any]:
    rng = get_invariant("lead_vs_harmony_db_range", [2.5, 7.5], explicit_path)
    return {
        "range": [float(rng[0]), float(rng[1])],
        "target_db": float(get_invariant("lead_vs_harmony_target_db", 4.0, explicit_path)),
    }


def tuning_invariants(explicit_path: str | None = None) -> dict[str, float]:
    """v3.0: strict on foundation stems (<= 6.0 ct), tolerant on expressive vibrato
    (mean <= 5.0 ct, peak <= 12.0 ct) so vibrato is not mistaken for being out of tune."""
    return {
        "foundation_max_ct": float(get_invariant("tuning_offset_max_ct_foundation", 6.0, explicit_path)),
        "expressive_mean_max_ct": float(get_invariant("tuning_offset_expressive_mean_max_ct", 5.0, explicit_path)),
        "expressive_peak_ct": float(get_invariant("tuning_offset_expressive_peak_ct", 12.0, explicit_path)),
    }


def resolve_limiter_params(
    ceiling_db: float | None,
    gain_db: float | None,
    explicit_path: str | None = None,
) -> tuple[float, float]:
    """Fill None defaults from `config/mix_profiles.json` invariants (v3.0 single source).

    Explicit values are respected (only None defaults are resolved from the config), so
    legacy callers passing explicit ceilings/gains keep their behavior.
    """
    inv = limiter_invariants(explicit_path)
    ceiling = float(ceiling_db) if ceiling_db is not None else inv["ceiling_dbtp"]
    gain = float(gain_db) if gain_db is not None else inv["max_gain_db"]
    return ceiling, gain


def get_profile_config(profile: str, valence: float = 0.0, explicit_path: str | None = None) -> dict[str, Any]:
    """v3.1: merged config for a genre profile, dark-mode aware.

    Neutral mode returns the profile values; when `valence < valence_dark_threshold`
    the dark overrides are applied (air_min_dark_pct, sharpness_max_dark, ...).
    """
    ps = profile_settings(profile, explicit_path)
    out: dict[str, Any] = dict(ps)
    inv = load_mix_config(explicit_path).get("global_invariants", {})
    dark_thr = float(inv.get("valence_dark_threshold", -0.2))
    if float(valence) < dark_thr:
        for k, dark_k in (("air_min_pct", "air_min_dark_pct"), ("sharpness_max", "sharpness_max_dark")):
            if dark_k in ps:
                out[k] = float(ps[dark_k])
        out["dark_mode"] = True
    else:
        out["dark_mode"] = False
    out["dark_threshold"] = dark_thr
    return out


def tuning_tolerance(explicit_path: str | None = None) -> dict[str, float]:
    t = get_invariant("tuning_tolerance_ct", {}, explicit_path) or {}
    return {
        "foundation_stem_max": float(t.get("foundation_stem_max", 6.0)),
        "expressive_peak_max": float(t.get("expressive_peak_max", 12.0)),
        "master_warning": float(t.get("master_warning", 6.5)),
    }


def notch_budget(explicit_path: str | None = None) -> dict[str, Any]:
    nb = get_invariant("notch_budget", {}, explicit_path) or {}
    return {
        "max_per_track": int(nb.get("max_per_track", 2)),
        "max_total": int(nb.get("max_total", 3)),
        "min_excess_db": float(nb.get("min_excess_db", 6.0)),
        "max_width_bands": int(nb.get("max_width_bands", 2)),
    }


def stem_hierarchy_bounds(explicit_path: str | None = None) -> dict[str, list[float]]:
    sh = get_invariant("stem_hierarchy_bounds_db", {}, explicit_path) or {}
    return {
        "lead_vs_harmony": [float(x) for x in sh.get("lead_vs_harmony", [3.0, 9.5])],
        "bass_vs_kick": [float(x) for x in sh.get("bass_vs_kick", [-3.5, 2.0])],
    }


def autofix_step_bounds(explicit_path: str | None = None) -> dict[str, float]:
    return {
        "max_step_db": float(get_invariant("autofix_max_step_db", 4.0, explicit_path)),
        "max_cumulative_db": float(get_invariant("autofix_max_cumulative_db", 8.0, explicit_path)),
    }
