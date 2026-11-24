#!/usr/bin/env python3
"""
Project-wide musical safety audit of song blueprints (dependency-free: stdlib only).

`blueprint_musical_safety_issues(bp, catalog_path)` checks EVERY `source_type == "sample"`
track of a `song_blueprint.json` BEFORE any build/render, so the "ChordKey dissonant /
LeadBell imperceptible" class of errors is caught at the project level, not per track:

  - structural: a sample track must declare `sample_path` + `root_note` (runtime guard
    complementing the JSON-schema `allOf`; hand-edited blueprints bypass the schema).
    Without `root_note`, `gl.sampler` falls back to C4=60 -> systematic dissonance.
  - catalog cross-check (when `<projet>/samples_catalog.json` exists): the declared
    `root_note` must match the detected `midi_root` of the same sample (pitch-class
    equality; octave errors are harmless, partial-lock errors are not), and a sample
    flagged `playable_chromatically == False` (chord / melodic sequence) must not sit
    in a melodic role.
  - key-fit: a melodic, non-loop sample whose root pitch class is far outside the song
    key scale will sound "etrange" if its root is wrong; each issue carries the
    recommended `root_note` shift in semitones.

Every issue has a `severity` ("error" blocks the render quality gate, "warning" is only
reported) and a concrete `fix`. Wired into `song_builder.load_blueprint`,
`loop_first_runner.apply_palette_to_blueprint` and the `render_and_audit` quality gate.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

TONIC_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11, "B#": 0,
}

MAJOR_SCALE_PCS = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE_PCS = (0, 2, 3, 5, 7, 8, 10)

MODE_SCALE_PCS: dict[str, tuple[int, ...]] = {
    "ionian": MAJOR_SCALE_PCS,
    "major": MAJOR_SCALE_PCS,
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "aeolian": MINOR_SCALE_PCS,
    "minor": MINOR_SCALE_PCS,
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "pentatonic_major": (0, 2, 4, 7, 9),
    "pentatonic_minor": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
}

# Roles whose samples are transients: pitch class is irrelevant for them.
PERCUSSIVE_ROLE_SUBSTRINGS = (
    "kick", "snare", "clap", "hat", "perc", "drum", "tom", "rim", "shaker", "cymbal",
)

_KEY_QUALITY_RE = re.compile(
    r"\b(min|minor|maj|major|m|ionian|lydian|mixolydian|dorian|aeolian|phrygian|locrian"
    r"|harmonic|melodic|pentatonic|blues)\b",
    re.IGNORECASE,
)


def key_str_scale_pcs(key: str | None) -> set[int] | None:
    """Scale pitch classes from a key string: 'F minor', 'C major', 'Bb dorian', 'Am', 'C'."""
    if not key or not key.strip():
        return None
    m = re.match(r"^\s*([A-Ga-g])([#b]?)\s*([A-Za-z]*)", key.strip())
    if not m:
        return None
    pc = TONIC_TO_PC.get(m.group(1).upper() + m.group(2))
    if pc is None:
        return None
    qual = m.group(3).lower()
    scale = MODE_SCALE_PCS.get(qual)
    if scale is None:
        scale = MINOR_SCALE_PCS if qual == "m" else MAJOR_SCALE_PCS
    return {(pc + iv) % 12 for iv in scale}


def blueprint_key_scale_pcs(bp: dict[str, Any]) -> set[int] | None:
    """Mode-aware scale pitch classes of a blueprint.

    An explicit quality/mode in the top-level `key` ("F minor") wins; otherwise combine
    `harmonic_core.key` + `harmonic_core.mode` (e.g. key="F", mode="aeolian" -> F minor).
    """
    if not isinstance(bp, dict):
        return None
    hc = bp.get("harmonic_core") or {}
    top = str(bp.get("key") or "")
    if _KEY_QUALITY_RE.search(top):
        pcs = key_str_scale_pcs(top)
        if pcs:
            return pcs
    tonic = str(hc.get("key") or top or "")
    m = re.match(r"^\s*([A-Ga-g])([#b]?)", tonic)
    pc = TONIC_TO_PC.get((m.group(1).upper() + m.group(2))) if m else None
    if pc is None:
        return key_str_scale_pcs(top)
    scale = MODE_SCALE_PCS.get(str(hc.get("mode") or "").strip().lower())
    if scale is None:
        return key_str_scale_pcs(tonic or top)
    return {(pc + iv) % 12 for iv in scale}


def key_fit_penalty_from_pcs(pc: int, pcs: set[int]) -> float:
    """0.0 if pc is in scale, 0.03 if within <= 2 st, else 0.12 (would sound dissonant)."""
    if pc in pcs:
        return 0.0
    d = min(min((pc - q) % 12, (q - pc) % 12) for q in pcs)
    return 0.03 if d <= 2 else 0.12


def nearest_scale_shift_st(pc: int, pcs: set[int]) -> tuple[int, int]:
    """Return (nearest scale pc, signed semitone shift in [-6, +6] from pc to reach it)."""
    best_q, best_d = pc, 0
    for q in sorted(pcs):
        d = (q - pc) % 12
        if d > 6:
            d -= 12
        if best_q == pc or abs(d) < abs(best_d):
            best_q, best_d = q, d
    return best_q, best_d


def is_percussive_role(role: str) -> bool:
    r = (role or "").lower()
    return any(s in r for s in PERCUSSIVE_ROLE_SUBSTRINGS)


def load_catalog_index(catalog_path: str) -> dict[str, dict[str, Any]]:
    """Index a `samples_catalog.json` by absolute path and by basename (lower)."""
    idx: dict[str, dict[str, Any]] = {}
    if not catalog_path or not os.path.isfile(catalog_path):
        return idx
    try:
        with open(catalog_path, encoding="utf-8") as fh:
            catalog = json.load(fh)
    except Exception:
        return idx
    proj_dir = os.path.dirname(os.path.abspath(catalog_path))
    for s in catalog.get("samples") or []:
        if not isinstance(s, dict):
            continue
        fp = str(s.get("filepath") or "")
        if fp:
            idx[os.path.normpath(os.path.abspath(fp))] = s
            idx.setdefault(os.path.basename(fp).lower(), s)
        # also index project-relative paths ("samples/x.wav")
        if fp:
            rel = os.path.relpath(os.path.abspath(fp), proj_dir)
            idx[os.path.normpath(rel).lower()] = s
            idx.setdefault(os.path.basename(fp).lower(), s)
    return idx


def _lookup_catalog_entry(idx: dict[str, dict[str, Any]], sample_path: str, proj_dir: str) -> dict[str, Any] | None:
    p = str(sample_path or "").strip()
    if not p:
        return None
    cands = [
        os.path.normpath(os.path.abspath(p)),
        os.path.normpath(os.path.join(proj_dir, p)).lower() if not os.path.isabs(p) else os.path.normpath(os.path.abspath(p)),
        p.lower(),
        os.path.basename(p).lower(),
    ]
    for c in cands:
        if c in idx:
            return idx[c]
    return None


def blueprint_musical_safety_issues(
    bp: dict[str, Any],
    catalog_path: str | None = None,
) -> list[dict[str, Any]]:
    """Audit every `source_type == "sample"` track of a blueprint (see module docstring).

    Returns a list of issues; each has `severity` ("error" | "warning"), `code`, `track`,
    `role` and a human-readable `fix`. `catalog_path` (optional `<proj>/samples_catalog.json`)
    enables the detected-`midi_root` / playability cross-check.
    """
    issues: list[dict[str, Any]] = []
    if not isinstance(bp, dict):
        return issues
    pcs = blueprint_key_scale_pcs(bp)
    hc = bp.get("harmonic_core") or {}
    key_label = str(bp.get("key") or hc.get("key") or "")
    proj_dir = os.path.dirname(os.path.abspath(catalog_path)) if catalog_path else ""
    cat_idx = load_catalog_index(catalog_path) if catalog_path else {}

    for i, tr in enumerate(bp.get("tracks") or []):
        if not isinstance(tr, dict):
            continue
        name = str(tr.get("name") or f"track[{i}]")
        role = str(tr.get("role") or "")
        st = str(tr.get("source_type") or "")
        if st != "sample":
            continue
        base = {"track": name, "role": role}
        sample_path = str(tr.get("sample_path") or "")
        if not sample_path:
            issues.append({
                **base,
                "severity": "error",
                "code": "sample_missing_path",
                "fix": "set sample_path to an existing user sample (user_samples_locked: no auto-fetch)",
            })
        rn = tr.get("root_note")
        if rn is None:
            issues.append({
                **base,
                "severity": "error",
                "code": "sample_missing_root_note",
                "fix": "fill root_note from the detected midi_root (palette/catalog); never leave the C4=60 default",
            })
            continue
        try:
            rn_i = int(rn)
        except (TypeError, ValueError):
            issues.append({**base, "severity": "error", "code": "root_note_invalid", "root_note": rn,
                           "fix": "root_note must be an integer MIDI note (0-127)"})
            continue

        entry = _lookup_catalog_entry(cat_idx, sample_path, proj_dir) if cat_idx else None
        if entry is not None:
            if entry.get("hard_reject"):
                issues.append({
                    **base, "severity": "warning", "code": "sample_hard_reject_in_catalog",
                    "sample_path": sample_path,
                    "fix": "this sample is hard-rejected by the indexer; prefer another palette candidate",
                })
            detected = entry.get("midi_root")
            if detected is not None:
                try:
                    det_i = int(detected)
                except (TypeError, ValueError):
                    det_i = None
                if det_i is not None and (det_i % 12) != (rn_i % 12):
                    rec = det_i + 12 * int(round((rn_i - det_i) / 12.0))
                    issues.append({
                        **base,
                        "severity": "error",
                        "code": "root_note_pc_mismatch",
                        "sample_path": sample_path,
                        "root_note": rn_i,
                        "detected_midi_root": det_i,
                        "recommended_root_note": int(rec),
                        "fix": (
                            f"root_note={rn_i} does not match the detected midi_root={det_i} of this sample; "
                            f"set root_note={int(rec)} (same pitch class, closest octave) — a wrong root makes "
                            "every played note land off-scale (dissonance)"
                        ),
                    })
            if not is_percussive_role(role) and not bool(tr.get("is_loop")) and not bool(entry.get("is_loop")):
                if entry.get("playable_chromatically") is False:
                    issues.append({
                        **base,
                        "severity": "error",
                        "code": "non_playable_sample_in_melodic_role",
                        "sample_path": sample_path,
                        "sample_nature": entry.get("sample_nature"),
                        "fix": (
                            "this sample is not a single stable pitch (chord/sequence); playing it chromatically "
                            "in gl.sampler sounds dissonant — swap it (SWAP_SAMPLE), extract one clean note via "
                            "studio_lib.extract_clean_oneshot(...), or use a native gl.* synth"
                        ),
                    })

        if is_percussive_role(role) or bool(tr.get("is_loop")):
            continue
        if not pcs:
            continue
        pc = rn_i % 12
        if pc in pcs:
            continue
        pen = key_fit_penalty_from_pcs(pc, pcs)
        _, shift = nearest_scale_shift_st(pc, pcs)
        issues.append({
            **base,
            "severity": "error" if pen >= 0.12 else "warning",
            "code": "key_fit_out_of_scale",
            "key": key_label or None,
            "root_note": rn_i,
            "root_pc": pc,
            "scale_pcs": sorted(pcs),
            "key_fit_penalty": pen,
            "recommended_root_note_shift_st": int(shift),
            "fix": (
                f"root_note={rn_i} (pc {pc}) is outside the '{key_label}' scale; if root_note is the true "
                f"detected root this is harmless (playback is relative), otherwise shift root_note by "
                f"{int(shift):+d} st (-> {rn_i + int(shift)}) or swap for an in-scale palette sample"
            ),
        })
    return issues


def summarize_musical_safety(bp: dict[str, Any], catalog_path: str | None = None) -> dict[str, Any]:
    """Structured summary of `blueprint_musical_safety_issues` (wireable into reports)."""
    issues = blueprint_musical_safety_issues(bp, catalog_path=catalog_path)
    pcs = blueprint_key_scale_pcs(bp) if isinstance(bp, dict) else None
    hc = (bp.get("harmonic_core") or {}) if isinstance(bp, dict) else {}
    errors = sum(1 for i in issues if i.get("severity") == "error")
    return {
        "checked": isinstance(bp, dict),
        "key": str(bp.get("key") or hc.get("key") or "") if isinstance(bp, dict) else "",
        "scale_pcs": sorted(pcs) if pcs else None,
        "catalog_used": bool(catalog_path),
        "issues": issues,
        "errors": errors,
        "warnings": len(issues) - errors,
        "ok": errors == 0,
    }
