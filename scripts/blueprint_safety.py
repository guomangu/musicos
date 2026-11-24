#!/usr/bin/env python3
"""
Project-wide musical safety audit of song blueprints (dependency-free: stdlib only).

`blueprint_musical_safety_issues(bp, catalog_path, notes_by_track)` checks EVERY
`source_type == "sample"` track of a `song_blueprint.json` BEFORE any build/render, so the
"ChordKey dissonant / LeadBell imperceptible" class of errors is caught at project level:

  - structural: a sample track must declare `sample_path` + `root_note` (runtime guard
    complementing the JSON-schema `allOf`; hand-edited blueprints bypass the schema).
    Without `root_note`, `gl.sampler` falls back to C4=60 -> systematic dissonance.
  - strict root lock (v3.0): the declared `root_note` must EQUAL the detected `midi_root`
    of the same sample in `samples_catalog.json`. `gl.sampler` transposes by varispeed, so
    a +/-12 st error doubles/halves the playback speed (chipmunk or smudged timbre) and a
    sub-80 Hz 808 shifted down becomes inaudible infra-sub that eats the limiter headroom.
    Any non-zero delta is a blocking error with `recommended_root_note = midi_root`.
  - max semitone shift (v3.0): for non-percussive sample tracks, S_max = max |n - root_note|
    over the generated notes must respect the per-category cap (acoustic <= 5 st,
    synth/pluck/bell/808 <= 7 st); beyond it the fix is auto-multizone
    (`sample_prep_lib.build_multizone_samples`) or `SWAP_SAMPLE` / native `gl.*` synth.
  - loops (v3.0): `percussive_loop` stays exempt from key-fit, but `melodic_loop` is NOT
    exempt: its CQT chroma is checked against the song scale (R_out > 0.22 -> try
    transpositions delta in {-2,-1,+1,+2}; if one reaches <= 0.15 the fix is an automatic
    repitch, else the loop is rejected LOOP_KEY_MISMATCH). BPM lock: only half/double tempo
    ratios are allowed and the effective stretch must stay in [0.88, 1.14]; any residual
    > 0.05 BPM forces a mandatory `time_stretch_stereo` with exact bar-multiple duration.
  - inharmonic bells/FM (v3.0): confined to polyphony_max=1 and stable degrees {1, 5, 8}
    (their non-integer partials, e.g. 2.76x f0, clash on 2nd/3rd/6th/7th degrees).
  - catalog cross-check: a sample flagged `playable_chromatically == False` (chord /
    melodic sequence) must not sit in a melodic role.

Every issue has a `severity` ("error" blocks the render quality gate, "warning" is only
reported) and a concrete `fix`. Wired into `song_builder.load_blueprint`,
`loop_first_runner.apply_palette_to_blueprint` and the `render_and_audit` quality gate.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

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

# v3.0 -- per-category mono-zone transposition caps (semitones).
SAMPLE_SHIFT_CAP_ST = {"acoustic": 5, "synth": 7}
ACOUSTIC_CATEGORY_TOKENS = ("piano", "guitar", "string", "violin", "cello", "brass", "woodwind", "flute", "voice", "vocal", "choir", "acoustic", "rhodes", "organ", "harp", "cello")
# Auto-multizone is only possible when the total ambitus stays small enough.
MULTIZONE_MAX_AMBITUS_ST = {"acoustic": 18, "synth": 20}

# v3.0 -- loop guards.
PERCUSSIVE_LOOP_MAX_HARM_RATIO = 0.25   # E_harm / (E_harm + E_perc)
PERCUSSIVE_LOOP_MAX_HNR_DB = 3.0
LOOP_KEY_MAX_OUT_OF_SCALE = 0.22        # R_out above this -> try repitch, else reject
LOOP_REPITCH_MIN_OUT_OF_SCALE = 0.15    # acceptable after transposition delta
LOOP_REPITCH_DELTAS_ST = (-2, -1, 1, 2)
BPM_ALLOWED_NORM_RATIOS = (0.5, 1.0, 2.0)   # half / same / double tempo only
BPM_STRETCH_MIN, BPM_STRETCH_MAX = 0.88, 1.14
BPM_LOCK_TOLERANCE = 0.05               # BPM; below this no stretch is needed

# v3.0 -- inharmonic bell confinement: stable degrees {1, 5, 8} -> pitch-class offsets.
BELL_CONFINED_DEGREE_OFFSETS = (0, 7, 12)
BELL_INHARMONICITY_MAX = 0.15

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


# ---------------------------------------------------------------------------
# v3.0 -- sample category / max semitone shift
# ---------------------------------------------------------------------------

def sample_shift_category(entry: dict[str, Any] | None) -> str:
    """'acoustic' (piano/guitar/strings/voice) vs 'synth' (pluck/bell/808/synth/FM)."""
    text = " ".join(
        str(entry.get(k) or "") for k in ("category", "timbre_card", "filename", "filepath", "sample_nature")
    ).lower() if entry else ""
    if any(t in text for t in ACOUSTIC_CATEGORY_TOKENS):
        return "acoustic"
    return "synth"


def max_semitone_shift(root_note: int, note_midis: Iterable[int]) -> int:
    """S_max = max |n_midi - root_note| over the notes actually played on the track."""
    return max((abs(int(n) - int(root_note)) for n in note_midis), default=0)


def decide_shift_action(s_max: int, ambitus_st: int, category: str) -> str:
    """'ok' | 'multizone' | 'swap' according to the per-category mono-zone cap."""
    cap = SAMPLE_SHIFT_CAP_ST.get(category, SAMPLE_SHIFT_CAP_ST["synth"])
    if s_max <= cap:
        return "ok"
    if ambitus_st <= MULTIZONE_MAX_AMBITUS_ST.get(category, 20):
        return "multizone"
    return "swap"


# ---------------------------------------------------------------------------
# v3.0 -- loop typing, key-fit, BPM lock
# ---------------------------------------------------------------------------

def classify_loop_nature(harmonic_energy_ratio: float, hnr_db: float) -> str:
    """'percussive_loop' only if HPSS harmonic share < 0.25 AND HNR < 3.0 dB, else 'melodic_loop'."""
    if harmonic_energy_ratio < PERCUSSIVE_LOOP_MAX_HARM_RATIO and hnr_db < PERCUSSIVE_LOOP_MAX_HNR_DB:
        return "percussive_loop"
    return "melodic_loop"


def chroma_out_of_scale_ratio(chroma: Iterable[float], scale_pcs: set[int]) -> float:
    """R_out = fraction of chroma energy on pitch classes outside the target scale."""
    vec = [max(0.0, float(c)) for c in chroma]
    if len(vec) != 12 or not any(vec):
        return 0.0
    tot = sum(vec)
    out = sum(v for i, v in enumerate(vec) if (i % 12) not in scale_pcs)
    return out / tot


def best_loop_transposition(chroma: Iterable[float], scale_pcs: set[int]) -> tuple[int, float]:
    """Best delta in {-2,-1,+1,+2} st minimizing R_out; returns (delta, R_out_after)."""
    base = chroma_out_of_scale_ratio(chroma, scale_pcs)
    best_delta, best_r = 0, base
    for d in LOOP_REPITCH_DELTAS_ST:
        shifted = [0.0] * 12
        for i, v in enumerate(chroma):
            shifted[(i + d) % 12] += max(0.0, float(v))
        r = chroma_out_of_scale_ratio(shifted, scale_pcs)
        if r < best_r - 1e-9:
            best_delta, best_r = d, r
    return best_delta, best_r


def bpm_compatibility(bpm_proj: float | None, bpm_loop: float | None) -> dict[str, Any]:
    """v3.0 BPM lock for loops.

    Only half/same/double tempo is allowed (rhythmic octave normalization): the ratio
    r = bpm_proj / bpm_loop is snapped to the closest of {0.5, 1.0, 2.0}; the effective
    stretch r_eff = r / norm must stay in [0.88, 1.14] (phase/transient artifacts beyond),
    and any residual > 0.05 BPM forces a mandatory exact time-stretch.
    """
    if not bpm_proj or not bpm_loop or float(bpm_loop) <= 0:
        return {"ok": True, "ratio": None, "norm_ratio": None, "r_eff": 1.0, "needs_stretch": False, "reason": "unknown bpm"}
    r = float(bpm_proj) / float(bpm_loop)
    norm = min(BPM_ALLOWED_NORM_RATIOS, key=lambda c: abs(r - c))
    r_eff = r / norm
    ok = BPM_STRETCH_MIN <= r_eff <= BPM_STRETCH_MAX
    residual_bpm = abs(float(bpm_proj) - float(bpm_loop) * norm)
    needs_stretch = ok and residual_bpm > BPM_LOCK_TOLERANCE
    return {
        "ok": ok,
        "ratio": round(r, 4),
        "norm_ratio": norm,
        "r_eff": round(r_eff, 4),
        "residual_bpm": round(residual_bpm, 3),
        "needs_stretch": needs_stretch,
        "reason": None if ok else f"effective stretch r_eff={r_eff:.3f} outside [{BPM_STRETCH_MIN}, {BPM_STRETCH_MAX}]",
    }


def is_melodic_loop(entry: dict[str, Any] | None, track: dict[str, Any]) -> bool:
    """A loop is melodic (key-fit applies) unless classified percussive."""
    if not (track.get("is_loop") or (entry or {}).get("is_loop")):
        return False
    nature = str((entry or {}).get("loop_nature") or "").lower()
    if nature:
        return nature != "percussive_loop"
    return not is_percussive_role(str(track.get("role") or ""))


def allowed_degrees_for_sample(entry: dict[str, Any] | None) -> list[int] | None:
    """v3.0: confined degree offsets {1,5,8} for inharmonic bells/FM, else None (free)."""
    if not entry:
        return None
    nature = str(entry.get("sample_nature") or "")
    inh = float(entry.get("inharmonicity") or 0.0)
    if nature == "inharmonic_bell_or_fx" or inh > BELL_INHARMONICITY_MAX:
        return list(BELL_CONFINED_DEGREE_OFFSETS)
    return None


# ---------------------------------------------------------------------------
# catalog index
# ---------------------------------------------------------------------------

def load_catalog_index(catalog_path: str) -> dict[str, dict[str, Any]]:
    """Index a `samples_catalog.json` by absolute path, project-relative path and basename."""
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
        if not fp:
            continue
        abs_p = os.path.normpath(os.path.abspath(fp))
        idx[abs_p] = s
        idx[abs_p.lower()] = s
        rel = os.path.normpath(os.path.relpath(abs_p, proj_dir))
        idx[rel.lower()] = s
        idx.setdefault(os.path.basename(fp).lower(), s)
    return idx


def _lookup_catalog_entry(idx: dict[str, dict[str, Any]], sample_path: str, proj_dir: str) -> dict[str, Any] | None:
    p = str(sample_path or "").strip()
    if not p:
        return None
    cands = [p.lower(), os.path.basename(p).lower()]
    if os.path.isabs(p):
        cands.insert(0, os.path.normpath(p).lower())
    else:
        cands.insert(0, os.path.normpath(os.path.join(proj_dir, p)).lower())
    for c in cands:
        if c in idx:
            return idx[c]
    return None


# ---------------------------------------------------------------------------
# main audit
# ---------------------------------------------------------------------------

def blueprint_musical_safety_issues(
    bp: dict[str, Any],
    catalog_path: str | None = None,
    notes_by_track: dict[str, Iterable[int]] | None = None,
) -> list[dict[str, Any]]:
    """Audit every `source_type == "sample"` track of a blueprint (see module docstring).

    `catalog_path` (optional `<proj>/samples_catalog.json`) enables the strict root lock
    vs detected `midi_root`, playability, loop-nature and bell-confinement cross-checks.
    `notes_by_track` (optional {track name: [midi notes]}) enables the MAX_SEMITONE_SHIFT
    check (S_max vs per-category cap).
    """
    issues: list[dict[str, Any]] = []
    if not isinstance(bp, dict):
        return issues
    pcs = blueprint_key_scale_pcs(bp)
    hc = bp.get("harmonic_core") or {}
    key_label = str(bp.get("key") or hc.get("key") or "")
    proj_dir = os.path.dirname(os.path.abspath(catalog_path)) if catalog_path else ""
    cat_idx = load_catalog_index(catalog_path) if catalog_path else {}
    proj_bpm = bp.get("bpm")
    notes_by_track = notes_by_track or {}

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
            # v3.0 strict root lock: gl.sampler varispeed -> octave errors are NOT harmless.
            detected = entry.get("midi_root")
            det_i: int | None = None
            try:
                det_i = int(detected) if detected is not None else None
            except (TypeError, ValueError):
                det_i = None
            if det_i is not None and det_i != rn_i:
                issues.append({
                    **base,
                    "severity": "error",
                    "code": "root_note_mismatch",
                    "sample_path": sample_path,
                    "root_note": rn_i,
                    "detected_midi_root": det_i,
                    "delta_st": int(det_i) - rn_i,
                    "recommended_root_note": int(det_i),
                    "fix": (
                        f"root_note={rn_i} != detected midi_root={det_i} (delta {int(det_i) - rn_i:+d} st); "
                        f"set root_note={int(det_i)} — varispeed playback makes any octave error audible "
                        "(chipmunk / half-speed smudge / infra-sub 808)"
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
            # v3.0 bell confinement
            confined = allowed_degrees_for_sample(entry)
            if confined is not None and not is_percussive_role(role):
                issues.append({
                    **base,
                    "severity": "warning",
                    "code": "inharmonic_bell_confined",
                    "sample_path": sample_path,
                    "sample_nature": entry.get("sample_nature"),
                    "allowed_degree_offsets": confined,
                    "fix": (
                        "inharmonic bell/FM: force polyphony_max=1 and restrict motif degrees to {1,5,8} "
                        "(tonic, fifth, octave) — its non-integer partials clash on 2/3/6/7 degrees"
                    ),
                })
            # v3.0 melodic loops are NOT exempt from key-fit (CQT chroma when available)
            if is_melodic_loop(entry, tr) and pcs:
                chroma = entry.get("chroma_cqt")
                if chroma and len(chroma) == 12:
                    r_out = chroma_out_of_scale_ratio(chroma, pcs)
                    if r_out > LOOP_KEY_MAX_OUT_OF_SCALE:
                        delta, r_after = best_loop_transposition(chroma, pcs)
                        if r_after <= LOOP_REPITCH_MIN_OUT_OF_SCALE:
                            issues.append({
                                **base,
                                "severity": "warning",
                                "code": "loop_repitch_recommended",
                                "sample_path": sample_path,
                                "key": key_label or None,
                                "r_out": round(r_out, 4),
                                "recommended_transposition_st": int(delta),
                                "r_out_after": round(r_after, 4),
                                "fix": f"melodic loop is out of the '{key_label}' scale (R_out={r_out:.2f}); repitch by {delta:+d} st in sample_prep_lib (R_out -> {r_after:.2f})",
                            })
                        else:
                            issues.append({
                                **base,
                                "severity": "error",
                                "code": "loop_key_mismatch",
                                "sample_path": sample_path,
                                "key": key_label or None,
                                "r_out": round(r_out, 4),
                                "fix": f"melodic loop out of the '{key_label}' scale (R_out={r_out:.2f} > {LOOP_KEY_MAX_OUT_OF_SCALE}) and no +/-2 st repitch fixes it — reject the loop (LOOP_KEY_MISMATCH)",
                            })
                elif pcs:
                    pc = rn_i % 12
                    pen = key_fit_penalty_from_pcs(pc, pcs)
                    if pen >= 0.12:
                        _, shift = nearest_scale_shift_st(pc, pcs)
                        issues.append({
                            **base,
                            "severity": "error",
                            "code": "loop_key_mismatch",
                            "sample_path": sample_path,
                            "key": key_label or None,
                            "root_note": rn_i,
                            "root_pc": pc,
                            "recommended_transposition_st": int(shift),
                            "fix": f"melodic loop root pc {pc} far outside '{key_label}' scale; repitch by {int(shift):+d} st or reject",
                        })
            # v3.0 BPM lock for loops
            loop_bpm = tr.get("bpm") or (entry or {}).get("bpm")
            if (tr.get("is_loop") or (entry or {}).get("is_loop")) and loop_bpm and proj_bpm:
                comp = bpm_compatibility(float(proj_bpm), float(loop_bpm))
                if not comp["ok"]:
                    issues.append({
                        **base,
                        "severity": "error",
                        "code": "loop_bpm_incompatible",
                        "sample_path": sample_path,
                        "bpm_loop": float(loop_bpm),
                        "bpm_project": float(proj_bpm),
                        **{k: comp[k] for k in ("ratio", "norm_ratio", "r_eff", "reason") if comp.get(k) is not None},
                        "fix": "loop tempo incompatible (only 0.5x/1x/2x allowed with r_eff in [0.88, 1.14]) — reject the loop",
                    })
                elif comp["needs_stretch"]:
                    issues.append({
                        **base,
                        "severity": "warning",
                        "code": "loop_time_stretch_required",
                        "sample_path": sample_path,
                        "bpm_loop": float(loop_bpm),
                        "bpm_project": float(proj_bpm),
                        "stretch_rate": comp["r_eff"],
                        "residual_bpm": comp["residual_bpm"],
                        "fix": f"run time_stretch_stereo(rate={comp['r_eff']:.4f}) and verify the WAV duration is an exact multiple of 4*60/{float(proj_bpm):g} BPM bars (+/-2 ms)",
                    })

        percussive = is_percussive_role(role)
        melodic_loop = is_melodic_loop(entry, tr)
        is_loop = bool(tr.get("is_loop")) or bool((entry or {}).get("is_loop"))
        if percussive or (is_loop and not melodic_loop):
            # percussive roles and percussive loops: pitch class is irrelevant
            continue
        if melodic_loop:
            # melodic loops already checked above via CQT chroma / root pc
            continue
        # v3.0 MAX_SEMITONE_SHIFT for non-percussive one-shots (before the key-fit continue)
        if not percussive and not is_loop:
            note_midis = list(notes_by_track.get(name) or [])
            if note_midis:
                cat = sample_shift_category(entry)
                cap = SAMPLE_SHIFT_CAP_ST.get(cat, SAMPLE_SHIFT_CAP_ST["synth"])
                s_max = max_semitone_shift(rn_i, note_midis)
                ambitus = (max(int(n) for n in note_midis) - min(int(n) for n in note_midis)) if note_midis else 0
                action = decide_shift_action(s_max, ambitus, cat)
                if action != "ok":
                    issues.append({
                        **base,
                        "severity": "error",
                        "code": "max_semitone_shift_exceeded",
                        "sample_path": sample_path,
                        "category": cat,
                        "s_max_st": int(s_max),
                        "cap_st": cap,
                        "ambitus_st": int(ambitus),
                        "fix": (
                            f"S_max={s_max} st > {cap} st cap for {cat} one-shot; "
                            + ("auto-trigger build_multizone_samples (zones every 6 st, max +/-3 st/zone, formants preserved)"
                               if action == "multizone" else
                               "ambitus too wide for multizone — SWAP_SAMPLE or use a native gl.* synth")
                        ),
                    })
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


def summarize_musical_safety(
    bp: dict[str, Any],
    catalog_path: str | None = None,
    notes_by_track: dict[str, Iterable[int]] | None = None,
) -> dict[str, Any]:
    """Structured summary of `blueprint_musical_safety_issues` (wireable into reports)."""
    issues = blueprint_musical_safety_issues(bp, catalog_path=catalog_path, notes_by_track=notes_by_track)
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
