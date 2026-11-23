#!/usr/bin/env python3
"""
Deterministic MIDI Auto-Fix engine (Axe 3 §3.1).

Applies safe, auditable repairs before failing `verify_midi_gate`:
  1. `out_of_scale` -> snap non-drum pitches to closest allowed scale PC.
  2. `too_high_count` -> transpose down by octaves (-12 st) into <= max_pitch.
  3. `too_low_count` -> transpose up by octaves (+12 st) into >= min_pitch.
  4. `sub_collisions` -> re-voice `Chords` (< MIDI 48) via `optimize_voice_leading(low_midi=48, high_midi=67)` and lift low colliding non-bass notes +12 st.
  5. `strong_beat_chord_tone_ratio` -> project offending strong-beat topline notes (<= 2 st) onto active chord/bass pitch classes.

Writes a corrected MIDI file, emits a structured `fix_report`, and re-runs `audit_midi_file`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pretty_midi

ROOT = "/home/gamo/Bureau/musicos"
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
if ANALYZER_DIR not in sys.path:
    sys.path.insert(0, ANALYZER_DIR)

from deep_diagnostics import audit_midi_file, resolve_allowed_pcs
from musicality_metrics import (
    _PERC_TRACK_RE,
    _compute_strong_beat_chord_tone_ratio,
    _extract_monophonic_topline,
    _infer_roles,
    _resolve_time_signature,
    _strong_beat_times,
    optimize_voice_leading,
)

STRUCTURAL_PREFIXES = (
    "stepwise_ratio",
    "gap_fill_compliance",
    "melody_rest_ratio",
    "contrary_or_oblique_ratio",
    "lead_overlap_conflict",
    "minor_2nd_clashes",
)


def _is_perc_inst(inst: pretty_midi.Instrument) -> bool:
    name = (inst.name or "").lower()
    return bool(inst.is_drum) or bool(_PERC_TRACK_RE.search(name))


def _nearest_pitch_for_pcs(pitch: int, pcs: set[int], min_pitch: int, max_pitch: int) -> int | None:
    if not pcs:
        return None
    best: int | None = None
    best_dist = 999
    for delta in range(-12, 13):
        cand = pitch + delta
        if cand < 0 or cand > 127:
            continue
        if (cand % 12) not in pcs:
            continue
        in_range_penalty = 0 if (min_pitch <= cand <= max_pitch) else 24
        score = abs(delta) + in_range_penalty
        if score < best_dist:
            best_dist = score
            best = cand
    return best


def _fix_out_of_scale(
    pm: pretty_midi.PrettyMIDI,
    allowed_pcs: set[int],
    min_pitch: int,
    max_pitch: int,
    changes: list[dict[str, Any]],
) -> int:
    fixed = 0
    for inst in pm.instruments:
        if _is_perc_inst(inst):
            continue
        tr_name = inst.name or f"Track_{inst.program}"
        for note in inst.notes:
            if (note.pitch % 12) in allowed_pcs:
                continue
            cand = _nearest_pitch_for_pcs(note.pitch, allowed_pcs, min_pitch, max_pitch)
            if cand is not None and cand != note.pitch:
                old_p = int(note.pitch)
                note.pitch = int(cand)
                fixed += 1
                changes.append(
                    {
                        "rule": "out_of_scale",
                        "track": tr_name,
                        "start_s": round(float(note.start), 3),
                        "old_pitch": old_p,
                        "new_pitch": int(cand),
                        "delta_st": int(cand) - old_p,
                    }
                )
    return fixed


def _fix_pitch_bounds(
    pm: pretty_midi.PrettyMIDI,
    min_pitch: int,
    max_pitch: int,
    changes: list[dict[str, Any]],
) -> tuple[int, int]:
    high_fixed = 0
    low_fixed = 0
    for inst in pm.instruments:
        if _is_perc_inst(inst):
            continue
        tr_name = inst.name or f"Track_{inst.program}"
        for note in inst.notes:
            old_p = int(note.pitch)
            if note.pitch > max_pitch:
                new_p = old_p
                while new_p > max_pitch and new_p - 12 >= 0:
                    new_p -= 12
                if new_p != old_p and new_p <= max_pitch:
                    note.pitch = new_p
                    high_fixed += 1
                    changes.append(
                        {
                            "rule": "too_high_count",
                            "track": tr_name,
                            "start_s": round(float(note.start), 3),
                            "old_pitch": old_p,
                            "new_pitch": new_p,
                            "delta_st": new_p - old_p,
                        }
                    )
            elif note.pitch < min_pitch:
                new_p = old_p
                while new_p < min_pitch and new_p + 12 <= 127:
                    new_p += 12
                if new_p != old_p and new_p <= max_pitch:
                    note.pitch = new_p
                    low_fixed += 1
                    changes.append(
                        {
                            "rule": "too_low_count",
                            "track": tr_name,
                            "start_s": round(float(note.start), 3),
                            "old_pitch": old_p,
                            "new_pitch": new_p,
                            "delta_st": new_p - old_p,
                        }
                    )
    return high_fixed, low_fixed


def _group_chord_note_slices(notes: list[pretty_midi.Note], quantize_s: float = 0.08) -> list[list[pretty_midi.Note]]:
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: (n.start, n.pitch))
    slices: list[list[pretty_midi.Note]] = [[ordered[0]]]
    anchor_t = ordered[0].start
    for n in ordered[1:]:
        if n.start - anchor_t <= quantize_s:
            slices[-1].append(n)
        else:
            slices.append([n])
            anchor_t = n.start
    return slices


def _fix_sub_collisions(
    pm: pretty_midi.PrettyMIDI,
    max_pitch: int,
    changes: list[dict[str, Any]],
) -> int:
    roles = _infer_roles(pm)
    fixed = 0
    chord_high = max(48, min(67, max_pitch))

    explicit_bass = {
        id(i)
        for i in pm.instruments
        if any(k in (i.name or "").lower() for k in ("808", "bass", "sub"))
    }
    bass_set = {id(i) for i in roles["bass"]} | explicit_bass
    chord_candidates: list[pretty_midi.Instrument] = list(roles["chords"])
    seen_ids = {id(i) for i in chord_candidates}
    for inst in pm.instruments:
        if id(inst) in seen_ids or id(inst) in explicit_bass or _is_perc_inst(inst):
            continue
        if any(k in (inst.name or "").lower() for k in ("chord", "pad", "string", "rhodes", "harmony", "keys", "ep")):
            chord_candidates.append(inst)
            seen_ids.add(id(inst))

    for inst in chord_candidates:
        slices = _group_chord_note_slices(inst.notes)
        poly_slices = [sl for sl in slices if len({n.pitch % 12 for n in sl}) >= 2]
        if not poly_slices:
            continue
        needs_revoice = any(n.pitch < 48 for sl in poly_slices for n in sl)
        if not needs_revoice:
            continue
        chord_pcs = [sorted({n.pitch % 12 for n in sl}) for sl in poly_slices]
        voiced = optimize_voice_leading(chord_pcs, low_midi=48, high_midi=chord_high)
        tr_name = inst.name or f"Track_{inst.program}"
        for sl, new_voicing in zip(poly_slices, voiced):
            if not any(n.pitch < 48 for n in sl) or not new_voicing:
                continue
            sl_sorted = sorted(sl, key=lambda n: n.pitch)
            for idx, note in enumerate(sl_sorted):
                target_p = new_voicing[min(idx, len(new_voicing) - 1)]
                if len(sl_sorted) != len(new_voicing):
                    pc_matches = [v for v in new_voicing if (v % 12) == (note.pitch % 12)]
                    target_p = pc_matches[0] if pc_matches else min(new_voicing, key=lambda v: abs(v - note.pitch))
                if target_p != note.pitch:
                    old_p = int(note.pitch)
                    note.pitch = int(target_p)
                    fixed += 1
                    changes.append(
                        {
                            "rule": "sub_collisions",
                            "track": tr_name,
                            "start_s": round(float(note.start), 3),
                            "old_pitch": old_p,
                            "new_pitch": int(target_p),
                            "delta_st": int(target_p) - old_p,
                        }
                    )

    # Secondary safety check: lift non-bass notes < 48 that still overlap another < 48 note
    for inst in pm.instruments:
        if _is_perc_inst(inst) or id(inst) in bass_set:
            continue
        tr_name = inst.name or f"Track_{inst.program}"
        for note in inst.notes:
            if note.pitch < 48 and note.pitch + 12 <= max_pitch:
                old_p = int(note.pitch)
                note.pitch += 12
                fixed += 1
                changes.append(
                    {
                        "rule": "sub_collisions",
                        "track": tr_name,
                        "start_s": round(float(note.start), 3),
                        "old_pitch": old_p,
                        "new_pitch": int(note.pitch),
                        "delta_st": 12,
                    }
                )
    return fixed


def _fix_strong_beat_chord_tones(
    pm: pretty_midi.PrettyMIDI,
    allowed_pcs: set[int],
    min_pitch: int,
    max_pitch: int,
    valence: float,
    tension: float,
    time_signature: tuple[int, int],
    phrase_plan: str,
    changes: list[dict[str, Any]],
) -> int:
    roles = _infer_roles(pm)
    has_explicit_lead = bool(roles["lead"]) and any(inst not in roles["chords"] for inst in roles["lead"])
    if not has_explicit_lead:
        return 0
    lead_notes = [n for inst in roles["lead"] for n in inst.notes]
    chord_notes = [n for inst in roles["chords"] for n in inst.notes]
    bass_notes = [n for inst in roles["bass"] for n in inst.notes]
    if not lead_notes or (not chord_notes and not bass_notes):
        return 0

    topline = _extract_monophonic_topline(lead_notes)
    dark_tense = (valence < -0.3) or (tension >= 0.5)
    is_free_ambient = (phrase_plan or "").strip().lower() == "free_ambient"
    target_min = 0.65 if (dark_tense or is_free_ambient) else 0.80
    cur_ratio = _compute_strong_beat_chord_tone_ratio(
        pm, topline, chord_notes, bass_notes, time_signature=time_signature
    )
    if cur_ratio >= target_min:
        return 0

    strong_beats = _strong_beat_times(pm, time_signature[0], time_signature[1], total_dur=pm.get_end_time())
    lead_note_track: dict[int, str] = {
        id(n): (inst.name or f"Track_{inst.program}")
        for inst in roles["lead"]
        for n in inst.notes
    }

    fixed = 0
    for note in topline:
        if np.min(np.abs(strong_beats - note.start)) > 0.09:
            continue
        active_pcs = {
            c.pitch % 12
            for c in (chord_notes + bass_notes)
            if c.start - 0.05 <= note.start <= c.end + 0.05
        }
        if not active_pcs or (note.pitch % 12) in active_pcs:
            continue
        valid_pcs = (active_pcs & allowed_pcs) or active_pcs
        candidates = [
            note.pitch + d
            for d in (-1, 1, -2, 2)
            if min_pitch <= note.pitch + d <= max_pitch and ((note.pitch + d) % 12) in valid_pcs
        ]
        if not candidates:
            continue
        new_p = min(candidates, key=lambda p: abs(p - note.pitch))
        old_p = int(note.pitch)
        note.pitch = int(new_p)
        fixed += 1
        changes.append(
            {
                "rule": "strong_beat_chord_tone_ratio",
                "track": lead_note_track.get(id(note), "Lead"),
                "start_s": round(float(note.start), 3),
                "old_pitch": old_p,
                "new_pitch": int(new_p),
                "delta_st": int(new_p) - old_p,
            }
        )
    return fixed


def autofix_midi(
    midi_path: str,
    expected_key: str = "",
    min_pitch: int = 31,
    max_pitch: int = 74,
    strict_musicality: bool = True,
    valence: float = 0.0,
    tension: float = 0.3,
    time_signature: tuple[int, int] | str | None = None,
    phrase_plan: str = "",
    out_path: str | None = None,
    in_place: bool = False,
) -> dict[str, Any]:
    """
    Runs deterministic MIDI repairs, writes a corrected MIDI file, re-audits via `audit_midi_file`,
    and returns a structured `fix_report`.
    """
    abs_in = os.path.abspath(os.path.expanduser(midi_path))
    if not os.path.isfile(abs_in):
        return {"error": f"Fichier MIDI introuvable : {abs_in}", "gate_passed": False}

    initial_audit = audit_midi_file(
        midi_path=abs_in,
        expected_key=expected_key,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        strict_musicality=strict_musicality,
        valence=valence,
        tension=tension,
        time_signature=time_signature,
        phrase_plan=phrase_plan,
    )

    ref_key = (expected_key or "").strip() or str(initial_audit.get("detected_key") or "C minor")
    allowed_pcs = resolve_allowed_pcs(ref_key)
    pm = pretty_midi.PrettyMIDI(abs_in)
    resolved_ts = _resolve_time_signature(pm, time_signature)

    changes: list[dict[str, Any]] = []
    oos_fixed = _fix_out_of_scale(pm, allowed_pcs, min_pitch, max_pitch, changes)
    high_fixed, low_fixed = _fix_pitch_bounds(pm, min_pitch, max_pitch, changes)
    sub_fixed = _fix_sub_collisions(pm, max_pitch, changes)
    sb_fixed = _fix_strong_beat_chord_tones(
        pm=pm,
        allowed_pcs=allowed_pcs,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        valence=valence,
        tension=tension,
        time_signature=resolved_ts,
        phrase_plan=phrase_plan,
        changes=changes,
    )

    if out_path:
        abs_out = os.path.abspath(os.path.expanduser(out_path))
    elif in_place:
        abs_out = abs_in
    else:
        stem, ext = os.path.splitext(abs_in)
        abs_out = f"{stem}_fixed{ext or '.mid'}"

    os.makedirs(os.path.dirname(abs_out) or ".", exist_ok=True)
    pm.write(abs_out)

    post_audit = audit_midi_file(
        midi_path=abs_out,
        expected_key=expected_key,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        strict_musicality=strict_musicality,
        valence=valence,
        tension=tension,
        time_signature=resolved_ts,
        phrase_plan=phrase_plan,
    )

    remaining_structural = [
        f
        for f in post_audit.get("musicality_failures", [])
        if any(f.startswith(prefix) for prefix in STRUCTURAL_PREFIXES)
    ]

    return {
        "source_midi_path": abs_in,
        "fixed_midi_path": abs_out,
        "expected_key": ref_key,
        "time_signature": f"{resolved_ts[0]}/{resolved_ts[1]}",
        "counts_by_rule": {
            "out_of_scale": oos_fixed,
            "too_high_count": high_fixed,
            "too_low_count": low_fixed,
            "sub_collisions": sub_fixed,
            "strong_beat_chord_tone_ratio": sb_fixed,
        },
        "total_changes": len(changes),
        "changes": changes,
        "initial_gate_passed": bool(initial_audit.get("gate_passed")),
        "gate_passed": bool(post_audit.get("gate_passed")),
        "remaining_structural_failures": remaining_structural,
        "initial_audit": initial_audit,
        "post_audit": post_audit,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic MIDI Auto-Fix tool (Axe 3 §3.1)")
    ap.add_argument("--midi", required=True, help="Input MIDI path")
    ap.add_argument("--key", default="", help="Expected key (e.g. 'C minor', 'F# phrygian')")
    ap.add_argument("--out", default="", help="Output MIDI path (default: <stem>_fixed.mid)")
    ap.add_argument("--in-place", action="store_true", help="Overwrite input MIDI file")
    ap.add_argument("--min-pitch", type=int, default=31)
    ap.add_argument("--max-pitch", type=int, default=74)
    ap.add_argument("--valence", type=float, default=0.0)
    ap.add_argument("--tension", type=float, default=0.3)
    ap.add_argument("--time-signature", default="", help="Optional meter override like '3/4' or '6/8'")
    ap.add_argument("--phrase-plan", default="", help="Optional phrase plan (e.g. 'free_ambient')")
    args = ap.parse_args()

    res = autofix_midi(
        midi_path=args.midi,
        expected_key=args.key,
        min_pitch=args.min_pitch,
        max_pitch=args.max_pitch,
        valence=args.valence,
        tension=args.tension,
        time_signature=args.time_signature or None,
        phrase_plan=args.phrase_plan,
        out_path=args.out or None,
        in_place=args.in_place,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if not res.get("gate_passed"):
        sys.exit(2)


if __name__ == "__main__":
    main()
