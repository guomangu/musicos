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

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
if ANALYZER_DIR not in sys.path:
    sys.path.insert(0, ANALYZER_DIR)

from deep_diagnostics import audit_midi_file, resolve_allowed_pcs
from musicality_metrics import (
    _PERC_TRACK_RE,
    _bar_quarters,
    _compute_strong_beat_chord_tone_ratio,
    _extract_monophonic_topline,
    _infer_roles,
    _note_quarter_pos,
    _resolve_time_signature,
    _strong_beat_times,
    compute_motif_cohesion,
    compute_rhythmic_interlocking,
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


def contour_preserving_cost(
    cand: int,
    n_k: int,
    n_prev: int | None,
    b_k: int | None,
    b_prev: int | None,
    w: tuple[float, float, float, float] = (1.0, 3.0, 4.0, 5.0),
) -> float:
    """v3.0 contour-preserving autofix cost J(c) = w1|c-nk| + w2*I(sign change) + w3*I(c==n_prev)
    + w4*I(parallel 5th/8ve with the bass). Minimizes -> keeps the melodic intention."""
    j = w[0] * abs(cand - n_k)
    if n_prev is not None:
        if (cand - n_prev) * (n_k - n_prev) < 0:
            j += w[1]
        if cand == n_prev:
            j += w[2]
        if b_k is not None and b_prev is not None:
            prev_iv = (n_prev - b_prev) % 12
            cand_iv = (cand - b_k) % 12
            if prev_iv in (7, 0) and cand_iv == prev_iv and (cand - n_prev) * ((b_k - b_prev)) > 0:
                j += w[3]
    return j


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
        # v3.0: candidates in a +-5 st window, chosen by contour-preserving cost J(c)
        # (not by raw distance) so an ascending phrase is not flattened into repeats.
        candidates = [
            note.pitch + d
            for d in range(-5, 6)
            if d != 0 and min_pitch <= note.pitch + d <= max_pitch and ((note.pitch + d) % 12) in valid_pcs
        ]
        if not candidates:
            continue
        topline_sorted = sorted(topline, key=lambda n: n.start)
        idx_k = topline_sorted.index(note)
        n_prev = int(topline_sorted[idx_k - 1].pitch) if idx_k > 0 else None
        n_next = int(topline_sorted[idx_k + 1].pitch) if idx_k + 1 < len(topline_sorted) else None
        del n_next  # reserved for future look-ahead terms
        b_k = b_prev = None
        if bass_notes:
            bs = sorted(bass_notes, key=lambda n: n.start)
            cur = [b for b in bs if b.start - 0.05 <= note.start <= b.end + 0.05]
            if cur:
                b_k = int(cur[0].pitch)
                prev_b = [b for b in bs if b.end <= note.start + 0.05]
                b_prev = int(prev_b[-1].pitch) if prev_b else None
        new_p = min(
            candidates,
            key=lambda p: contour_preserving_cost(int(p), int(note.pitch), n_prev, b_k, b_prev),
        )
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


def fix_motif_clone(
    pm: pretty_midi.PrettyMIDI,
    allowed_pcs: set[int],
    min_pitch: int = 31,
    max_pitch: int = 74,
    time_signature: tuple[int, int] | str | None = None,
    phrase_plan: str = "",
    changes: list[dict[str, Any]] | None = None,
) -> int:
    """
    Phase 6 §3.3 (`S3/H`): Mutate bar 2 (`A'`) and/or bar 3 (`A''`) of a cloned
    4-bar Lead phrase (`ROBOTIC_LOOP_CLONE`) with diatonic chord/scale degree shifts
    and a rhythmic half-beat variation while preserving stepwise motion and rests.
    """
    ch_log = changes if changes is not None else []
    resolved_ts = _resolve_time_signature(pm, time_signature)
    motif = compute_motif_cohesion(pm, time_signature=resolved_ts, phrase_plan=phrase_plan)
    if motif.get("issue") != "ROBOTIC_LOOP_CLONE":
        return 0

    roles = _infer_roles(pm)
    lead_insts = [i for i in roles["lead"] if i not in roles["chords"]]
    if not lead_insts:
        return 0

    chord_notes = [n for inst in roles["chords"] for n in inst.notes]
    bass_notes = [n for inst in roles["bass"] for n in inst.notes]
    bar_q = _bar_quarters(resolved_ts[0], resolved_ts[1])
    eff_min = max(48, int(min_pitch))

    def _bar_notes_map(inst: pretty_midi.Instrument) -> dict[int, list[pretty_midi.Note]]:
        out: dict[int, list[pretty_midi.Note]] = {}
        for n in sorted(inst.notes, key=lambda x: x.start):
            q = _note_quarter_pos(pm, n.start)
            b_idx = max(0, int((q + 1e-6) // max(bar_q, 1e-4)))
            out.setdefault(b_idx, []).append(n)
        return out

    def _safe_diatonic_neighbor(note: pretty_midi.Note, target_delta: int, prev_p: int | None, next_p: int | None) -> int | None:
        # Prefer active chord tones if on strong beat, else allowed scale PCs without b2 clash
        active_harm = [
            c.pitch for c in (chord_notes + bass_notes)
            if c.start - 0.04 <= note.start <= c.end + 0.04
        ]
        harm_pcs = {p % 12 for p in active_harm}
        cands: list[tuple[int, int]] = []
        for delta in (-4, -3, -2, -1, 1, 2, 3, 4):
            cand = int(note.pitch) + delta
            if not (eff_min <= cand <= max_pitch):
                continue
            if allowed_pcs and (cand % 12) not in allowed_pcs:
                continue
            # Avoid minor 2nd / 9th clash against active harmony
            if any(abs(cand - hp) in (1, 13) for hp in active_harm):
                continue
            # Keep stepwise smoothness
            step_ok = True
            if prev_p is not None and abs(cand - prev_p) > 4:
                step_ok = False
            if next_p is not None and abs(cand - next_p) > 4:
                step_ok = False
            if not step_ok:
                continue
            score = abs(delta - target_delta) * 10 - (3 if (cand % 12) in harm_pcs else 0)
            cands.append((score, cand))
        if not cands:
            return None
        cands.sort(key=lambda x: x[0])
        return cands[0][1]

    fixed = 0
    for block_start_1 in motif.get("clone_start_bars") or []:
        b0 = int(block_start_1) - 1
        for inst in lead_insts:
            tr_name = inst.name or f"Track_{inst.program}"
            bmap = _bar_notes_map(inst)

            # Mutate bar A' (b0 + 1) and bar A'' (b0 + 2) differently
            for offset_bar, target_delta, rhy_shift_q in ((1, 2, 0.5), (2, 3, -0.5)):
                notes_b = bmap.get(b0 + offset_bar, [])
                if not notes_b:
                    continue
                # Pick a non-first note if possible (preserves strong-beat anchor)
                pick_idx = min(len(notes_b) - 1, 1 if len(notes_b) >= 2 else 0)
                if len(notes_b) >= 3 and offset_bar == 2:
                    pick_idx = len(notes_b) - 1
                target_note = notes_b[pick_idx]
                prev_p = notes_b[pick_idx - 1].pitch if pick_idx > 0 else None
                next_p = notes_b[pick_idx + 1].pitch if pick_idx + 1 < len(notes_b) else None

                new_p = _safe_diatonic_neighbor(target_note, target_delta, prev_p, next_p)
                if new_p is None:
                    new_p = _safe_diatonic_neighbor(target_note, -target_delta, prev_p, next_p)
                if new_p is not None and new_p != target_note.pitch:
                    old_p = int(target_note.pitch)
                    target_note.pitch = int(new_p)
                    fixed += 1
                    ch_log.append({
                        "rule": "fix_motif_clone",
                        "track": tr_name,
                        "start_s": round(float(target_note.start), 3),
                        "old_pitch": old_p,
                        "new_pitch": int(new_p),
                        "delta_st": int(new_p) - old_p,
                    })

                # Also shift one rhythmic onset by half a beat without overlapping neighbors
                rhy_idx = min(len(notes_b) - 1, 1 if len(notes_b) >= 2 else 0)
                rnote = notes_b[rhy_idx]
                ppq = float(pm.resolution or 480)
                q_cur = _note_quarter_pos(pm, rnote.start)
                q_end = _note_quarter_pos(pm, rnote.end)
                bar_start_q = float(b0 + offset_bar) * bar_q
                bar_end_q = bar_start_q + bar_q
                q_new = q_cur + rhy_shift_q
                min_q = (_note_quarter_pos(pm, notes_b[rhy_idx - 1].end) + 0.05) if rhy_idx > 0 else bar_start_q
                max_q = (_note_quarter_pos(pm, notes_b[rhy_idx + 1].start) - 0.20) if rhy_idx + 1 < len(notes_b) else (bar_end_q - 0.25)
                if min_q <= q_new <= max_q:
                    new_start_s = float(pm.tick_to_time(int(round(q_new * ppq))))
                    dur_s = max(0.08, float(rnote.end - rnote.start))
                    next_lim_s = float(notes_b[rhy_idx + 1].start) - 0.02 if rhy_idx + 1 < len(notes_b) else float(pm.tick_to_time(int(round(bar_end_q * ppq))))
                    new_end_s = min(next_lim_s, new_start_s + dur_s)
                    if new_end_s > new_start_s + 0.06:
                        old_s = round(float(rnote.start), 3)
                        rnote.start = new_start_s
                        rnote.end = new_end_s
                        fixed += 1
                        ch_log.append({
                            "rule": "fix_motif_clone_rhythm",
                            "track": tr_name,
                            "old_start_s": old_s,
                            "new_start_s": round(new_start_s, 3),
                        })

    return fixed


def fix_chords_lead_overlap(
    pm: pretty_midi.PrettyMIDI,
    time_signature: tuple[int, int] | str | None = None,
    max_overlap: float = 0.30,
    changes: list[dict[str, Any]] | None = None,
) -> int:
    """
    Phase 6 §3.3 (`S3/V`): Relieve excessive `chords_lead_attack_overlap` (> `max_overlap`)
    by anticipating (preferred: preserves strong-beat chord harmony) or delaying colliding
    `Chords` slices by 1/16th of a bar so Lead attacks speak cleanly.
    """
    ch_log = changes if changes is not None else []
    resolved_ts = _resolve_time_signature(pm, time_signature)
    inter = compute_rhythmic_interlocking(
        pm, time_signature=resolved_ts, max_chords_lead_overlap=max_overlap
    )
    cur_overlap = inter.get("chords_lead_attack_overlap")
    if cur_overlap is None or cur_overlap <= float(max_overlap):
        return 0

    roles = _infer_roles(pm)
    chord_insts = roles["chords"]
    lead_insts = roles["lead"]
    if not chord_insts or not lead_insts:
        return 0

    bar_q = _bar_quarters(resolved_ts[0], resolved_ts[1])
    ppq = float(pm.resolution or 480)
    step_q = bar_q / 16.0

    def _slot_of_time(t_s: float) -> tuple[int, int]:
        q = _note_quarter_pos(pm, t_s)
        b_idx = max(0, int((q + 1e-6) // max(bar_q, 1e-4)))
        ph = (q - b_idx * bar_q) / max(bar_q, 1e-4)
        return b_idx, min(15, max(0, int(round(ph * 16.0))))

    lead_slots: set[tuple[int, int]] = {
        _slot_of_time(n.start) for inst in lead_insts for n in inst.notes
    }

    fixed = 0
    for inst in chord_insts:
        tr_name = inst.name or f"Track_{inst.program}"
        slices = _group_chord_note_slices(inst.notes)
        for sl_idx, sl in enumerate(slices):
            # Re-evaluate stopping criterion
            inter_now = compute_rhythmic_interlocking(
                pm, time_signature=resolved_ts, max_chords_lead_overlap=max_overlap
            )
            if (inter_now.get("chords_lead_attack_overlap") or 0.0) <= float(max_overlap):
                return fixed

            t_ref = min(n.start for n in sl)
            b_idx, slot = _slot_of_time(t_ref)
            if (b_idx, slot) not in lead_slots:
                continue

            q_ref = _note_quarter_pos(pm, t_ref)
            # Option A: Anticipate by 1/16th (keeps chord sounding ON the strong beat!)
            shifted = False
            if q_ref >= step_q * 0.9:
                q_ant = max(0.0, q_ref - step_q)
                t_ant = float(pm.tick_to_time(int(round(q_ant * ppq))))
                ant_slot = _slot_of_time(t_ant)
                prev_end = max(n.end for n in slices[sl_idx - 1]) if sl_idx > 0 else -1.0
                if ant_slot not in lead_slots:
                    # Trim previous slice slightly if needed so anticipation doesn't overlap
                    if sl_idx > 0 and prev_end > t_ant - 0.01:
                        for pn in slices[sl_idx - 1]:
                            if pn.end > t_ant - 0.01 and pn.start < t_ant - 0.05:
                                pn.end = max(pn.start + 0.05, t_ant - 0.01)
                    delta_s = t_ant - t_ref
                    for cn in sl:
                        cn.start = max(0.0, cn.start + delta_s)
                    shifted = True
                    fixed += 1
                    ch_log.append({
                        "rule": "fix_chords_lead_overlap",
                        "track": tr_name,
                        "mode": "anticipate_16th",
                        "old_start_s": round(float(t_ref), 3),
                        "new_start_s": round(float(t_ant), 3),
                    })

            # Option B: Delay by 1/16th if anticipation wasn't possible (e.g. bar 1 beat 0)
            if not shifted:
                q_del = q_ref + step_q
                t_del = float(pm.tick_to_time(int(round(q_del * ppq))))
                del_slot = _slot_of_time(t_del)
                min_end = min(n.end for n in sl)
                if del_slot not in lead_slots and t_del < min_end - 0.05:
                    delta_s = t_del - t_ref
                    for cn in sl:
                        cn.start = cn.start + delta_s
                    fixed += 1
                    ch_log.append({
                        "rule": "fix_chords_lead_overlap",
                        "track": tr_name,
                        "mode": "delay_16th",
                        "old_start_s": round(float(t_ref), 3),
                        "new_start_s": round(float(t_del), 3),
                    })

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
    fix_s3: bool = False,
    strict_s3: bool = False,
) -> dict[str, Any]:
    """
    Runs deterministic MIDI repairs, writes a corrected MIDI file, re-audits via `audit_midi_file`,
    and returns a structured `fix_report`.
    """
    abs_in = os.path.abspath(os.path.expanduser(midi_path))
    if not os.path.isfile(abs_in):
        return {"error": f"Fichier MIDI introuvable : {abs_in}", "gate_passed": False}

    eff_strict_s3 = bool(strict_s3)
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
        strict_s3=eff_strict_s3,
    )

    ref_key = (expected_key or "").strip() or str(initial_audit.get("detected_key") or "C minor")
    allowed_pcs = resolve_allowed_pcs(ref_key)
    pm = pretty_midi.PrettyMIDI(abs_in)
    resolved_ts = _resolve_time_signature(pm, time_signature)

    changes: list[dict[str, Any]] = []
    oos_fixed = _fix_out_of_scale(pm, allowed_pcs, min_pitch, max_pitch, changes)
    high_fixed, low_fixed = _fix_pitch_bounds(pm, min_pitch, max_pitch, changes)
    sub_fixed = _fix_sub_collisions(pm, max_pitch, changes)
    motif_fixed = 0
    overlap_fixed = 0
    if fix_s3 or eff_strict_s3:
        motif_fixed = fix_motif_clone(
            pm,
            allowed_pcs=allowed_pcs,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            time_signature=resolved_ts,
            phrase_plan=phrase_plan,
            changes=changes,
        )
        overlap_fixed = fix_chords_lead_overlap(
            pm,
            time_signature=resolved_ts,
            changes=changes,
        )
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
        strict_s3=eff_strict_s3,
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
            "fix_motif_clone": motif_fixed,
            "fix_chords_lead_overlap": overlap_fixed,
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


# ---------------------------------------------------------------------------
# v3.1 IR-02 — autofix direct sur ScoreIR (court-circuite `_infer_roles`) +
# construction d'un ScoreIR inféré depuis un MIDI externe (compat ascendante).
# ---------------------------------------------------------------------------

def autofix_score_ir(score_ir: Any, thresholds: dict[str, Any] | None = None,
                     guard_ms: float = 20.0) -> tuple[Any, dict[str, Any]]:
    """v3.1 IR-02 : réparations déterministes sur le ScoreIR (snap gamme, clamp de
    frontière harmonique, trim des chevauchements mono)."""
    from score_ir import autofix_score_ir as _af
    return _af(score_ir, thresholds, guard_ms=guard_ms)


def score_ir_from_midi(midi_path: str, expected_key: str = "", bpm: float | None = None) -> Any:
    """v3.1 IR-02 : ScoreIR inféré depuis un MIDI importé (soundbanks/) — délègue à
    `musicality_metrics.infer_score_ir_from_midi` qui utilise `_infer_roles`."""
    from musicality_metrics import infer_score_ir_from_midi
    return infer_score_ir_from_midi(midi_path, expected_key=expected_key, bpm=bpm)
