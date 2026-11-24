#!/usr/bin/env python3
"""
Groove & Micro-Humanization Engine (Axe 5 §5.1).

Operates on 0-indexed beat note event lists (`[{"start", "length", "pitch", "velocity"}, ...]`,
also accepting `"duration"` fallback cleanly) before MIDI export & Gridline compilation:
  - `apply_micro_strum`: strums simultaneous chord stacks ascending by 10-25 ms & softens inner voices.
  - `apply_haas_pocket`: psychoacoustic role timing shifts in ms converted cleanly to beats.
  - `apply_anti_machine_gun`: metric accent curve + deterministic jitter for subdivisions >= 1/16.
  - `enforce_frequency_inertia`: prevents rapid/short < MIDI 45 (< 100 Hz) sub-bass chatter unless staccato.
"""
from __future__ import annotations

import copy
import random
from typing import Any


def ms_to_beats(ms: float, bpm: float) -> float:
    """Convert milliseconds to beats using `ms / (60000 / bpm)`."""
    return float(ms) * float(bpm) / 60000.0


def _get_len(note: dict[str, Any]) -> float:
    if "length" in note:
        return float(note["length"])
    return float(note.get("duration", 0.25))


def _set_len(note: dict[str, Any], new_len: float) -> None:
    if "length" in note or "duration" not in note:
        note["length"] = round(float(new_len), 5)
    else:
        note["duration"] = round(float(new_len), 5)


def _clamp_vel(vel: float) -> int:
    return max(15, min(127, int(round(vel))))


def apply_micro_strum(
    notes: list[dict[str, Any]],
    strum_ms: float = 18.0,
    bpm: float = 120.0,
    inner_vel_drop: int = 8,
    chord_tol_beats: float = 0.04,
) -> list[dict[str, Any]]:
    """
    Detect simultaneous chord voicings, stagger ascending notes smoothly over `strum_ms`
    (recommended 10-25 ms total spread), and attenuate inner-voice velocities cleanly.
    """
    if not notes:
        return []
    copied = [copy.copy(n) for n in notes]
    copied.sort(key=lambda n: (float(n["start"]), int(n["pitch"])))

    groups: list[list[dict[str, Any]]] = [[copied[0]]]
    anchor = float(copied[0]["start"])
    for n in copied[1:]:
        if abs(float(n["start"]) - anchor) <= chord_tol_beats:
            groups[-1].append(n)
        else:
            groups.append([n])
            anchor = float(n["start"])

    total_spread_beats = ms_to_beats(max(0.0, float(strum_ms)), bpm)
    for grp in groups:
        if len(grp) < 2:
            continue
        distinct_pitches = {int(n["pitch"]) for n in grp}
        if len(distinct_pitches) < 2:
            continue
        grp.sort(key=lambda n: int(n["pitch"]))
        n_voices = len(grp)
        step_beats = total_spread_beats / float(n_voices - 1)
        for idx, note in enumerate(grp):
            offset = idx * step_beats
            orig_start = float(note["start"])
            orig_len = _get_len(note)
            note["start"] = round(orig_start + offset, 5)
            # Keep original note-off alignment so legato chords don't bleed across barline
            _set_len(note, max(0.05, orig_len - offset))
            if 0 < idx < n_voices - 1 and inner_vel_drop > 0:
                note["velocity"] = _clamp_vel(float(note.get("velocity", 80)) - float(inner_vel_drop))

    copied.sort(key=lambda n: (float(n["start"]), int(n["pitch"])))
    return copied


# Haas pocket timing offsets (ms) per instrument role keyword across genre profiles.
# Spec guidelines: Kick 0 ms, Snare -5..+10 ms, Hats +5..+20 ms, Bass +10..+25 ms.
HAAS_POCKET_MS_PROFILES: dict[str, dict[str, float]] = {
    "trap_rap": {
        "kick": 0.0,
        "clap": 2.0,
        "snare": 5.0,
        "hat": 12.0,
        "perc": 10.0,
        "808": 15.0,
        "bass": 15.0,
        "sub": 15.0,
        "chord": 8.0,
        "pad": 10.0,
        "lead": 4.0,
        "counter": 7.0,
    },
    "lofi_chill": {
        "kick": 0.0,
        "clap": 4.0,
        "snare": 8.0,
        "hat": 16.0,
        "perc": 14.0,
        "808": 20.0,
        "bass": 18.0,
        "chord": 12.0,
        "keys": 10.0,
        "pad": 12.0,
        "lead": 6.0,
    },
    "pop_rnb": {
        "kick": 0.0,
        "clap": 0.0,
        "snare": -3.0,
        "hat": 8.0,
        "perc": 7.0,
        "bass": 12.0,
        "808": 12.0,
        "chord": 6.0,
        "keys": 6.0,
        "lead": 2.0,
    },
    "edm_club": {
        "kick": 0.0,
        "clap": 1.0,
        "snare": 2.0,
        "hat": 6.0,
        "perc": 6.0,
        "bass": 10.0,
        "sub": 10.0,
        "chord": 5.0,
        "lead": 0.0,
    },
    "rock_metal": {
        "kick": 0.0,
        "snare": -2.0,
        "hat": 5.0,
        "perc": 5.0,
        "bass": 10.0,
        "guitar": 4.0,
        "chord": 4.0,
        "lead": 0.0,
    },
    "jazz_acoustic": {
        "kick": 0.0,
        "snare": 6.0,
        "hat": 14.0,
        "ride": 12.0,
        "bass": 16.0,
        "piano": 8.0,
        "chord": 8.0,
        "lead": 4.0,
    },
    "classical_ambient": {
        "piano": 5.0,
        "chord": 6.0,
        "pad": 8.0,
        "strings": 6.0,
        "lead": 0.0,
        "bass": 10.0,
    },
}


def _pocket_shift_for_track(track_name: str, profile: str) -> float:
    prof = HAAS_POCKET_MS_PROFILES.get(profile, HAAS_POCKET_MS_PROFILES["trap_rap"])
    low = track_name.lower()
    for kw, ms in prof.items():
        if kw in low:
            return float(ms)
    return 0.0


def apply_haas_pocket(
    tracks_dict: dict[str, list[dict[str, Any]]],
    bpm: float,
    profile: str = "trap_rap",
) -> dict[str, list[dict[str, Any]]]:
    """
    Apply subtle role-based psychoacoustic timing offsets in ms (converted accurately to beats)
    to create pocket groove & separate Kick/Bass transients cleanly.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    beat_ms = 60000.0 / float(bpm) if bpm else 500.0
    max_shift_ms = 0.05 * beat_ms  # never smear more than 5% of a beat (fast tempos)
    for tr_name, notes in tracks_dict.items():
        shift_ms = min(_pocket_shift_for_track(tr_name, profile), max_shift_ms)
        shift_beats = ms_to_beats(shift_ms, bpm)
        updated: list[dict[str, Any]] = []
        for n in notes:
            nc = copy.copy(n)
            nc["start"] = round(max(0.0, float(n["start"]) + shift_beats), 5)
            updated.append(nc)
        out[tr_name] = updated
    return out


def apply_swing(
    notes: list[dict[str, Any]],
    swing: float = 0.0,
    beats_per_bar: float = 4.0,
) -> list[dict[str, Any]]:
    """
    Delay off-beat 8th notes by `swing * 0.25` beats (swing in [0, 1), blueprint convention).

    Only notes sitting exactly on an off-beat 8th grid position (odd multiple of 0.5 beat)
    are moved, so syncopated notes keep their exact placement.
    """
    if not notes or swing <= 0.0:
        return notes
    delay_beats = max(0.0, min(0.75, float(swing))) * 0.25
    out = []
    for n in notes:
        nc = copy.copy(n)
        t = float(nc["start"])
        eighth_idx = t / 0.5
        if abs(eighth_idx - round(eighth_idx)) < 1e-3 and int(round(eighth_idx)) % 2 == 1:
            nc["start"] = round(t + delay_beats, 5)
        out.append(nc)
    return out


def apply_anti_machine_gun(
    notes: list[dict[str, Any]],
    subdivision_thresh_beats: float = 0.25,
    vel_jitter_pct: float = 5.0,
    beats_per_bar: float = 4.0,
    seed: int = 42,
    humanize_sparse: bool = False,
) -> list[dict[str, Any]]:
    """
    Apply a metric accent curve + deterministic velocity jitter (`seed`) to any note sequence
    containing dense subdivisions (`IOI <= subdivision_thresh_beats`, i.e., >= 1/16 notes).
    With `humanize_sparse=True`, sparse notes also get a gentle metric accent (no jitter) so
    long chords/leads do not sound like a drum machine.
    """
    if not notes:
        return []
    copied = [copy.copy(n) for n in notes]
    copied.sort(key=lambda n: (float(n["start"]), int(n["pitch"])))
    rng = random.Random(seed)
    frac_jit = max(0.0, float(vel_jitter_pct)) / 100.0

    in_rapid = [False] * len(copied)
    for i in range(len(copied) - 1):
        ioi = float(copied[i + 1]["start"]) - float(copied[i]["start"])
        if 1e-5 < ioi <= subdivision_thresh_beats + 1e-4:
            in_rapid[i] = True
            in_rapid[i + 1] = True

    for i, note in enumerate(copied):
        t = float(note["start"])
        bar_phase = t % max(1.0, float(beats_per_bar))
        beat_frac = bar_phase % 1.0
        if in_rapid[i]:
            # Strong quarter beat > 8th offbeat > 16th in-between
            if beat_frac < 0.06 or beat_frac > 0.94:
                metric_mult = 1.06 if (bar_phase < 0.06 or abs(bar_phase - 2.0) < 0.06) else 1.02
            elif abs(beat_frac - 0.5) < 0.06:
                metric_mult = 0.92
            else:
                metric_mult = 0.82
            jit_mult = 1.0 + rng.uniform(-frac_jit, frac_jit)
        elif humanize_sparse:
            # Gentle metric accent only (no jitter) for sparse notes
            if beat_frac < 0.06 or beat_frac > 0.94:
                metric_mult = 1.04 if (bar_phase < 0.06 or abs(bar_phase - 2.0) < 0.06) else 1.01
            elif abs(beat_frac - 0.5) < 0.06:
                metric_mult = 0.96
            else:
                metric_mult = 0.90
            jit_mult = 1.0
        else:
            continue
        orig_v = float(note.get("velocity", 90))
        note["velocity"] = _clamp_vel(orig_v * metric_mult * jit_mult)

    return copied


def enforce_frequency_inertia(
    bass_notes: list[dict[str, Any]],
    bpm: float = 120.0,
    max_sub_rate_beats: float = 0.5,
    staccato: bool = False,
) -> list[dict[str, Any]]:
    """
    Acoustic physics guard for low sub-bass (`pitch < 45` ~ < 110 Hz):
    merges or lengthens sub notes shorter than `max_sub_rate_beats` (1/8 note default)
    unless `staccato=True`, avoiding low-frequency click/flap artifacts.
    """
    del bpm  # kept in signature for API parity & time-aware extensions
    if not bass_notes:
        return []
    copied = [copy.copy(n) for n in bass_notes]
    copied.sort(key=lambda n: (float(n["start"]), int(n["pitch"])))
    if staccato:
        return copied

    out: list[dict[str, Any]] = []
    i = 0
    while i < len(copied):
        cur = copy.copy(copied[i])
        cur_pitch = int(cur["pitch"])
        cur_start = float(cur["start"])
        cur_len = _get_len(cur)
        if cur_pitch < 45 and cur_len < max_sub_rate_beats:
            # Merge rapid back-to-back repeats on identical sub pitch
            while (
                i + 1 < len(copied)
                and int(copied[i + 1]["pitch"]) == cur_pitch
                and float(copied[i + 1]["start"]) - (cur_start + cur_len) <= 0.12
                and cur_len < max_sub_rate_beats
            ):
                nxt = copied[i + 1]
                cur_len = (float(nxt["start"]) + _get_len(nxt)) - cur_start
                i += 1
            # Extend remaining sub note up to `max_sub_rate_beats` without overlapping next note
            if cur_len < max_sub_rate_beats:
                if i + 1 < len(copied):
                    avail = float(copied[i + 1]["start"]) - cur_start - 0.02
                    cur_len = max(cur_len, min(max_sub_rate_beats, avail))
                else:
                    cur_len = max_sub_rate_beats
            _set_len(cur, cur_len)
        out.append(cur)
        i += 1
    return out


MONO_ROLE_SUBSTRINGS = ("bass", "808", "sub", "lead_mono")


def sanitize_note_overlaps(
    tracks: dict[str, list[dict[str, Any]]],
    bpm: float = 120.0,
    guard_ms: float = 10.0,
    mono_roles: set[str] | None = None,
    voices_by_role: dict[str, int] | None = None,
    exempt_pairs: set[tuple[str, float]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """v3.0 mandatory final pipeline step: no note-stealing on monophonic tracks and no
    same-pitch overlaps anywhere (run AFTER swing/strum/haas/inertia).

    Mono tracks (role matches MONO_ROLE_SUBSTRINGS / mono_roles, or voices_by_role == 1):
    notes are sorted by start and each end is clamped to `next_start - guard` with
    `guard = ms_to_beats(guard_ms, bpm)` (10 ms default) so a Voices=1 sampler never
    steals the next note's transient. Polyphonic tracks: an earlier note of the SAME
    pitch is clipped when it overlaps the next one.
    """
    guard = ms_to_beats(float(guard_ms), float(bpm))
    mono_roles = {str(r).lower() for r in (mono_roles or ())}
    voices_by_role = voices_by_role or {}
    exempt_pairs = exempt_pairs or set()
    out: dict[str, list[dict[str, Any]]] = {}
    for key, notes in tracks.items():
        rl = str(key).lower()
        is_mono = (
            rl in mono_roles
            or any(m in rl for m in MONO_ROLE_SUBSTRINGS)
            or voices_by_role.get(rl) == 1
            or voices_by_role.get(str(key)) == 1
        )
        copied = sorted((copy.copy(n) for n in notes), key=lambda n: (float(n["start"]), int(n["pitch"])))
        if is_mono:
            for i in range(len(copied) - 1):
                cur, nxt = copied[i], copied[i + 1]
                cur_end = float(cur["start"]) + _get_len(cur)
                nxt_start = float(nxt["start"])
                if (key, round(float(cur["start"]), 5)) in exempt_pairs:
                    continue  # v3.2 : paire glide 808 volontaire (glide_to_next), non coupée
                if cur_end > nxt_start - guard:
                    _set_len(cur, max(0.02, nxt_start - guard - float(cur["start"])))
        else:
            by_pitch: dict[int, list[dict[str, Any]]] = {}
            for n in copied:
                by_pitch.setdefault(int(n["pitch"]), []).append(n)
            for ns in by_pitch.values():
                ns.sort(key=lambda n: float(n["start"]))
                for i in range(len(ns) - 1):
                    cur, nxt = ns[i], ns[i + 1]
                    if float(cur["start"]) + _get_len(cur) > float(nxt["start"]):
                        _set_len(cur, max(0.02, float(nxt["start"]) - float(cur["start"])))
        out[key] = sorted(copied, key=lambda n: (float(n["start"]), int(n["pitch"])))
    return out
