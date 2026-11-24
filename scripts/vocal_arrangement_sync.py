#!/usr/bin/env python3
"""
Text-Music Prosodic Synchronization & Vocal Formant Ducking (`vocal_arrangement_sync.py`, Phase 7 P6).

Deterministic CPU-only workflow (0 tokens):
  1. `extract_vocal_phrases`: extracts active sung phrase intervals `[start_beat, end_beat]`
     from OpenUTAU `.ustx` notes/silences (priority) or `samples/vocals.wav` 50 ms RMS (> -35 dBFS,
     merging breaths < 250 ms).
  2. `mute_counter_during_vocals`: removes counter-melody notes overlapping active vocals or gaps
     shorter than `0.5` beat, keeping instrumental call-and-response inside breathing windows;
     gently attenuates overlapping secondary lead ornaments.
  3. `build_formant_ducking_automation`: generates `gl_dsl.eq_gain_to_auto` automation for
     `BUS Music` (`mixer=19, slot=2, band=5` peak `1850 Hz, Q=1.4`) ducking `0 -> -3.5 dB`
     (`40 ms` attack / `120 ms` release) during sung phrases only.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np
import soundfile as sf
import yaml

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from gl_dsl import eq_band, eq_gain_to_auto, eq_params


def _merge_intervals(
    intervals: list[tuple[float, float]],
    gap_tol_beats: float,
    source: str,
) -> list[dict[str, Any]]:
    if not intervals:
        return []
    ordered = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    if not ordered:
        return []
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for s_b, e_b in ordered[1:]:
        if s_b - merged[-1][1] <= gap_tol_beats:
            merged[-1][1] = max(merged[-1][1], e_b)
        else:
            merged.append([s_b, e_b])
    return [
        {
            "start_beat": round(a, 4),
            "end_beat": round(b, 4),
            "duration_beats": round(b - a, 4),
            "source": source,
        }
        for a, b in merged
    ]


def _extract_from_ustx(ustx_path: str, bpm: float, merge_gap_beats: float) -> list[dict[str, Any]]:
    with open(ustx_path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        return []
    ppq = max(1.0, float(doc.get("resolution") or 480.0))
    raw_intervals: list[tuple[float, float]] = []

    for part in doc.get("voice_parts") or []:
        if not isinstance(part, dict):
            continue
        part_pos = float(part.get("position") or 0.0)
        for note in part.get("notes") or []:
            if not isinstance(note, dict):
                continue
            lyric = str(note.get("lyric") or "").strip().lower()
            if lyric in ("r", "sil", "pau", "-", ""):
                continue
            pos = part_pos + float(note.get("position") or 0.0)
            dur = float(note.get("duration") or 0.0)
            if dur <= 0.0:
                continue
            s_beat = pos / ppq
            e_beat = (pos + dur) / ppq
            raw_intervals.append((s_beat, e_beat))

    return _merge_intervals(raw_intervals, merge_gap_beats, source="ustx")


def _extract_from_audio(
    audio_path: str,
    bpm: float,
    merge_gap_beats: float,
    threshold_db: float,
) -> list[dict[str, Any]]:
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim == 2:
        mono = np.mean(data, axis=1)
    else:
        mono = data
    if len(mono) < 64:
        return []

    win_n = max(64, int(round(0.050 * sr)))
    n_frames = len(mono) // win_n
    if n_frames == 0:
        rms = float(np.sqrt(np.mean(mono ** 2)))
        if 20.0 * math.log10(rms + 1e-12) > threshold_db:
            dur_beats = (len(mono) / float(sr)) * (float(bpm) / 60.0)
            return _merge_intervals([(0.0, dur_beats)], merge_gap_beats, source="audio")
        return []

    frames = mono[: n_frames * win_n].reshape(n_frames, win_n)
    f_rms = np.sqrt(np.einsum("ij,ij->i", frames, frames, dtype=np.float64) / float(win_n))
    f_db = 20.0 * np.log10(f_rms + 1e-12)
    active = f_db > float(threshold_db)

    sec_per_frame = float(win_n) / float(sr)
    beats_per_sec = max(0.25, float(bpm) / 60.0)
    raw_intervals: list[tuple[float, float]] = []
    cur_start: int | None = None
    for idx, flag in enumerate(active):
        if flag and cur_start is None:
            cur_start = idx
        elif not flag and cur_start is not None:
            s_b = (cur_start * sec_per_frame) * beats_per_sec
            e_b = (idx * sec_per_frame) * beats_per_sec
            raw_intervals.append((s_b, e_b))
            cur_start = None
    if cur_start is not None:
        s_b = (cur_start * sec_per_frame) * beats_per_sec
        e_b = (n_frames * sec_per_frame) * beats_per_sec
        raw_intervals.append((s_b, e_b))

    return _merge_intervals(raw_intervals, merge_gap_beats, source="audio")


def extract_vocal_phrases(
    ustx_path: str = "",
    audio_path: str = "",
    bpm: float = 120.0,
    *,
    merge_gap_ms: float = 250.0,
    threshold_db: float = -35.0,
) -> list[dict[str, Any]]:
    """
    Extract active vocal phrase intervals `[{start_beat, end_beat, duration_beats, source}]`.
    Reads `.ustx` first when available; falls back to 50 ms RMS envelope (`> -35 dBFS`,
    merging gaps `< 250 ms`) on `audio_path`.
    """
    eff_bpm = max(30.0, float(bpm))
    merge_gap_beats = (float(merge_gap_ms) / 1000.0) * (eff_bpm / 60.0)

    if ustx_path and os.path.isfile(os.path.expanduser(ustx_path)):
        phrases = _extract_from_ustx(os.path.expanduser(ustx_path), eff_bpm, merge_gap_beats)
        if phrases:
            return phrases

    if audio_path and os.path.isfile(os.path.expanduser(audio_path)):
        return _extract_from_audio(os.path.expanduser(audio_path), eff_bpm, merge_gap_beats, float(threshold_db))

    return []


def _compute_breathing_windows(
    vocal_phrases: list[dict[str, Any]],
    *,
    min_breath_beats: float,
    horizon_beats: float = 10000.0,
) -> list[tuple[float, float]]:
    if not vocal_phrases:
        return [(0.0, horizon_beats)]
    ordered = sorted(vocal_phrases, key=lambda p: float(p["start_beat"]))
    windows: list[tuple[float, float]] = []
    cursor = 0.0
    for ph in ordered:
        s_b = float(ph["start_beat"])
        e_b = float(ph["end_beat"])
        if s_b - cursor >= min_breath_beats - 1e-6:
            windows.append((cursor, s_b))
        cursor = max(cursor, e_b)
    if horizon_beats - cursor >= min_breath_beats - 1e-6:
        windows.append((cursor, horizon_beats))
    return windows


def _note_overlaps_any(s_n: float, e_n: float, intervals: list[tuple[float, float]]) -> bool:
    for s_p, e_p in intervals:
        if e_n > s_p + 1e-6 and s_n < e_p - 1e-6:
            return True
    return False


def _note_inside_breathing(s_n: float, e_n: float, breath_windows: list[tuple[float, float]]) -> bool:
    for s_w, e_w in breath_windows:
        if s_n >= s_w - 1e-6 and e_n <= e_w + 1e-6:
            return True
    return False


def mute_counter_during_vocals(
    tracks: dict[str, list[Any]],
    vocal_phrases: list[dict[str, Any]],
    *,
    min_breath_beats: float = 0.5,
    attenuate_lead_ornaments: bool = True,
) -> dict[str, list[Any]]:
    """
    Silence counter-melody notes overlapping active vocals or gaps `< min_breath_beats` (0.5 beat),
    preserving call-and-response replies placed in >= 0.5 beat breathing windows.
    Gently attenuates secondary `lead` ornaments during sung phrases.
    """
    if not vocal_phrases or not tracks:
        return dict(tracks)

    active_spans = [(float(p["start_beat"]), float(p["end_beat"])) for p in vocal_phrases]
    max_end = max(e for _, e in active_spans) if active_spans else 0.0
    breath_windows = _compute_breathing_windows(
        vocal_phrases,
        min_breath_beats=float(min_breath_beats),
        horizon_beats=max(10000.0, max_end + 1000.0),
    )

    out: dict[str, list[Any]] = {}
    for role, events in tracks.items():
        r_low = str(role).lower()
        is_counter = any(k in r_low for k in ("counter", "contre", "reply", "response"))
        is_ornament = any(k in r_low for k in ("ornament", "lead2", "arp_fill"))

        if not is_counter and not (attenuate_lead_ornaments and (is_ornament or r_low == "lead")):
            out[role] = list(events)
            continue

        filtered: list[Any] = []
        for ev in events:
            if isinstance(ev, dict):
                b_s = float(ev.get("start", 0.0))
                b_d = float(ev.get("duration", ev.get("length", 0.25)))
                vel = int(ev.get("velocity", 90))
                b_e = b_s + max(0.05, b_d)
                if is_counter:
                    if not _note_overlaps_any(b_s, b_e, active_spans) and _note_inside_breathing(b_s, b_e, breath_windows):
                        filtered.append(dict(ev))
                elif is_ornament:
                    if not _note_overlaps_any(b_s, b_e, active_spans):
                        filtered.append(dict(ev))
                elif r_low == "lead" and attenuate_lead_ornaments:
                    new_ev = dict(ev)
                    if _note_overlaps_any(b_s, b_e, active_spans) and (b_d <= 0.375 or vel <= 85):
                        new_ev["velocity"] = max(38, int(round(vel * 0.65)))
                    filtered.append(new_ev)
                else:
                    filtered.append(dict(ev))
            elif isinstance(ev, (tuple, list)) and len(ev) >= 4:
                b_s, b_d, pitch, vel = float(ev[0]), float(ev[1]), int(ev[2]), int(ev[3])
                b_e = b_s + max(0.05, b_d)
                if is_counter:
                    # Keep ONLY if completely outside active vocals AND inside a >= 0.5 beat breathing window
                    if not _note_overlaps_any(b_s, b_e, active_spans) and _note_inside_breathing(b_s, b_e, breath_windows):
                        filtered.append((b_s, b_d, pitch, vel))
                elif is_ornament:
                    if not _note_overlaps_any(b_s, b_e, active_spans):
                        filtered.append((b_s, b_d, pitch, vel))
                elif r_low == "lead" and attenuate_lead_ornaments:
                    # Soften quiet/grace lead ornaments during vocal phrases instead of dropping main motif
                    if _note_overlaps_any(b_s, b_e, active_spans) and (b_d <= 0.375 or vel <= 85):
                        new_vel = max(38, int(round(vel * 0.65)))
                        filtered.append((b_s, b_d, pitch, new_vel))
                    else:
                        filtered.append((b_s, b_d, pitch, vel))
                else:
                    filtered.append(ev)
            else:
                filtered.append(ev)
        out[role] = filtered

    return out


def formant_ducking_eq_params(
    mixer_idx: int = 19,
    slot_idx: int = 2,
    band_idx: int = 5,
    freq_hz: float = 1850.0,
    q: float = 1.4,
) -> list[tuple[str, Any]]:
    """Build parameter pairs configuring `BUS Music` Band 5 as a `1850 Hz` (`Q=1.4`) peak band."""
    return eq_params(int(mixer_idx), [eq_band(int(band_idx), "peak", float(freq_hz), 0.0, q=float(q))], slot=int(slot_idx), quality=0)


def build_formant_ducking_automation(
    vocal_phrases: list[dict[str, Any]],
    *,
    bpm: float = 120.0,
    mixer_idx: int = 19,
    slot_idx: int = 2,
    band_idx: int = 5,
    duck_db: float = -3.5,
    attack_ms: float = 40.0,
    release_ms: float = 120.0,
    track_idx: int = 28,
    name: str = "BUS Music Formant Duck 1.85kHz",
) -> dict[str, Any]:
    """
    Generate Gridline `add_automation` payload driving `Band 5 gain` on `BUS Music` (`mixer=19, slot=2`)
    from `0.0 dB` in vocal silences to `-3.5 dB` (`40 ms` attack / `120 ms` release) during vocals.
    """
    eff_bpm = max(30.0, float(bpm))
    beats_per_sec = eff_bpm / 60.0
    att_beats = max(0.01, (float(attack_ms) / 1000.0) * beats_per_sec)
    rel_beats = max(0.02, (float(release_ms) / 1000.0) * beats_per_sec)

    v_idle = round(eq_gain_to_auto(0.0), 5)
    v_duck = round(eq_gain_to_auto(float(duck_db)), 5)

    # Merge phrases closer than release + attack so envelope never collides
    raw_spans = [(float(p["start_beat"]), float(p["end_beat"])) for p in vocal_phrases if float(p["end_beat"]) > float(p["start_beat"])]
    merged = _merge_intervals(raw_spans, gap_tol_beats=att_beats + rel_beats + 1e-4, source="merged")

    pts: list[tuple[float, float]] = [(0.0, v_idle)]
    for ph in merged:
        s_b = float(ph["start_beat"])
        e_b = float(ph["end_beat"])
        pre_b = max(0.0, s_b - att_beats)
        post_b = e_b + rel_beats
        if pre_b <= 0.005:
            # Vocal starts immediately near beat 0
            if s_b > 0.002:
                pts.append((0.0, v_idle))
                pts.append((s_b, v_duck))
            else:
                pts[0] = (0.0, v_duck)
        else:
            pts.append((pre_b, v_idle))
            pts.append((s_b, v_duck))
        pts.append((e_b, v_duck))
        pts.append((post_b, v_idle))

    # Deduplicate / enforce strictly increasing point positions >= 0
    clean_pts: list[dict[str, Any]] = []
    last_pos = -1.0
    for pos, val in pts:
        rp = round(max(0.0, float(pos)), 4)
        if rp <= last_pos + 1e-4:
            if clean_pts:
                clean_pts[-1]["value"] = round(float(val), 5)
            continue
        clean_pts.append({"position": rp, "value": round(float(val), 5), "shape": "curve"})
        last_pos = rp

    if not clean_pts:
        clean_pts = [{"position": 0.0, "value": v_idle, "shape": "curve"}]
    clip_len = max(4.0, float(clean_pts[-1]["position"]) + 0.5)

    return {
        "name": name,
        "mixer": int(mixer_idx),
        "slot": int(slot_idx),
        "param": f"Band {int(band_idx)} gain",
        "track": int(track_idx),
        "start": 0.0,
        "length": round(clip_len, 4),
        "points": clean_pts,
    }
