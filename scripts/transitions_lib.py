#!/usr/bin/env python3
"""
Macro Expression & Structural Transition Primitives (Axe 5 §5.2).

Generates normalized Gridline `add_automation` payloads (via `gl_dsl.db_to_auto`) and
clean beat-accurate note dropouts:
  - `beat4_dropout`: clears last beat before a Drop/Chorus on Drums/Bass and ducks FX Room volume to 0.0.
  - `filter_sweep`: smooth exp/linear automation curve on EQ (`Band 7 freq`) or Synth (`Cutoff`).
  - `section_gain_arc`: macro section loudness arcs (-2.5 dB Intro/Bridge, 0 dB Verse/Chorus, +1.0 dB Climax).
  - `outro_fade`: smooth logarithmic tail fade to exact `0.0` at track end (no abrupt tail cuts).
"""
from __future__ import annotations

import math
from typing import Any, Literal

from gl_dsl import clamp, db_to_auto

_DRUM_KEYWORDS = ("kick", "clap", "snare", "hat", "perc", "crash", "rim", "drum", "shaker", "tom")
_BASS_KEYWORDS = ("808", "bass", "sub")


def _note_len(ev: dict[str, Any]) -> float:
    return float(ev.get("length", ev.get("duration", 0.25)))


def _set_note_len(ev: dict[str, Any], val: float) -> None:
    if "length" in ev or "duration" not in ev:
        ev["length"] = round(float(val), 5)
    else:
        ev["duration"] = round(float(val), 5)


def _find_mixer_vol_auto(automations: list[dict[str, Any]], mixer_idx: int) -> dict[str, Any] | None:
    for a in automations:
        if a.get("mixer") == mixer_idx and a.get("slot") is None and a.get("param") == "volume":
            return a
    return None


def _upsert_point(points: list[dict[str, Any]], position: float, value: float, shape: str = "curve") -> None:
    pos = round(max(0.0, float(position)), 4)
    val = round(clamp(float(value), 0.0, 1.0), 4)
    for pt in points:
        if abs(float(pt["position"]) - pos) < 1e-3:
            pt["value"] = val
            pt["shape"] = shape
            return
    points.append({"position": pos, "value": val, "shape": shape})
    points.sort(key=lambda p: float(p["position"]))


def beat4_dropout(
    tracks: dict[str, list[dict[str, Any]]],
    automations: list[dict[str, Any]],
    bar_idx: int,
    beats_per_bar: int = 4,
    mute_drums: bool = True,
    mute_bass: bool = True,
    cut_reverb: bool = True,
    fx_room_mixer: int = 20,
    fx_room_base_auto: float = 0.28,
    track_idx: int = 10,
) -> dict[str, Any]:
    """
    Clears the last beat of `bar_idx` (`0-indexed`, i.e., beats `[drop_start, drop_end)`)
    across Drums and/or Bass tracks and drops `FX Room` volume to `0.0` over that beat.
    """
    bar_start = float(bar_idx) * float(beats_per_bar)
    drop_end = float(bar_idx + 1) * float(beats_per_bar)
    drop_start = max(bar_start, drop_end - 1.0)
    removed_notes = 0
    clipped_notes = 0

    for tr_name, notes in tracks.items():
        low = tr_name.lower()
        is_drum = any(k in low for k in _DRUM_KEYWORDS)
        is_bass = any(k in low for k in _BASS_KEYWORDS)
        if not ((mute_drums and is_drum) or (mute_bass and is_bass)):
            continue
        kept: list[dict[str, Any]] = []
        for ev in notes:
            st = float(ev["start"])
            ln = _note_len(ev)
            en = st + ln
            if drop_start - 1e-4 <= st < drop_end - 1e-4:
                removed_notes += 1
                continue
            if st < drop_start - 1e-4 and en > drop_start + 1e-4:
                _set_note_len(ev, max(0.05, drop_start - st))
                clipped_notes += 1
            kept.append(ev)
        tracks[tr_name] = kept

    if cut_reverb:
        base_val = round(clamp(float(fx_room_base_auto), 0.0, 1.0), 4)
        existing = _find_mixer_vol_auto(automations, fx_room_mixer)
        if existing is not None and float(existing.get("start", 0.0)) == 0.0:
            pts = existing.setdefault("points", [])
            _upsert_point(pts, max(0.0, drop_start - 0.125), base_val)
            _upsert_point(pts, drop_start, 0.0, shape="hold")
            _upsert_point(pts, drop_end - 0.05, 0.0, shape="hold")
            _upsert_point(pts, drop_end, base_val)
            existing["length"] = max(float(existing.get("length", drop_end)), drop_end)
        else:
            clip_start = max(0.0, drop_start - 0.125)
            automations.append(
                {
                    "name": f"Beat4 Dropout Bar {bar_idx + 1}",
                    "mixer": int(fx_room_mixer),
                    "param": "volume",
                    "track": int(track_idx),
                    "start": round(clip_start, 4),
                    "length": round(drop_end - clip_start + 0.05, 4),
                    "points": [
                        {"position": 0.0, "value": base_val},
                        {"position": round(drop_start - clip_start, 4), "value": 0.0, "shape": "hold"},
                        {"position": round(drop_end - clip_start - 0.02, 4), "value": 0.0, "shape": "hold"},
                        {"position": round(drop_end - clip_start, 4), "value": base_val},
                    ],
                }
            )

    return {
        "drop_start_beat": drop_start,
        "drop_end_beat": drop_end,
        "removed_notes": removed_notes,
        "clipped_notes": clipped_notes,
        "cut_reverb": cut_reverb,
    }


def _hz_to_normalized(hz: float, param: str, is_channel: bool) -> float:
    if is_channel and "cutoff" in param.lower():
        if hz <= 1.0:
            return round(clamp(hz, 0.0, 1.0), 4)
        lo, hi = 20.0, 20000.0
        val = math.log10(clamp(hz, lo, hi) / lo) / math.log10(hi / lo)
        return round(clamp(val, 0.0, 1.0), 4)
    # gl.eq Band N freq linear parameter range [20 .. 20000 Hz]
    return round(clamp((float(hz) - 20.0) / 19980.0, 0.0, 1.0), 4)


def filter_sweep(
    automations: list[dict[str, Any]],
    mixer_idx: int | None,
    slot_idx: int | None,
    start_beat: float,
    end_beat: float,
    start_hz: float,
    end_hz: float,
    curve: Literal["exp", "linear"] = "exp",
    param: str = "Band 7 freq",
    channel_idx: int | None = None,
    track_idx: int = 11,
    steps: int = 8,
    name: str | None = None,
) -> dict[str, Any]:
    """
    Generate normalized filter sweep automation points (`0..1`) over `[start_beat, end_beat]`.
    Supports mixer EQ slots (`mixer=..., slot=..., param='Band 7 freq'`) and channels (`channel=..., param='Cutoff'`).
    """
    dur = max(0.25, float(end_beat) - float(start_beat))
    n_steps = max(2, int(steps))
    points: list[dict[str, Any]] = []
    s_hz = max(20.0 if channel_idx is None else 0.001, float(start_hz))
    e_hz = max(20.0 if channel_idx is None else 0.001, float(end_hz))
    is_channel = channel_idx is not None

    for i in range(n_steps + 1):
        alpha = i / float(n_steps)
        if curve == "exp" and s_hz > 0.0 and e_hz > 0.0:
            hz = s_hz * ((e_hz / s_hz) ** alpha)
        else:
            hz = s_hz + (e_hz - s_hz) * alpha
        points.append(
            {
                "position": round(alpha * dur, 4),
                "value": _hz_to_normalized(hz, param, is_channel),
                "shape": "curve",
            }
        )

    clip: dict[str, Any] = {
        "name": name or f"Filter Sweep {int(round(s_hz))}->{int(round(e_hz))}",
        "param": param,
        "track": int(track_idx),
        "start": round(float(start_beat), 4),
        "length": round(dur, 4),
        "points": points,
    }
    if channel_idx is not None:
        clip["channel"] = int(channel_idx)
    else:
        clip["mixer"] = int(mixer_idx if mixer_idx is not None else 19)
        if slot_idx is not None:
            clip["slot"] = int(slot_idx)
    automations.append(clip)
    return clip


def _default_section_db(name: str, explicit_db: float | None = None) -> float:
    if explicit_db is not None:
        return float(explicit_db)
    low = name.lower()
    if any(k in low for k in ("climax", "final", "peak", "drop")):
        return 1.0
    if any(k in low for k in ("intro", "pont", "bridge", "break", "outro")):
        return -2.5
    return 0.0


def section_gain_arc(
    automations: list[dict[str, Any]],
    bus_mixer_idx: int,
    sections_plan: list[Any],
    beats_per_bar: float = 4.0,
    track_idx: int = 9,
    base_offset_db: float = 0.0,
    ramp_beats: float = 0.5,
) -> dict[str, Any]:
    """
    Generate macro section gain arc automation on `bus_mixer_idx` using `gl_dsl.db_to_auto`:
      - Intro / Bridge / Pont: -2.5 dB
      - Verse / Chorus:         0.0 dB
      - Climax:                +1.0 dB
    Accepts `sections_plan` entries as `(name, start_bar_1b, end_bar_1b[, role_or_db])`
    or `{"section"|"name": ..., "start_bar": ..., "bars": ..., "gain_db": ...}`.
    """
    parsed: list[tuple[float, float, str, float]] = []
    for item in sections_plan:
        if isinstance(item, dict):
            sec_name = str(item.get("section") or item.get("name") or "Section")
            b0 = int(item.get("start_bar", 1))
            b1 = int(item.get("end_bar", b0 + int(item.get("bars", 1)) - 1))
            exp_db = item.get("gain_db")
            db = _default_section_db(sec_name, None if exp_db is None else float(exp_db))
        elif isinstance(item, (tuple, list)):
            sec_name = str(item[0])
            b0 = int(item[1])
            b1 = int(item[2])
            exp_db = float(item[3]) if len(item) >= 4 and isinstance(item[3], (int, float)) else None
            tag = str(item[3]) if len(item) >= 4 and isinstance(item[3], str) else sec_name
            db = _default_section_db(f"{sec_name} {tag}", exp_db)
        else:
            continue
        s_beat = float(b0 - 1) * float(beats_per_bar)
        e_beat = float(b1) * float(beats_per_bar)
        parsed.append((s_beat, e_beat, sec_name, db))

    parsed.sort(key=lambda x: x[0])
    points: list[dict[str, Any]] = []
    total_len = 0.0
    for idx, (s_beat, e_beat, _nm, db) in enumerate(parsed):
        val = round(db_to_auto(db + base_offset_db), 4)
        if idx > 0:
            ramp_start = max(parsed[idx - 1][0], s_beat - float(ramp_beats))
            prev_val = round(db_to_auto(parsed[idx - 1][3] + base_offset_db), 4)
            _upsert_point(points, ramp_start, prev_val)
        _upsert_point(points, s_beat, val)
        total_len = max(total_len, e_beat)
    if parsed:
        last_val = round(db_to_auto(parsed[-1][3] + base_offset_db), 4)
        _upsert_point(points, total_len, last_val)

    clip = {
        "name": f"Macro Arc Mixer {bus_mixer_idx}",
        "mixer": int(bus_mixer_idx),
        "param": "volume",
        "track": int(track_idx),
        "start": 0.0,
        "length": round(max(1.0, total_len), 4),
        "points": points,
    }
    automations.append(clip)
    return clip


def _find_track_key(tracks: dict[str, list[dict[str, Any]]], keywords: tuple[str, ...]) -> str | None:
    for k in tracks:
        low = k.lower()
        if any(kw in low for kw in keywords):
            return k
    return None


def _remove_or_clip(tracks: dict[str, list[dict[str, Any]]], key: str, t0: float, t1: float) -> int:
    """Remove notes starting in `[t0, t1)` and clip those crossing `t0`; return removed count."""
    removed = 0
    kept: list[dict[str, Any]] = []
    for ev in tracks.get(key, []):
        st = float(ev["start"])
        ln = _note_len(ev)
        en = st + ln
        if t0 - 1e-4 <= st < t1 - 1e-4:
            removed += 1
            continue
        if st < t0 - 1e-4 and en > t0 + 1e-4:
            _set_note_len(ev, max(0.05, t0 - st))
        kept.append(ev)
    tracks[key] = kept
    return removed


def generate_drum_fill(
    tracks: dict[str, list[dict[str, Any]]],
    transition_beat: float,
    bpb: float = 4.0,
    profile: str = "trap_rap",
    style: str = "auto",
    bpm: float = 140.0,
) -> dict[str, Any]:
    """
    Phase 8 Q5.1 — deterministic idiomatic drum fill over the last 2 beats before
    `transition_beat` (0-indexed beats), profile-aware (`trap_rap`, `edm_club`,
    `pop_rnb`, `rock_metal`, `lofi_chill`, `jazz_acoustic`, `classical_ambient`).
    """
    t_end = float(transition_beat)
    t_start = max(0.0, t_end - 2.0)
    snare_key = _find_track_key(tracks, ("snare", "clap", "rim"))
    kick_key = _find_track_key(tracks, ("kick", "bd"))
    if snare_key is None:
        snare_key = "Snare"
        tracks.setdefault(snare_key, [])
    if kick_key is None:
        kick_key = "Kick"
        tracks.setdefault(kick_key, [])

    prof = str(profile or "trap_rap").strip().lower()
    stl = "auto" if str(style).strip().lower() in ("", "auto") else str(style).strip().lower()
    added = 0
    kind = "snare_roll"
    if prof == "classical_ambient":
        return {"profile": prof, "added_notes": 0, "kind": "noop", "transition_beat": t_end}

    if prof in ("trap_rap", "edm_club") or stl in ("snare_roll", "trap_roll"):
        kind = "snare_roll"
        # Accelerando 1/8 -> 1/16 -> 1/32 over the last 2 beats
        offsets: list[float] = []
        step_groups = ((0.0, 0.5, 1.0), (1.0, 0.25, 1.5), (1.5, 0.125, 2.0))
        for base, step, span_end in step_groups:
            n = max(1, int(round((span_end - base) / step)))
            offsets.extend([base + i * step for i in range(n)])
        base_pitch = 38
        for i, off in enumerate(offsets):
            st = round(t_start + off, 5)
            if st >= t_end - 1e-4:
                continue
            vel = int(round(55 + (112 - 55) * (i / max(1, len(offsets) - 1))))
            pitch = base_pitch - (i - (len(offsets) - 4) + 1) if i >= len(offsets) - 4 else base_pitch
            tracks[snare_key].append({"start": st, "length": 0.08, "pitch": int(max(24, pitch)), "velocity": max(24, min(127, vel))})
            added += 1
        _remove_or_clip(tracks, kick_key, max(t_start, t_end - 0.5), t_end)
    elif prof in ("pop_rnb", "rock_metal", "lofi_chill") or stl in ("tom_fill", "tom"):
        kind = "tom_fill"
        tom_pitches = [52, 50, 47, 45]
        for i, (off, pitch) in enumerate(zip((0.0, 0.5, 1.0, 1.5), tom_pitches)):
            st = round(t_start + off, 5)
            tracks[snare_key].append({"start": st, "length": 0.12, "pitch": int(pitch), "velocity": int(78 + i * 10)})
            added += 1
        # Ghost notes + final accent on the last half-beat
        for off in (0.75, 1.25, 1.75):
            st = round(t_start + off, 5)
            tracks[snare_key].append({"start": st, "length": 0.06, "pitch": int(38), "velocity": 42})
            added += 1
    else:
        kind = "pickup"
        st = round(max(t_start, t_end - 0.5), 5)
        tracks[snare_key].append({"start": st, "length": 0.1, "pitch": 40, "velocity": 62})
        added += 1

    tracks[snare_key].sort(key=lambda ev: (float(ev["start"]), int(ev["pitch"])))
    return {
        "profile": prof,
        "kind": kind,
        "transition_beat": round(t_end, 3),
        "added_notes": added,
        "snare_track": snare_key,
        "kick_cut": prof in ("trap_rap", "edm_club"),
        "bpm": float(bpm),
    }


def add_transition_fx(
    tracks: dict[str, list[dict[str, Any]]],
    automations: list[dict[str, Any]],
    transition_beat: float,
    bpb: float = 4.0,
    kind: str = "riser_impact",
    fx_track_idx: int | None = None,
    *,
    fx_channel_idx: int | None = None,
    fx_mixer_idx: int | None = None,
    fx_track_key: str | None = None,
    impact: bool = True,
) -> dict[str, Any]:
    """
    Phase 8 Q5.1 — 4-beat riser (filter 300 Hz -> 9000 Hz + rising volume), 0.25-beat
    breathing silence before the downbeat, then an impact/crash on `transition_beat`.
    """
    t_end = float(transition_beat)
    t_start = max(0.0, t_end - 4.0)
    key = fx_track_key or _find_track_key(tracks, ("riser", "fx", "noise", "impact", "sweep"))
    if key is None:
        key = "FX"
        tracks.setdefault(key, [])
    k_norm = str(kind or "riser_impact").strip().lower()
    summary: dict[str, Any] = {"kind": k_norm, "transition_beat": round(t_end, 3), "fx_track": key}
    if "riser" in k_norm or "sweep" in k_norm:
        tracks[key].append({"start": round(t_start, 5), "length": round(max(0.25, 3.75), 5), "pitch": 55, "velocity": 96})
        lane = int(fx_track_idx) if fx_track_idx is not None else 11
        if fx_channel_idx is not None:
            filter_sweep(
                automations,
                mixer_idx=None,
                slot_idx=None,
                start_beat=t_start,
                end_beat=max(t_start + 0.25, t_end - 0.25),
                start_hz=300.0,
                end_hz=9000.0,
                curve="exp",
                param="Cutoff",
                channel_idx=int(fx_channel_idx),
                track_idx=lane,
                name="Riser Cutoff",
            )
        elif fx_mixer_idx is not None:
            filter_sweep(
                automations,
                mixer_idx=int(fx_mixer_idx),
                slot_idx=1,
                start_beat=t_start,
                end_beat=max(t_start + 0.25, t_end - 0.25),
                start_hz=300.0,
                end_hz=9000.0,
                curve="exp",
                track_idx=lane,
                name="Riser Cutoff",
            )
        auto_clip: dict[str, Any] = {
            "name": "Riser Volume",
            "param": "volume",
            "track": lane,
            "start": round(t_start, 4),
            "length": round(max(0.5, t_end - t_start), 4),
            "points": [
                {"position": 0.0, "value": 0.30},
                {"position": round(max(0.25, t_end - t_start - 0.5), 4), "value": 0.88},
                {"position": round(max(0.26, t_end - t_start - 0.27), 4), "value": 0.90},
                {"position": round(max(0.27, t_end - t_start - 0.25), 4), "value": 0.0, "shape": "hold"},
            ],
        }
        if fx_channel_idx is not None:
            auto_clip["channel"] = int(fx_channel_idx)
        elif fx_mixer_idx is not None:
            auto_clip["mixer"] = int(fx_mixer_idx)
        if fx_channel_idx is not None or fx_mixer_idx is not None:
            automations.append(auto_clip)
        summary["riser"] = True
    if impact and ("impact" in k_norm or k_norm == "riser_impact"):
        tracks[key].append({"start": round(t_end, 5), "length": 0.5, "pitch": 48, "velocity": 120})
        summary["impact"] = True
    tracks[key].sort(key=lambda ev: (float(ev["start"]), int(ev["pitch"])))
    return summary


def apply_harmonic_anticipation(
    chords_notes: list[dict[str, Any]],
    bass_notes: list[dict[str, Any]],
    bar_boundaries: list[float],
    offset_beats: float = 0.5,
    ratio: float = 0.35,
    lead_notes: list[dict[str, Any]] | None = None,
    counter_notes: list[dict[str, Any]] | None = None,
    allowed_pcs_by_pivot: dict[float, set[int]] | None = None,
    cut_margin_beats: float = 0.05,
) -> dict[str, Any]:
    """
    Phase 8 Q5.1 — Anticipate chord/bass notes by one eighth on a deterministic fraction
    of phrase pivots, shortening the previous note so no overlap or negative start remains.

    v3.0 synchronous anticipation: `lead_notes` / `counter_notes` crossing the anticipation
    frontier [pivot - offset, pivot] are cleanly cut (margin `cut_margin_beats`) UNLESS their
    pitch class belongs to the new chord (`allowed_pcs_by_pivot[pivot]`), so the lead never
    rubs against the new chord for a full eighth.
    """
    pivots = sorted({round(float(b), 4) for b in bar_boundaries if float(b) > float(offset_beats) + 1e-4})
    if not pivots or not chords_notes and not bass_notes:
        return {"applied": 0, "pivots": pivots, "moved_chords": 0, "moved_bass": 0}
    n_apply = max(1, int(round(len(pivots) * clamp(float(ratio), 0.0, 1.0))))
    stride = max(1, len(pivots) // n_apply)
    selected = pivots[::stride][:n_apply]

    moved_ch = moved_bs = 0

    def _process(notes: list[dict[str, Any]], pivot: float) -> int:
        moved = 0
        for ev in notes:
            st = float(ev["start"])
            if abs(st - pivot) > 0.06:
                continue
            new_start = round(max(0.0, pivot - float(offset_beats)), 5)
            # Shorten any previous overlapping note to end just before the anticipation
            prev = [p for p in notes if float(p["start"]) < new_start - 1e-4]
            ok = True
            if prev:
                pe = max(prev, key=lambda p: float(p["start"]))
                p_end = float(pe["start"]) + _note_len(pe)
                if p_end > new_start + 1e-4:
                    new_len = round(new_start - float(pe["start"]) - 0.01, 5)
                    if new_len < 0.05:
                        ok = False
                    else:
                        _set_note_len(pe, new_len)
            if ok:
                ev["start"] = new_start
                moved += 1
        return moved

    cut_lead = cut_counter = 0

    def _cut_crossing(notes: list[dict[str, Any]], pivot: float) -> int:
        frontier = pivot - float(offset_beats)
        allowed = (allowed_pcs_by_pivot or {}).get(round(float(pivot), 4)) or (allowed_pcs_by_pivot or {}).get(pivot)
        cut = 0
        for ev in notes:
            st = float(ev["start"])
            en = st + _note_len(ev)
            if st < frontier + 1e-4 and en > frontier + float(cut_margin_beats):
                if allowed is not None and int(ev["pitch"]) % 12 in allowed:
                    continue  # note already belongs to the new chord: keep it
                new_len = round(max(0.05, frontier - float(cut_margin_beats) - st), 5)
                _set_note_len(ev, new_len)
                cut += 1
        return cut

    for p in selected:
        moved_ch += _process(chords_notes, p)
        moved_bs += _process(bass_notes, p)
        if lead_notes:
            cut_lead += _cut_crossing(lead_notes, p)
        if counter_notes:
            cut_counter += _cut_crossing(counter_notes, p)
    chords_notes.sort(key=lambda ev: (float(ev["start"]), int(ev["pitch"])))
    bass_notes.sort(key=lambda ev: (float(ev["start"]), int(ev["pitch"])))
    if lead_notes:
        lead_notes.sort(key=lambda ev: (float(ev["start"]), int(ev["pitch"])))
    if counter_notes:
        counter_notes.sort(key=lambda ev: (float(ev["start"]), int(ev["pitch"])))
    return {
        "applied": len(selected),
        "pivots": selected,
        "moved_chords": moved_ch,
        "moved_bass": moved_bs,
        "cut_lead": cut_lead,
        "cut_counter": cut_counter,
        "offset_beats": float(offset_beats),
        "ratio": float(ratio),
    }


def outro_fade(
    automations: list[dict[str, Any]],
    master_or_bus_idx: int,
    start_beat: float,
    end_beat: float,
    base_db: float = 0.0,
    track_idx: int = 9,
    steps: int = 6,
) -> dict[str, Any]:
    """
    Append a smooth logarithmic fade-out to exact `0.0` at `end_beat`.
    If an existing volume clip targets `master_or_bus_idx` from beat 0, appends fade points cleanly.
    """
    st = max(0.0, float(start_beat))
    en = max(st + 0.25, float(end_beat))
    dur = en - st
    n_steps = max(2, int(steps))

    abs_pts: list[tuple[float, float]] = []
    for i in range(n_steps + 1):
        alpha = i / float(n_steps)
        pos = round(st + alpha * dur, 4)
        if i == n_steps:
            val = 0.0
        else:
            # Smooth decibel fade (0 -> -24 dB) then drop into 0.0
            db = float(base_db) - 24.0 * (alpha ** 1.35)
            val = round(db_to_auto(db), 4)
        abs_pts.append((pos, val))

    existing = _find_mixer_vol_auto(automations, int(master_or_bus_idx))
    if existing is not None and float(existing.get("start", 0.0)) == 0.0:
        pts = existing.setdefault("points", [])
        pts[:] = [p for p in pts if float(p["position"]) < st - 1e-3]
        for pos, val in abs_pts:
            _upsert_point(pts, pos, val)
        existing["length"] = round(max(float(existing.get("length", en)), en), 4)
        return existing

    clip = {
        "name": f"Outro Fade Mixer {master_or_bus_idx}",
        "mixer": int(master_or_bus_idx),
        "param": "volume",
        "track": int(track_idx),
        "start": round(st, 4),
        "length": round(dur, 4),
        "points": [
            {"position": round(pos - st, 4), "value": val, "shape": "curve"}
            for pos, val in abs_pts
        ],
    }
    automations.append(clip)
    return clip


def apply_outro_ritardando(
    tracks: dict[str, list[dict[str, Any]]],
    bpm: float,
    bars: int = 2,
    amount: float = 0.06,
    tracks_meta: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """v3.2 ExpressionArticulations — respiration finale organique (`outro_ritardando`) :
    décélération douce de `amount` (4-8 %) sur les `bars` dernières mesures (styles
    organiques lofi/jazz/classical) par étirement symbolique monotone des starts/lengths
    dans la fenêtre (le début de fenêtre reste ancré ; l'ordre des notes est préservé)."""
    amount = max(0.0, min(0.15, float(amount)))
    if amount <= 0.0 or bars <= 0:
        return {"applied": 0, "amount": amount}
    total = 0.0
    for notes in tracks.values():
        for n in notes:
            total = max(total, float(n["start"]) + _note_len(n))
    if total <= 0.0:
        return {"applied": 0, "amount": amount}
    plan = None
    try:
        from dsp_interaction_guards import plan_safe_ritardando
        sir_like = {"tracks": {k: {"notes": [{"start": float(n.get("start", 0.0)), "duration_beats": _note_len(n)} for n in notes]}
                                for k, notes in tracks.items()}}
        plan = plan_safe_ritardando(sir_like, tracks_meta or [], amount, bars=bars)
    except Exception:
        plan = None
    bpb = 4.0
    win_start = max(0.0, total - bars * bpb)
    win_len = total - win_start
    applied = 0
    for notes in tracks.values():
        for n in notes:
            st = float(n["start"])
            if st < win_start - 1e-6:
                continue
            t = (st - win_start) / win_len
            new_st = win_start + win_len * (t * (1.0 + amount))
            n["start"] = round(max(0.0, new_st), 5)
            _set_note_len(n, _note_len(n) * (1.0 + amount * t))
            applied += 1
    return {"applied": applied, "amount": amount, "window_beats": [round(win_start, 3), round(total, 3)], "plan": plan}
