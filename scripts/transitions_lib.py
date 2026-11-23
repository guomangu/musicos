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
