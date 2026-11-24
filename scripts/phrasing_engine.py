#!/usr/bin/env python3
"""
Instrumental Prosody & Micro-Dynamic Phrasing Engine (`scripts/phrasing_engine.py`, Phase 7 P2).

Enriches 0-indexed beat MIDI tracks and emits calibrated `gl_dsl` LPF automations:
  1. Intra-phrase velocity arch (4-bar phrases peaking around phi=0.65, cadence drop phi>0.85)
     plus matching LPF opening/closing automation for sustained harmony tracks (`pad`/`strings`/`chords`).
  2. Cadential idiomatic ornaments at bars 4 & 8:
     - Bass/808 diatonic approach note (0.25 beat, -18% velocity, >= MIDI 31) before >=3 st jumps
     - Lead upper-degree short appoggiatura (0.125 beat, -14 velocity) before bar-3 apex in high-energy sections,
       verified with `musicality_metrics.detect_minor_2nd_clashes`.
  3. Section-dependent dynamic pocket shift:
     - `intro`/`verse`/`bridge`: +3.5 ms laid-back on `snare`, `chords`, `lead`
     - `chorus`/`drop`/`climax`: -2.0 ms on-top on `snare`, `hihat`.
"""
from __future__ import annotations

import copy
import math
import os
import sys
from typing import Any

import pretty_midi

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
ANALYZER_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from audio_metrics import parse_key_tonic_pc
from gl_dsl import _default_cutoff_hz, clamp
from groove_lib import ms_to_beats
from musicality_metrics import detect_minor_2nd_clashes
from transitions_lib import _hz_to_normalized

_SCALES: dict[str, list[int]] = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
}

_LAID_BACK_KINDS = ("intro", "verse", "couplet", "bridge", "pont", "break", "pre_chorus")
_ON_TOP_KINDS = ("chorus", "refrain", "drop", "climax", "hook", "peak")


def _note_len(ev: dict[str, Any]) -> float:
    return float(ev.get("length", ev.get("duration", 0.25)))


def _set_note_len(ev: dict[str, Any], val: float) -> None:
    if "length" in ev or "duration" not in ev:
        ev["length"] = round(float(val), 5)
    else:
        ev["duration"] = round(float(val), 5)


def _parse_time_sig(ts: Any) -> tuple[int, int]:
    if isinstance(ts, dict):
        return int(ts.get("numerator", 4)), int(ts.get("denominator", 4))
    if isinstance(ts, (tuple, list)) and len(ts) >= 2:
        return int(ts[0]), int(ts[1])
    if isinstance(ts, str) and "/" in ts:
        a, b = ts.split("/", 1)
        return int(a), int(b)
    if isinstance(ts, (int, float)):
        return max(1, int(ts)), 4
    return 4, 4


def _normalize_sections(sections: list[Any]) -> list[dict[str, Any]]:
    """Normalize tuples `(name, start_bar_0idx, bars, kind)` or dicts into `start_bar` 0-indexed."""
    if not sections:
        return []
    dicts: list[dict[str, Any]] = []
    min_sb = None
    has_blueprint_shape = False
    for s in sections:
        if isinstance(s, (tuple, list)) and len(s) >= 3:
            sb = int(s[1])
            dicts.append({
                "name": str(s[0]),
                "start_bar": sb,
                "bars": int(s[2]),
                "kind": str(s[3] if len(s) >= 4 else s[0]).lower(),
            })
            min_sb = sb if min_sb is None else min(min_sb, sb)
        elif isinstance(s, dict):
            sb = int(s.get("start_bar", 0))
            if "section" in s and "kind" not in s:
                has_blueprint_shape = True
            name = str(s.get("name") or s.get("section") or "Section")
            kind = str(s.get("kind") or name).lower()
            dicts.append({
                "name": name,
                "start_bar": sb,
                "bars": max(1, int(s.get("bars", 4))),
                "kind": kind,
            })
            min_sb = sb if min_sb is None else min(min_sb, sb)

    if has_blueprint_shape and min_sb is not None and min_sb >= 1:
        for d in dicts:
            d["start_bar"] = max(0, d["start_bar"] - 1)
    return dicts


def _resolve_scale_pcs(harmonic_core: dict[str, Any] | None) -> tuple[set[int], str]:
    hc = harmonic_core or {}
    key_str = str(hc.get("key") or "C")
    mode_raw = str(hc.get("mode") or "minor").strip().lower().replace(" ", "_")
    tonic_pc = parse_key_tonic_pc(key_str)
    if tonic_pc is None:
        tonic_pc = 0
    if mode_raw not in _SCALES:
        mode_raw = "major" if "maj" in key_str.lower() else "minor"
    intervals = _SCALES[mode_raw]
    pcs = {(int(tonic_pc) + iv) % 12 for iv in intervals}
    return pcs, mode_raw


def _build_temp_pm(
    tracks: dict[str, list[dict[str, Any]]],
    bpm: float,
    bpb: int,
    denom: int,
) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    if bpb != 4 or denom != 4:
        pm.time_signature_changes.append(pretty_midi.TimeSignature(bpb, denom, 0.0))
    b2s = 60.0 / max(40.0, float(bpm))
    for role, evs in tracks.items():
        if not evs:
            continue
        rl = role.lower()
        is_drum = any(k in rl for k in ("kick", "snare", "clap", "hat", "perc", "rim", "tom", "drum", "fx"))
        inst_name = "Bass" if any(k in rl for k in ("bass", "808", "sub")) else (
            "Chords" if any(k in rl for k in ("chord", "pad", "string")) else (
                "Lead" if "lead" in rl else role.capitalize()
            )
        )
        inst = pretty_midi.Instrument(program=0, is_drum=is_drum, name=inst_name)
        for ev in evs:
            st = max(0.0, float(ev["start"])) * b2s
            en = max(st + 0.02, (float(ev["start"]) + _note_len(ev)) * b2s)
            inst.notes.append(pretty_midi.Note(
                velocity=int(max(1, min(127, ev.get("velocity", 80)))),
                pitch=int(max(0, min(127, ev["pitch"]))),
                start=st,
                end=en,
            ))
        pm.instruments.append(inst)
    return pm


def _classify_section_pocket(sec: dict[str, Any]) -> str:
    hay = f"{sec.get('kind', '')} {sec.get('name', '')}".lower()
    if any(k in hay for k in _ON_TOP_KINDS):
        return "on_top"
    if any(k in hay for k in _LAID_BACK_KINDS):
        return "laid_back"
    return "neutral"


def apply_instrumental_prosody(
    tracks: dict[str, list[dict[str, Any]]],
    sections: list[Any],
    harmonic_core: dict[str, Any] | None,
    emotion_vector: dict[str, Any] | None,
    bpm: float,
    time_signature: Any = (4, 4),
    *,
    track_idx_start: int = 12,
    channel_mixer_map: dict[str, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    Apply intra-phrase dynamics/LPF breathing, cadential ornaments (bars 4 & 8),
    and section-level pocket shifts deterministically to beat-indexed MIDI tracks.
    Returns `(enriched_tracks, gl_dsl_automations)`.
    """
    eff_bpm = max(40.0, float(bpm))
    bpb_num, bpb_den = _parse_time_sig(time_signature)
    bpb = float(bpb_num)
    ev_vec = emotion_vector or {}
    valence = clamp(float(ev_vec.get("valence", 0.0)), -1.0, 1.0)
    energy = clamp(float(ev_vec.get("energy", 0.65)), 0.0, 1.0)
    tension = clamp(float(ev_vec.get("tension", 0.40)), 0.0, 1.0)

    out_tracks: dict[str, list[dict[str, Any]]] = {
        k: [copy.copy(ev) for ev in v] for k, v in (tracks or {}).items()
    }
    norm_secs = _normalize_sections(sections)
    if not norm_secs:
        max_b = max(
            [float(ev["start"]) + _note_len(ev) for evs in out_tracks.values() for ev in evs] or [bpb * 4.0]
        )
        n_bars = max(1, int(math.ceil(max_b / bpb)))
        norm_secs = [{"name": "Main", "start_bar": 0, "bars": n_bars, "kind": "chorus"}]

    allowed_pcs, _mode = _resolve_scale_pcs(harmonic_core)
    a_expr = 6.0 + 8.0 * clamp(0.6 * energy + 0.4 * tension, 0.0, 1.0)
    max_lead_pitch = 72 if valence < -0.3 else 75

    # 1. Intra-phrase velocity expression arch (4-bar phrases) on lead, chords, pad/strings, counter
    phrased_keys = [
        k for k in out_tracks
        if any(w in k.lower() for w in ("lead", "melody", "chord", "pad", "string", "keys", "counter"))
    ]
    phrase_spans: list[tuple[float, float]] = []
    for sec in norm_secs:
        sb = sec["start_bar"]
        nb = sec["bars"]
        cursor = sb
        while cursor < sb + nb:
            p_bars = min(4, (sb + nb) - cursor)
            p_start = float(cursor) * bpb
            p_end = float(cursor + p_bars) * bpb
            phrase_spans.append((p_start, p_end))
            cursor += p_bars

    for r_key in phrased_keys:
        for ev in out_tracks[r_key]:
            st = float(ev["start"])
            for p_start, p_end in phrase_spans:
                if p_start - 1e-4 <= st < p_end - 1e-4:
                    phi = clamp((st - p_start) / max(1e-6, p_end - p_start), 0.0, 1.0)
                    delta = round(a_expr * math.sin(math.pi * (phi ** 0.85)) - (8.0 if phi > 0.85 else 0.0))
                    orig_v = float(ev.get("velocity", 80))
                    ev["velocity"] = int(max(38, min(118, round(orig_v + delta))))
                    break

    # 1b. LPF breathing automation curves for sustained harmonic tracks (`pad`, `strings`, `chords`)
    autos: list[dict[str, Any]] = []
    sustained_keys = [
        k for k in out_tracks
        if out_tracks[k] and any(w in k.lower() for w in ("pad", "string", "chord"))
    ]
    open_factor = 1.15 + 0.15 * clamp(0.6 * energy + 0.4 * tension, 0.0, 1.0)
    auto_tr_idx = int(track_idx_start)
    for s_key in sustained_keys:
        role_cat = "pad" if any(w in s_key.lower() for w in ("pad", "string")) else "chords"
        base_hz = _default_cutoff_hz(role_cat, valence)
        peak_hz = min(14000.0, base_hz * open_factor)
        close_hz = max(400.0, base_hz * 0.94)
        mx_idx = int((channel_mixer_map or {}).get(s_key, 19))
        slot_idx = 2 if mx_idx == 19 else 1

        pts: list[dict[str, Any]] = []
        tot_end = 0.0
        for p_start, p_end in phrase_spans:
            dur = p_end - p_start
            if dur <= 0.25:
                continue
            tot_end = max(tot_end, p_end)
            for frac, hz in ((0.0, base_hz), (0.65, peak_hz), (0.88, close_hz), (0.99, base_hz)):
                t_abs = round(p_start + frac * dur, 4)
                val_n = _hz_to_normalized(hz, "Band 7 freq", is_channel=False)
                if not pts or abs(pts[-1]["position"] - t_abs) > 1e-3:
                    pts.append({"position": t_abs, "value": val_n, "shape": "curve"})
        if pts and tot_end > 0.0:
            autos.append({
                "name": f"Prosody LPF {s_key}",
                "mixer": mx_idx,
                "slot": slot_idx,
                "param": "Band 7 freq",
                "track": min(31, auto_tr_idx),
                "start": 0.0,
                "length": round(tot_end, 4),
                "points": pts,
            })
            auto_tr_idx = min(31, auto_tr_idx + 1)

    # 2. Cadential idiomatic ornaments at bars 4 & 8
    bass_keys = [k for k in out_tracks if any(w in k.lower() for w in ("bass", "808", "sub"))]
    for bk in bass_keys:
        evs = sorted(out_tracks[bk], key=lambda e: float(e["start"]))
        if len(evs) < 2:
            continue
        additions: list[dict[str, Any]] = []
        for sec in norm_secs:
            sb = sec["start_bar"]
            nb = sec["bars"]
            for rel_b in range(nb):
                if (rel_b + 1) % 4 != 0:
                    continue
                abs_b = sb + rel_b
                bar_s = float(abs_b) * bpb
                bar_e = float(abs_b + 1) * bpb
                in_bar = [e for e in evs if bar_s - 1e-4 <= float(e["start"]) < bar_e - 1e-4]
                after_bar = [e for e in evs if bar_e - 1e-4 <= float(e["start"]) < bar_e + bpb]
                if not in_bar or not after_bar:
                    continue
                last_ev = in_bar[-1]
                nxt_ev = after_bar[0]
                last_end = float(last_ev["start"]) + _note_len(last_ev)
                nxt_start = float(nxt_ev["start"])
                cur_p = int(last_ev["pitch"])
                nxt_p = int(nxt_ev["pitch"])
                if abs(nxt_p - cur_p) < 3:
                    continue
                if nxt_start - last_end < 0.5 - 1e-4:
                    continue
                if last_end > bar_e - 0.25 + 1e-4:
                    continue

                direction = 1 if nxt_p > cur_p else -1
                cands = [
                    p for p in range(max(31, nxt_p - 3), min(60, nxt_p + 4))
                    if p != nxt_p and (p % 12) in allowed_pcs
                ]
                if not cands:
                    continue
                # Prefer diatonic leading tone in the direction of motion (1-2 st from nxt_p)
                app_p = min(
                    cands,
                    key=lambda p: (
                        0 if (nxt_p - p) * direction > 0 else 1,
                        abs(nxt_p - p),
                    ),
                )
                app_vel = int(max(38, min(118, round(float(last_ev.get("velocity", 90)) * 0.82))))
                additions.append({
                    "start": round(bar_e - 0.25, 5),
                    "length": 0.25,
                    "pitch": int(app_p),
                    "velocity": app_vel,
                })
        if additions:
            evs.extend(additions)
            evs.sort(key=lambda e: (float(e["start"]), int(e["pitch"])))
            out_tracks[bk] = evs

    # 2b. Lead short appoggiatura before bar-3 apex in high-energy sections (clash-checked)
    lead_keys = [k for k in out_tracks if any(w in k.lower() for w in ("lead", "melody", "topline"))]
    pm_base = _build_temp_pm(out_tracks, eff_bpm, bpb_num, bpb_den)
    _, base_blocking = detect_minor_2nd_clashes(
        pm_base,
        allowed_pcs=allowed_pcs,
        valence=valence,
        tension=tension,
        time_signature=(bpb_num, bpb_den),
    )
    base_clash_count = len(base_blocking)

    for lk in lead_keys:
        evs = sorted(out_tracks[lk], key=lambda e: float(e["start"]))
        if not evs:
            continue
        for sec in norm_secs:
            if _classify_section_pocket(sec) != "on_top":
                continue
            sb = sec["start_bar"]
            nb = sec["bars"]
            cursor = sb
            while cursor < sb + nb:
                p_bars = min(4, (sb + nb) - cursor)
                if p_bars >= 3:
                    bar3_s = float(cursor + 2) * bpb
                    bar3_e = float(cursor + 3) * bpb
                    b3_notes = [e for e in evs if bar3_s - 1e-4 <= float(e["start"]) < bar3_e - 1e-4]
                    if b3_notes:
                        apex = max(b3_notes, key=lambda e: (int(e["pitch"]), -float(e["start"])))
                        ap_st = float(apex["start"])
                        if ap_st >= 0.125:
                            prev_ends = [
                                float(e["start"]) + _note_len(e)
                                for e in evs
                                if float(e["start"]) < ap_st - 1e-4
                            ]
                            if not prev_ends or max(prev_ends) <= ap_st - 0.125 + 1e-4:
                                upper_cands = [
                                    p for p in range(int(apex["pitch"]) + 1, min(max_lead_pitch, int(apex["pitch"]) + 4) + 1)
                                    if (p % 12) in allowed_pcs
                                ]
                                if upper_cands:
                                    app_p = upper_cands[0]
                                    cand_ev = {
                                        "start": round(ap_st - 0.125, 5),
                                        "length": 0.125,
                                        "pitch": int(app_p),
                                        "velocity": int(max(38, min(118, int(apex.get("velocity", 90)) - 14))),
                                    }
                                    trial_tracks = {**out_tracks, lk: sorted([*evs, cand_ev], key=lambda e: float(e["start"]))}
                                    pm_trial = _build_temp_pm(trial_tracks, eff_bpm, bpb_num, bpb_den)
                                    _, trial_blocking = detect_minor_2nd_clashes(
                                        pm_trial,
                                        allowed_pcs=allowed_pcs,
                                        valence=valence,
                                        tension=tension,
                                        time_signature=(bpb_num, bpb_den),
                                    )
                                    if len(trial_blocking) <= base_clash_count:
                                        evs.append(cand_ev)
                                        evs.sort(key=lambda e: (float(e["start"]), int(e["pitch"])))
                                        out_tracks[lk] = evs
                cursor += p_bars

    # 3. Section-dependent dynamic pocket shift
    laid_shift_beats = ms_to_beats(3.5, eff_bpm)
    ontop_shift_beats = ms_to_beats(-2.0, eff_bpm)
    for r_key, evs in out_tracks.items():
        rl = r_key.lower()
        is_laid_target = any(w in rl for w in ("snare", "chord", "lead", "melody"))
        is_ontop_target = any(w in rl for w in ("snare", "hat", "hh"))
        if not (is_laid_target or is_ontop_target):
            continue
        for ev in evs:
            st = float(ev["start"])
            for sec in norm_secs:
                s_b = float(sec["start_bar"]) * bpb
                e_b = float(sec["start_bar"] + sec["bars"]) * bpb
                if s_b - 1e-4 <= st < e_b - 1e-4:
                    p_kind = _classify_section_pocket(sec)
                    if p_kind == "laid_back" and is_laid_target:
                        ev["start"] = round(max(0.0, st + laid_shift_beats), 5)
                    elif p_kind == "on_top" and is_ontop_target:
                        ev["start"] = round(max(0.0, st + ontop_shift_beats), 5)
                    break
        evs.sort(key=lambda e: (float(e["start"]), int(e["pitch"])))

    return out_tracks, autos


def velocity_cutoff_automation(
    tracks: dict[str, list[dict[str, Any]]],
    sections_tuples: list[tuple[str, int, int, str]],
    autos: list[dict[str, Any]],
    bpm: float,
    *,
    track_key: str = "chords",
    mixer_idx: int = 19,
    track_idx: int = 30,
    lo_hz: float = 6500.0,
    hi_hz: float = 12000.0,
    phrase_bars: int = 4,
) -> dict[str, Any]:
    """v3.2 ExpressionArticulations — couplage vélocité -> Cutoff (dynamique timbrale) :
    par phrase de `phrase_bars` mesures, le LPF du canal harmonie suit l'enveloppe de
    vélocité moyenne (passages intimes vel<75 plus chaleureux/fermés, sommets ouverts).
    N'émet que des automations `filter_sweep` douces ; ne touche aucune note."""
    from transitions_lib import filter_sweep

    notes = [n for n in tracks.get(track_key, []) if n]
    if not notes:
        return {"applied": 0, "reason": "no notes"}
    bpb = 4.0
    total = max(float(n["start"]) + _note_len(n) for n in notes)
    applied = 0
    p = 0.0
    while p < total - 1e-6:
        pend = p + phrase_bars * bpb
        in_phrase = [n for n in notes if p - 1e-6 <= float(n["start"]) < pend]
        if in_phrase:
            mean_v = sum(float(n.get("velocity", 80)) for n in in_phrase) / len(in_phrase)
            t = max(0.0, min(1.0, (mean_v - 40.0) / 60.0))  # vel 40 -> fermé, 100 -> ouvert
            hz = lo_hz + t * (hi_hz - lo_hz)
            filter_sweep(
                autos,
                mixer_idx=mixer_idx,
                slot_idx=1,
                start_beat=round(p, 4),
                end_beat=round(pend, 4),
                start_hz=hz,
                end_hz=hz,
                curve="linear",
                track_idx=track_idx,
                steps=2,
                name=f"{track_key} vel->cutoff",
            )
            applied += 1
        p += phrase_bars * bpb
    return {"applied": applied, "track": track_key, "lo_hz": lo_hz, "hi_hz": hi_hz}
