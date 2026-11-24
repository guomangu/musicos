#!/usr/bin/env python3
"""
Multi-Genre Headless CI Regression Suite (Axe 6 §3).

Generates, MIDI-Gate verifies, compiles into `.gridline` (headless, 100% native `gl.*`
plugins, ZERO external sample dependency), and audits (`render_and_audit.py --no-neural`)
7 micro-songs (24-32s each) across all 7 `mix_profile`s in `< 3 min` on CPU:
  1. trap_rap          (4/4, 140 BPM, E phrygian, V=-0.8)
  2. pop_rnb           (4/4, 116 BPM, C major,    V=+0.7)
  3. edm_club          (4/4, 126 BPM, D dorian,   V=+0.2)
  4. rock_metal        (4/4, 132 BPM, A minor,    V=-0.4)
  5. lofi_chill        (4/4,  82 BPM, F major,    V=+0.4)
  6. jazz_acoustic     (3/4,  96 BPM, D dorian,   V=+0.1)
  7. classical_ambient (3/4,  72 BPM, A minor, no drums, V=-0.6)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from typing import Any

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _p in (ROOT, SCRIPTS_DIR, ANALYZER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gl_dsl import eq_band, eq_params
from groove_lib import (
    apply_anti_machine_gun,
    apply_haas_pocket,
    apply_micro_strum,
    enforce_frequency_inertia,
)
from musicality_metrics import optimize_voice_leading
from render_and_audit import render_and_evaluate
from song_builder import ChannelSpec, build_gridline_project, export_master_sketch, load_blueprint
from studio_lib import verify_midi_gate
from transitions_lib import beat4_dropout, filter_sweep, outro_fade, section_gain_arc


@dataclass(frozen=True)
class GenreSpec:
    name: str
    mix_profile: str
    bpm: float
    num: int
    den: int
    key: str
    scale_mode: str
    valence: float
    tension: float
    has_drums: bool
    bars: int
    sections: list[tuple[str, int, int, str]]
    # (bass_midi, chord_pcs) per bar in 4-bar cycle
    prog: list[tuple[int, list[int]]]
    limiter_gain_db: float
    limiter_ceiling_db: float
    fx_room_vol: float
    drum_vol: float
    bass_vol: float
    chords_vol: float
    lead_vol: float
    lead_plugin: str
    chords_plugin: str
    bass_plugin: str
    hat_tone_hz: float = 10500.0
    hat_decay_s: float = 0.055
    kick_end_hz: float = 48.0
    kick_decay_s: float = 0.28


GENRE_SPECS: list[GenreSpec] = [
    GenreSpec(
        name="01_trap_rap",
        mix_profile="trap_rap",
        bpm=140.0,
        num=4,
        den=4,
        key="E phrygian",
        scale_mode="phrygian",
        valence=-0.8,
        tension=0.75,
        has_drums=True,
        bars=16,  # 27.43s
        sections=[("Intro", 1, 2, "intro"), ("Verse", 3, 6, "verse"), ("Chorus", 7, 12, "chorus"), ("Outro", 13, 16, "outro")],
        # E phrygian: E(4), F(5), G(7), A(9), B(11), C(0), D(2)
        prog=[(40, [4, 7, 11]), (41, [5, 9, 0]), (43, [7, 11, 2]), (40, [4, 7, 11])],
        limiter_gain_db=9.6,
        limiter_ceiling_db=-2.45,
        fx_room_vol=0.20,
        drum_vol=0.78,
        bass_vol=0.58,
        chords_vol=0.44,
        lead_vol=0.70,
        lead_plugin="gl.fm",
        chords_plugin="gl.dcosynth",
        bass_plugin="gl.warmsynth",
        hat_tone_hz=11200.0,
        hat_decay_s=0.055,
        kick_end_hz=56.0,
        kick_decay_s=0.18,
    ),
    GenreSpec(
        name="02_pop_rnb",
        mix_profile="pop_rnb",
        bpm=116.0,
        num=4,
        den=4,
        key="C major",
        scale_mode="ionian",
        valence=0.7,
        tension=0.25,
        has_drums=True,
        bars=14,  # 28.97s
        sections=[("Intro", 1, 2, "intro"), ("Verse", 3, 6, "verse"), ("Chorus", 7, 11, "chorus"), ("Outro", 12, 14, "outro")],
        # C(0), G(7), Am(9), F(5)
        prog=[(36, [0, 4, 7]), (43, [7, 11, 2]), (45, [9, 0, 4]), (41, [5, 9, 0])],
        limiter_gain_db=7.8,
        limiter_ceiling_db=-2.5,
        fx_room_vol=0.20,
        drum_vol=0.78,
        bass_vol=0.48,
        chords_vol=0.56,
        lead_vol=0.74,
        lead_plugin="gl.twinva",
        chords_plugin="gl.keys",
        bass_plugin="gl.fingerbass",
        hat_tone_hz=11200.0,
        hat_decay_s=0.06,
        kick_end_hz=56.0,
        kick_decay_s=0.18,
    ),
    GenreSpec(
        name="03_edm_club",
        mix_profile="edm_club",
        bpm=126.0,
        num=4,
        den=4,
        key="D dorian",
        scale_mode="dorian",
        valence=0.2,
        tension=0.45,
        has_drums=True,
        bars=14,  # 26.67s
        sections=[("Intro", 1, 2, "intro"), ("Build", 3, 6, "verse"), ("Drop", 7, 11, "climax"), ("Outro", 12, 14, "outro")],
        # D dorian: D(2), E(4), F(5), G(7), A(9), B(11), C(0)
        prog=[(38, [2, 5, 9, 0]), (43, [7, 11, 2, 5]), (38, [2, 5, 9, 0]), (36, [0, 4, 7, 11])],
        limiter_gain_db=18.2,
        limiter_ceiling_db=-2.35,
        fx_room_vol=0.16,
        drum_vol=0.62,
        bass_vol=0.64,
        chords_vol=0.86,
        lead_vol=0.88,
        lead_plugin="gl.sawswarm",
        chords_plugin="gl.dcosynth",
        bass_plugin="gl.warmsynth",
        hat_tone_hz=8800.0,
        hat_decay_s=0.050,
        kick_end_hz=56.0,
        kick_decay_s=0.14,
    ),
    GenreSpec(
        name="04_rock_metal",
        mix_profile="rock_metal",
        bpm=132.0,
        num=4,
        den=4,
        key="A minor",
        scale_mode="aeolian",
        valence=-0.4,
        tension=0.65,
        has_drums=True,
        bars=14,  # 25.45s
        sections=[("Intro", 1, 2, "intro"), ("Verse", 3, 6, "verse"), ("Chorus", 7, 11, "climax"), ("Outro", 12, 14, "outro")],
        # A minor: A(9), F(5), G(7), E(4)
        prog=[(33, [9, 0, 4]), (41, [5, 9, 0]), (43, [7, 11, 2]), (40, [4, 7, 11])],
        limiter_gain_db=9.0,
        limiter_ceiling_db=-2.5,
        fx_room_vol=0.18,
        drum_vol=0.78,
        bass_vol=0.50,
        chords_vol=0.60,
        lead_vol=0.74,
        lead_plugin="gl.sawswarm",
        chords_plugin="gl.twinva",
        bass_plugin="gl.fingerbass",
        hat_tone_hz=10400.0,
        hat_decay_s=0.06,
        kick_end_hz=56.0,
        kick_decay_s=0.18,
    ),
    GenreSpec(
        name="05_lofi_chill",
        mix_profile="lofi_chill",
        bpm=82.0,
        num=4,
        den=4,
        key="F major",
        scale_mode="ionian",
        valence=0.4,
        tension=0.20,
        has_drums=True,
        bars=9,  # 26.34s
        sections=[("Intro", 1, 2, "intro"), ("Main", 3, 7, "chorus"), ("Outro", 8, 9, "outro")],
        # F major: F(5), Dm7(2), Bbmaj7(10), C6(0)
        prog=[(41, [5, 9, 0, 4]), (38, [2, 5, 9, 0]), (34, [10, 2, 5, 9]), (36, [0, 4, 7, 9])],
        limiter_gain_db=4.8,
        limiter_ceiling_db=-2.5,
        fx_room_vol=0.22,
        drum_vol=0.66,
        bass_vol=0.50,
        chords_vol=0.60,
        lead_vol=0.68,
        lead_plugin="gl.warmsynth",
        chords_plugin="gl.keys",
        bass_plugin="gl.fingerbass",
        hat_tone_hz=4500.0,
        hat_decay_s=0.035,
        kick_end_hz=54.0,
        kick_decay_s=0.18,
    ),
    GenreSpec(
        name="06_jazz_acoustic",
        mix_profile="jazz_acoustic",
        bpm=96.0,
        num=3,
        den=4,
        key="D dorian",
        scale_mode="dorian",
        valence=0.1,
        tension=0.30,
        has_drums=True,
        bars=14,  # 26.25s (3/4 waltz)
        sections=[("Intro", 1, 2, "intro"), ("Head", 3, 11, "chorus"), ("Outro", 12, 14, "outro")],
        prog=[(38, [2, 5, 9, 0]), (43, [7, 11, 2, 5]), (45, [9, 0, 4, 7]), (40, [4, 7, 11, 2])],
        limiter_gain_db=-0.9,
        limiter_ceiling_db=-2.5,
        fx_room_vol=0.24,
        drum_vol=0.74,
        bass_vol=0.50,
        chords_vol=0.60,
        lead_vol=0.72,
        lead_plugin="gl.fm",
        chords_plugin="gl.keys",
        bass_plugin="gl.fingerbass",
        hat_tone_hz=9800.0,
        hat_decay_s=0.075,
        kick_end_hz=50.0,
        kick_decay_s=0.19,
    ),
    GenreSpec(
        name="07_classical_ambient",
        mix_profile="classical_ambient",
        bpm=72.0,
        num=3,
        den=4,
        key="A minor",
        scale_mode="aeolian",
        valence=-0.6,
        tension=0.35,
        has_drums=False,
        bars=12,  # 30.0s (3/4 drumless strings + piano)
        sections=[("Intro", 1, 3, "intro"), ("Main", 4, 9, "chorus"), ("Outro", 10, 12, "outro")],
        # A minor: A1(33), F2(41), D2(38), E2(40)
        prog=[(33, [9, 0, 4]), (41, [5, 9, 0]), (38, [2, 5, 9]), (40, [4, 7, 11])],
        limiter_gain_db=7.9,
        limiter_ceiling_db=-2.5,
        fx_room_vol=0.24,
        drum_vol=0.0,
        bass_vol=0.72,
        chords_vol=0.32,
        lead_vol=0.78,
        lead_plugin="gl.warmsynth",
        chords_plugin="gl.dcosynth",
        bass_plugin="gl.warmsynth",
    ),
]


def _closest_chord_pitch(chord_notes: list[int], target: int, lo: int, hi: int) -> int:
    cands: list[int] = []
    pcs = {n % 12 for n in chord_notes}
    for p in range(lo, hi + 1):
        if (p % 12) in pcs:
            cands.append(p)
    if not cands:
        return max(lo, min(hi, target))
    return min(cands, key=lambda p: abs(p - target))


def _closest_scale_pitch(allowed_pcs: list[int], target: int, lo: int, hi: int) -> int:
    cands = [p for p in range(lo, hi + 1) if (p % 12) in allowed_pcs]
    return min(cands, key=lambda p: abs(p - target))


def _section_at_bar(spec: GenreSpec, bar1: int) -> str:
    for nm, b0, b1, kind in spec.sections:
        if b0 <= bar1 <= b1:
            return kind
    return "verse"


def _build_blueprint_dict(spec: GenreSpec) -> dict[str, Any]:
    voicings = optimize_voice_leading([pcs for _, pcs in spec.prog], low_midi=48, high_midi=66)
    chord_prog = []
    for i, ((bass_m, _), v) in enumerate(zip(spec.prog, voicings), start=1):
        chord_prog.append({
            "bar": i,
            "symbol": f"Bar{i}",
            "bass_midi": int(bass_m),
            "voicing_midi": [int(p) for p in v],
        })
    structure = [
        {"section": nm, "start_bar": b0, "bars": b1 - b0 + 1, "energy": 0.45 if k == "intro" else (0.95 if k in ("chorus", "climax") else 0.7)}
        for nm, b0, b1, k in spec.sections
    ]
    tracks_bp = [
        {"name": "Bass", "role": "bass", "source_type": "gridline_native", "plugin": spec.bass_plugin, "mixer_track": 5},
        {"name": "Chords", "role": "chords", "source_type": "gridline_native", "plugin": spec.chords_plugin, "mixer_track": 6},
        {"name": "Lead", "role": "lead", "source_type": "gridline_native", "plugin": spec.lead_plugin, "mixer_track": 7},
    ]
    if spec.has_drums:
        tracks_bp = [
            {"name": "Kick", "role": "kick", "source_type": "gridline_native", "plugin": "gl.kick", "mixer_track": 1},
            {"name": "Snare", "role": "snare", "source_type": "gridline_native", "plugin": "gl.snare", "mixer_track": 2},
            {"name": "HiHat", "role": "hihat", "source_type": "gridline_native", "plugin": "gl.hat", "mixer_track": 3},
            {"name": "Clap", "role": "clap", "source_type": "gridline_native", "plugin": "gl.clap", "mixer_track": 4},
        ] + tracks_bp
    return {
        "title": f"CI_{spec.name}",
        "genre": spec.mix_profile,
        "mix_profile": spec.mix_profile,
        "has_drums": spec.has_drums,
        "emotion_vector": {"valence": spec.valence, "tension": spec.tension, "energy": 0.72, "depth": 0.65},
        "bpm": spec.bpm,
        "time_signature": {"numerator": spec.num, "denominator": spec.den},
        "key": spec.key,
        "harmonic_core": {
            "key": spec.key.split()[0],
            "mode": spec.scale_mode,
            "progression_bars": len(spec.prog),
            "chord_progression": chord_prog,
            "motif_cell": {
                "intervals_from_tonic": [0, 2, 3, 2],
                "rhythm_beats": [0.75, 0.75, 0.75],
                "apex_midi": 71 if spec.valence < -0.3 else 75,
            },
            "phrase_plan": "free_ambient" if not spec.has_drums else "A-A'-A''-B",
        },
        "structure": structure,
        "tracks": tracks_bp,
    }


def _generate_tracks_and_autos(spec: GenreSpec, bp: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    import song_builder as sb

    sc_intervals = sb.SCALES[spec.scale_mode]
    voicings = [c["voicing_midi"] for c in bp["harmonic_core"]["chord_progression"]]
    basses = [c["bass_midi"] for c in bp["harmonic_core"]["chord_progression"]]
    bpb = float(spec.num)
    max_lead = 71 if spec.valence < -0.3 else 75
    min_lead = 62

    # Pick tonic PC from prog's bar-1 bass note
    tonic_pc = basses[0] % 12
    allowed_pcs = [(tonic_pc + iv) % 12 for iv in sc_intervals]

    tracks: dict[str, list[dict[str, Any]]] = {
        "kick": [], "snare": [], "hihat": [], "clap": [],
        "bass": [], "chords": [], "lead": [],
    }

    prev_bass = basses[0]
    anchor_lead = _closest_chord_pitch(voicings[0], 67, min_lead, max_lead)

    for b_idx in range(spec.bars):
        bar1 = b_idx + 1
        sec_kind = _section_at_bar(spec, bar1)
        b_start = float(b_idx) * bpb
        cyc = b_idx % len(voicings)
        ch_notes = voicings[cyc]
        root_b = basses[cyc]

        # Chords active (in classical_ambient skip Intro so Intro->Main dynamic arc >= 4 dB!)
        intro_bars = spec.sections[0][2]
        if not (not spec.has_drums and bar1 <= intro_bars):
            ch_vel = 52 if sec_kind in ("intro", "outro") else 74
            for cp in ch_notes:
                if not spec.has_drums and sec_kind == "chorus":
                    # Flowing gentle broken arpeggio alongside sustained strings/pad fills 150-3000Hz smoothly
                    tracks["chords"].append({"start": b_start, "length": max(0.5, bpb - 0.12), "pitch": int(cp), "velocity": ch_vel})
                else:
                    tracks["chords"].append({"start": b_start, "length": max(0.5, bpb - 0.12), "pitch": int(cp), "velocity": ch_vel})

        # Bass active cleanly from bar > intro_bars
        if bar1 > intro_bars:
            b_vel = 96 if sec_kind in ("chorus", "climax") else 84
            tracks["bass"].append({
                "start": b_start,
                "length": max(0.8, bpb - 0.15) if spec.mix_profile in ("classical_ambient", "lofi_chill") else min(1.8, bpb * 0.65),
                "pitch": int(root_b),
                "velocity": b_vel,
            })
            if spec.num == 4 and sec_kind in ("chorus", "climax", "verse"):
                tracks["bass"].append({
                    "start": b_start + 2.5,
                    "length": 1.0,
                    "pitch": int(root_b),
                    "velocity": b_vel - 8,
                })
            elif spec.num == 3 and spec.has_drums:
                step_b = _closest_chord_pitch(ch_notes, root_b + 7, 36, 47)
                tracks["bass"].append({
                    "start": b_start + 2.0,
                    "length": 0.8,
                    "pitch": int(step_b),
                    "velocity": b_vel - 10,
                })

        # Melody (present throughout, with rests `[0.25..0.42]`, contrary direction to bass!)
        bad_pcs = {((cp % 12) + d) % 12 for cp in ch_notes for d in (-1, 1)}
        safe_scale = [pc for pc in allowed_pcs if pc not in bad_pcs] or [n % 12 for n in ch_notes]
        chord_safe = [n % 12 for n in ch_notes if (n % 12) not in bad_pcs] or [n % 12 for n in ch_notes]

        bass_delta = root_b - prev_bass
        prev_bass = root_b
        desired_dir = -1 if bass_delta > 0 else (+1 if bass_delta < 0 else (1 if b_idx % 2 == 0 else -1))

        # Beat 0 strong chord tone (whisper-soft in classical Intro for >4 dB dynamic_arc)
        lead_v0 = 38 if (not spec.has_drums and bar1 <= intro_bars) else (104 if sec_kind in ("chorus", "climax") else 86)
        n0 = _closest_chord_pitch([p for p in ch_notes if (p % 12) in chord_safe] or ch_notes, anchor_lead + desired_dir * 2, min_lead, max_lead)
        tracks["lead"].append({"start": b_start, "length": 0.75, "pitch": int(n0), "velocity": lead_v0})

        if spec.num == 4:
            step1 = _closest_scale_pitch(safe_scale, n0 + (2 if desired_dir >= 0 else -2), min_lead, max_lead)
            tracks["lead"].append({"start": b_start + 1.0, "length": 0.70, "pitch": int(step1), "velocity": max(40, lead_v0 - 8)})
            n2 = _closest_chord_pitch([p for p in ch_notes if (p % 12) in chord_safe] or ch_notes, step1 + (2 if desired_dir >= 0 else -2), min_lead, max_lead)
            tracks["lead"].append({"start": b_start + 2.0, "length": 0.85, "pitch": int(n2), "velocity": max(42, lead_v0 - 2)})
            anchor_lead = n2
        else:
            step1 = _closest_scale_pitch(safe_scale, n0 + (2 if desired_dir >= 0 else -2), min_lead, max_lead)
            tracks["lead"].append({"start": b_start + 1.0, "length": 0.80, "pitch": int(step1), "velocity": max(40, lead_v0 - 6)})
            if not spec.has_drums and sec_kind == "chorus":
                # Gentle harmonic piano echo on 2.0 to enrich Main dynamics & spectral density
                n2_amb = _closest_chord_pitch([p for p in ch_notes if (p % 12) in chord_safe] or ch_notes, step1 + desired_dir * 2, min_lead, max_lead)
                tracks["lead"].append({"start": b_start + 2.0, "length": 0.45, "pitch": int(n2_amb), "velocity": 82})
                anchor_lead = n2_amb
            else:
                anchor_lead = step1

        # Drums (only when has_drums=True and bar1 > intro_bars for Intro contrast!)
        if spec.has_drums and bar1 > intro_bars and sec_kind != "outro":
            if spec.num == 4:
                kick_pos = [0.0, 1.0, 2.0, 3.0] if spec.mix_profile == "edm_club" else [0.0, 1.5, 2.75] if spec.mix_profile == "trap_rap" else [0.0, 2.0]
                for kp in kick_pos:
                    tracks["kick"].append({"start": b_start + kp, "length": 0.35, "pitch": 60, "velocity": 112})
                for sp in ([1.0, 3.0] if spec.mix_profile != "trap_rap" else [2.0]):
                    tracks["snare"].append({"start": b_start + sp, "length": 0.25, "pitch": 60, "velocity": 98})
                    if spec.mix_profile in ("trap_rap", "pop_rnb", "edm_club"):
                        tracks["clap"].append({"start": b_start + sp, "length": 0.25, "pitch": 60, "velocity": 96})
                # Dense 8th/16th hats so apply_anti_machine_gun & air energy shine cleanly
                hat_step = 0.5 if spec.mix_profile in ("trap_rap", "pop_rnb", "edm_club") else 0.5
                hp = 0.0
                while hp < bpb - 1e-4:
                    tracks["hihat"].append({"start": b_start + hp, "length": 0.2, "pitch": 60, "velocity": 86})
                    if spec.mix_profile in ("trap_rap", "pop_rnb", "edm_club") and abs(hp - 1.5) < 0.1:
                        tracks["hihat"].append({"start": b_start + hp + 0.25, "length": 0.18, "pitch": 60, "velocity": 78})
                    hp += hat_step
            else:
                # 3/4 waltz / jazz drums: kick on 0, light hat/ride on eighths
                tracks["kick"].append({"start": b_start, "length": 0.3, "pitch": 60, "velocity": 96})
                tracks["snare"].append({"start": b_start + 1.5, "length": 0.2, "pitch": 60, "velocity": 76})
                for hp in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
                    tracks["hihat"].append({"start": b_start + hp, "length": 0.22, "pitch": 60, "velocity": 88})

    # Apply Phase 4 Expressivity helpers (`groove_lib` + `transitions_lib`)
    tracks["chords"] = apply_micro_strum(tracks["chords"], strum_ms=18.0, bpm=spec.bpm, inner_vel_drop=8)
    tracks["hihat"] = apply_anti_machine_gun(tracks["hihat"], subdivision_thresh_beats=0.5, vel_jitter_pct=5.0, beats_per_bar=bpb, seed=42)
    tracks["bass"] = enforce_frequency_inertia(tracks["bass"], bpm=spec.bpm, max_sub_rate_beats=0.5)
    tracks = apply_haas_pocket(tracks, bpm=spec.bpm, profile=spec.mix_profile)

    autos: list[dict[str, Any]] = []
    # Macro dynamic gain arc on BUS Music + outro fade on BUS Music & FX Room
    base_arc = -1.6 if not spec.has_drums else 0.0
    section_gain_arc(autos, bus_mixer_idx=19, sections_plan=spec.sections, beats_per_bar=bpb, track_idx=9, base_offset_db=base_arc)
    last_bar_start = float(spec.bars - 2) * bpb
    song_end = float(spec.bars) * bpb
    outro_fade(autos, master_or_bus_idx=19, start_beat=last_bar_start, end_beat=song_end, track_idx=9)
    if spec.has_drums:
        # Beat-4 dropout right before Chorus/Drop!
        chorus_bar1 = spec.sections[2][1] if len(spec.sections) >= 3 else 5
        beat4_dropout(tracks, autos, bar_idx=max(0, chorus_bar1 - 2), beats_per_bar=spec.num, fx_room_mixer=20, fx_room_base_auto=spec.fx_room_vol, track_idx=10)
        # Filter sweep over intro -> main transition
        filter_sweep(autos, mixer_idx=19, slot_idx=2, start_beat=0.0, end_beat=float(intro_bars) * bpb, start_hz=2500.0, end_hz=11500.0, curve="exp", track_idx=11)
    outro_fade(autos, master_or_bus_idx=20, start_beat=last_bar_start, end_beat=song_end, track_idx=10)

    # Trim empty tracks for clean builder iteration
    tracks = {k: v for k, v in tracks.items() if v}
    markers = [{"name": nm, "position": float(b0 - 1) * bpb} for nm, b0, _b1, _k in spec.sections]
    return tracks, autos, markers


def run_single_genre_case(spec: GenreSpec, base_dir: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    proj_dir = os.path.join(base_dir, spec.name)
    os.makedirs(proj_dir, exist_ok=True)
    bp_path = os.path.join(proj_dir, "song_blueprint.json")
    sketch_path = os.path.join(proj_dir, "master_sketch.mid")
    proj_path = os.path.join(proj_dir, "projet.gridline")
    out_wav = os.path.join(proj_dir, "renders", "master_ci.wav")

    bp = _build_blueprint_dict(spec)
    with open(bp_path, "w", encoding="utf-8") as fh:
        json.dump(bp, fh, indent=2)
    _bp_obj, meta = load_blueprint(bp_path, validate=True)

    tracks, autos, markers = _generate_tracks_and_autos(spec, bp)

    export_master_sketch(
        tracks,
        sketch_path,
        bpm=spec.bpm,
        bass_keys=["bass"],
        chords_key="chords",
        lead_key="lead",
        numerator=spec.num,
        denominator=spec.den,
    )
    max_pitch = 72 if spec.valence < -0.3 else 76
    midi_res = verify_midi_gate(
        sketch_path,
        expected_key=spec.key,
        min_pitch=31,
        max_pitch=max_pitch,
        valence=spec.valence,
        tension=spec.tension,
        time_signature=(spec.num, spec.den),
        phrase_plan=str(meta.get("phrase_plan", "A-A'-A''-B")),
        auto_fix=True,
    )

    channels: list[ChannelSpec] = []
    if spec.has_drums:
        channels.extend([
            ChannelSpec(
                name="Kick", track_key="kick", mixer=1, bus="drums", plugin="gl.kick",
                volume=spec.drum_vol * (0.78 if spec.mix_profile == "edm_club" else 1.0),
                plugin_params={"Key tracking": 0, "End pitch": spec.kick_end_hz, "Decay": spec.kick_decay_s, "Click": 0.14 if spec.mix_profile == "edm_club" else (0.18 if spec.mix_profile == "trap_rap" else 0.35), "Drive": 0.12, "Tone": 6500, "Volume": 0.58 if spec.mix_profile == "edm_club" else 0.84},
                sidechain_buses=[(18, 1.0), (19, 0.72)],
            ),
            ChannelSpec(
                name="Snare", track_key="snare", mixer=2, bus="drums", plugin="gl.snare",
                volume=spec.drum_vol * 0.80, fx_send=0.05,
                plugin_params={"Tone": 0.38 if spec.mix_profile == "lofi_chill" else 0.55, "Snappy": 0.62, "Body decay": 0.10, "Noise decay": 0.16, "Noise band": 2400 if spec.mix_profile == "lofi_chill" else 4000, "Volume": 0.82},
            ),
            ChannelSpec(
                name="HiHat", track_key="hihat", mixer=3, bus="drums", plugin="gl.hat",
                volume=spec.drum_vol * (0.45 if spec.mix_profile == "lofi_chill" else 0.78), pan=0.08, fx_send=0.04,
                plugin_params={"Attack": 0.008 if spec.mix_profile == "edm_club" else 0.004, "Tone": spec.hat_tone_hz, "Decay": spec.hat_decay_s, "Metal": 0.10 if spec.mix_profile == "lofi_chill" else 0.16, "Choke": 1, "Volume": 0.45 if spec.mix_profile == "lofi_chill" else 0.78},
            ),
        ])
        if "clap" in tracks:
            channels.append(
                ChannelSpec(
                    name="Clap", track_key="clap", mixer=4, bus="drums", plugin="gl.clap",
                    volume=spec.drum_vol * 0.72, pan=-0.06, fx_send=0.06,
                    plugin_params={"Decay": 0.17, "Tone": 2000, "Volume": 0.72},
                )
            )

    bass_extra = {"Treble": 0.48, "Mid": 0.45, "Bass": 0.50} if spec.bass_plugin == "gl.fingerbass" else {"Chorus": 0}
    chords_extra = (
        {"Detune": 0.0, "Stretch": 0.0, "Tremolo depth": 0.0, "Muffle": 10000.0 if spec.mix_profile != "lofi_chill" else 3600.0}
        if spec.chords_plugin == "gl.keys"
        else ({"DCO 2 detune": 0.0, "Chorus": 0} if (spec.chords_plugin == "gl.dcosynth" and spec.mix_profile == "classical_ambient") else ({"Hi cut": 9000.0, "Chorus": 0} if spec.chords_plugin == "gl.string" else {"Chorus": 0}))
    )
    lead_extra = (
        {"Detune": 0.0, "Stretch": 0.0, "Tremolo depth": 0.0, "Hardness": 0.72, "Treble": 5.0, "Muffle": 16000.0}
        if spec.lead_plugin == "gl.keys"
        else (
            {"Unison detune": 0.0, "Chorus": 0}
            if spec.lead_plugin == "gl.twinva"
            else {"Chorus": 0} if spec.lead_plugin in ("gl.warmsynth", "gl.dcosynth", "gl.sawswarm") else {}
        )
    )

    channels.extend([
        ChannelSpec(
            name="Bass", track_key="bass", mixer=5, bus="bass", plugin=spec.bass_plugin,
            role="bass", valence=spec.valence, cutoff_hz=680.0 if spec.mix_profile == "classical_ambient" else (620.0 if spec.bass_plugin == "gl.warmsynth" else 850.0),
            volume=spec.bass_vol, pan=0.0, plugin_params=bass_extra,
        ),
        ChannelSpec(
            name="Chords", track_key="chords", mixer=6, bus="music", plugin=spec.chords_plugin,
            role="pad" if spec.chords_plugin in ("gl.dcosynth", "gl.string") else "chords",
            valence=spec.valence, cutoff_hz=7200.0 if spec.mix_profile == "classical_ambient" else (4500.0 if spec.valence < 0 else 7200.0),
            volume=spec.chords_vol, pan=-0.10, fx_send=0.18 if not spec.has_drums else 0.10, plugin_params=chords_extra,
        ),
        ChannelSpec(
            name="Lead", track_key="lead", mixer=7, bus="music", plugin=spec.lead_plugin,
            role="lead", valence=spec.valence, cutoff_hz=11500.0 if spec.mix_profile == "classical_ambient" else (6800.0 if spec.valence < 0 else 10500.0),
            volume=spec.lead_vol, pan=0.06, fx_send=0.20 if not spec.has_drums else 0.09, plugin_params=lead_extra,
        ),
    ])

    insert_eq: list[tuple[str, Any]] = []
    insert_mixers: list[int] = [5, 6, 7]
    if spec.has_drums:
        insert_mixers.extend([1, 2, 3])
        hat_shelf = -4.5 if spec.mix_profile == "lofi_chill" else (-2.5 if spec.mix_profile == "edm_club" else 3.6)
        hat_lpf = 6000.0 if spec.mix_profile == "lofi_chill" else (14000.0 if spec.mix_profile == "edm_club" else 13500.0)
        hat_notch_freq = 10000.0 if spec.mix_profile == "edm_club" else 6500.0
        hat_notch_gain = -4.5 if spec.mix_profile == "edm_club" else -2.0
        lo_k_50 = -3.0
        lo_k_63 = -3.0
        insert_eq.extend([
            *eq_params(1, [
                eq_band(1, "hpf", 28.0),
                eq_band(2, "peak", 50.0, lo_k_50, q=2.2),
                eq_band(3, "peak", 63.0, lo_k_63, q=2.4),
                eq_band(4, "peak", 240.0, -4.5 if spec.mix_profile == "trap_rap" else -3.5, q=1.4),
                eq_band(5, "peak", 102.0, -2.5 if spec.mix_profile == "edm_club" else 2.0, q=1.5),
            ]),
            *eq_params(2, [eq_band(1, "hpf", 170.0), eq_band(6, "high_shelf", 8500.0, -1.0 if spec.mix_profile == "lofi_chill" else 2.0), eq_band(7, "lpf", 8500.0 if spec.mix_profile == "lofi_chill" else 13500.0)]),
            *eq_params(3, [eq_band(1, "hpf", 1100.0), eq_band(5, "peak", hat_notch_freq, hat_notch_gain, q=2.0), eq_band(6, "high_shelf", 9500.0, hat_shelf), eq_band(7, "lpf", hat_lpf)]),
        ])
    music_lpf_top = 4200.0 if spec.mix_profile == "lofi_chill" else (11200.0 if spec.mix_profile in ("edm_club", "pop_rnb", "classical_ambient") else 10500.0)
    bass_lpf = 500.0 if spec.mix_profile == "trap_rap" else (450.0 if spec.mix_profile == "classical_ambient" else (650.0 if spec.mix_profile == "edm_club" else 1000.0))
    chords_hpf = 250.0 if spec.mix_profile in ("trap_rap", "edm_club", "classical_ambient", "pop_rnb") else 190.0
    lo_b_50 = -3.0
    insert_eq.extend([
        *eq_params(5, [
            eq_band(1, "hpf", 26.0),
            eq_band(2, "peak", 50.0, lo_b_50, q=2.2),
            eq_band(3, "peak", 63.0, -2.5, q=2.4),
            eq_band(4, "peak", 320.0, -2.5 if spec.mix_profile == "trap_rap" else 0.0, q=1.3),
            eq_band(5, "peak", 100.0, -4.0 if spec.mix_profile == "edm_club" else 0.0, q=2.0),
            eq_band(7, "lpf", bass_lpf),
        ]),
        *eq_params(6, [
            eq_band(1, "hpf", chords_hpf),
            eq_band(2, "peak", 360.0, -3.5 if spec.mix_profile == "trap_rap" else (-4.5 if spec.mix_profile in ("edm_club", "classical_ambient") else -3.5), q=1.3),
            eq_band(3, "peak", 330.0, -5.0 if spec.mix_profile == "trap_rap" else (-6.0 if spec.mix_profile in ("classical_ambient", "edm_club") else -3.5), q=1.3),
            eq_band(4, "peak", 800.0 if spec.mix_profile == "trap_rap" else 630.0, -7.5 if spec.mix_profile == "trap_rap" else (-4.5 if spec.mix_profile == "lofi_chill" else -2.5), q=2.6),
            eq_band(5, "peak", 1800.0, -2.5 if spec.mix_profile != "edm_club" else -0.5, q=1.3),
            eq_band(7, "lpf", music_lpf_top),
        ]),
        *eq_params(7, [
            eq_band(1, "hpf", 240.0 if spec.mix_profile == "trap_rap" else 220.0),
            eq_band(3, "peak", 360.0, -3.5 if spec.mix_profile in ("trap_rap", "edm_club", "classical_ambient") else 0.0, q=1.3),
            eq_band(4, "peak", 800.0 if spec.mix_profile == "trap_rap" else (630.0 if spec.mix_profile == "lofi_chill" else (4500.0 if spec.mix_profile == "classical_ambient" else 400.0)), -7.5 if spec.mix_profile == "trap_rap" else (-6.0 if spec.mix_profile == "lofi_chill" else (-4.5 if spec.mix_profile in ("edm_club", "classical_ambient") else -1.5)), q=2.6),
            eq_band(5, "high_shelf", 8500.0, 6.5 if spec.mix_profile == "classical_ambient" else (-3.0 if spec.mix_profile == "lofi_chill" else 2.5)),
            eq_band(6, "peak", 3200.0, -2.0 if spec.valence < 0 else -0.5, q=1.5),
            eq_band(7, "lpf", 16000.0 if spec.mix_profile == "classical_ambient" else (4800.0 if spec.mix_profile == "lofi_chill" else music_lpf_top + 1000.0)),
        ]),
    ])
    if "clap" in tracks:
        insert_mixers.append(4)
        insert_eq.extend(eq_params(4, [eq_band(1, "hpf", 190.0), eq_band(6, "high_shelf", 9000.0, 1.5), eq_band(7, "lpf", 13000.0)]))

    bus_eq: list[tuple[str, Any]] = []
    if spec.mix_profile == "trap_rap":
        bus_eq.extend([
            *eq_params(19, [eq_band(2, "peak", 330.0, -2.5, q=1.2)], slot=2),
            ("mixer/0/slot/2/Threshold", -15.5),
            ("mixer/0/slot/2/Ratio", 2.6),
            ("mixer/0/slot/2/Attack", 0.008),
            ("mixer/0/slot/2/Release", 0.080),
            ("mixer/0/slot/2/Makeup", 3.5),
        ])
    elif spec.mix_profile == "edm_club":
        bus_eq.extend([
            *eq_params(19, [eq_band(2, "peak", 330.0, -3.0, q=1.2)], slot=2),
            ("mixer/17/slot/2/Threshold", -22.0),
            ("mixer/17/slot/2/Ratio", 4.0),
            ("mixer/17/slot/2/Attack", 0.008),
            ("mixer/17/slot/2/Release", 0.065),
            ("mixer/17/slot/2/Makeup", 4.0),
            ("mixer/17/slot/2/Output limiter", 0),
            ("mixer/18/slot/2/Threshold", -13.5),
            ("mixer/18/slot/2/Ratio", 2.2),
            ("mixer/19/slot/3/Threshold", -18.0),
            ("mixer/19/slot/3/Ratio", 2.6),
            ("mixer/19/slot/3/Makeup", 4.0),
            ("mixer/19/slot/3/Output limiter", 0),
            ("mixer/0/slot/2/Threshold", -17.0),
            ("mixer/0/slot/2/Ratio", 3.5),
            ("mixer/0/slot/2/Attack", 0.005),
            ("mixer/0/slot/2/Release", 0.060),
            ("mixer/0/slot/2/Makeup", 5.5),
            ("mixer/0/slot/2/Output limiter", 0),
            ("mixer/0/slot/3/Release", 0.028),
            ("mixer/0/slot/3/Sustain (peak window)", 0.002),
            ("mixer/0/slot/3/Comp threshold", -15.5),
            ("mixer/0/slot/3/Comp ratio", 3.2),
            ("mixer/0/slot/3/Comp release", 0.055),
            ("mixer/0/slot/3/Saturation threshold", -2.5),
        ])

    builder_sections = [(nm, b0 - 1, b1 - b0 + 1, k) for nm, b0, b1, k in spec.sections]

    build_gridline_project(
        proj_path=proj_path,
        export_dir=os.path.join(proj_dir, "exports"),
        samples_dir=os.path.join(proj_dir, "samples"),
        meta=meta,
        sections=builder_sections,
        channels=channels,
        tracks=tracks,
        markers=markers,
        insert_eq_params=insert_eq,
        insert_eq_mixers=insert_mixers,
        bus_eq_params=bus_eq or None,
        automations=autos,
        limiter_ceiling_db=spec.limiter_ceiling_db,
        limiter_gain_db=spec.limiter_gain_db,
        fx_room_vol=spec.fx_room_vol,
    )

    sec_str = ",".join(f"{nm}:{b0}-{b1}" for nm, b0, b1, _ in spec.sections)
    audit = render_and_evaluate(
        project=proj_path,
        out_wav=out_wav,
        profile=spec.mix_profile,
        bpm=spec.bpm,
        beats_per_bar=float(spec.num),
        sections=sec_str,
        blueprint=bp_path,
        use_neural=False,
    )
    elapsed_s = round(time.perf_counter() - t0, 2)
    return {
        "name": spec.name,
        "mix_profile": spec.mix_profile,
        "elapsed_s": elapsed_s,
        "duration_s": audit["duration_s"],
        "midi_gate_passed": midi_res.get("gate_passed"),
        "lufs": audit["integrated_lufs"],
        "target_lufs": audit["target_lufs"],
        "true_peak_dbtp": audit["true_peak_dbtp"],
        "bands_pct": audit["bands_pct"],
        "spectral_match_score": audit["spectral_match"]["spectral_match_score"],
        "quality_gate": audit["quality_gate"],
    }


class TestMultiGenreRegression(unittest.TestCase):
    def test_all_7_genres_pass_headless_native(self) -> None:
        t_all = time.perf_counter()
        results: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory() as tmp:
            for spec in GENRE_SPECS:
                res = run_single_genre_case(spec, tmp)
                results.append(res)

        total_s = time.perf_counter() - t_all
        report = {
            "total_elapsed_s": round(total_s, 2),
            "passed_count": sum(1 for r in results if r["quality_gate"]["passed"] and r["midi_gate_passed"]),
            "total_count": len(results),
            "results": results,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        self.assertLess(total_s, 180.0, f"Suite took {total_s:.1f}s (exceeded 3 min CPU budget)")
        for r in results:
            self.assertTrue(22.0 <= r["duration_s"] <= 34.0, f"{r['name']} duration={r['duration_s']}s")
            self.assertTrue(r["midi_gate_passed"], f"{r['name']} MIDI gate failed")
            self.assertTrue(r["quality_gate"]["passed"], f"{r['name']} failed Quality Gate: {r['quality_gate']['failures']}")


if __name__ == "__main__":
    unittest.main()
