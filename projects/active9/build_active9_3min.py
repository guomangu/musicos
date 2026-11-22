#!/usr/bin/env python3
"""
Build script for 3-minute enriched "Glass Horizon" (projects/active9/projet.gridline)
129 bars @ 172 BPM (180.00 s = 3:00 exact) Atmospheric Liquid Drum and Bass in F minor.

Honors user's stylistic curation in projet.gridline:
- Keeps the cleaner, deeper aesthetic (no Lead Synth, no Noisehats, no Glass Shimmer)
- Preserves the user's warm Deep 808 + crisp Hi-Hat / Tech Snare / Glass Waves balance
- Enriches & expands to 3:00 using the 3 user-provided samples (and clean tuned derivatives):
  1. Headroom Piano (Headroom Piano Clean.sfz + tuned HEADROOM PIANO LEVEL1 CLOSE 99 sparkle)
  2. Vocal 3 (vocal3_phrase_g4.wav, vocal3_sustain_g4.wav, vocal3_swell_rev.wav)
  3. Chant Misc 003 Hi (chant_hi_ab3.wav, chant_hi_c4.wav)
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import mido

sys.path.insert(0, "/home/gamo/Bureau/musicos/scripts")
from studio_lib import GridlineMCP, TEMPLATE_EMPTY_MIXER, setup_standard_buses, verify_midi_gate

ROOT = "/home/gamo/Bureau/musicos"
PROJ_DIR = f"{ROOT}/projects/active9"
SAMPLES = f"{PROJ_DIR}/samples"
RENDERS = f"{PROJ_DIR}/renders"
BP_PATH = f"{PROJ_DIR}/song_blueprint.json"
SFZ_PIANO = f"{ROOT}/soundbanks/sfz/headroom-piano/Headroom Piano Clean.sfz"

rng = random.Random(172093)


def humanize(notes: list[dict], vel_jit: int = 7, time_jit: float = 0.0) -> list[dict]:
    out = []
    for n in notes:
        item = dict(n)
        v = int(item.get("velocity", 90)) + rng.randint(-vel_jit, vel_jit)
        item["velocity"] = max(25, min(124, v))
        if time_jit > 0.0:
            item["start"] = max(0.0, round(float(item["start"]) + rng.uniform(-time_jit, time_jit), 4))
        out.append(item)
    return out


def main() -> None:
    os.makedirs(RENDERS, exist_ok=True)
    bp = json.load(open(BP_PATH))
    chords = bp["harmonic_core"]["chord_progression"]  # Fm7, Dbmaj9, Abmaj7, Eb6

    # ------------------------------------------------------------------
    # 1. Motif & Phrase Generators (verified Top-Down against F minor)
    # ------------------------------------------------------------------
    # Motif A (Main Liquid Horizon theme, C5..Ab5)
    lead_a = [
        [(72, 0.0, 0.85, 92), (75, 1.0, 0.85, 88), (77, 2.0, 0.70, 98), (75, 2.75, 0.55, 85)],
        [(73, 0.0, 0.85, 90), (72, 1.0, 0.85, 86), (68, 2.0, 0.80, 88)],
        [(72, 0.0, 0.85, 90), (75, 1.0, 0.85, 94), (80, 2.0, 0.85, 104), (79, 3.0, 0.60, 90)],
        [(79, 0.0, 0.85, 92), (77, 1.0, 0.75, 86), (75, 2.0, 0.85, 90)],
    ]
    # Motif C (Warm Nocturne register Eb4..Eb5 for Breaks 2 & 3; zero minor-9th vs G3)
    lead_c = [
        [(65, 0.0, 0.70, 78), (67, 0.75, 0.70, 80), (68, 1.5, 0.70, 84), (72, 2.25, 0.85, 88)],
        [(73, 0.0, 0.85, 84), (72, 1.0, 0.85, 80), (68, 2.0, 0.85, 82)],
        [(67, 0.0, 0.85, 82), (75, 1.0, 0.85, 88), (72, 2.0, 0.85, 86), (70, 3.0, 0.60, 82)],
        [(70, 0.0, 1.05, 84), (67, 1.25, 0.85, 78), (70, 2.25, 0.85, 80)],
    ]

    def build_chords(bars: int, vel: int = 72, cut_last_beat: bool = True) -> list[dict]:
        notes = []
        for b in range(bars):
            ch = chords[b % 4]
            dur = 2.85 if (cut_last_beat and b == bars - 1) else 3.55
            for vp in ch["voicing_midi"]:
                notes.append({"pitch": vp, "start": b * 4.0, "length": dur, "velocity": vel})
        return humanize(notes, vel_jit=5, time_jit=0.008)

    def build_piano_lead(
        bars: int,
        vel_scale: float = 1.0,
        sparse_mode: bool = False,
        nocturne: bool = False,
        cut_last_beat: bool = True,
    ) -> list[dict]:
        notes = []
        src = lead_c if nocturne else lead_a
        for b in range(bars):
            if sparse_mode and (b % 2 == 1):
                p, sb, db, v = src[b % 4][0]
                notes.append({"pitch": p, "start": b * 4.0 + sb, "length": 1.30, "velocity": int(v * vel_scale)})
                continue
            for p, sb, db, v in src[b % 4]:
                if cut_last_beat and b == bars - 1 and sb >= 2.5:
                    continue
                notes.append({"pitch": p, "start": b * 4.0 + sb, "length": db, "velocity": int(v * vel_scale)})
        return humanize(notes, vel_jit=6, time_jit=0.010)

    def build_sparkle_99(bars: int, vel: int = 72, sparse: bool = False) -> list[dict]:
        """Crystalline upper piano notes (MIDI 72..80 with Root=87 -> C6..Ab6) placed in rests."""
        # Safe consonant notes per bar (avoiding minor-9th with chords/lead):
        # Bar 0 (Fm7): C6(72), Eb6(75)
        # Bar 1 (Dbmaj9): F6(77) at 3.25
        # Bar 2 (Abmaj7): Eb6(75) at 1.5, C6(72) at 3.5
        # Bar 3 (Eb6): Bb5(70) or G6(79) at 3.25
        notes = []
        for b in range(bars):
            if sparse and b % 2 == 0:
                continue
            if b == bars - 1:
                continue
            m = b % 4
            if m == 0:
                notes.append({"pitch": 75, "start": b * 4.0 + 3.5, "length": 0.45, "velocity": vel})
            elif m == 1:
                notes.append({"pitch": 77, "start": b * 4.0 + 3.25, "length": 0.65, "velocity": vel - 4})
            elif m == 2:
                notes.append({"pitch": 75, "start": b * 4.0 + 1.5, "length": 0.45, "velocity": vel - 6})
                notes.append({"pitch": 72, "start": b * 4.0 + 3.5, "length": 0.45, "velocity": vel})
            elif m == 3:
                notes.append({"pitch": 79, "start": b * 4.0 + 3.25, "length": 0.65, "velocity": vel - 2})
        return humanize(notes, vel_jit=5, time_jit=0.008)

    def build_vocal_counter(bars: int, vel: int = 80, soaring: bool = False) -> list[dict]:
        """
        Call-and-response sustained vocal lines using vocal3_sustain_g4.wav (Root=67 G4).
        Placed strictly in Piano Lead rests (>= 0.15s away from Piano attacks) to guarantee 0 overlap conflict!
        """
        notes = []
        for b in range(bars):
            if b == bars - 1:
                continue
            m = b % 4
            if m == 1:
                # Dbmaj9: piano ends at 2.80 -> Vocal enters at beat 3.0 on F4 (65)
                notes.append({"pitch": 65, "start": b * 4.0 + 3.0, "length": 0.90, "velocity": vel})
            elif m == 3:
                # Eb6: piano ends at 2.85 -> Vocal enters at beat 3.0 on G4 (67) (or Eb4->G4)
                p = 67 if (not soaring or b < 4) else 70  # Bb4 soaring peak in second half
                notes.append({"pitch": p, "start": b * 4.0 + 3.0, "length": 0.90, "velocity": vel + 4})
            elif soaring and m == 0:
                # Gentle sustained G4 pickup at 3.5 into Dbmaj9
                notes.append({"pitch": 67, "start": b * 4.0 + 3.5, "length": 0.45, "velocity": vel - 6})
        return humanize(notes, vel_jit=4, time_jit=0.006)

    def build_808(bars: int, octave_bounce: bool = False, cut_last_beat: bool = True) -> list[dict]:
        notes = []
        for b in range(bars):
            bm = chords[b % 4]["bass_midi"]
            if cut_last_beat and b == bars - 1:
                notes.append({"pitch": bm, "start": b * 4.0 + 0.0, "length": 2.25, "velocity": 106})
                continue
            notes.append({"pitch": bm, "start": b * 4.0 + 0.0, "length": 2.25, "velocity": 108})
            if octave_bounce and (b % 4 == 3):
                notes.append({"pitch": bm, "start": b * 4.0 + 2.5, "length": 0.65, "velocity": 98})
                notes.append({"pitch": bm + 12, "start": b * 4.0 + 3.25, "length": 0.55, "velocity": 94})
            else:
                notes.append({"pitch": bm, "start": b * 4.0 + 2.5, "length": 1.35, "velocity": 98})
        return humanize(notes, vel_jit=4, time_jit=0.0)

    def build_sausage(bars: int, start_bar: int = 4, end_bar: int = 7) -> list[dict]:
        notes = []
        for b in range(start_bar, end_bar):
            bm = chords[b % 4]["bass_midi"]
            notes.append({"pitch": bm, "start": b * 4.0 + 0.0, "length": 3.70, "velocity": 66})
        return humanize(notes, vel_jit=3, time_jit=0.0)

    def build_dnb_kick(bars: int, cut_last_beat: bool = True, rolling: bool = False) -> list[dict]:
        notes = []
        for b in range(bars):
            if cut_last_beat and b == bars - 1:
                notes.append({"pitch": 60, "start": b * 4.0 + 0.0, "length": 0.45, "velocity": 108})
                continue
            notes.append({"pitch": 60, "start": b * 4.0 + 0.0, "length": 0.45, "velocity": 110})
            notes.append({"pitch": 60, "start": b * 4.0 + 2.5, "length": 0.45, "velocity": 106})
            if rolling and (b % 4 == 2):
                notes.append({"pitch": 60, "start": b * 4.0 + 3.5, "length": 0.25, "velocity": 86})
            elif b % 2 == 1:
                notes.append({"pitch": 60, "start": b * 4.0 + 3.75, "length": 0.22, "velocity": 76})
        return humanize(notes, vel_jit=4, time_jit=0.0)

    def build_dnb_snare(bars: int, cut_last_beat: bool = True, fill_build: bool = False) -> list[dict]:
        notes = []
        if fill_build:
            for b in range(bars):
                if b == bars - 2:
                    for sb, v in [(1.0, 68), (3.0, 76)]:
                        notes.append({"pitch": 60, "start": b * 4.0 + sb, "length": 0.35, "velocity": v})
                elif b == bars - 1:
                    for i, sb in enumerate([0.0, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75]):
                        notes.append({"pitch": 60, "start": b * 4.0 + sb, "length": 0.22, "velocity": 62 + i * 6})
            return notes
        for b in range(bars):
            notes.append({"pitch": 60, "start": b * 4.0 + 1.0, "length": 0.40, "velocity": 108})
            if not (cut_last_beat and b == bars - 1):
                notes.append({"pitch": 60, "start": b * 4.0 + 3.0, "length": 0.40, "velocity": 110})
            else:
                notes.append({"pitch": 60, "start": b * 4.0 + 3.0, "length": 0.35, "velocity": 102})
            if b % 2 == 0 and b < bars - 1:
                notes.append({"pitch": 60, "start": b * 4.0 + 1.75, "length": 0.20, "velocity": 54})
        return humanize(notes, vel_jit=4, time_jit=0.0)

    def build_hats(bars: int, dense: bool = True, cut_last_beat: bool = True, vel_scale: float = 1.0) -> list[dict]:
        notes = []
        step = 0.5
        for b in range(bars):
            for s in range(8):
                sb = s * step
                if cut_last_beat and b == bars - 1 and sb >= 3.0:
                    continue
                v = int((86 if (s % 2 == 0) else 56) * vel_scale)
                notes.append({"pitch": 60, "start": b * 4.0 + sb, "length": 0.18, "velocity": v})
                if dense and (s in (1, 5)):
                    notes.append({"pitch": 60, "start": b * 4.0 + sb + 0.25, "length": 0.14, "velocity": int(48 * vel_scale)})
        return humanize(notes, vel_jit=6, time_jit=0.006)

    def build_perc(bars: int, cut_last_beat: bool = True, vel_scale: float = 1.0) -> list[dict]:
        notes = []
        for b in range(bars):
            if cut_last_beat and b == bars - 1:
                continue
            for sb, v in [(0.75, 74), (1.75, 68), (3.5, 78)]:
                notes.append({"pitch": 60, "start": b * 4.0 + sb, "length": 0.25, "velocity": int(v * vel_scale)})
        return humanize(notes, vel_jit=6, time_jit=0.008)

    # ------------------------------------------------------------------
    # 2. Define Section Clips & Verify Full 129-Bar Orchestrated MIDI Gate
    # ------------------------------------------------------------------
    # Macro structure (129 bars @ 172 BPM = 180.00 s):
    #   Intro:            bars  1..8   (start beat   0, 8 bars)
    #   Drop 1A:          bars  9..16  (start beat  32, 8 bars)
    #   Drop 1B:          bars 17..24  (start beat  64, 8 bars)
    #   Break 1:          bars 25..32  (start beat  96, 8 bars)
    #   Drop 2A:          bars 33..40  (start beat 128, 8 bars)
    #   Drop 2B Sparkle:  bars 41..48  (start beat 160, 8 bars)
    #   Break 2A Nocturne:bars 49..56  (start beat 192, 8 bars)
    #   Break 2B Build:   bars 57..64  (start beat 224, 8 bars)
    #   Drop 3A Rolling:  bars 65..72  (start beat 256, 8 bars)
    #   Drop 3B Soaring:  bars 73..80  (start beat 288, 8 bars)
    #   Break 3A Space:   bars 81..88  (start beat 320, 8 bars)
    #   Break 3B Build:   bars 89..96  (start beat 352, 8 bars)
    #   Drop 4A Climax:   bars 97..104 (start beat 384, 8 bars)
    #   Drop 4B Peak:     bars 105..112(start beat 416, 8 bars)
    #   Drop 4C Release:  bars 113..120(start beat 448, 8 bars)
    #   Outro:            bars 121..129(start beat 480..516, 9 bars)

    outro_chords = []
    for b in range(9):
        ch = chords[b % 4]
        vel = max(36, 64 - b * 3)
        dur = 3.6 if b < 8 else 3.85
        for vp in ch["voicing_midi"]:
            outro_chords.append({"pitch": vp, "start": b * 4.0, "length": dur, "velocity": vel})
    outro_piano = [
        # Bars 0..3 sparse motif A
        {"pitch": 72, "start": 0.0, "length": 0.9, "velocity": 76},
        {"pitch": 75, "start": 1.0, "length": 0.9, "velocity": 72},
        {"pitch": 77, "start": 2.0, "length": 1.4, "velocity": 78},
        {"pitch": 73, "start": 4.0, "length": 1.4, "velocity": 70},
        {"pitch": 72, "start": 8.0, "length": 0.9, "velocity": 72},
        {"pitch": 75, "start": 9.0, "length": 0.9, "velocity": 74},
        {"pitch": 80, "start": 10.0, "length": 1.2, "velocity": 80},
        {"pitch": 79, "start": 12.0, "length": 1.5, "velocity": 70},
        # Bars 4..8 tender final resolution
        {"pitch": 72, "start": 16.0, "length": 0.9, "velocity": 68},
        {"pitch": 75, "start": 17.0, "length": 0.9, "velocity": 66},
        {"pitch": 77, "start": 18.0, "length": 1.5, "velocity": 70},
        {"pitch": 75, "start": 20.0, "length": 1.8, "velocity": 62},
        {"pitch": 72, "start": 24.0, "length": 2.4, "velocity": 58},
        {"pitch": 72, "start": 32.0, "length": 3.5, "velocity": 52},
    ]
    outro_vocal = [
        {"pitch": 65, "start": 7.0, "length": 0.9, "velocity": 68},
        {"pitch": 67, "start": 15.0, "length": 1.2, "velocity": 64},
        {"pitch": 67, "start": 23.0, "length": 2.5, "velocity": 54},
    ]

    # Build patterns list: (name, bars, channel, notes, playlist_track, start_beat)
    pat_specs: list[tuple[str, int, int, list[dict], int, float]] = [
        # --- INTRO (Bars 1..8, beat 0) ---
        ("Intro Glass Intro", 8, 21, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 98}], 16, 0.0),
        ("Intro Chords", 8, 9, build_chords(8, vel=64, cut_last_beat=True), 9, 0.0),
        ("Intro Piano", 4, 10, build_piano_lead(4, vel_scale=0.84, sparse_mode=True), 10, 16.0),
        ("Intro Vocal Air", 4, 13, [{"pitch": 67, "start": 3.0, "length": 5.0, "velocity": 72}], 12, 16.0),
        ("Intro Chant Cue", 1, 15, [{"pitch": 68, "start": 3.5, "length": 0.4, "velocity": 74}], 13, 28.0),
        ("Intro Riser", 1, 20, [{"pitch": 65, "start": 0.0, "length": 3.85, "velocity": 90}], 15, 28.0),

        # --- DROP 1A (Bars 9..16, beat 32) ---
        ("D1A Kick", 8, 1, build_dnb_kick(8), 1, 32.0),
        ("D1A Snare", 8, 2, build_dnb_snare(8), 2, 32.0),
        ("D1A Tech Snare", 8, 3, build_dnb_snare(8), 3, 32.0),
        ("D1A Hats", 8, 4, build_hats(8, dense=True), 4, 32.0),
        ("D1A Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 96}], 5, 32.0),
        ("D1A Deep808", 8, 7, build_808(8, octave_bounce=False), 7, 32.0),
        ("D1A Chords", 8, 9, build_chords(8, vel=70), 9, 32.0),
        ("D1A Piano Lead", 8, 10, build_piano_lead(8, vel_scale=0.92), 10, 32.0),
        ("D1A GlassPad", 8, 17, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 94}], 14, 32.0),

        # --- DROP 1B (Bars 17..24, beat 64) ---
        ("D1B Kick", 8, 1, build_dnb_kick(8), 1, 64.0),
        ("D1B Snare", 8, 2, build_dnb_snare(8), 2, 64.0),
        ("D1B Tech Snare", 8, 3, build_dnb_snare(8), 3, 64.0),
        ("D1B Hats", 8, 4, build_hats(8, dense=True), 4, 64.0),
        ("D1B Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 98}], 5, 64.0),
        ("D1B Perc", 8, 6, build_perc(8), 6, 64.0),
        ("D1B Deep808", 8, 7, build_808(8, octave_bounce=True), 7, 64.0),
        ("D1B Chords", 8, 9, build_chords(8, vel=74), 9, 64.0),
        ("D1B Piano Lead", 8, 10, build_piano_lead(8, vel_scale=0.96), 10, 64.0),
        ("D1B Vocal Counter", 8, 13, build_vocal_counter(8, vel=78), 12, 64.0),
        ("D1B Chant Ab3", 8, 15, [{"pitch": 68, "start": 15.5, "length": 0.4, "velocity": 82}], 13, 64.0),
        ("D1B Chant C4", 8, 16, [{"pitch": 60, "start": 31.5, "length": 0.4, "velocity": 84}], 13, 64.0),
        ("D1B GlassPadB", 8, 18, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 96}], 14, 64.0),
        ("D1B BFNK Cloud", 8, 19, [
            {"pitch": 65, "start": 7.0, "length": 2.2, "velocity": 76},
            {"pitch": 65, "start": 23.0, "length": 2.2, "velocity": 76},
        ], 15, 64.0),
        ("D1B Riser", 1, 20, [{"pitch": 65, "start": 0.0, "length": 3.85, "velocity": 96}], 15, 92.0),

        # --- BREAK 1 PLANANT (Bars 25..32, beat 96) ---
        ("B1 Snare Fill", 8, 2, build_dnb_snare(8, fill_build=True), 2, 96.0),
        ("B1 Hats", 8, 4, build_hats(8, dense=False, vel_scale=0.68), 4, 96.0),
        ("B1 SausageSub", 8, 8, build_sausage(8, 3, 7), 8, 96.0),
        ("B1 Chords", 8, 9, build_chords(8, vel=56), 9, 96.0),
        ("B1 Piano", 8, 10, build_piano_lead(8, vel_scale=0.76, sparse_mode=True), 10, 96.0),
        ("B1 Vocal3 Phrase", 6, 12, [{"pitch": 67, "start": 1.0, "length": 14.0, "velocity": 78}], 12, 100.0),
        ("B1 GlassPad", 8, 17, [{"pitch": 65, "start": 0.0, "length": 30.8, "velocity": 66}], 14, 96.0),
        ("B1 BFNK Cloud", 8, 19, [
            {"pitch": 65, "start": 7.0, "length": 2.2, "velocity": 62},
            {"pitch": 65, "start": 23.0, "length": 2.2, "velocity": 62},
        ], 15, 96.0),
        ("B1 Riser", 1, 20, [{"pitch": 65, "start": 0.0, "length": 3.85, "velocity": 78}], 15, 124.0),

        # --- DROP 2A (Bars 33..40, beat 128) ---
        ("D2A Kick", 8, 1, build_dnb_kick(8), 1, 128.0),
        ("D2A Snare", 8, 2, build_dnb_snare(8), 2, 128.0),
        ("D2A Tech Snare", 8, 3, build_dnb_snare(8), 3, 128.0),
        ("D2A Hats", 8, 4, build_hats(8, dense=True), 4, 128.0),
        ("D2A Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 100}], 5, 128.0),
        ("D2A Perc", 8, 6, build_perc(8), 6, 128.0),
        ("D2A Deep808", 8, 7, build_808(8, octave_bounce=True), 7, 128.0),
        ("D2A Chords", 8, 9, build_chords(8, vel=76), 9, 128.0),
        ("D2A Piano Lead", 8, 10, build_piano_lead(8, vel_scale=0.98), 10, 128.0),
        ("D2A GlassPadB", 8, 18, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 98}], 14, 128.0),

        # --- DROP 2B SPARKLE (Bars 41..48, beat 160) ---
        ("D2B Kick", 8, 1, build_dnb_kick(8, rolling=True), 1, 160.0),
        ("D2B Snare", 8, 2, build_dnb_snare(8), 2, 160.0),
        ("D2B Tech Snare", 8, 3, build_dnb_snare(8), 3, 160.0),
        ("D2B Hats", 8, 4, build_hats(8, dense=True), 4, 160.0),
        ("D2B Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 102}], 5, 160.0),
        ("D2B Perc", 8, 6, build_perc(8), 6, 160.0),
        ("D2B Deep808", 8, 7, build_808(8, octave_bounce=True), 7, 160.0),
        ("D2B Chords", 8, 9, build_chords(8, vel=76), 9, 160.0),
        ("D2B Piano Lead", 8, 10, build_piano_lead(8, vel_scale=1.00), 10, 160.0),
        ("D2B Headroom99 Sparkle", 8, 11, build_sparkle_99(8, vel=74), 11, 160.0),
        ("D2B Vocal Counter", 8, 13, build_vocal_counter(8, vel=82, soaring=True), 12, 160.0),
        ("D2B Chant Ab3", 8, 15, [{"pitch": 68, "start": 15.5, "length": 0.4, "velocity": 82}], 13, 160.0),
        ("D2B Chant C4", 8, 16, [{"pitch": 60, "start": 31.5, "length": 0.4, "velocity": 86}], 13, 160.0),
        ("D2B GlassPadB", 8, 18, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 100}], 14, 160.0),

        # --- BREAK 2 NOCTURNE (Bars 49..64, beat 192..256) ---
        ("B2A Perc Soft", 8, 6, build_perc(8, vel_scale=0.78), 6, 192.0),
        ("B2A Chords", 8, 9, build_chords(8, vel=64), 9, 192.0),
        ("B2A Piano Nocturne", 8, 10, build_piano_lead(8, vel_scale=0.88, nocturne=True), 10, 192.0),
        ("B2A Headroom99 Echo", 8, 11, build_sparkle_99(8, vel=70, sparse=True), 11, 192.0),
        ("B2A Vocal3 Phrase", 6, 12, [{"pitch": 67, "start": 1.0, "length": 15.0, "velocity": 88}], 12, 196.0),
        ("B2A Glass Intro", 8, 21, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 98}], 16, 192.0),

        ("B2B Snare Build", 8, 2, build_dnb_snare(8, fill_build=True), 2, 224.0),
        ("B2B Hats", 8, 4, build_hats(8, dense=False, vel_scale=0.72), 4, 224.0),
        ("B2B SausageSub", 8, 8, build_sausage(8, 2, 7), 8, 224.0),
        ("B2B Chords", 8, 9, build_chords(8, vel=58), 9, 224.0),
        ("B2B Piano Nocturne", 8, 10, build_piano_lead(8, vel_scale=0.86, nocturne=True), 10, 224.0),
        ("B2B Vocal3 Swell", 1, 14, [{"pitch": 67, "start": 0.0, "length": 3.85, "velocity": 88}], 12, 252.0),
        ("B2B Chant Cue", 1, 16, [{"pitch": 60, "start": 3.5, "length": 0.4, "velocity": 82}], 13, 252.0),
        ("B2B GlassPadA", 8, 17, [{"pitch": 65, "start": 0.0, "length": 30.8, "velocity": 68}], 14, 224.0),
        ("B2B Riser", 1, 20, [{"pitch": 65, "start": 0.0, "length": 3.85, "velocity": 86}], 15, 252.0),

        # --- DROP 3A ROLLING (Bars 65..72, beat 256) ---
        ("D3A Kick", 8, 1, build_dnb_kick(8, rolling=True), 1, 256.0),
        ("D3A Snare", 8, 2, build_dnb_snare(8), 2, 256.0),
        ("D3A Tech Snare", 8, 3, build_dnb_snare(8), 3, 256.0),
        ("D3A Hats", 8, 4, build_hats(8, dense=True), 4, 256.0),
        ("D3A Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 100}], 5, 256.0),
        ("D3A Perc", 8, 6, build_perc(8), 6, 256.0),
        ("D3A Deep808", 8, 7, build_808(8, octave_bounce=True), 7, 256.0),
        ("D3A Chords", 8, 9, build_chords(8, vel=76), 9, 256.0),
        ("D3A Piano Lead", 8, 10, build_piano_lead(8, vel_scale=0.98), 10, 256.0),
        ("D3A Chant Ab3", 8, 15, [{"pitch": 68, "start": 15.5, "length": 0.4, "velocity": 82}], 13, 256.0),
        ("D3A GlassPadB", 8, 18, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 98}], 14, 256.0),

        # --- DROP 3B SOARING (Bars 73..80, beat 288) ---
        ("D3B Kick", 8, 1, build_dnb_kick(8, rolling=True), 1, 288.0),
        ("D3B Snare", 8, 2, build_dnb_snare(8), 2, 288.0),
        ("D3B Tech Snare", 8, 3, build_dnb_snare(8), 3, 288.0),
        ("D3B Hats", 8, 4, build_hats(8, dense=True), 4, 288.0),
        ("D3B Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 102}], 5, 288.0),
        ("D3B Perc", 8, 6, build_perc(8), 6, 288.0),
        ("D3B Deep808", 8, 7, build_808(8, octave_bounce=True), 7, 288.0),
        ("D3B Chords", 8, 9, build_chords(8, vel=78), 9, 288.0),
        ("D3B Piano Lead", 8, 10, build_piano_lead(8, vel_scale=1.02), 10, 288.0),
        ("D3B Headroom99 Sparkle", 8, 11, build_sparkle_99(8, vel=78), 11, 288.0),
        ("D3B Vocal Counter", 8, 13, build_vocal_counter(8, vel=86, soaring=True), 12, 288.0),
        ("D3B Chant C4", 8, 16, [{"pitch": 60, "start": 31.5, "length": 0.4, "velocity": 86}], 13, 288.0),
        ("D3B GlassPadB", 8, 18, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 100}], 14, 288.0),
        ("D3B BFNK Cloud", 8, 19, [
            {"pitch": 65, "start": 7.0, "length": 2.2, "velocity": 76},
            {"pitch": 65, "start": 23.0, "length": 2.2, "velocity": 76},
        ], 15, 288.0),

        # --- BREAK 3 SUSPENSION (Bars 81..96, beat 320..384) ---
        ("B3A Chords", 8, 9, build_chords(8, vel=64), 9, 320.0),
        ("B3A Piano Nocturne", 8, 10, build_piano_lead(8, vel_scale=0.86, sparse_mode=True, nocturne=True), 10, 320.0),
        ("B3A Headroom99 Echo", 8, 11, build_sparkle_99(8, vel=72), 11, 320.0),
        ("B3A Vocal3 Phrase", 6, 12, [{"pitch": 67, "start": 1.0, "length": 15.0, "velocity": 88}], 12, 324.0),
        ("B3A Glass Intro", 8, 21, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 98}], 16, 320.0),
        ("B3A BFNK Cloud", 8, 19, [{"pitch": 65, "start": 15.0, "length": 2.2, "velocity": 72}], 15, 320.0),

        ("B3B Snare Build", 8, 2, build_dnb_snare(8, fill_build=True), 2, 352.0),
        ("B3B Hats", 8, 4, build_hats(8, dense=False, vel_scale=0.72), 4, 352.0),
        ("B3B SausageSub", 8, 8, build_sausage(8, 2, 7), 8, 352.0),
        ("B3B Chords", 8, 9, build_chords(8, vel=58), 9, 352.0),
        ("B3B Piano Nocturne", 8, 10, build_piano_lead(8, vel_scale=0.86, nocturne=True), 10, 352.0),
        ("B3B Vocal3 Swell", 1, 14, [{"pitch": 67, "start": 0.0, "length": 3.85, "velocity": 92}], 12, 380.0),
        ("B3B Chant Cue", 1, 16, [{"pitch": 60, "start": 3.5, "length": 0.4, "velocity": 86}], 13, 380.0),
        ("B3B GlassPadA", 8, 17, [{"pitch": 65, "start": 0.0, "length": 30.8, "velocity": 70}], 14, 352.0),
        ("B3B Riser", 1, 20, [{"pitch": 65, "start": 0.0, "length": 3.85, "velocity": 94}], 15, 380.0),

        # --- DROP 4A CLIMAX (Bars 97..104, beat 384) ---
        ("D4A Kick", 8, 1, build_dnb_kick(8, rolling=True), 1, 384.0),
        ("D4A Snare", 8, 2, build_dnb_snare(8), 2, 384.0),
        ("D4A Tech Snare", 8, 3, build_dnb_snare(8), 3, 384.0),
        ("D4A Hats", 8, 4, build_hats(8, dense=True), 4, 384.0),
        ("D4A Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 102}], 5, 384.0),
        ("D4A Perc", 8, 6, build_perc(8), 6, 384.0),
        ("D4A Deep808", 8, 7, build_808(8, octave_bounce=True), 7, 384.0),
        ("D4A Chords", 8, 9, build_chords(8, vel=78), 9, 384.0),
        ("D4A Piano Lead", 8, 10, build_piano_lead(8, vel_scale=1.02), 10, 384.0),
        ("D4A Chant Ab3", 8, 15, [{"pitch": 68, "start": 15.5, "length": 0.4, "velocity": 84}], 13, 384.0),
        ("D4A GlassPadB", 8, 18, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 100}], 14, 384.0),

        # --- DROP 4B EUPHORIC PEAK (Bars 105..112, beat 416) ---
        ("D4B Kick", 8, 1, build_dnb_kick(8, rolling=True), 1, 416.0),
        ("D4B Snare", 8, 2, build_dnb_snare(8), 2, 416.0),
        ("D4B Tech Snare", 8, 3, build_dnb_snare(8), 3, 416.0),
        ("D4B Hats", 8, 4, build_hats(8, dense=True), 4, 416.0),
        ("D4B Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 104}], 5, 416.0),
        ("D4B Perc", 8, 6, build_perc(8), 6, 416.0),
        ("D4B Deep808", 8, 7, build_808(8, octave_bounce=True), 7, 416.0),
        ("D4B Chords", 8, 9, build_chords(8, vel=80), 9, 416.0),
        ("D4B Piano Lead", 8, 10, build_piano_lead(8, vel_scale=1.05), 10, 416.0),
        ("D4B Headroom99 Peak", 8, 11, build_sparkle_99(8, vel=82), 11, 416.0),
        ("D4B Vocal Counter", 8, 13, build_vocal_counter(8, vel=88, soaring=True), 12, 416.0),
        ("D4B Chant Ab3", 8, 15, [{"pitch": 68, "start": 15.5, "length": 0.4, "velocity": 86}], 13, 416.0),
        ("D4B Chant C4", 8, 16, [{"pitch": 60, "start": 31.5, "length": 0.4, "velocity": 88}], 13, 416.0),
        ("D4B GlassPadB", 8, 18, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 102}], 14, 416.0),
        ("D4B BFNK Cloud", 8, 19, [
            {"pitch": 65, "start": 7.0, "length": 2.2, "velocity": 80},
            {"pitch": 65, "start": 23.0, "length": 2.2, "velocity": 80},
        ], 15, 416.0),

        # --- DROP 4C RELEASE (Bars 113..120, beat 448) ---
        ("D4C Kick", 8, 1, build_dnb_kick(8), 1, 448.0),
        ("D4C Snare", 8, 2, build_dnb_snare(8), 2, 448.0),
        ("D4C Tech Snare", 8, 3, build_dnb_snare(8), 3, 448.0),
        ("D4C Hats", 8, 4, build_hats(8, dense=False, vel_scale=0.85), 4, 448.0),
        ("D4C Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 86}], 5, 448.0),
        ("D4C Deep808", 8, 7, build_808(8, octave_bounce=False), 7, 448.0),
        ("D4C Chords", 8, 9, build_chords(8, vel=68), 9, 448.0),
        ("D4C Piano Lead", 8, 10, build_piano_lead(8, vel_scale=0.88), 10, 448.0),
        ("D4C GlassPadA", 8, 17, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 84}], 14, 448.0),

        # --- OUTRO (Bars 121..129, beat 480..516 = 180.00 s) ---
        ("Outro Hats", 2, 4, build_hats(2, dense=False, cut_last_beat=True, vel_scale=0.52), 4, 480.0),
        ("Outro SausageSub", 3, 8, build_sausage(3, 0, 2), 8, 480.0),
        ("Outro Glass Intro", 8, 21, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 88}], 16, 480.0),
        ("Outro Chords", 9, 9, outro_chords, 9, 480.0),
        ("Outro Piano", 9, 10, outro_piano, 10, 480.0),
        ("Outro Vocal Sigh", 9, 13, outro_vocal, 12, 480.0),
    ]

    # Export full orchestrated MIDI and pass strict gate BEFORE building Gridline project
    ch_names_by_idx = {
        1: "DnB Kick", 2: "DnB Snare", 3: "Tech Snare Layer", 4: "Hi-Hat Crisp",
        5: "Tech Top Break FX", 6: "Perc Ghost", 7: "Deep 808 Sub", 8: "Sausage Sub Swell",
        9: "Chords Synth", 10: "Headroom Piano Lead", 11: "Headroom 99 Sparkle",
        12: "Vocal 3 Phrase FX", 13: "Counter Harmony (Voice3)", 14: "Vocal 3 Swell FX",
        15: "Chant Hi Ab3 FX", 16: "Chant Hi C4 FX", 17: "Glass Waves Pad FX",
        18: "Glass Waves Pad B FX", 19: "BFNK Fm Cloud FX", 20: "Glass Riser FX",
        21: "Glass Waves Intro FX",
    }
    midi_by_ch: dict[int, list[dict]] = {}
    for _, _, ch_idx, notes, _, start_b in pat_specs:
        # Exclude FX loop/percussion triggers from pitched harmony audit, keep all harmonic/bass/piano/vocal tracks
        if ch_idx in (7, 8, 9, 10, 11, 13):
            lst = midi_by_ch.setdefault(ch_idx, [])
            for n in notes:
                lst.append({
                    "pitch": int(n["pitch"]),
                    "start": float(n["start"]) + start_b,
                    "length": float(n["length"]),
                    "velocity": int(n["velocity"]),
                })

    mid = mido.MidiFile(ticks_per_beat=480)
    tempo_us = mido.bpm2tempo(172)
    first_tr = True
    for ch_idx, evs in sorted(midi_by_ch.items()):
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=ch_names_by_idx[ch_idx], time=0))
        if first_tr:
            tr.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
            first_tr = False
        raw = []
        for n in evs:
            t0 = int(round(n["start"] * 480))
            t1 = int(round((n["start"] + n["length"]) * 480))
            raw.append((t0, "note_on", n["pitch"], n["velocity"]))
            raw.append((t1, "note_off", n["pitch"], 0))
        raw.sort(key=lambda x: (x[0], 0 if x[1] == "note_off" else 1))
        last = 0
        for t, typ, p, v in raw:
            tr.append(mido.Message(typ, note=p, velocity=v, time=max(0, t - last)))
            last = t
        tr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr)

    full_mid_path = f"{PROJ_DIR}/glass_horizon_3min.mid"
    mid.save(full_mid_path)
    gate = verify_midi_gate(full_mid_path, "F minor", max_pitch=84)
    print("FULL 129-BAR ORCHESTRATED MIDI GATE PASSED:", json.dumps(gate["musicality_metrics"], indent=2))

    # ------------------------------------------------------------------
    # 3. Build Gridline Project cleanly from TEMPLATE_EMPTY_MIXER
    # ------------------------------------------------------------------
    proj_path = f"{PROJ_DIR}/projet.gridline"
    shutil.copyfile(TEMPLATE_EMPTY_MIXER, proj_path)
    mcp = GridlineMCP(proj_path, RENDERS)
    mcp.call("set_song", {
        "title": "Glass Horizon",
        "genre": "Atmospheric Liquid Drum and Bass",
        "tempo": 172,
        "numerator": 4,
        "denominator": 4,
        "swing": 0.02,
    })

    BUS_DRUMS, BUS_808, BUS_MUSIC, FX_ROOM = 18, 19, 20, 21
    mcp.batch([
        {"tool": "add_mixer_track", "arguments": {"name": nm}}
        for nm in ["Glass Intro", "BUS Drums", "BUS 808", "BUS Music", "FX Room"]
    ])

    # Channels 1..21 (mixer tracks 1..17)
    ch_defs = [
        ("DnB Kick", "gl.sampler", 1, f"{SAMPLES}/dnb_kick_punch.wav", 0.80, 0.0),
        ("DnB Snare", "gl.sampler", 2, f"{SAMPLES}/dnb_snare_crisp.wav", 0.80, 0.0),
        ("Tech Snare Layer", "gl.sampler", 3, f"{SAMPLES}/tech_dnb_snare_clean.wav", 0.72, -0.05),
        ("Hi-Hat Crisp", "gl.hat", 4, None, 0.78, 0.14),
        ("Tech Top Break FX", "gl.sampler", 5, f"{SAMPLES}/tech_dnb_top_8bar_172_clean.wav", 0.68, 0.0),
        ("Perc Ghost", "gl.sampler", 6, f"{SAMPLES}/dnb_perc_ghost.wav", 0.62, -0.22),
        ("Deep 808 Sub", "gl.sampler", 7, f"{SAMPLES}/deep_808_clean.wav", 0.72, 0.0),
        ("Sausage Sub Swell", "gl.sampler", 8, f"{SAMPLES}/sausage_bass_clean.wav", 0.58, 0.0),
        ("Chords Synth", "gl.synth", 9, None, 0.60, 0.0),
        ("Headroom Piano Lead", "gl.multisampler", 10, SFZ_PIANO, 0.85, 0.04),
        ("Headroom 99 Sparkle", "gl.sampler", 11, f"{SAMPLES}/headroom_99_eb7_clean.wav", 0.72, 0.16),
        ("Vocal 3 Phrase FX", "gl.sampler", 12, f"{SAMPLES}/vocal3_phrase_g4.wav", 0.78, -0.08),
        ("Counter Harmony (Voice3)", "gl.sampler", 12, f"{SAMPLES}/vocal3_sustain_g4.wav", 0.76, -0.08),
        ("Vocal 3 Swell FX", "gl.sampler", 12, f"{SAMPLES}/vocal3_swell_rev.wav", 0.66, 0.0),
        ("Chant Hi Ab3 FX", "gl.sampler", 13, f"{SAMPLES}/chant_hi_ab3.wav", 0.64, -0.18),
        ("Chant Hi C4 FX", "gl.sampler", 13, f"{SAMPLES}/chant_hi_c4.wav", 0.64, 0.18),
        ("Glass Waves Pad FX", "gl.sampler", 14, f"{SAMPLES}/glass_waves_pad_a_8bar.wav", 0.66, 0.0),
        ("Glass Waves Pad B FX", "gl.sampler", 14, f"{SAMPLES}/glass_waves_pad_b_8bar.wav", 0.66, 0.0),
        ("BFNK Fm Cloud FX", "gl.sampler", 15, f"{SAMPLES}/bfnk_pad_fm_shot.wav", 0.50, -0.20),
        ("Glass Riser FX", "gl.sampler", 16, f"{SAMPLES}/glass_waves_riser_1bar.wav", 0.56, 0.0),
        ("Glass Waves Intro FX", "gl.sampler", 17, f"{SAMPLES}/glass_waves_intro_8bar.wav", 0.66, 0.0),
    ]

    mcp.batch([
        {"tool": "add_channel", "arguments": {"plugin": plug, "name": name, "mixer": mx, "volume": vol, "pan": pan}}
        for name, plug, mx, _, vol, pan in ch_defs
    ])

    for idx, (name, plug, mx, spath, _, _) in enumerate(ch_defs, start=1):
        if spath:
            mcp.call("patch", {"path": f"channel/{idx}", "set": {"plugin": {"data": spath}}})

    param_changes = [
        # Drums
        ("channel/1/Declick", 1), ("channel/1/Remove DC", 1),
        ("channel/2/Declick", 1), ("channel/2/Remove DC", 1),
        ("channel/3/Declick", 1), ("channel/3/Remove DC", 1),
        ("channel/4/Decay", 0.055), ("channel/4/Tone", 10500), ("channel/4/Metal", 0.28), ("channel/4/Width", 0.65), ("channel/4/Attack", 0.004),
        ("channel/5/Declick", 1), ("channel/5/Root note", 65), ("channel/5/One shot", 0), ("channel/5/Release", 0.05),
        ("channel/6/Declick", 1),
        # Deep 808 Sub (mono, tuned F1, Glide 35ms)
        ("channel/7/Root note", 24), ("channel/7/Fine tune", -14),
        ("channel/7/Voices", 1), ("channel/7/One shot", 0),
        ("channel/7/Remove DC", 1), ("channel/7/Declick", 1),
        ("channel/7/Release", 0.045), ("channel/7/Glide", 0.035),
        # Sausage Sub Swell (mono, tuned A1 +40ct)
        ("channel/8/Root note", 33), ("channel/8/Fine tune", 40),
        ("channel/8/Voices", 1), ("channel/8/One shot", 0),
        ("channel/8/Remove DC", 1), ("channel/8/Declick", 1),
        ("channel/8/Attack", 0.035), ("channel/8/Release", 0.060), ("channel/8/Glide", 0.050),
        # Chords Synth (Warm Dreamy Liquid Pad/Rhodes)
        ("channel/9/Osc 1 shape", 1), ("channel/9/Osc 1 level", 0.78),
        ("channel/9/Osc 2 shape", 0), ("channel/9/Osc 2 level", 0.55), ("channel/9/Osc 2 coarse", 0), ("channel/9/Osc 2 fine", 5), ("channel/9/Osc 2 pan", -0.20),
        ("channel/9/Osc 3 shape", 0), ("channel/9/Osc 3 level", 0.32), ("channel/9/Osc 3 coarse", 12), ("channel/9/Osc 3 fine", -5), ("channel/9/Osc 3 pan", 0.22),
        ("channel/9/Attack", 0.045), ("channel/9/Decay", 0.85), ("channel/9/Sustain", 0.52), ("channel/9/Release", 0.42),
        ("channel/9/Cutoff", 2400), ("channel/9/Resonance", 0.08),
        ("channel/9/Filter env", 0.16), ("channel/9/Filter decay", 0.55),
        ("channel/9/Voices", 12), ("channel/9/Velocity to cutoff", 0.25),
        # Headroom Piano Lead (gl.multisampler, warm liquid acoustic piano)
        ("channel/10/Polyphony", 16), ("channel/10/Velocity", 0.65), ("channel/10/Interpolation", 1),
        ("channel/10/Attack", 0.002), ("channel/10/Release", 0.45),
        ("channel/10/Filter", 1), ("channel/10/Cutoff", 6200), ("channel/10/Resonance", 0.05),
        # Headroom 99 Sparkle (Root=87 so MIDI 72..80 plays C6..Ab6 crystalline shimmer)
        ("channel/11/Root note", 87), ("channel/11/One shot", 0),
        ("channel/11/Attack", 0.004), ("channel/11/Release", 0.48), ("channel/11/Remove DC", 1), ("channel/11/Declick", 1),
        # Vocal 3 Phrase / Counter / Swell (Root=67 G4)
        ("channel/12/Root note", 67), ("channel/12/One shot", 0), ("channel/12/Attack", 0.04), ("channel/12/Release", 0.38), ("channel/12/Remove DC", 1),
        ("channel/13/Root note", 67), ("channel/13/Voices", 1), ("channel/13/One shot", 0), ("channel/13/Attack", 0.06), ("channel/13/Release", 0.35), ("channel/13/Glide", 0.055), ("channel/13/Remove DC", 1),
        ("channel/14/Root note", 67), ("channel/14/One shot", 0), ("channel/14/Attack", 0.02), ("channel/14/Release", 0.10), ("channel/14/Remove DC", 1),
        # Chant Hi Ab3 / C4
        ("channel/15/Root note", 68), ("channel/15/One shot", 1), ("channel/15/Declick", 1), ("channel/15/Remove DC", 1),
        ("channel/16/Root note", 60), ("channel/16/One shot", 1), ("channel/16/Declick", 1), ("channel/16/Remove DC", 1),
        # Pads & Risers
        ("channel/17/Root note", 65), ("channel/17/One shot", 0), ("channel/17/Attack", 0.03), ("channel/17/Release", 0.18), ("channel/17/Remove DC", 1),
        ("channel/18/Root note", 65), ("channel/18/One shot", 0), ("channel/18/Attack", 0.03), ("channel/18/Release", 0.18), ("channel/18/Remove DC", 1),
        ("channel/19/Root note", 65), ("channel/19/One shot", 0), ("channel/19/Attack", 0.04), ("channel/19/Release", 0.25), ("channel/19/Remove DC", 1),
        ("channel/20/Root note", 65), ("channel/20/One shot", 0), ("channel/20/Attack", 0.01), ("channel/20/Release", 0.08),
        ("channel/21/Root note", 65), ("channel/21/One shot", 0), ("channel/21/Attack", 0.04), ("channel/21/Release", 0.22), ("channel/21/Remove DC", 1),
    ]
    mcp.set_params(param_changes)

    # ------------------------------------------------------------------
    # 4. Mixer Routing, EQs, Sidechain & Standard Buses
    # ------------------------------------------------------------------
    setup_standard_buses(
        mcp,
        bpm=172.0,
        bus_drums=BUS_DRUMS,
        bus_808=BUS_808,
        bus_music=BUS_MUSIC,
        fx_room=FX_ROOM,
        limiter_ceiling_db=-3.05,
        limiter_gain_db=8.9,
        fx_room_vol=0.40,
    )

    # Respecting user's fader preferences in projet.gridline: warm Deep 808, crisp Hats/Tech Snare, soft DnB Snare, rich Pad
    mixer_defs = [
        (1, "DnB Kick", BUS_DRUMS, 0.78, 0.0),
        (2, "DnB Snare", BUS_DRUMS, 0.62, 0.0),
        (3, "Tech Snare Layer", BUS_DRUMS, 0.86, -0.05),
        (4, "Hi-Hat Crisp", BUS_DRUMS, 1.06, 0.14),
        (5, "Tech Top Break", BUS_DRUMS, 0.74, 0.0),
        (6, "Perc Ghost", BUS_DRUMS, 0.62, -0.22),
        (7, "Deep 808 Sub", BUS_808, 0.76, 0.0),
        (8, "Sausage Sub", BUS_808, 0.60, 0.0),
        (9, "Chords Synth", BUS_MUSIC, 0.74, 0.0),
        (10, "Headroom Piano", BUS_MUSIC, 0.98, 0.04),
        (11, "Headroom99 Sparkle", BUS_MUSIC, 0.78, 0.16),
        (12, "Vocal 3", BUS_MUSIC, 0.64, -0.08),
        (13, "Chant Hi", BUS_MUSIC, 0.58, 0.0),
        (14, "Glass Waves Pad", BUS_MUSIC, 0.82, 0.0),
        (15, "BFNK Fm Cloud", BUS_MUSIC, 0.56, -0.20),
        (16, "Glass Riser", BUS_MUSIC, 0.54, 0.0),
        (17, "Glass Intro", BUS_MUSIC, 0.90, 0.0),
        (BUS_DRUMS, "BUS Drums", 0, 0.94, 0.0),
        (BUS_808, "BUS 808", 0, 0.72, 0.0),
        (BUS_MUSIC, "BUS Music", 0, 0.90, 0.0),
    ]
    mcp.batch([
        {"tool": "edit_mixer_track", "arguments": {"mixer": m, "name": nm, "routeTo": rt, "volume": v, "pan": p}}
        for m, nm, rt, v, p in mixer_defs
    ])

    # Narrow stereo separation slightly on Headroom Piano & Glass Waves Pad for tight low-mid phase
    mcp.set_params([("mixer/10/stereo", -0.22), ("mixer/14/stereo", -0.18)])

    mcp.batch([
        {"tool": "add_plugin", "arguments": {"mixer": 1, "plugin": "gl.eq"}},   # Kick punch vs 808
        {"tool": "add_plugin", "arguments": {"mixer": 2, "plugin": "gl.eq"}},   # Snare body/air
        {"tool": "add_plugin", "arguments": {"mixer": 7, "plugin": "gl.dist"}}, # Subtle 808 warmth
        {"tool": "add_plugin", "arguments": {"mixer": 9, "plugin": "gl.eq"}},   # Chords carve
        {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.eq"}},  # Piano HPF 165Hz & warm presence
        {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.eq"}},  # Vocal 3 HPF 230Hz & air polish
        {"tool": "add_plugin", "arguments": {"mixer": 14, "plugin": "gl.eq"}},  # Glass Waves carve
    ])
    mcp.set_params([
        ("mixer/0/slot/1/Band 7 freq", 14500),
        ("mixer/0/slot/2/Threshold", -9.5), ("mixer/0/slot/2/Makeup", 1.7),
        ("mixer/0/slot/3/Attack (look-ahead)", 0.002),
        # Buses
        (f"mixer/{BUS_DRUMS}/slot/1/Band 6 type", 7), (f"mixer/{BUS_DRUMS}/slot/1/Band 6 freq", 8200), (f"mixer/{BUS_DRUMS}/slot/1/Band 6 gain", 5.0),
        (f"mixer/{BUS_DRUMS}/slot/1/Band 7 type", 3), (f"mixer/{BUS_DRUMS}/slot/1/Band 7 freq", 14200),
        (f"mixer/{BUS_808}/slot/3/Band 3 type", 6), (f"mixer/{BUS_808}/slot/3/Band 3 freq", 64), (f"mixer/{BUS_808}/slot/3/Band 3 gain", -9.5), (f"mixer/{BUS_808}/slot/3/Band 3 Q (legacy)", 0.85),
        (f"mixer/{BUS_MUSIC}/slot/1/Band 3 type", 6), (f"mixer/{BUS_MUSIC}/slot/1/Band 3 freq", 480), (f"mixer/{BUS_MUSIC}/slot/1/Band 3 gain", -3.4), (f"mixer/{BUS_MUSIC}/slot/1/Band 3 Q (legacy)", 1.5),
        (f"mixer/{BUS_MUSIC}/slot/1/Band 6 type", 7), (f"mixer/{BUS_MUSIC}/slot/1/Band 6 freq", 8500), (f"mixer/{BUS_MUSIC}/slot/1/Band 6 gain", 4.0),
        (f"mixer/{BUS_MUSIC}/slot/1/Band 7 freq", 14000),
        (f"mixer/{BUS_MUSIC}/slot/3/Threshold", -13.5), (f"mixer/{BUS_MUSIC}/slot/3/Makeup", 1.2),
        (f"mixer/{FX_ROOM}/slot/1/Decay time", 0.348), (f"mixer/{FX_ROOM}/slot/1/High cut", 10500), (f"mixer/{FX_ROOM}/slot/1/Width", 0.90),
        # Kick EQ
        ("mixer/1/slot/1/Band 1 type", 1), ("mixer/1/slot/1/Band 1 freq", 46),
        ("mixer/1/slot/1/Band 2 type", 6), ("mixer/1/slot/1/Band 2 freq", 63), ("mixer/1/slot/1/Band 2 gain", -7.4), ("mixer/1/slot/1/Band 2 Q (legacy)", 1.1),
        ("mixer/1/slot/1/Band 3 type", 6), ("mixer/1/slot/1/Band 3 freq", 115), ("mixer/1/slot/1/Band 3 gain", 2.0), ("mixer/1/slot/1/Band 3 Q (legacy)", 1.4),
        # Snare EQ
        ("mixer/2/slot/1/Band 1 type", 1), ("mixer/2/slot/1/Band 1 freq", 150),
        ("mixer/2/slot/1/Band 3 type", 6), ("mixer/2/slot/1/Band 3 freq", 220), ("mixer/2/slot/1/Band 3 gain", 1.5),
        ("mixer/2/slot/1/Band 5 type", 6), ("mixer/2/slot/1/Band 5 freq", 3500), ("mixer/2/slot/1/Band 5 gain", -1.5),
        ("mixer/2/slot/1/Band 7 type", 7), ("mixer/2/slot/1/Band 7 freq", 8500), ("mixer/2/slot/1/Band 7 gain", 1.5),
        # Deep 808 warm 2nd/3rd harmonics
        ("mixer/7/slot/1/Mode", 0), ("mixer/7/slot/1/Drive", 4.2), ("mixer/7/slot/1/Mix", 0.30),
        ("mixer/7/slot/1/Oversampling", 2), ("mixer/7/slot/1/Remove DC", 1),
        # Chords Synth carve
        ("mixer/9/slot/1/Band 1 type", 1), ("mixer/9/slot/1/Band 1 freq", 165),
        ("mixer/9/slot/1/Band 3 type", 6), ("mixer/9/slot/1/Band 3 freq", 280), ("mixer/9/slot/1/Band 3 gain", -2.6),
        ("mixer/9/slot/1/Band 5 type", 6), ("mixer/9/slot/1/Band 5 freq", 1600), ("mixer/9/slot/1/Band 5 gain", -2.5),
        # Headroom Piano Lead: HPF 165 Hz, tidy 320 Hz (-2.0 dB), gentle warmth 2.2 kHz (+1.6 dB)
        ("mixer/10/slot/1/Band 1 type", 1), ("mixer/10/slot/1/Band 1 freq", 165),
        ("mixer/10/slot/1/Band 3 type", 6), ("mixer/10/slot/1/Band 3 freq", 320), ("mixer/10/slot/1/Band 3 gain", -2.0),
        ("mixer/10/slot/1/Band 5 type", 6), ("mixer/10/slot/1/Band 5 freq", 2200), ("mixer/10/slot/1/Band 5 gain", 1.6),
        # Vocal 3: HPF 230 Hz, tame 784 Hz harmonic (-7.2 dB), airy presence 2.2 kHz (+1.5 dB) & 7.5 kHz (+1.5 dB)
        ("mixer/12/slot/1/Band 1 type", 1), ("mixer/12/slot/1/Band 1 freq", 230),
        ("mixer/12/slot/1/Band 3 type", 6), ("mixer/12/slot/1/Band 3 freq", 784), ("mixer/12/slot/1/Band 3 gain", -7.2), ("mixer/12/slot/1/Band 3 Q (legacy)", 1.6),
        ("mixer/12/slot/1/Band 5 type", 6), ("mixer/12/slot/1/Band 5 freq", 2200), ("mixer/12/slot/1/Band 5 gain", 1.5),
        ("mixer/12/slot/1/Band 7 type", 7), ("mixer/12/slot/1/Band 7 freq", 7500), ("mixer/12/slot/1/Band 7 gain", 1.5),
        # Glass Waves Pad carve
        ("mixer/14/slot/1/Band 1 type", 1), ("mixer/14/slot/1/Band 1 freq", 190),
        ("mixer/14/slot/1/Band 3 type", 6), ("mixer/14/slot/1/Band 3 freq", 260), ("mixer/14/slot/1/Band 3 gain", -3.0),
        ("mixer/14/slot/1/Band 5 type", 6), ("mixer/14/slot/1/Band 5 freq", 1600), ("mixer/14/slot/1/Band 5 gain", -2.6),
    ])

    sends = [
        {"tool": "set_send", "arguments": {"mixer": 1, "to": BUS_808, "level": 1.0, "preFader": True, "sidechain": True}},
        {"tool": "set_send", "arguments": {"mixer": 1, "to": BUS_MUSIC, "level": 1.0, "preFader": True, "sidechain": True}},
        {"tool": "set_send", "arguments": {"mixer": 2, "to": FX_ROOM, "level": 0.08}},
        {"tool": "set_send", "arguments": {"mixer": 3, "to": FX_ROOM, "level": 0.12}},
        {"tool": "set_send", "arguments": {"mixer": 6, "to": FX_ROOM, "level": 0.14}},
        {"tool": "set_send", "arguments": {"mixer": 9, "to": FX_ROOM, "level": 0.22}},
        {"tool": "set_send", "arguments": {"mixer": 10, "to": FX_ROOM, "level": 0.24}},
        {"tool": "set_send", "arguments": {"mixer": 11, "to": FX_ROOM, "level": 0.28}},
        {"tool": "set_send", "arguments": {"mixer": 12, "to": FX_ROOM, "level": 0.30}},
        {"tool": "set_send", "arguments": {"mixer": 13, "to": FX_ROOM, "level": 0.24}},
        {"tool": "set_send", "arguments": {"mixer": 14, "to": FX_ROOM, "level": 0.18}},
        {"tool": "set_send", "arguments": {"mixer": 15, "to": FX_ROOM, "level": 0.26}},
        {"tool": "set_send", "arguments": {"mixer": 16, "to": FX_ROOM, "level": 0.20}},
        {"tool": "set_send", "arguments": {"mixer": 17, "to": FX_ROOM, "level": 0.20}},
    ]
    mcp.batch(sends)

    # ------------------------------------------------------------------
    # 5. Create Per-Instrument Patterns & Playlist Clips (129 bars)
    # ------------------------------------------------------------------
    for p_idx, (pname, bars, ch_idx, notes, pl_track, start_b) in enumerate(pat_specs, start=1):
        if p_idx == 1:
            mcp.call("edit_pattern", {"pattern": 1, "name": pname, "bars": bars})
        else:
            mcp.call("add_pattern", {"name": pname, "bars": bars})
        mcp.call("set_notes", {"pattern": p_idx, "channel": ch_idx, "notes": notes, "mode": "replace"})
        mcp.call("add_clip", {"pattern": p_idx, "track": pl_track, "start": start_b})

    # Name playlist tracks cleanly
    track_names = [
        (1, "DnB Kick"), (2, "DnB Snare"), (3, "Tech Snare Layer"), (4, "Hi-Hat Crisp"),
        (5, "Tech Top Break"), (6, "Perc Ghost"), (7, "Deep 808 Sub"), (8, "Sausage Sub Swell"),
        (9, "Chords Synth"), (10, "Headroom Piano Lead"), (11, "Headroom 99 Sparkle"),
        (12, "Vocal 3"), (13, "Chant Hi"), (14, "Glass Waves Pad"),
        (15, "BFNK Cloud / Risers"), (16, "Glass Waves Intro"), (17, "Automation FX Room"),
    ]
    mcp.batch([{"tool": "patch", "arguments": {"path": f"track/{t}", "set": {"name": nm}}} for t, nm in track_names])

    # FX Room volume automation: clean tail cuts at 8-bar pivot beats + Outro fade
    nom = round(0.42 / 1.25, 4)
    auto_pts = [{"position": 0.0, "value": nom}]
    for piv in [31.0, 63.0, 95.0, 127.0, 159.0, 191.0, 223.0, 255.0, 287.0, 319.0, 351.0, 383.0, 415.0, 447.0, 479.0]:
        auto_pts.extend([
            {"position": piv - 0.1, "value": nom},
            {"position": piv, "value": 0.0, "shape": "hold"},
            {"position": piv + 0.95, "value": 0.0},
            {"position": piv + 1.0, "value": nom},
        ])
    auto_pts.extend([
        {"position": 512.0, "value": nom},
        {"position": 516.0, "value": 0.0},
    ])
    mcp.call("add_automation", {
        "mixer": FX_ROOM,
        "param": "volume",
        "name": "FX Room Pivot Cuts",
        "points": auto_pts,
        "track": 17,
        "start": 0.0,
        "length": 516.0,
    })

    # Section Markers
    for mk_name, mk_beat in [
        ("Intro", 0.0), ("Drop 1A", 32.0), ("Drop 1B", 64.0),
        ("Break 1 Planant", 96.0), ("Drop 2A", 128.0), ("Drop 2B Sparkle", 160.0),
        ("Break 2 Nocturne", 192.0), ("Break 2 Build", 224.0),
        ("Drop 3A Rolling", 256.0), ("Drop 3B Soaring", 288.0),
        ("Break 3 Suspension", 320.0), ("Break 3 Build", 352.0),
        ("Drop 4A Climax", 384.0), ("Drop 4B Peak", 416.0), ("Drop 4C Release", 448.0),
        ("Outro", 480.0),
    ]:
        mcp.call("add_marker", {"name": mk_name, "position": mk_beat})

    mcp.call("save_project", {"path": "projet.gridline"})
    mcp.close()
    print("Saved 3-minute projects/active9/projet.gridline successfully!")


if __name__ == "__main__":
    main()
