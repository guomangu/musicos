#!/usr/bin/env python3
"""
Build script for "Glass Horizon" (projects/active9/projet.gridline)
43 bars @ 172 BPM (60.00 s) Atmospheric Liquid Drum and Bass in F minor,
using exclusively the sounds & plugins present in testfinal.gridline and their
derived/tuned/M-S-stabilized WAVs in projects/active9/samples/.
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

rng = random.Random(17209)


def humanize(notes: list[dict], vel_jit: int = 8, time_jit: float = 0.0) -> list[dict]:
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
    chords = bp["harmonic_core"]["chord_progression"]  # 4-bar cycle: Fm7, Dbmaj9, Abmaj7, Eb6

    # ------------------------------------------------------------------
    # 1. Generate exact musical note tables for all 43 bars
    # ------------------------------------------------------------------
    # Sections (1-based bars):
    #   Intro:   1..8   (8 bars, start beat 0)
    #   Drop1:   9..16  (8 bars, start beat 32)
    #   Drop1B: 17..24  (8 bars, start beat 64)
    #   Break:  25..32  (8 bars, start beat 96)
    #   Drop2:  33..40  (8 bars, start beat 128)
    #   Outro:  41..43  (3 bars, start beat 160..172)

    lead_4bar = [
        # Bar 0 (Fm7): C5, Eb5, F5, Eb5 -> breath beat 3.5..4.0
        [(72, 0.0, 0.85, 92), (75, 1.0, 0.85, 88), (77, 2.0, 0.70, 98), (75, 2.75, 0.65, 85)],
        # Bar 1 (Dbmaj9): Db5, C5, Ab4 -> breath beat 3.0..4.0
        [(73, 0.0, 0.85, 90), (72, 1.0, 0.85, 86), (68, 2.0, 0.90, 88)],
        # Bar 2 (Abmaj7): C5, Eb5, Ab5 (Apex!), G5 -> breath beat 3.75..4.0
        [(72, 0.0, 0.85, 90), (75, 1.0, 0.85, 94), (80, 2.0, 0.85, 104), (79, 3.0, 0.65, 90)],
        # Bar 3 (Eb6): G5, F5, Eb5 -> breath beat 3.0..4.0
        [(79, 0.0, 0.85, 92), (77, 1.0, 0.75, 86), (75, 2.0, 0.95, 90)],
    ]

    def build_chords(bars: int, vel: int = 72, octave_shift: int = 0, cut_last_beat: bool = False) -> list[dict]:
        notes = []
        for b in range(bars):
            ch = chords[b % 4]
            dur = 2.85 if (cut_last_beat and b == bars - 1) else 3.60
            for vp in ch["voicing_midi"]:
                notes.append({"pitch": vp + octave_shift, "start": b * 4.0, "length": dur, "velocity": vel})
        return humanize(notes, vel_jit=5, time_jit=0.008)

    def build_lead(bars: int, vel_scale: float = 1.0, sparse_mode: bool = False, cut_last_beat: bool = False) -> list[dict]:
        notes = []
        for b in range(bars):
            if sparse_mode and (b % 2 == 1):
                # Keep only bar-downbeat anchor on odd bars for spaciousness
                p, sb, db, v = lead_4bar[b % 4][0]
                notes.append({"pitch": p, "start": b * 4.0 + sb, "length": 1.35, "velocity": int(v * vel_scale)})
                continue
            for p, sb, db, v in lead_4bar[b % 4]:
                if cut_last_beat and b == bars - 1 and sb >= 2.5:
                    continue
                notes.append({"pitch": p, "start": b * 4.0 + sb, "length": db, "velocity": int(v * vel_scale)})
        return humanize(notes, vel_jit=6, time_jit=0.010)

    def build_808(bars: int, octave_bounce: bool = False, cut_last_beat: bool = True) -> list[dict]:
        notes = []
        for b in range(bars):
            bm = chords[b % 4]["bass_midi"]
            # Pivot bar: cut at beat 3.0 to give clean pre-drop breath
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

    def build_sausage(bars: int) -> list[dict]:
        notes = []
        # Bars 0..3: pure floating drumless/subless space; bars 4..6: soft warm sub swell; bar 7: riser silence
        for b in range(4, bars - 1):
            bm = chords[b % 4]["bass_midi"]
            notes.append({"pitch": bm, "start": b * 4.0 + 0.0, "length": 3.70, "velocity": 64})
        return humanize(notes, vel_jit=3, time_jit=0.0)

    def build_dnb_kick(bars: int, cut_last_beat: bool = True) -> list[dict]:
        notes = []
        for b in range(bars):
            if cut_last_beat and b == bars - 1:
                notes.append({"pitch": 60, "start": b * 4.0 + 0.0, "length": 0.45, "velocity": 108})
                continue
            notes.append({"pitch": 60, "start": b * 4.0 + 0.0, "length": 0.45, "velocity": 110})
            notes.append({"pitch": 60, "start": b * 4.0 + 2.5, "length": 0.45, "velocity": 106})
            if b % 2 == 1:
                notes.append({"pitch": 60, "start": b * 4.0 + 3.75, "length": 0.22, "velocity": 76})
        return humanize(notes, vel_jit=4, time_jit=0.0)

    def build_dnb_snare(bars: int, cut_last_beat: bool = True, fill_build: bool = False) -> list[dict]:
        notes = []
        if fill_build:
            # Break snare build on bars 6..7 (0-indexed)
            for b in range(bars):
                if b == 6:
                    for sb, v in [(1.0, 68), (3.0, 74)]:
                        notes.append({"pitch": 60, "start": b * 4.0 + sb, "length": 0.35, "velocity": v})
                elif b == 7:
                    for i, sb in enumerate([0.0, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75]):
                        notes.append({"pitch": 60, "start": b * 4.0 + sb, "length": 0.22, "velocity": 62 + i * 6})
            return notes
        for b in range(bars):
            notes.append({"pitch": 60, "start": b * 4.0 + 1.0, "length": 0.40, "velocity": 108})
            if not (cut_last_beat and b == bars - 1):
                notes.append({"pitch": 60, "start": b * 4.0 + 3.0, "length": 0.40, "velocity": 110})
            else:
                notes.append({"pitch": 60, "start": b * 4.0 + 3.0, "length": 0.35, "velocity": 102})
            # Ghost snare at 1.75 every 2 bars
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

    def build_perc(bars: int, cut_last_beat: bool = True) -> list[dict]:
        notes = []
        for b in range(bars):
            if cut_last_beat and b == bars - 1:
                continue
            for sb, v in [(0.75, 74), (1.75, 68), (3.5, 78)]:
                notes.append({"pitch": 60, "start": b * 4.0 + sb, "length": 0.25, "velocity": v})
        return humanize(notes, vel_jit=6, time_jit=0.008)

    # ------------------------------------------------------------------
    # 2. Export & Verify full 43-bar orchestrated MIDI Gate
    # ------------------------------------------------------------------
    midi_tracks_data = {
        "Deep 808 Sub": [],
        "Sausage Sub Swell": [],
        "Chords Synth": [],
        "Lead Synth": [],
    }

    def add_offset(lst: list[dict], notes: list[dict], bar_offset: int) -> None:
        b_off = bar_offset * 4.0
        for n in notes:
            lst.append({"pitch": n["pitch"], "start": n["start"] + b_off, "length": n["length"], "velocity": n["velocity"]})

    # Intro (bars 0..8)
    intro_chords = build_chords(8, vel=56, cut_last_beat=True)
    intro_lead = build_lead(4, vel_scale=0.72, sparse_mode=True)  # bars 4..8
    add_offset(midi_tracks_data["Chords Synth"], intro_chords, 0)
    add_offset(midi_tracks_data["Lead Synth"], intro_lead, 4)

    # Drop 1 (bars 8..16)
    d1_808 = build_808(8, octave_bounce=False, cut_last_beat=True)
    d1_chords = build_chords(8, vel=72, cut_last_beat=True)
    d1_lead = build_lead(8, vel_scale=0.95, cut_last_beat=True)
    add_offset(midi_tracks_data["Deep 808 Sub"], d1_808, 8)
    add_offset(midi_tracks_data["Chords Synth"], d1_chords, 8)
    add_offset(midi_tracks_data["Lead Synth"], d1_lead, 8)

    # Drop 1B (bars 16..24)
    d1b_808 = build_808(8, octave_bounce=True, cut_last_beat=True)
    d1b_chords = build_chords(8, vel=76, cut_last_beat=True)
    d1b_lead = build_lead(8, vel_scale=1.00, cut_last_beat=True)
    add_offset(midi_tracks_data["Deep 808 Sub"], d1b_808, 16)
    add_offset(midi_tracks_data["Chords Synth"], d1b_chords, 16)
    add_offset(midi_tracks_data["Lead Synth"], d1b_lead, 16)

    # Break (bars 24..32) — true 3.5+ dB breath before Drop 2 Climax
    brk_sausage = build_sausage(8)
    brk_chords = build_chords(8, vel=40, cut_last_beat=True)
    brk_lead = build_lead(8, vel_scale=0.56, sparse_mode=True, cut_last_beat=True)
    add_offset(midi_tracks_data["Sausage Sub Swell"], brk_sausage, 24)
    add_offset(midi_tracks_data["Chords Synth"], brk_chords, 24)
    add_offset(midi_tracks_data["Lead Synth"], brk_lead, 24)

    # Drop 2 Climax (bars 32..40)
    d2_808 = build_808(8, octave_bounce=True, cut_last_beat=True)
    d2_chords = build_chords(8, vel=78, cut_last_beat=True)
    d2_lead = build_lead(8, vel_scale=1.04, cut_last_beat=True)
    add_offset(midi_tracks_data["Deep 808 Sub"], d2_808, 32)
    add_offset(midi_tracks_data["Chords Synth"], d2_chords, 32)
    add_offset(midi_tracks_data["Lead Synth"], d2_lead, 32)

    # Outro (bars 40..43, 3 bars)
    outro_chords = [
        {"pitch": p, "start": 0.0, "length": 3.6, "velocity": 60} for p in chords[0]["voicing_midi"]
    ] + [
        {"pitch": p, "start": 4.0, "length": 3.6, "velocity": 54} for p in chords[1]["voicing_midi"]
    ] + [
        {"pitch": p, "start": 8.0, "length": 3.8, "velocity": 48} for p in chords[0]["voicing_midi"]
    ]
    outro_lead = [
        {"pitch": 72, "start": 0.0, "length": 0.9, "velocity": 76},
        {"pitch": 75, "start": 1.0, "length": 0.9, "velocity": 72},
        {"pitch": 77, "start": 2.0, "length": 1.6, "velocity": 78},
        {"pitch": 75, "start": 4.0, "length": 1.6, "velocity": 68},
        {"pitch": 72, "start": 8.0, "length": 3.6, "velocity": 62},
    ]
    add_offset(midi_tracks_data["Chords Synth"], outro_chords, 40)
    add_offset(midi_tracks_data["Lead Synth"], outro_lead, 40)

    mid = mido.MidiFile(ticks_per_beat=480)
    tempo_us = mido.bpm2tempo(172)
    for tname, evs in midi_tracks_data.items():
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=tname, time=0))
        if tname == "Deep 808 Sub":
            tr.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        raw = []
        for n in evs:
            t0 = int(round(n["start"] * 480))
            t1 = int(round((n["start"] + n["length"]) * 480))
            raw.append((t0, "note_on", int(n["pitch"]), int(n["velocity"])))
            raw.append((t1, "note_off", int(n["pitch"]), 0))
        raw.sort(key=lambda x: (x[0], 0 if x[1] == "note_off" else 1))
        last = 0
        for t, typ, p, v in raw:
            tr.append(mido.Message(typ, note=p, velocity=v, time=max(0, t - last)))
            last = t
        tr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr)
    full_mid_path = f"{PROJ_DIR}/glass_horizon.mid"
    mid.save(full_mid_path)
    gate = verify_midi_gate(full_mid_path, "F minor", max_pitch=84)
    print("FULL 43-BAR MIDI GATE PASSED:", json.dumps(gate["musicality_metrics"], indent=2))

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

    # Channels 1..15
    ch_defs = [
        ("DnB Kick", "gl.sampler", 1, f"{SAMPLES}/dnb_kick_punch.wav", 0.80, 0.0),
        ("DnB Snare", "gl.sampler", 2, f"{SAMPLES}/dnb_snare_crisp.wav", 0.84, 0.0),
        ("Tech Snare Layer", "gl.sampler", 3, f"{SAMPLES}/tech_dnb_snare_clean.wav", 0.60, -0.05),
        ("Hi-Hat Crisp", "gl.hat", 4, None, 0.76, 0.16),
        ("Tech Top Break FX", "gl.sampler", 5, f"{SAMPLES}/tech_dnb_top_8bar_172.wav", 0.68, 0.0),
        ("Noisehats Air FX", "gl.sampler", 6, f"{SAMPLES}/noisehats_8bar_172.wav", 0.84, -0.14),
        ("Perc Ghost", "gl.sampler", 7, f"{SAMPLES}/dnb_perc_ghost.wav", 0.62, -0.24),
        ("Deep 808 Sub", "gl.sampler", 8, f"{SAMPLES}/deep_808_clean.wav", 0.60, 0.0),
        ("Sausage Sub Swell", "gl.sampler", 9, f"{SAMPLES}/sausage_bass_clean.wav", 0.58, 0.0),
        ("Chords Synth", "gl.synth", 10, None, 0.60, 0.0),
        ("Lead Synth", "gl.synth", 11, None, 0.72, 0.04),
        ("Glass Waves Pad FX", "gl.sampler", 12, f"{SAMPLES}/glass_waves_pad_a_8bar.wav", 0.62, 0.0),
        ("Glass Waves Shimmer FX", "gl.sampler", 13, f"{SAMPLES}/glass_waves_high12_8bar.wav", 0.52, 0.0),
        ("BFNK Fm Cloud FX", "gl.sampler", 14, f"{SAMPLES}/bfnk_pad_fm_shot.wav", 0.50, -0.22),
        ("Glass Riser FX", "gl.sampler", 15, f"{SAMPLES}/glass_waves_riser_1bar.wav", 0.56, 0.0),
        ("Glass Waves Intro FX", "gl.sampler", 16, f"{SAMPLES}/glass_waves_intro_8bar.wav", 0.66, 0.0),
        ("Glass Waves Pad B FX", "gl.sampler", 12, f"{SAMPLES}/glass_waves_pad_b_8bar.wav", 0.62, 0.0),
    ]

    mcp.batch([
        {"tool": "add_channel", "arguments": {"plugin": plug, "name": name, "mixer": mx, "volume": vol, "pan": pan}}
        for name, plug, mx, _, vol, pan in ch_defs
    ])

    for idx, (name, plug, mx, spath, _, _) in enumerate(ch_defs, start=1):
        if spath:
            mcp.call("patch", {"path": f"channel/{idx}", "set": {"plugin": {"data": spath}}})

    # Configure Sampler / Synth parameters
    param_changes = [
        # Drums samplers (Declick Transient)
        ("channel/1/Declick", 1), ("channel/1/Remove DC", 1),
        ("channel/2/Declick", 1), ("channel/2/Remove DC", 1),
        ("channel/3/Declick", 1), ("channel/3/Remove DC", 1),
        ("channel/5/Declick", 1), ("channel/5/Root note", 65), ("channel/5/One shot", 0), ("channel/5/Release", 0.05),
        ("channel/6/Declick", 1), ("channel/6/Root note", 65), ("channel/6/One shot", 0), ("channel/6/Release", 0.08),
        ("channel/7/Declick", 1),
        # Hi-Hat Crisp (gl.hat): tight airy tick (55 ms) with 4 ms soft onset (0 micro-glitches)
        ("channel/4/Decay", 0.055), ("channel/4/Tone", 12200), ("channel/4/Metal", 0.30), ("channel/4/Width", 0.65), ("channel/4/Attack", 0.004),
        # Deep 808 Sub (mono, tuned F1, Glide 35ms)
        ("channel/8/Root note", 24), ("channel/8/Fine tune", -14),
        ("channel/8/Voices", 1), ("channel/8/One shot", 0),
        ("channel/8/Remove DC", 1), ("channel/8/Declick", 1),
        ("channel/8/Release", 0.045), ("channel/8/Glide", 0.035),
        # Sausage Sub Swell (mono, tuned A1 +40ct, soft attack)
        ("channel/9/Root note", 33), ("channel/9/Fine tune", 40),
        ("channel/9/Voices", 1), ("channel/9/One shot", 0),
        ("channel/9/Remove DC", 1), ("channel/9/Declick", 1),
        ("channel/9/Attack", 0.035), ("channel/9/Release", 0.060), ("channel/9/Glide", 0.050),
        # Chords Synth (gl.synth - Warm Dreamy Liquid Pad/Rhodes with subtle high pan)
        ("channel/10/Osc 1 shape", 1), ("channel/10/Osc 1 level", 0.78),
        ("channel/10/Osc 2 shape", 0), ("channel/10/Osc 2 level", 0.55), ("channel/10/Osc 2 coarse", 0), ("channel/10/Osc 2 fine", 5), ("channel/10/Osc 2 pan", -0.20),
        ("channel/10/Osc 3 shape", 0), ("channel/10/Osc 3 level", 0.32), ("channel/10/Osc 3 coarse", 12), ("channel/10/Osc 3 fine", -5), ("channel/10/Osc 3 pan", 0.22),
        ("channel/10/Attack", 0.045), ("channel/10/Decay", 0.85), ("channel/10/Sustain", 0.52), ("channel/10/Release", 0.42),
        ("channel/10/Cutoff", 2600), ("channel/10/Resonance", 0.08),
        ("channel/10/Filter env", 0.18), ("channel/10/Filter decay", 0.55),
        ("channel/10/Voices", 12), ("channel/10/Velocity to cutoff", 0.25),
        # Lead Synth (gl.synth - Silky Liquid Pluck/Bell)
        ("channel/11/Osc 1 shape", 1), ("channel/11/Osc 1 level", 0.85),
        ("channel/11/Osc 2 shape", 0), ("channel/11/Osc 2 level", 0.60), ("channel/11/Osc 2 coarse", 12), ("channel/11/Osc 2 fine", 4), ("channel/11/Osc 2 pan", 0.15),
        ("channel/11/Osc 3 shape", 5), ("channel/11/Osc 3 level", 0.24), ("channel/11/Osc 3 coarse", 0), ("channel/11/Osc 3 fine", -4), ("channel/11/Osc 3 pan", -0.15),
        ("channel/11/Attack", 0.006), ("channel/11/Decay", 0.44), ("channel/11/Sustain", 0.26), ("channel/11/Release", 0.34),
        ("channel/11/Cutoff", 3000), ("channel/11/Resonance", 0.10),
        ("channel/11/Filter env", 0.42), ("channel/11/Filter decay", 0.30),
        ("channel/11/Voices", 8), ("channel/11/Velocity to cutoff", 0.35),
        # Loop / Pad FX Samplers (Root=65 F4, smooth release, not one-shot so length controls cutoff!)
        ("channel/12/Root note", 65), ("channel/12/One shot", 0), ("channel/12/Attack", 0.03), ("channel/12/Release", 0.18), ("channel/12/Remove DC", 1),
        ("channel/13/Root note", 65), ("channel/13/One shot", 0), ("channel/13/Attack", 0.06), ("channel/13/Release", 0.22), ("channel/13/Remove DC", 1),
        ("channel/14/Root note", 65), ("channel/14/One shot", 0), ("channel/14/Attack", 0.04), ("channel/14/Release", 0.25), ("channel/14/Remove DC", 1),
        ("channel/15/Root note", 65), ("channel/15/One shot", 0), ("channel/15/Attack", 0.01), ("channel/15/Release", 0.08),
        ("channel/16/Root note", 65), ("channel/16/One shot", 0), ("channel/16/Attack", 0.04), ("channel/16/Release", 0.22), ("channel/16/Remove DC", 1),
        ("channel/17/Root note", 65), ("channel/17/One shot", 0), ("channel/17/Attack", 0.03), ("channel/17/Release", 0.18), ("channel/17/Remove DC", 1),
    ]
    mcp.set_params(param_changes)

    # ------------------------------------------------------------------
    # 4. Mixer Routing, EQs, Sidechain & Standard Buses
    # ------------------------------------------------------------------
    BUS_DRUMS, BUS_808, BUS_MUSIC, FX_ROOM = 17, 18, 19, 20
    mcp.batch([
        {"tool": "add_mixer_track", "arguments": {"name": "BUS Drums"}},
        {"tool": "add_mixer_track", "arguments": {"name": "BUS 808"}},
        {"tool": "add_mixer_track", "arguments": {"name": "BUS Music"}},
        {"tool": "add_mixer_track", "arguments": {"name": "FX Room"}},
    ])
    setup_standard_buses(
        mcp,
        bpm=172.0,
        bus_drums=BUS_DRUMS,
        bus_808=BUS_808,
        bus_music=BUS_MUSIC,
        fx_room=FX_ROOM,
        limiter_ceiling_db=-2.55,
        limiter_gain_db=8.9,
        fx_room_vol=0.42,
    )

    mixer_defs = [
        (1, "DnB Kick", BUS_DRUMS, 0.76, 0.0),
        (2, "DnB Snare", BUS_DRUMS, 0.80, 0.0),
        (3, "Tech Snare Layer", BUS_DRUMS, 0.62, -0.05),
        (4, "Hi-Hat Crisp", BUS_DRUMS, 0.84, 0.16),
        (5, "Tech Top Break", BUS_DRUMS, 0.72, 0.0),
        (6, "Noisehats Air", BUS_DRUMS, 1.12, -0.16),
        (7, "Perc Ghost", BUS_DRUMS, 0.58, -0.24),
        (8, "Deep 808 Sub", BUS_808, 0.56, 0.0),
        (9, "Sausage Sub", BUS_808, 0.42, 0.0),
        (10, "Chords Synth", BUS_MUSIC, 0.76, 0.0),
        (11, "Lead Synth", BUS_MUSIC, 0.90, 0.04),
        (12, "Glass Waves Pad", BUS_MUSIC, 0.72, 0.0),
        (13, "Glass Shimmer", BUS_MUSIC, 0.72, 0.0),
        (14, "BFNK Fm Cloud", BUS_MUSIC, 0.56, -0.22),
        (15, "Glass Riser", BUS_MUSIC, 0.60, 0.0),
        (16, "Glass Intro", BUS_MUSIC, 0.74, 0.0),
        (BUS_DRUMS, "BUS Drums", 0, 0.90, 0.0),
        (BUS_808, "BUS 808", 0, 0.70, 0.0),
        (BUS_MUSIC, "BUS Music", 0, 0.88, 0.0),
    ]
    mcp.batch([
        {"tool": "edit_mixer_track", "arguments": {"mixer": m, "name": nm, "routeTo": rt, "volume": v, "pan": p}}
        for m, nm, rt, v, p in mixer_defs
    ])

    # Specific surgical EQs (keep CPU light: only tracks that need carving!)
    mcp.batch([
        {"tool": "add_plugin", "arguments": {"mixer": 1, "plugin": "gl.eq"}},   # Kick punch vs 808
        {"tool": "add_plugin", "arguments": {"mixer": 2, "plugin": "gl.eq"}},   # Snare body/air
        {"tool": "add_plugin", "arguments": {"mixer": 8, "plugin": "gl.dist"}}, # Subtle 808 warmth (80-200Hz audibility)
        {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.eq"}},  # Chords carve 1.6kHz + HPF 165Hz
        {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.eq"}},  # Glass Waves carve 260Hz + 1.6kHz
    ])
    mcp.set_params([
        # Master: clean LPF at 15.0 kHz prevents Nyquist ISP spikes; 5 ms look-ahead limiter with -2.55 dB ceiling
        ("mixer/0/slot/1/Band 7 freq", 15000),
        ("mixer/0/slot/2/Threshold", -9.5), ("mixer/0/slot/2/Makeup", 1.5),
        ("mixer/0/slot/3/Attack (look-ahead)", 0.005),
        # Tune BUS Drums air shelf (+6.0 dB @ 8.5 kHz), BUS 808 64 Hz Peak dip (-6.5 dB Q=1.0), BUS Music 500 Hz dip + Air shelf
        (f"mixer/{BUS_DRUMS}/slot/1/Band 7 type", 7), (f"mixer/{BUS_DRUMS}/slot/1/Band 7 freq", 8500), (f"mixer/{BUS_DRUMS}/slot/1/Band 7 gain", 6.0),
        (f"mixer/{BUS_808}/slot/3/Band 3 type", 6), (f"mixer/{BUS_808}/slot/3/Band 3 freq", 64), (f"mixer/{BUS_808}/slot/3/Band 3 gain", -6.5), (f"mixer/{BUS_808}/slot/3/Band 3 Q (legacy)", 1.0),
        (f"mixer/{BUS_MUSIC}/slot/1/Band 3 type", 6), (f"mixer/{BUS_MUSIC}/slot/1/Band 3 freq", 500), (f"mixer/{BUS_MUSIC}/slot/1/Band 3 gain", -3.6), (f"mixer/{BUS_MUSIC}/slot/1/Band 3 Q (legacy)", 1.5),
        (f"mixer/{BUS_MUSIC}/slot/1/Band 6 type", 7), (f"mixer/{BUS_MUSIC}/slot/1/Band 6 freq", 8500), (f"mixer/{BUS_MUSIC}/slot/1/Band 6 gain", 4.5),
        (f"mixer/{BUS_MUSIC}/slot/1/Band 7 freq", 15000),
        (f"mixer/{BUS_MUSIC}/slot/3/Threshold", -15.5), (f"mixer/{BUS_MUSIC}/slot/3/Makeup", 2.0),
        (f"mixer/{FX_ROOM}/slot/1/Decay time", 0.348), (f"mixer/{FX_ROOM}/slot/1/High cut", 10500), (f"mixer/{FX_ROOM}/slot/1/Width", 0.92),
        # Kick EQ: HPF 44 Hz, carve 64 Hz (-4.5dB), punch 108 Hz (+2.4dB)
        ("mixer/1/slot/1/Band 1 type", 1), ("mixer/1/slot/1/Band 1 freq", 44),
        ("mixer/1/slot/1/Band 2 type", 6), ("mixer/1/slot/1/Band 2 freq", 64), ("mixer/1/slot/1/Band 2 gain", -4.5), ("mixer/1/slot/1/Band 2 Q (legacy)", 1.2),
        ("mixer/1/slot/1/Band 3 type", 6), ("mixer/1/slot/1/Band 3 freq", 108), ("mixer/1/slot/1/Band 3 gain", 2.4), ("mixer/1/slot/1/Band 3 Q (legacy)", 1.5),
        # Snare EQ: HPF 150 Hz, gentle punch 220 Hz, tame 3.5 kHz, crisp air 8.5 kHz
        ("mixer/2/slot/1/Band 1 type", 1), ("mixer/2/slot/1/Band 1 freq", 150),
        ("mixer/2/slot/1/Band 3 type", 6), ("mixer/2/slot/1/Band 3 freq", 220), ("mixer/2/slot/1/Band 3 gain", 1.5),
        ("mixer/2/slot/1/Band 5 type", 6), ("mixer/2/slot/1/Band 5 freq", 3500), ("mixer/2/slot/1/Band 5 gain", -1.5),
        ("mixer/2/slot/1/Band 7 type", 7), ("mixer/2/slot/1/Band 7 freq", 8500), ("mixer/2/slot/1/Band 7 gain", 1.5),
        # Deep 808 warm 2nd/3rd harmonics (bridges 63 Hz -> 125-200 Hz cleanly)
        ("mixer/8/slot/1/Mode", 0), ("mixer/8/slot/1/Drive", 4.5), ("mixer/8/slot/1/Mix", 0.34),
        ("mixer/8/slot/1/Oversampling", 2), ("mixer/8/slot/1/Remove DC", 1),
        # Chords Synth dynamic space: HPF 165 Hz, cut 280 Hz (-2.5dB), carve 1.6 kHz (-2.5dB) for Lead
        ("mixer/10/slot/1/Band 1 type", 1), ("mixer/10/slot/1/Band 1 freq", 165),
        ("mixer/10/slot/1/Band 3 type", 6), ("mixer/10/slot/1/Band 3 freq", 280), ("mixer/10/slot/1/Band 3 gain", -2.5),
        ("mixer/10/slot/1/Band 5 type", 6), ("mixer/10/slot/1/Band 5 freq", 1600), ("mixer/10/slot/1/Band 5 gain", -2.5),
        # Glass Waves Pad: HPF 190 Hz, cut 250 Hz (-3.0dB), carve 1.6 kHz (-2.8dB)
        ("mixer/12/slot/1/Band 1 type", 1), ("mixer/12/slot/1/Band 1 freq", 190),
        ("mixer/12/slot/1/Band 3 type", 6), ("mixer/12/slot/1/Band 3 freq", 250), ("mixer/12/slot/1/Band 3 gain", -3.0),
        ("mixer/12/slot/1/Band 5 type", 6), ("mixer/12/slot/1/Band 5 freq", 1600), ("mixer/12/slot/1/Band 5 gain", -2.8),
    ])

    # Pre-fader sidechains (Kick -> BUS 808 & BUS Music) + FX Room sends
    sends = [
        {"tool": "set_send", "arguments": {"mixer": 1, "to": BUS_808, "level": 1.0, "preFader": True, "sidechain": True}},
        {"tool": "set_send", "arguments": {"mixer": 1, "to": BUS_MUSIC, "level": 1.0, "preFader": True, "sidechain": True}},
        {"tool": "set_send", "arguments": {"mixer": 2, "to": FX_ROOM, "level": 0.08}},
        {"tool": "set_send", "arguments": {"mixer": 3, "to": FX_ROOM, "level": 0.12}},
        {"tool": "set_send", "arguments": {"mixer": 7, "to": FX_ROOM, "level": 0.14}},
        {"tool": "set_send", "arguments": {"mixer": 10, "to": FX_ROOM, "level": 0.22}},
        {"tool": "set_send", "arguments": {"mixer": 11, "to": FX_ROOM, "level": 0.28}},
        {"tool": "set_send", "arguments": {"mixer": 12, "to": FX_ROOM, "level": 0.18}},
        {"tool": "set_send", "arguments": {"mixer": 13, "to": FX_ROOM, "level": 0.26}},
        {"tool": "set_send", "arguments": {"mixer": 14, "to": FX_ROOM, "level": 0.26}},
        {"tool": "set_send", "arguments": {"mixer": 15, "to": FX_ROOM, "level": 0.20}},
        {"tool": "set_send", "arguments": {"mixer": 16, "to": FX_ROOM, "level": 0.20}},
    ]
    mcp.batch(sends)

    # ------------------------------------------------------------------
    # 5. Create Per-Instrument Patterns & Place on Dedicated Playlist Tracks
    # ------------------------------------------------------------------
    # Pattern ID 1 already exists -> edit it; add more patterns as needed.
    # Let's define each pattern: (name, bars, channel, notes, playlist_track, start_beat)
    pat_specs = [
        # Intro (bars 1..8, start=0)
        ("Intro Glass Intro", 8, 16, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 90}], 16, 0.0),
        ("Intro Chords", 8, 10, intro_chords, 10, 0.0),
        ("Intro Lead", 4, 11, intro_lead, 11, 16.0),
        ("Intro Noisehats", 4, 6, [{"pitch": 65, "start": 0.0, "length": 15.0, "velocity": 66}], 6, 16.0),
        ("Intro Riser", 1, 15, [{"pitch": 65, "start": 0.0, "length": 3.85, "velocity": 95}], 15, 28.0),

        # Drop 1 (bars 9..16, start=32)
        ("Drop1 Kick", 8, 1, build_dnb_kick(8, cut_last_beat=True), 1, 32.0),
        ("Drop1 Snare", 8, 2, build_dnb_snare(8, cut_last_beat=True), 2, 32.0),
        ("Drop1 Tech Snare", 8, 3, build_dnb_snare(8, cut_last_beat=True), 3, 32.0),
        ("Drop1 Hats", 8, 4, build_hats(8, dense=True, cut_last_beat=True), 4, 32.0),
        ("Drop1 Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 96}], 5, 32.0),
        ("Drop1 Noisehats", 8, 6, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 86}], 6, 32.0),
        ("Drop1 Deep808", 8, 8, d1_808, 8, 32.0),
        ("Drop1 Chords", 8, 10, d1_chords, 10, 32.0),
        ("Drop1 Lead", 8, 11, d1_lead, 11, 32.0),
        ("Drop1 GlassPad", 8, 12, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 95}], 12, 32.0),

        # Drop 1B (bars 17..24, start=64)
        ("Drop1B Kick", 8, 1, build_dnb_kick(8, cut_last_beat=True), 1, 64.0),
        ("Drop1B Snare", 8, 2, build_dnb_snare(8, cut_last_beat=True), 2, 64.0),
        ("Drop1B Tech Snare", 8, 3, build_dnb_snare(8, cut_last_beat=True), 3, 64.0),
        ("Drop1B Hats", 8, 4, build_hats(8, dense=True, cut_last_beat=True), 4, 64.0),
        ("Drop1B Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 98}], 5, 64.0),
        ("Drop1B Noisehats", 8, 6, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 90}], 6, 64.0),
        ("Drop1B Perc", 8, 7, build_perc(8, cut_last_beat=True), 7, 64.0),
        ("Drop1B Deep808", 8, 8, d1b_808, 8, 64.0),
        ("Drop1B Chords", 8, 10, d1b_chords, 10, 64.0),
        ("Drop1B Lead", 8, 11, d1b_lead, 11, 64.0),
        ("Drop1B GlassPadB", 8, 17, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 98}], 12, 64.0),
        ("Drop1B BFNK Cloud", 8, 14, [
            {"pitch": 65, "start": 7.0, "length": 2.2, "velocity": 78},
            {"pitch": 65, "start": 15.0, "length": 2.2, "velocity": 78},
            {"pitch": 65, "start": 23.0, "length": 2.2, "velocity": 78},
        ], 14, 64.0),
        ("Drop1B Riser", 1, 15, [{"pitch": 65, "start": 0.0, "length": 3.85, "velocity": 100}], 15, 92.0),

        # Break Planant (bars 25..32, start=96) — airy & dynamic breath
        ("Break Snare Fill", 8, 2, build_dnb_snare(8, fill_build=True), 2, 96.0),
        ("Break Hats", 8, 4, build_hats(8, dense=False, cut_last_beat=True, vel_scale=0.65), 4, 96.0),
        ("Break Noisehats", 7, 6, [{"pitch": 65, "start": 0.0, "length": 27.0, "velocity": 42}], 6, 96.0),
        ("Break SausageSub", 8, 9, brk_sausage, 9, 96.0),
        ("Break Chords", 8, 10, brk_chords, 10, 96.0),
        ("Break Lead", 8, 11, brk_lead, 11, 96.0),
        ("Break GlassPad", 8, 12, [{"pitch": 65, "start": 0.0, "length": 30.8, "velocity": 44}], 12, 96.0),
        ("Break Shimmer", 8, 13, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 52}], 13, 96.0),
        ("Break BFNK Cloud", 8, 14, [
            {"pitch": 65, "start": 7.0, "length": 2.2, "velocity": 50},
            {"pitch": 65, "start": 23.0, "length": 2.2, "velocity": 50},
        ], 14, 96.0),
        ("Break Riser", 1, 15, [{"pitch": 65, "start": 0.0, "length": 3.85, "velocity": 76}], 15, 124.0),

        # Drop 2 Climax (bars 33..40, start=128)
        ("Drop2 Kick", 8, 1, build_dnb_kick(8, cut_last_beat=True), 1, 128.0),
        ("Drop2 Snare", 8, 2, build_dnb_snare(8, cut_last_beat=True), 2, 128.0),
        ("Drop2 Tech Snare", 8, 3, build_dnb_snare(8, cut_last_beat=True), 3, 128.0),
        ("Drop2 Hats", 8, 4, build_hats(8, dense=True, cut_last_beat=True), 4, 128.0),
        ("Drop2 Tech Top", 8, 5, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 102}], 5, 128.0),
        ("Drop2 Noisehats", 8, 6, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 94}], 6, 128.0),
        ("Drop2 Perc", 8, 7, build_perc(8, cut_last_beat=True), 7, 128.0),
        ("Drop2 Deep808", 8, 8, d2_808, 8, 128.0),
        ("Drop2 Chords", 8, 10, d2_chords, 10, 128.0),
        ("Drop2 Lead", 8, 11, d2_lead, 11, 128.0),
        ("Drop2 GlassPadB", 8, 17, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 102}], 12, 128.0),
        ("Drop2 Shimmer", 8, 13, [{"pitch": 65, "start": 0.0, "length": 31.0, "velocity": 98}], 13, 128.0),

        # Outro (bars 41..43, start=160..172)
        ("Outro GlassIntro", 3, 16, [{"pitch": 65, "start": 0.0, "length": 11.6, "velocity": 82}], 16, 160.0),
        ("Outro Noisehats", 1, 6, [{"pitch": 65, "start": 0.0, "length": 3.5, "velocity": 54}], 6, 160.0),
        ("Outro Chords", 3, 10, outro_chords, 10, 160.0),
        ("Outro Lead", 3, 11, outro_lead, 11, 160.0),
    ]

    for p_idx, (pname, bars, ch_idx, notes, pl_track, start_b) in enumerate(pat_specs, start=1):
        if p_idx == 1:
            mcp.call("edit_pattern", {"pattern": 1, "name": pname, "bars": bars})
        else:
            mcp.call("add_pattern", {"name": pname, "bars": bars})
        mcp.call("set_notes", {"pattern": p_idx, "channel": ch_idx, "notes": notes, "mode": "replace"})
        mcp.call("add_clip", {"pattern": p_idx, "track": pl_track, "start": start_b})

    # Add FX Room volume automation to cut reverb tails cleanly at pivot beats (31, 63, 95, 127, 159)
    # Normalized 0..1 of 0..1.25 -> 0.42 / 1.25 = 0.336
    nom = round(0.42 / 1.25, 4)
    auto_pts = [{"position": 0.0, "value": nom}]
    for piv in [31.0, 63.0, 95.0, 127.0, 159.0]:
        auto_pts.extend([
            {"position": piv - 0.1, "value": nom},
            {"position": piv, "value": 0.0, "shape": "hold"},
            {"position": piv + 0.95, "value": 0.0},
            {"position": piv + 1.0, "value": nom},
        ])
    auto_pts.extend([
        {"position": 170.5, "value": nom},
        {"position": 172.0, "value": 0.0},
    ])
    mcp.call("add_automation", {
        "mixer": FX_ROOM,
        "param": "volume",
        "name": "FX Room Pivot Cuts",
        "points": auto_pts,
        "track": 18,
        "start": 0.0,
        "length": 172.0,
    })

    # Add Section Markers
    for mk_name, mk_beat in [
        ("Intro", 0.0), ("Drop 1", 32.0), ("Drop 1B", 64.0),
        ("Break Planant", 96.0), ("Drop 2 Climax", 128.0), ("Outro", 160.0),
    ]:
        mcp.call("add_marker", {"name": mk_name, "position": mk_beat})

    # Save project cleanly
    mcp.call("save_project", {"path": "projet.gridline"})
    mcp.close()
    print("Saved projects/active9/projet.gridline successfully!")


if __name__ == "__main__":
    main()
