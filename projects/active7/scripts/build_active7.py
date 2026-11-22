#!/usr/bin/env python3
"""
Build script for projects/active7/projet.gridline
"Midnight Drift" — 28 bars @ 150 BPM (~45.0 s), Bb minor.
100% built from sounds present in projects/active7/Untitled.gridline
(plus BPM-locked / DC-cleaned / tuned derivatives in projects/active7/samples/).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys

import pretty_midi

ROOT = "/home/gamo/Bureau/musicos"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/mcp-servers/audio-analyzer")

from deep_diagnostics import audit_midi_file
from musicality_metrics import optimize_voice_leading
from scripts.studio_lib import GridlineMCP, TEMPLATE_EMPTY_MIXER, setup_standard_buses

PROJ_DIR = f"{ROOT}/projects/active7"
PROJ_FILE = f"{PROJ_DIR}/projet.gridline"
SKETCH_MIDI = f"{PROJ_DIR}/master_sketch.mid"
ORCH_MIDI = f"{PROJ_DIR}/midnight_drift.mid"
RENDERS_DIR = f"{PROJ_DIR}/renders"
SAMPLES_DIR = f"{PROJ_DIR}/samples"

BPM = 150.0
random.seed(77)


def build_master_sketch_and_gate() -> tuple[list[list[int]], list[tuple[int, float, float, int]]]:
    """Compose Top-Down 3-voice reduction and enforce blocking 7-metric musicality gate."""
    chord_pcs = [
        [10, 1, 5, 8],   # Bbm7
        [8, 10, 3],      # Absus2
        [6, 10, 1, 5],   # Gbmaj7
        [6, 10, 1, 5],   # Gbmaj7/F
    ]
    voicings = optimize_voice_leading(chord_pcs, low_midi=51, high_midi=63)

    lead_phrase = [
        # Bar 0 (A on Bbm7, Bass=Bb1=34)
        (0, 0.0, 0.45, 70), (0, 0.5, 0.45, 73), (0, 1.0, 0.45, 77), (0, 1.5, 0.45, 75), (0, 2.0, 0.75, 73),
        # Bar 1 (A' on Absus2, Bass=Ab1=32 -> contrary motion)
        (1, 0.0, 0.45, 75), (1, 0.5, 0.45, 77), (1, 1.0, 0.45, 75), (1, 1.5, 0.45, 73), (1, 2.0, 0.75, 70),
        # Bar 2 (A'' Apex Bb5=82 on Gbmaj7, Bass=Gb1=30)
        (2, 0.0, 0.45, 73), (2, 0.5, 0.45, 77), (2, 1.0, 0.45, 78), (2, 1.5, 0.45, 80), (2, 2.0, 0.75, 82), (2, 2.75, 0.40, 80),
        # Bar 3 (B cadence on Gbmaj7/F, Bass=Gb1=30 -> F1=29)
        (3, 0.0, 0.45, 78), (3, 0.5, 0.45, 77), (3, 1.0, 0.45, 73), (3, 1.5, 0.45, 75), (3, 2.0, 0.75, 77), (3, 3.0, 0.45, 73),
    ]

    beat_s = 60.0 / BPM
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    bass = pretty_midi.Instrument(program=33, name="Bass")
    chords = pretty_midi.Instrument(program=0, name="Chords")
    lead = pretty_midi.Instrument(program=0, name="Lead")

    for rep in range(2):
        bar0 = rep * 4
        for b in range(4):
            s = (bar0 + b) * 4.0
            if b == 0:
                bass.notes.append(pretty_midi.Note(100, 34, s * beat_s, (s + 3.8) * beat_s))
            elif b == 1:
                bass.notes.append(pretty_midi.Note(100, 32, s * beat_s, (s + 3.8) * beat_s))
            elif b == 2:
                bass.notes.append(pretty_midi.Note(100, 30, s * beat_s, (s + 3.8) * beat_s))
            else:
                bass.notes.append(pretty_midi.Note(100, 30, s * beat_s, (s + 1.9) * beat_s))
                bass.notes.append(pretty_midi.Note(95, 29, (s + 2.0) * beat_s, (s + 3.8) * beat_s))
            for p in voicings[b]:
                chords.notes.append(pretty_midi.Note(74, p, s * beat_s, (s + 2.85) * beat_s))
        for b, st, ln, pt in lead_phrase:
            t0 = ((bar0 + b) * 4.0 + st) * beat_s
            lead.notes.append(pretty_midi.Note(92, pt, t0, t0 + ln * beat_s))

    pm.instruments.extend([bass, chords, lead])
    pm.write(SKETCH_MIDI)

    gate = audit_midi_file(SKETCH_MIDI, expected_key="Bb minor", min_pitch=24, max_pitch=84, strict_musicality=True)
    print("Master Sketch Gate:", json.dumps({
        "verdict": gate["musicality_verdict"],
        "gate_passed": gate["gate_passed"],
        "metrics": gate["musicality_metrics"],
        "failures": gate["musicality_failures"],
    }, indent=2))
    if not gate["gate_passed"]:
        raise RuntimeError(f"BLOCKING GATE FAILED: {gate['musicality_failures']}")
    return voicings, lead_phrase


def build() -> None:
    voicings, lead_phrase = build_master_sketch_and_gate()
    shutil.copyfile(TEMPLATE_EMPTY_MIXER, PROJ_FILE)
    mcp = GridlineMCP(PROJ_FILE, RENDERS_DIR, client_name="build_active7")

    try:
        # 1. Song settings
        mcp.call("set_song", {
            "title": "Midnight Drift",
            "genre": "Atmospheric Melodic Trap",
            "author": "AI-Studio",
            "tempo": BPM,
            "swing": 0.06,
        })

        # 2. Mixer tracks 1..20
        mix_info = mcp.call("get_mixer", {})
        existing = {t["mixer"] for t in mix_info["tracks"]}
        adds = [{"tool": "add_mixer_track", "arguments": {}} for m in range(1, 21) if m not in existing]
        if adds:
            mcp.batch(adds)
        for tr in mix_info["tracks"]:
            if tr["mixer"] == 0:
                for fx in tr["effects"]:
                    mcp.call("remove_plugin", {"mixer": 0, "slot": fx["slot"]})

        # Routing:
        # 1,4,5,6,7 -> BUS Drums (17)
        # 2,3 -> BUS 808 (18)
        # 8,9,10,11,12,13 -> BUS Music (19)
        # 20 -> FX Room -> Master (0)
        mixer_defs = [
            (1, "Kick", 17, 0.76, 0.0, 0.0),
            (2, "808 Pluggg Main", 18, 0.80, 0.0, -0.8),
            (3, "808 Pluggg Climax", 18, 0.80, 0.0, -0.8),
            (4, "Clap", 17, 0.88, 0.0, 0.0),
            (5, "Snare Native", 17, 0.74, 0.0, 0.0),
            (6, "Hi-Hats Closed", 17, 0.86, 0.0, 0.0),
            (7, "OpenHat, Cym & Perc", 17, 0.80, 0.0, 0.0),
            (8, "BFNK Open", 19, 0.94, 0.0, -0.18),
            (9, "BFNK Filtered", 19, 0.44, 0.0, -0.18),
            (10, "BFNK High+12 & FX", 19, 0.76, 0.0, -0.12),
            (11, "VOX Ambient", 19, 0.84, 0.0, -0.18),
            (12, "Lead Synth", 19, 0.96, 0.04, -0.10),
            (13, "Chords Synth Pad", 19, 0.66, -0.06, -0.22),
            (17, "BUS Drums", 0, 0.96, 0.0, 0.0),
            (18, "BUS 808", 0, 0.76, 0.0, -1.0),
            (19, "BUS Music", 0, 0.96, 0.0, -0.16),
            (20, "FX Room", 0, 0.30, 0.0, 0.0),
        ]
        calls = []
        stereo_changes = []
        for m, name, route, vol, pan, stereo in mixer_defs:
            calls.append({"tool": "edit_mixer_track", "arguments": {"mixer": m, "name": name, "routeTo": route, "volume": vol, "pan": pan}})
            if stereo != 0.0:
                stereo_changes.append((f"mixer/{m}/stereo", stereo))
        mcp.batch(calls)
        if stereo_changes:
            mcp.set_params(stereo_changes)

        # Sidechains & FX Room sends
        sends = [
            (1, 18, 1.0, True),   # Kick -> BUS 808 sidechain
            (1, 19, 0.65, True),  # Kick -> BUS Music micro-ducking
            (4, 20, 0.05, False),
            (5, 20, 0.04, False),
            (7, 20, 0.05, False),
            (8, 20, 0.09, False),
            (9, 20, 0.10, False),
            (10, 20, 0.14, False),
            (11, 20, 0.18, False),
            (12, 20, 0.15, False),
            (13, 20, 0.15, False),
        ]
        mcp.batch([{"tool": "set_send", "arguments": {"mixer": m, "to": to, "level": lvl, "sidechain": sc}} for m, to, lvl, sc in sends])

        # 3. Channels 1..20 (100% Untitled.gridline sources + derived samples/)
        channels_spec = [
            # 1-2 Kicks
            ("Kick Native", "gl.kick", 1, None, 0.68, 0.0),
            ("Kick Sample", "gl.sampler", 1, f"{SAMPLES_DIR}/dt_-_farewell_(kick)_2_clean.wav", 0.72, 0.0),
            # 3-4 808 Pluggg_2
            ("808 Pluggg Main", "gl.sampler", 2, f"{SAMPLES_DIR}/(808)_pluggg_2_clean.wav", 0.84, 0.0),
            ("808 Pluggg Climax", "gl.sampler", 3, f"{SAMPLES_DIR}/(808)_pluggg_2_clean.wav", 0.84, 0.0),
            # 5-7 Claps & Snare
            ("Clap Boring", "gl.sampler", 4, f"{SAMPLES_DIR}/nm_-_boring_clap_clean.wav", 0.88, 0.0),
            ("Clap Native", "gl.clap", 4, None, 0.44, 0.0),
            ("Snare Native", "gl.snare", 5, None, 0.72, 0.0),
            # 8-12 Hats, Cym & Perc
            ("Hi-Hat Finesse", "gl.sampler", 6, f"{SAMPLES_DIR}/[pbs]_finesse_hat_clean.wav", 0.86, 0.06),
            ("Hi-Hat Native", "gl.hat", 6, None, 0.56, -0.06),
            ("Open-Hat Southside", "gl.sampler", 7, f"{SAMPLES_DIR}/[acd]_-_southsidetriangle_hat_2_clean.wav", 0.76, -0.14),
            ("Cymbal OG", "gl.sampler", 7, f"{SAMPLES_DIR}/[pbs]_og_cym_clean.wav", 0.72, 0.14),
            ("Perc 14", "gl.sampler", 7, f"{SAMPLES_DIR}/nm_-_perc_14_tuned.wav", 0.72, -0.20),
            # 13-17 BFNK Harmonies & Risers
            ("Chords BFNK Open", "gl.sampler", 8, f"{SAMPLES_DIR}/bfnk_full_150.wav", 0.80, 0.0),
            ("Chords BFNK Filt", "gl.sampler", 9, f"{SAMPLES_DIR}/bfnk_a_150.wav", 0.78, 0.0),
            ("Chords BFNK Chop", "gl.sampler", 9, f"{SAMPLES_DIR}/bfnk_chop_150.wav", 0.80, 0.0),
            ("Chords BFNK High+12", "gl.sampler", 10, f"{SAMPLES_DIR}/bfnk_high12_150.wav", 0.66, 0.0),
            ("FX Radiate Riser", "gl.sampler", 10, f"{SAMPLES_DIR}/radiate_riser_150.wav", 0.68, 0.0),
            # 18-19 Counter Choir (VOX samples) & FX
            ("Counter Choir Virtual", "gl.sampler", 11, f"{SAMPLES_DIR}/vox_virtual_bb4.wav", 0.74, -0.10),
            ("Counter Choir ReUsed", "gl.sampler", 11, f"{SAMPLES_DIR}/vox_reused_f5.wav", 0.66, 0.12),
            ("FX (4)_3", "gl.sampler", 10, f"{SAMPLES_DIR}/fx_4_3_tuned.wav", 0.64, 0.16),
            # 21-22 Native gl.synth (from Untitled.gridline)
            ("Lead Synth", "gl.synth", 12, None, 0.84, 0.04),
            ("Chords Synth Pad", "gl.synth", 13, None, 0.56, -0.06),
        ]

        mcp.batch([
            {"tool": "add_channel", "arguments": {"name": name, "plugin": plug, "mixer": mixer, "volume": vol, "pan": pan}}
            for name, plug, mixer, _, vol, pan in channels_spec
        ])
        mcp.batch([
            {"tool": "patch", "arguments": {"path": f"channel/{idx}", "set": {"plugin": {"data": spath}}}}
            for idx, (_, _, _, spath, _, _) in enumerate(channels_spec, 1) if spath
        ])

        # 4. Channel tuning, ADSR & sound shaping
        ch_params = [
            # Ch 1 Kick Native (tight 125ms)
            ("channel/1/Start pitch", 200), ("channel/1/End pitch", 54), ("channel/1/Sweep", 0.045),
            ("channel/1/Decay", 0.125), ("channel/1/Click", 0.28), ("channel/1/Drive", 0.16),
            # Ch 2 Kick Sample
            ("channel/2/One shot", 0), ("channel/2/Release", 0.02), ("channel/2/Declick", 2), ("channel/2/Remove DC", 1),
            # Ch 3-4 808 Pluggg_2: Root 24 (C1), Fine +4 ct, Voices=1 (Cut Itself!), Release 45ms, Glide 55ms
            ("channel/3/Root note", 24), ("channel/3/Fine tune", 4), ("channel/3/Voices", 1),
            ("channel/3/One shot", 0), ("channel/3/Attack", 0.002), ("channel/3/Decay", 10.0),
            ("channel/3/Sustain", 1.0), ("channel/3/Release", 0.045), ("channel/3/Glide", 0.055),
            ("channel/3/Remove DC", 1), ("channel/3/Declick", 2),
            ("channel/4/Root note", 24), ("channel/4/Fine tune", 4), ("channel/4/Voices", 1),
            ("channel/4/One shot", 0), ("channel/4/Attack", 0.002), ("channel/4/Decay", 10.0),
            ("channel/4/Sustain", 1.0), ("channel/4/Release", 0.045), ("channel/4/Glide", 0.055),
            ("channel/4/Remove DC", 1), ("channel/4/Declick", 2),
            # Ch 5-7 Claps & Snare
            ("channel/5/One shot", 0), ("channel/5/Release", 0.025), ("channel/5/Declick", 2),
            ("channel/6/Decay", 0.18), ("channel/6/Tone", 1450), ("channel/6/Width", 0.65),
            ("channel/7/Tone", 0.55), ("channel/7/Snappy", 0.68), ("channel/7/Body decay", 0.11),
            ("channel/7/Noise decay", 0.16), ("channel/7/Tune", 198),
            # Ch 8-12 Hats, Cym & Perc (tight gates)
            ("channel/8/One shot", 0), ("channel/8/Release", 0.015), ("channel/8/Declick", 2),
            ("channel/9/Decay", 0.045), ("channel/9/Tone", 8200),
            ("channel/10/One shot", 0), ("channel/10/Release", 0.030), ("channel/10/Declick", 2),
            ("channel/11/One shot", 0), ("channel/11/Release", 0.080), ("channel/11/Declick", 2),
            ("channel/12/One shot", 0), ("channel/12/Release", 0.040), ("channel/12/Declick", 2),
            # Ch 13-17 BFNK & Riser (Root 58 = Bb3, already tuned 440.0 Hz)
            ("channel/13/Root note", 58), ("channel/13/One shot", 0), ("channel/13/Attack", 0.012), ("channel/13/Release", 0.12), ("channel/13/Declick", 2),
            ("channel/14/Root note", 58), ("channel/14/One shot", 0), ("channel/14/Attack", 0.015), ("channel/14/Release", 0.16), ("channel/14/Declick", 2),
            ("channel/15/Root note", 58), ("channel/15/One shot", 0), ("channel/15/Attack", 0.010), ("channel/15/Release", 0.12), ("channel/15/Declick", 2),
            ("channel/16/Root note", 58), ("channel/16/One shot", 0), ("channel/16/Attack", 0.020), ("channel/16/Release", 0.15), ("channel/16/Declick", 2),
            ("channel/17/Root note", 58), ("channel/17/One shot", 0), ("channel/17/Attack", 0.010), ("channel/17/Release", 0.02), ("channel/17/Declick", 2),
            # Ch 18-20 VOX & FX
            ("channel/18/Root note", 70), ("channel/18/Fine tune", 8), ("channel/18/One shot", 0),
            ("channel/18/Attack", 0.035), ("channel/18/Release", 0.35), ("channel/18/Declick", 2),
            ("channel/19/Root note", 77), ("channel/19/Fine tune", 24), ("channel/19/One shot", 0),
            ("channel/19/Attack", 0.015), ("channel/19/Release", 0.25), ("channel/19/Declick", 2),
            ("channel/20/Root note", 70), ("channel/20/One shot", 0), ("channel/20/Release", 0.025), ("channel/20/Declick", 2),
            # Ch 21 Lead Synth (warm dreamy bell-pluck: Triangle + Sine +12 + soft detuned Rounded Saw)
            ("channel/21/Osc 1 shape", 1), ("channel/21/Osc 1 level", 0.85), ("channel/21/Osc 1 stereo detune", 4), ("channel/21/Osc 1 stereo phase", 0.0),
            ("channel/21/Osc 2 shape", 0), ("channel/21/Osc 2 level", 0.46), ("channel/21/Osc 2 coarse", 12), ("channel/21/Osc 2 fine", 0), ("channel/21/Osc 2 stereo detune", 6),
            ("channel/21/Osc 3 shape", 5), ("channel/21/Osc 3 level", 0.14), ("channel/21/Osc 3 coarse", 0), ("channel/21/Osc 3 fine", -5),
            ("channel/21/Attack", 0.012), ("channel/21/Decay", 0.40), ("channel/21/Sustain", 0.20), ("channel/21/Release", 0.32),
            ("channel/21/Cutoff", 3400), ("channel/21/Resonance", 0.06), ("channel/21/Filter env", 0.28), ("channel/21/Filter decay", 0.28),
            ("channel/21/Velocity to cutoff", 0.24), ("channel/21/Volume", 0.86), ("channel/21/Voices", 8),
            # Ch 22 Chords Synth Pad (soft floating pad)
            ("channel/22/Osc 1 shape", 0), ("channel/22/Osc 1 level", 0.72), ("channel/22/Osc 1 stereo detune", 3),
            ("channel/22/Osc 2 shape", 1), ("channel/22/Osc 2 level", 0.42), ("channel/22/Osc 2 coarse", 0), ("channel/22/Osc 2 fine", 4), ("channel/22/Osc 2 stereo detune", 5),
            ("channel/22/Osc 3 level", 0.0),
            ("channel/22/Attack", 0.085), ("channel/22/Decay", 1.1), ("channel/22/Sustain", 0.68), ("channel/22/Release", 0.35),
            ("channel/22/Cutoff", 1600), ("channel/22/Resonance", 0.04), ("channel/22/Filter env", 0.14),
            ("channel/22/Volume", 0.56), ("channel/22/Voices", 12),
        ]
        mcp.set_params(ch_params)

        # 5. Standard Buses + Selective Insert EQs
        setup_standard_buses(
            mcp,
            bpm=BPM,
            limiter_ceiling_db=-2.50,
            limiter_gain_db=7.8,
            fx_room_vol=0.30,
        )

        # Add selective insert EQs + soft 808 Climax drive
        insert_fx = [
            {"tool": "add_plugin", "arguments": {"mixer": 1, "plugin": "gl.eq"}},    # Kick carve 160Hz
            {"tool": "add_plugin", "arguments": {"mixer": 3, "plugin": "gl.dist"}},  # 808 Climax warm harmonics
            {"tool": "add_plugin", "arguments": {"mixer": 4, "plugin": "gl.eq"}},    # Clap
            {"tool": "add_plugin", "arguments": {"mixer": 5, "plugin": "gl.eq"}},    # Snare
            {"tool": "add_plugin", "arguments": {"mixer": 6, "plugin": "gl.eq"}},    # Hi-Hats
            {"tool": "add_plugin", "arguments": {"mixer": 8, "plugin": "gl.eq"}},    # BFNK Open
            {"tool": "add_plugin", "arguments": {"mixer": 9, "plugin": "gl.eq"}},    # BFNK Filtered
            {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.eq"}},   # BFNK High+12
            {"tool": "add_plugin", "arguments": {"mixer": 11, "plugin": "gl.eq"}},   # VOX Ambient
            {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.eq"}},   # Lead Synth
            {"tool": "add_plugin", "arguments": {"mixer": 13, "plugin": "gl.eq"}},   # Chords Synth Pad
        ]
        mcp.batch(insert_fx)

        fx_changes = [
            # Master EQ & Comp (raise Master Comp Threshold to -9.5 dB so Intro/Pont keep 3-5 dB dynamic arc!)
            ("mixer/0/slot/1/Band 7 freq", 16500),
            ("mixer/0/slot/2/Threshold", -9.5), ("mixer/0/slot/2/Ratio", 1.6), ("mixer/0/slot/2/Makeup", 1.4),

            # Enhance FX Room slightly for "planant" feel while keeping RT60 <= T_noire (0.400 s)
            ("mixer/20/slot/1/Room size", 0.36),
            ("mixer/20/slot/1/Decay time", 0.340),
            ("mixer/20/slot/1/Wet", 0.48), ("mixer/20/slot/1/Wet (full)", 48),
            ("mixer/20/slot/1/Width", 0.80),

            # BUS Drums: control 160 Hz + balanced air shelf @ 10 kHz
            ("mixer/17/slot/1/Band 3 type", 6), ("mixer/17/slot/1/Band 3 freq", 160), ("mixer/17/slot/1/Band 3 gain", -2.4), ("mixer/17/slot/1/Band 3 Q (legacy)", 1.3),
            ("mixer/17/slot/1/Band 7 type", 7), ("mixer/17/slot/1/Band 7 freq", 10000), ("mixer/17/slot/1/Band 7 gain", 1.8),

            # BUS 808: gentle control @ 50 Hz & 160 Hz, warm punch @ 105 Hz
            ("mixer/18/slot/3/Band 2 type", 6), ("mixer/18/slot/3/Band 2 freq", 50), ("mixer/18/slot/3/Band 2 gain", -1.6), ("mixer/18/slot/3/Band 2 Q (legacy)", 1.4),
            ("mixer/18/slot/3/Band 3 type", 6), ("mixer/18/slot/3/Band 3 freq", 105), ("mixer/18/slot/3/Band 3 gain", 1.4), ("mixer/18/slot/3/Band 3 Q (legacy)", 1.1),
            ("mixer/18/slot/3/Band 4 type", 6), ("mixer/18/slot/3/Band 4 freq", 160), ("mixer/18/slot/3/Band 4 gain", -3.6), ("mixer/18/slot/3/Band 4 Q (legacy)", 1.3),

            # BUS Music: lift 390 Hz dip (+2.8 dB) & tame 530 Hz (-3.8 dB) so 1/3-oct spectrum is smooth (0 resonances!),
            # and raise Glue Comp threshold (-14.0 dB, Makeup 1.4 dB) so Intro/Pont breathe dynamically!
            ("mixer/19/slot/1/Band 1 freq", 140),
            ("mixer/19/slot/1/Band 3 type", 6), ("mixer/19/slot/1/Band 3 freq", 390), ("mixer/19/slot/1/Band 3 gain", 2.8), ("mixer/19/slot/1/Band 3 Q (legacy)", 1.0),
            ("mixer/19/slot/1/Band 4 type", 6), ("mixer/19/slot/1/Band 4 freq", 530), ("mixer/19/slot/1/Band 4 gain", -3.8), ("mixer/19/slot/1/Band 4 Q (legacy)", 1.3),
            ("mixer/19/slot/1/Band 5 type", 6), ("mixer/19/slot/1/Band 5 freq", 1750), ("mixer/19/slot/1/Band 5 gain", 1.8), ("mixer/19/slot/1/Band 5 Q (legacy)", 0.75),
            ("mixer/19/slot/1/Band 7 freq", 14000),
            ("mixer/19/slot/3/Threshold", -14.0), ("mixer/19/slot/3/Makeup", 1.4),

            # Kick (mixer 1): tight 160Hz carve + punch click
            ("mixer/1/slot/1/Band 1 type", 1), ("mixer/1/slot/1/Band 1 freq", 32),
            ("mixer/1/slot/1/Band 3 type", 6), ("mixer/1/slot/1/Band 3 freq", 160), ("mixer/1/slot/1/Band 3 gain", -2.6), ("mixer/1/slot/1/Band 3 Q (legacy)", 1.3),
            ("mixer/1/slot/1/Band 5 type", 6), ("mixer/1/slot/1/Band 5 freq", 2600), ("mixer/1/slot/1/Band 5 gain", 1.5),

            # 808 Climax gentle warm saturation
            ("mixer/3/slot/1/Mode", 0), ("mixer/3/slot/1/Drive", 2.8), ("mixer/3/slot/1/Tone", 3500),
            ("mixer/3/slot/1/Mix", 0.16), ("mixer/3/slot/1/Oversampling", 2), ("mixer/3/slot/1/Remove DC", 1),

            # Clap & Snare
            ("mixer/4/slot/1/Band 1 type", 1), ("mixer/4/slot/1/Band 1 freq", 180),
            ("mixer/4/slot/1/Band 5 type", 6), ("mixer/4/slot/1/Band 5 freq", 2400), ("mixer/4/slot/1/Band 5 gain", 2.2),
            ("mixer/4/slot/1/Band 7 type", 3), ("mixer/4/slot/1/Band 7 freq", 13000),
            ("mixer/5/slot/1/Band 1 type", 1), ("mixer/5/slot/1/Band 1 freq", 145),
            ("mixer/5/slot/1/Band 7 type", 3), ("mixer/5/slot/1/Band 7 freq", 12500),

            # Hi-Hats (smooth 7.2 kHz, crisp air above 10.5 kHz without 6.3-8 kHz spike)
            ("mixer/6/slot/1/Band 1 type", 1), ("mixer/6/slot/1/Band 1 freq", 550),
            ("mixer/6/slot/1/Band 5 type", 6), ("mixer/6/slot/1/Band 5 freq", 7200), ("mixer/6/slot/1/Band 5 gain", -1.6), ("mixer/6/slot/1/Band 5 Q (legacy)", 0.9),
            ("mixer/6/slot/1/Band 7 type", 7), ("mixer/6/slot/1/Band 7 freq", 10500), ("mixer/6/slot/1/Band 7 gain", 1.8),

            # BFNK Open (mixer 8): preserve low_mids warmth (340 Hz) while cleaning 245 Hz & 500 Hz
            ("mixer/8/slot/1/Band 1 type", 1), ("mixer/8/slot/1/Band 1 freq", 175),
            ("mixer/8/slot/1/Band 3 type", 6), ("mixer/8/slot/1/Band 3 freq", 245), ("mixer/8/slot/1/Band 3 gain", -1.4), ("mixer/8/slot/1/Band 3 Q (legacy)", 1.0),
            ("mixer/8/slot/1/Band 4 type", 6), ("mixer/8/slot/1/Band 4 freq", 500), ("mixer/8/slot/1/Band 4 gain", -2.4), ("mixer/8/slot/1/Band 4 Q (legacy)", 1.2),
            ("mixer/8/slot/1/Band 5 type", 6), ("mixer/8/slot/1/Band 5 freq", 1500), ("mixer/8/slot/1/Band 5 gain", -1.8),
            ("mixer/8/slot/1/Band 7 type", 3), ("mixer/8/slot/1/Band 7 freq", 10500),

            # BFNK Filtered (mixer 9): Intro / Pont / Outro LPF 1500 Hz, tame 500 Hz
            ("mixer/9/slot/1/Band 1 type", 1), ("mixer/9/slot/1/Band 1 freq", 175),
            ("mixer/9/slot/1/Band 3 type", 6), ("mixer/9/slot/1/Band 3 freq", 245), ("mixer/9/slot/1/Band 3 gain", -1.6), ("mixer/9/slot/1/Band 3 Q (legacy)", 1.0),
            ("mixer/9/slot/1/Band 4 type", 6), ("mixer/9/slot/1/Band 4 freq", 500), ("mixer/9/slot/1/Band 4 gain", -2.4), ("mixer/9/slot/1/Band 4 Q (legacy)", 1.2),
            ("mixer/9/slot/1/Band 7 type", 3), ("mixer/9/slot/1/Band 7 freq", 1500),

            # BFNK High+12 & FX (mixer 10)
            ("mixer/10/slot/1/Band 1 type", 1), ("mixer/10/slot/1/Band 1 freq", 520),
            ("mixer/10/slot/1/Band 7 type", 3), ("mixer/10/slot/1/Band 7 freq", 11000),

            # VOX Ambient (mixer 11)
            ("mixer/11/slot/1/Band 1 type", 1), ("mixer/11/slot/1/Band 1 freq", 280),
            ("mixer/11/slot/1/Band 5 type", 6), ("mixer/11/slot/1/Band 5 freq", 2200), ("mixer/11/slot/1/Band 5 gain", 1.8),
            ("mixer/11/slot/1/Band 7 type", 3), ("mixer/11/slot/1/Band 7 freq", 9000),

            # Lead Synth (mixer 12)
            ("mixer/12/slot/1/Band 1 type", 1), ("mixer/12/slot/1/Band 1 freq", 280),
            ("mixer/12/slot/1/Band 5 type", 6), ("mixer/12/slot/1/Band 5 freq", 1600), ("mixer/12/slot/1/Band 5 gain", 1.8),
            ("mixer/12/slot/1/Band 7 type", 3), ("mixer/12/slot/1/Band 7 freq", 8500),

            # Chords Synth Pad (mixer 13)
            ("mixer/13/slot/1/Band 1 type", 1), ("mixer/13/slot/1/Band 1 freq", 220),
            ("mixer/13/slot/1/Band 3 type", 6), ("mixer/13/slot/1/Band 3 freq", 300), ("mixer/13/slot/1/Band 3 gain", -2.5),
            ("mixer/13/slot/1/Band 7 type", 3), ("mixer/13/slot/1/Band 7 freq", 5500),
        ]
        mcp.set_params(fx_changes)

        # 6. Playlist Tracks 1..10
        track_names = [
            (1, "808 Pluggg Main"),
            (2, "808 Pluggg Climax"),
            (3, "Kick Native + Sample"),
            (4, "Clap & Snare"),
            (5, "Hi-Hats Closed"),
            (6, "OpenHat, Cym & Perc 14"),
            (7, "BFNK Harmony & Chop"),
            (8, "BFNK High+12 & Risers"),
            (9, "Lead Synth (A-A'-A''-B)"),
            (10, "VOX Ambient & Room Cuts"),
        ]
        mcp.batch([{"tool": "patch", "arguments": {"path": f"track/{t}", "set": {"name": nm}}} for t, nm in track_names])

        # 7. Patterns & Clips across 28 bars (~44.8 s + tail = ~45.0 s @ 150 BPM)
        pat_counter = 1
        queued: list[dict] = [{"tool": "edit_pattern", "arguments": {"pattern": 1, "name": "BFNK Intro 4m (Filt)", "bars": 4}}]

        def new_pattern(name: str, bars: int) -> int:
            nonlocal pat_counter
            pat_counter += 1
            queued.append({"tool": "add_pattern", "arguments": {"name": name, "bars": bars}})
            return pat_counter

        def add_notes(pat: int, ch: int, notes: list[dict]) -> None:
            queued.append({"tool": "set_notes", "arguments": {"pattern": pat, "channel": ch, "notes": notes}})

        def place(pat: int, track: int, start_bar: int) -> None:
            queued.append({"tool": "add_clip", "arguments": {"pattern": pat, "track": track, "start": (start_bar - 1) * 4.0}})

        def make_voiceled_chords(bars: int = 4, vel: int = 68, mute_last_beat: bool = True) -> list[dict]:
            notes = []
            for b in range(bars):
                ln = 2.60 if (mute_last_beat and b == bars - 1) else 2.85
                m4 = b % 4
                for p in voicings[m4]:
                    notes.append({"start": b * 4.0, "length": ln, "pitch": p, "velocity": vel})
            return notes

        # ---- HARMONY: BFNK + CHORDS SYNTH PAD (Track 7 & Track 8) ----
        # M1-4 Intro (Ch 14 BFNK A Filt + Ch 22 Chords Synth Pad soft -> -14.5 LUFS arc)
        add_notes(1, 14, [{"start": 0.0, "length": 15.6, "pitch": 58, "velocity": 66}])
        add_notes(1, 22, make_voiceled_chords(4, vel=36, mute_last_beat=True))
        place(1, 7, 1)

        # M5-12 Refrain 1 (Ch 13 BFNK Full 8m, cut beat 4 M12 + subtle Ch 22 Pad support)
        p_bfnk_ref1 = new_pattern("BFNK Refrain 1 8m", 8)
        add_notes(p_bfnk_ref1, 13, [{"start": 0.0, "length": 30.85, "pitch": 58, "velocity": 108}])
        add_notes(p_bfnk_ref1, 22, make_voiceled_chords(8, vel=42, mute_last_beat=True))
        place(p_bfnk_ref1, 7, 5)

        # M13-16 Pont (Ch 15 BFNK Chop 4m + Ch 22 Chords Synth Pad swell)
        p_bfnk_pont = new_pattern("BFNK Pont Chop 4m", 4)
        add_notes(p_bfnk_pont, 15, [{"start": 0.0, "length": 14.85, "pitch": 58, "velocity": 72}])
        add_notes(p_bfnk_pont, 22, make_voiceled_chords(4, vel=44, mute_last_beat=True))
        place(p_bfnk_pont, 7, 13)

        # M17-24 Refrain 2 Climax (Ch 13 BFNK Full 8m + subtle Ch 22 Pad support)
        p_bfnk_climax = new_pattern("BFNK Climax 8m", 8)
        add_notes(p_bfnk_climax, 13, [{"start": 0.0, "length": 30.85, "pitch": 58, "velocity": 112}])
        add_notes(p_bfnk_climax, 22, make_voiceled_chords(8, vel=46, mute_last_beat=True))
        place(p_bfnk_climax, 7, 17)

        # M25-28 Outro (Ch 14 BFNK A Filt 4m + Ch 22 soft fade)
        p_bfnk_outro = new_pattern("BFNK Outro 4m", 4)
        add_notes(p_bfnk_outro, 14, [{"start": 0.0, "length": 15.6, "pitch": 58, "velocity": 86}])
        add_notes(p_bfnk_outro, 22, make_voiceled_chords(4, vel=46, mute_last_beat=False))
        place(p_bfnk_outro, 7, 25)

        # Track 8: BFNK High+12 Shimmer (M17-24) & Radiate Riser (M4, M16) & FX (4)_3 (M12, M24)
        p_bfnk_hi = new_pattern("BFNK High+12 Climax 8m", 8)
        add_notes(p_bfnk_hi, 16, [{"start": 0.0, "length": 30.85, "pitch": 58, "velocity": 96}])
        place(p_bfnk_hi, 8, 17)

        p_riser = new_pattern("Radiate Riser 1m", 1)
        add_notes(p_riser, 17, [{"start": 0.0, "length": 3.92, "pitch": 58, "velocity": 94}])
        place(p_riser, 8, 4)
        place(p_riser, 8, 16)

        p_fx_hit = new_pattern("FX (4)_3 Transition", 1)
        add_notes(p_fx_hit, 20, [{"start": 3.0, "length": 0.90, "pitch": 70, "velocity": 82}])
        place(p_fx_hit, 8, 12)
        place(p_fx_hit, 8, 24)

        # ---- LEAD SYNTH (Track 9, Ch 21) ----
        def make_lead(bars: int = 8, vel: int = 96, oct_shift: int = 0, mute_last_beat: bool = True, sparse: bool = False) -> list[dict]:
            notes = []
            for rep in range(bars // 4):
                off = rep * 4
                for b, st, ln, pt in lead_phrase:
                    ab = off + b
                    t = ab * 4.0 + st
                    if mute_last_beat and ab == bars - 1 and t >= bars * 4.0 - 1.25:
                        continue
                    if sparse and st not in (0.0, 2.0):
                        continue
                    jv = max(48, min(116, vel + random.randint(-5, 5)))
                    notes.append({"start": t, "length": ln, "pitch": pt + oct_shift, "velocity": jv})
            return notes

        p_lead_intro = new_pattern("Lead Synth - Intro 4m", 4)
        intro_notes = make_lead(4, vel=56, mute_last_beat=True, sparse=False)
        intro_notes.append({"start": 14.75, "length": 0.35, "pitch": 73, "velocity": 54})  # Db5 bridge -> Bb4
        add_notes(p_lead_intro, 21, intro_notes)
        place(p_lead_intro, 9, 1)

        p_lead_ref1 = new_pattern("Lead Synth - Refrain 1 8m", 8)
        ref1_notes = make_lead(8, vel=98, mute_last_beat=True)
        ref1_notes.append({"start": 30.75, "length": 0.35, "pitch": 73, "velocity": 86})  # smooth bridge
        add_notes(p_lead_ref1, 21, ref1_notes)
        place(p_lead_ref1, 9, 5)

        # Pont: Lead rests on M13-14 so VOX shines, plays soft cadence pickup on M15-16
        p_lead_pont = new_pattern("Lead Synth - Pont 4m", 4)
        add_notes(p_lead_pont, 21, [
            {"start": 8.0, "length": 0.75, "pitch": 73, "velocity": 72},
            {"start": 9.0, "length": 0.75, "pitch": 75, "velocity": 74},
            {"start": 10.0, "length": 0.90, "pitch": 77, "velocity": 76},
            {"start": 12.0, "length": 0.75, "pitch": 78, "velocity": 78},
            {"start": 13.0, "length": 0.75, "pitch": 77, "velocity": 76},
            {"start": 14.0, "length": 0.75, "pitch": 73, "velocity": 72},
        ])
        place(p_lead_pont, 9, 13)

        p_lead_climax = new_pattern("Lead Synth - Climax 8m", 8)
        climax_notes = make_lead(8, vel=108, mute_last_beat=True)
        climax_notes.append({"start": 30.75, "length": 0.35, "pitch": 73, "velocity": 92})
        add_notes(p_lead_climax, 21, climax_notes)
        place(p_lead_climax, 9, 17)

        p_lead_outro = new_pattern("Lead Synth - Outro 4m", 4)
        add_notes(p_lead_outro, 21, [
            {"start": 0.0, "length": 0.90, "pitch": 70, "velocity": 72},
            {"start": 1.0, "length": 0.90, "pitch": 73, "velocity": 70},
            {"start": 2.0, "length": 1.20, "pitch": 77, "velocity": 68},
            {"start": 4.0, "length": 0.90, "pitch": 75, "velocity": 64},
            {"start": 5.0, "length": 0.90, "pitch": 73, "velocity": 62},
            {"start": 6.0, "length": 1.40, "pitch": 70, "velocity": 60},
            {"start": 8.0, "length": 2.00, "pitch": 70, "velocity": 54},
        ])
        place(p_lead_outro, 9, 25)

        # ---- VOX AMBIENT (Track 10, Ch 18 VOX Virtual Bb4 & Ch 19 VOX ReUsed F5) ----
        # Strictly in Lead Synth's rests (beat 3.0) or during Pont!
        p_vox_intro = new_pattern("VOX Ambient - Intro 4m", 4)
        add_notes(p_vox_intro, 18, [
            {"start": 3.0, "length": 0.90, "pitch": 70, "velocity": 72},   # Bb4
            {"start": 11.0, "length": 0.90, "pitch": 73, "velocity": 74},  # Db5
        ])
        place(p_vox_intro, 10, 1)

        p_vox_ref1 = new_pattern("VOX Ambient - Refrain 1 8m", 8)
        add_notes(p_vox_ref1, 18, [
            {"start": 3.0, "length": 0.85, "pitch": 70, "velocity": 74},
            {"start": 11.0, "length": 0.85, "pitch": 73, "velocity": 76},
            {"start": 19.0, "length": 0.85, "pitch": 70, "velocity": 76},
            {"start": 27.0, "length": 0.85, "pitch": 77, "velocity": 78},
        ])
        place(p_vox_ref1, 10, 5)

        p_vox_pont = new_pattern("VOX Ambient - Pont 4m", 4)
        add_notes(p_vox_pont, 18, [
            {"start": 0.0, "length": 1.80, "pitch": 70, "velocity": 82},   # Bb4
            {"start": 4.0, "length": 1.80, "pitch": 72, "velocity": 84},   # C5 (9th over Absus2)
        ])
        add_notes(p_vox_pont, 19, [
            {"start": 2.5, "length": 1.10, "pitch": 77, "velocity": 78},   # F5
            {"start": 6.5, "length": 1.10, "pitch": 77, "velocity": 80},
            {"start": 14.5, "length": 0.90, "pitch": 77, "velocity": 76},
        ])
        place(p_vox_pont, 10, 13)

        p_vox_climax = new_pattern("VOX Ambient - Climax 8m", 8)
        add_notes(p_vox_climax, 18, [
            {"start": 3.0, "length": 0.85, "pitch": 70, "velocity": 82},
            {"start": 19.0, "length": 0.85, "pitch": 73, "velocity": 84},
        ])
        add_notes(p_vox_climax, 19, [
            {"start": 11.0, "length": 0.85, "pitch": 77, "velocity": 80},
            {"start": 27.0, "length": 0.85, "pitch": 77, "velocity": 82},
        ])
        place(p_vox_climax, 10, 17)

        p_vox_outro = new_pattern("VOX Ambient - Outro 4m", 4)
        add_notes(p_vox_outro, 18, [
            {"start": 3.0, "length": 1.50, "pitch": 70, "velocity": 68},
            {"start": 11.0, "length": 2.20, "pitch": 70, "velocity": 58},
        ])
        place(p_vox_outro, 10, 25)

        # ---- 808 PLUGGG (Track 1 Ch 3 Ref1, Track 2 Ch 4 Climax) ----
        # Bb1=34, Ab1=32, Gb1=30, F1=29, Db2=37, Bb2=46, Ab2=44
        def make_808(climax: bool = False) -> list[dict]:
            notes = []
            for b in range(8):
                m4 = b % 4
                is_last = (b == 7)
                if m4 == 0:
                    hits = [
                        (0.0, 1.40, 34, 112 if climax else 108, False),
                        (1.5, 1.15, 34, 102, False),
                        (2.75, 1.10, 46 if (climax or b == 4) else 34, 102, True),
                    ]
                elif m4 == 1:
                    hits = [
                        (0.0, 1.40, 32, 110 if climax else 106, False),
                        (1.5, 1.15, 32, 100, False),
                        (2.75, 1.10, 44 if climax and b == 5 else 32, 100, True),
                    ]
                elif m4 == 2:
                    hits = [
                        (0.0, 1.40, 30, 112 if climax else 108, False),
                        (1.5, 1.15, 30, 102, False),
                        (2.75, 1.10, 37 if climax else 30, 102, True),
                    ]
                else:
                    if is_last:
                        hits = [(0.0, 1.40, 30, 108, False), (1.5, 0.90, 29, 102, False)]  # cut beat 4!
                    else:
                        hits = [
                            (0.0, 1.40, 30, 108, False),
                            (2.0, 1.00, 29, 104, False),
                            (3.25, 0.65, 37 if climax else 29, 98, True),
                        ]
                for st, ln, pt, vel, porta in hits:
                    n: dict = {"start": b * 4.0 + st, "length": ln, "pitch": pt, "velocity": vel}
                    if porta:
                        n["porta"] = True
                    notes.append(n)
            return notes

        p_808_ref1 = new_pattern("808 Pluggg - Refrain 1 8m", 8)
        add_notes(p_808_ref1, 3, make_808(climax=False))
        place(p_808_ref1, 1, 5)

        p_808_climax = new_pattern("808 Pluggg - Climax 8m", 8)
        add_notes(p_808_climax, 4, make_808(climax=True))
        place(p_808_climax, 2, 17)

        # ---- KICKS (Track 3, Ch 1 + Ch 2) ----
        def make_kicks(bars: int = 8, dense: bool = False, mute_last_beat: bool = True) -> list[dict]:
            notes = []
            for b in range(bars):
                m4 = b % 4
                if b == bars - 1 and mute_last_beat:
                    times = [0.0, 1.5]
                elif m4 == 3:
                    times = [0.0, 2.0, 3.25] if dense else [0.0, 2.0]
                elif b % 2 == 0:
                    times = [0.0, 1.5, 2.75]
                else:
                    times = [0.0, 1.5, 2.75, 3.5] if dense else [0.0, 1.5, 2.75]
                for t in times:
                    notes.append({"start": b * 4.0 + t, "length": 0.35, "pitch": 60, "velocity": 112 if t == 0.0 else 102})
            return notes

        p_kick_ref1 = new_pattern("Kick - Refrain 1 8m", 8)
        k_r1 = make_kicks(8, dense=False, mute_last_beat=True)
        add_notes(p_kick_ref1, 1, k_r1)
        add_notes(p_kick_ref1, 2, k_r1)
        place(p_kick_ref1, 3, 5)

        p_kick_climax = new_pattern("Kick - Climax 8m", 8)
        k_cl = make_kicks(8, dense=True, mute_last_beat=True)
        add_notes(p_kick_climax, 1, k_cl)
        add_notes(p_kick_climax, 2, k_cl)
        place(p_kick_climax, 3, 17)

        # ---- CLAP & SNARE (Track 4, Ch 5 Clap Boring, Ch 6 Clap Native, Ch 7 Snare Native) ----
        def make_clap_snare(bars: int = 8, fill_end: bool = True, ghosts: bool = True, layer_native: bool = False):
            cl, cl_nat, sn = [], [], []
            for b in range(bars):
                cl.append({"start": b * 4.0 + 2.0, "length": 0.40, "pitch": 60, "velocity": 106})
                if layer_native:
                    cl_nat.append({"start": b * 4.0 + 2.0, "length": 0.35, "pitch": 60, "velocity": 86})
                if ghosts and b in (1, 3, 5):
                    sn.append({"start": b * 4.0 + 3.75, "length": 0.22, "pitch": 60, "velocity": 62})
                    sn.append({"start": b * 4.0 + 1.75, "length": 0.18, "pitch": 60, "velocity": 54})
                if fill_end and b == bars - 1:
                    for sub, vel in [(3.0, 72), (3.25, 82), (3.5, 92), (3.75, 102)]:
                        sn.append({"start": b * 4.0 + sub, "length": 0.18, "pitch": 60, "velocity": vel})
            return cl, cl_nat, sn

        p_cs_ref1 = new_pattern("Clap/Snare - Refrain 1 8m", 8)
        cl1, _, sn1 = make_clap_snare(8, fill_end=True, ghosts=True, layer_native=False)
        add_notes(p_cs_ref1, 5, cl1)
        add_notes(p_cs_ref1, 7, sn1)
        place(p_cs_ref1, 4, 5)

        p_cs_pont = new_pattern("Snare - Pont Buildup 4m", 4)
        psn = [{"start": b * 4.0 + 2.0, "length": 0.30, "pitch": 60, "velocity": 50 + b * 10} for b in range(3)]
        for k in range(6):
            psn.append({"start": 3 * 4.0 + k * 0.5, "length": 0.18, "pitch": 60, "velocity": 52 + k * 9})
        add_notes(p_cs_pont, 7, psn)
        place(p_cs_pont, 4, 13)

        p_cs_climax = new_pattern("Clap/Snare - Climax 8m", 8)
        cl3, cln3, sn3 = make_clap_snare(8, fill_end=False, ghosts=True, layer_native=True)
        add_notes(p_cs_climax, 5, cl3)
        add_notes(p_cs_climax, 6, cln3)
        add_notes(p_cs_climax, 7, sn3)
        place(p_cs_climax, 4, 17)

        # ---- HI-HATS CLOSED (Track 5, Ch 8 Finesse Hat & Ch 9 Hat Native) ----
        p_hh_intro = new_pattern("Hi-Hat Native - Intro M3-4", 2)
        add_notes(p_hh_intro, 9, [
            {"start": b * 4.0 + s * 0.5, "length": 0.14, "pitch": 60, "velocity": 54 if s % 2 == 0 else 38}
            for b in range(2) for s in range(8) if not (b == 1 and s >= 6)
        ])
        place(p_hh_intro, 5, 3)

        def make_hats(bars: int = 8, high: bool = False) -> list[dict]:
            hn = []
            for b in range(bars):
                for s in range(8):
                    t = b * 4.0 + s * 0.5
                    if b == bars - 1 and s >= 6:
                        continue
                    vel = (104 if s % 2 == 0 else 68) + random.randint(-5, 5)
                    if b % 2 == 1 and s == 3:
                        for k in range(3):
                            hn.append({"start": round(b * 4.0 + 1.5 + k * (0.5 / 3.0), 4), "length": 0.10, "pitch": 60 - k * 2, "velocity": 82 + k * 6})
                        continue
                    if high and b % 2 == 1 and s in (6, 7):
                        for k in range(4):
                            hn.append({"start": round(b * 4.0 + 3.0 + k * 0.25, 4), "length": 0.10, "pitch": 60 - k * 2, "velocity": 92 - k * 7})
                        continue
                    hn.append({"start": t, "length": 0.16, "pitch": 60, "velocity": vel})
                    if high and s in (1, 5):
                        hn.append({"start": t + 0.25, "length": 0.12, "pitch": 60, "velocity": 72})
            return hn

        p_hh_ref1 = new_pattern("Hi-Hat Finesse - Refrain 1 8m", 8)
        add_notes(p_hh_ref1, 8, make_hats(8, high=False))
        place(p_hh_ref1, 5, 5)

        p_hh_pont = new_pattern("Hi-Hat Native - Pont 4m", 4)
        add_notes(p_hh_pont, 9, [
            {"start": b * 4.0 + s * 0.5, "length": 0.14, "pitch": 60, "velocity": 76 if s % 2 == 0 else 50}
            for b in range(3) for s in range(8)
        ])
        place(p_hh_pont, 5, 13)

        p_hh_climax = new_pattern("Hi-Hat Finesse - Climax 8m", 8)
        add_notes(p_hh_climax, 8, make_hats(8, high=True))
        place(p_hh_climax, 5, 17)

        # ---- OPEN HAT, CYMBAL & PERC 14 (Track 6, Ch 10, 11, 12) ----
        p_perc_ref1 = new_pattern("OpenHat, Cym & Perc 14 - Refrain 1 8m", 8)
        add_notes(p_perc_ref1, 11, [{"start": 0.0, "length": 1.80, "pitch": 60, "velocity": 76}])  # OG Cym impact M5
        add_notes(p_perc_ref1, 10, [{"start": b * 4.0 + 0.5, "length": 0.25, "pitch": 60, "velocity": 78} for b in (0, 4)])
        add_notes(p_perc_ref1, 12, [{"start": b * 4.0 + t, "length": 0.25, "pitch": 60, "velocity": 78} for b in range(7) for t in (1.25, 3.5)])
        place(p_perc_ref1, 6, 5)

        p_perc_pont = new_pattern("Perc 14 - Pont 4m", 4)
        add_notes(p_perc_pont, 12, [{"start": b * 4.0 + 1.5, "length": 0.28, "pitch": 60, "velocity": 68} for b in range(3)])
        place(p_perc_pont, 6, 13)

        p_perc_climax = new_pattern("OpenHat, Cym & Perc 14 - Climax 8m", 8)
        add_notes(p_perc_climax, 11, [{"start": 0.0, "length": 1.80, "pitch": 60, "velocity": 82}, {"start": 16.0, "length": 1.80, "pitch": 60, "velocity": 80}])
        add_notes(p_perc_climax, 10, [{"start": b * 4.0 + 0.5, "length": 0.26, "pitch": 60, "velocity": 84} for b in (0, 2, 4, 6)])
        add_notes(p_perc_climax, 12, [{"start": b * 4.0 + t, "length": 0.24, "pitch": 60, "velocity": 82} for b in range(7) for t in (1.25, 2.75, 3.5)])
        place(p_perc_climax, 6, 17)

        # ---- FX ROOM & LEAD SYNTH SECTION AUTOMATION (Track 10) ----
        pts = [{"position": 0.0, "value": 0.30, "shape": "hold"}]
        for p_bar in [4, 12, 16, 24]:
            pts.append({"position": (p_bar - 1) * 4.0 + 3.0, "value": 0.0, "shape": "hold"})
            pts.append({"position": p_bar * 4.0, "value": 0.30, "shape": "hold"})
        pts.append({"position": 28 * 4.0, "value": 0.30, "shape": "hold"})
        queued.append({"tool": "add_automation", "arguments": {
            "mixer": 20, "param": "volume", "name": "FX Room Pivot Cuts",
            "points": pts, "track": 10, "start": 0.0, "length": 28 * 4.0,
        }})

        # Lead Synth dynamic arc: Intro(0.54) -> Refrain1(0.96) -> Pont(0.56) -> Climax(1.00) -> Outro(0.52)
        lead_pts = [
            {"position": 0.0, "value": 0.54, "shape": "hold"},
            {"position": 4 * 4.0, "value": 0.96, "shape": "hold"},
            {"position": 12 * 4.0, "value": 0.56, "shape": "hold"},
            {"position": 16 * 4.0, "value": 1.00, "shape": "hold"},
            {"position": 24 * 4.0, "value": 0.52, "shape": "hold"},
            {"position": 28 * 4.0, "value": 0.52, "shape": "hold"},
        ]
        queued.append({"tool": "add_automation", "arguments": {
            "mixer": 12, "param": "volume", "name": "Lead Synth Dynamic Arc",
            "points": lead_pts, "track": 9, "start": 0.0, "length": 28 * 4.0,
        }})

        mcp.batch(queued)
        mcp.call("save_project", {})
        mcp.call("export_project", {"format": "midi", "path": ORCH_MIDI})
        song = mcp.call("get_song", {})
        print("Built active7/projet.gridline:", json.dumps({
            "title": song["title"],
            "tempo": song["tempo"],
            "lengthBars": song["lengthBars"],
            "lengthSeconds": song["lengthSeconds"],
            "channels": len(song["channels"]),
            "patterns": song["counts"]["patterns"],
            "clips": song["counts"]["clips"],
        }, indent=2))
    finally:
        mcp.close()

    audit = audit_midi_file(ORCH_MIDI, expected_key="Bb minor", min_pitch=24, max_pitch=84, strict_musicality=True)
    print("Full Orchestrated MIDI Audit:", json.dumps({
        "verdict": audit["musicality_verdict"],
        "gate_passed": audit["gate_passed"],
        "metrics": audit["musicality_metrics"],
        "roles": audit["musicality_detected_roles"],
        "failures": audit["musicality_failures"],
        "minor_2nd_clashes": audit["minor_2nd_clashes"],
        "low_end_collisions": audit["low_end_simultaneous_collisions"],
    }, indent=2))


if __name__ == "__main__":
    build()
