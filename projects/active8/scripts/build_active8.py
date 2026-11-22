#!/usr/bin/env python3
"""
Build script for projects/active8/projet.gridline
"Simba d'Ea'alah" — 64 bars @ 144 BPM (72 BPM half-time feel, ~106.7 s), E minor.
PNL (Simba)-inspired French Cloud Rap built exclusively from Ea'alah (Family).mp3's
extracted stems & derived textures + tuned NIKOS/TriSamples drums & deep round 808.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys

ROOT = "/home/gamo/Bureau/musicos"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/mcp-servers/audio-analyzer")

from deep_diagnostics import audit_midi_file
from musicality_metrics import optimize_voice_leading
from scripts.studio_lib import GridlineMCP, TEMPLATE_EMPTY_MIXER, setup_standard_buses

PROJ_DIR = f"{ROOT}/projects/active8"
PROJ_FILE = f"{PROJ_DIR}/projet.gridline"
ORCH_MIDI = f"{PROJ_DIR}/simba_eaalah.mid"
RENDERS_DIR = f"{PROJ_DIR}/renders"
SAMPLES_DIR = f"{PROJ_DIR}/samples"

BPM = 144.0
random.seed(808)


def build() -> None:
    # Voice-led chords from blueprint harmonic_core (Em7 -> Cadd9/C -> Am7 -> Bm7)
    # Voiced cleanly to avoid B3-C4 / B3-C5 minor-2nd/9th clashes!
    voicings = [
        [52, 55, 59, 62],  # Em7:   E3, G3, B3, D4
        [52, 55, 60, 62],  # Cadd9: E3, G3, C4, D4 (ultra cloud-rap, B3->C4 smooth!)
        [52, 55, 57, 60],  # Am7:   E3, G3, A3, C4
        [50, 54, 57, 59],  # Bm7:   D3, F#3, A3, B3
    ]

    shutil.copyfile(TEMPLATE_EMPTY_MIXER, PROJ_FILE)
    mcp = GridlineMCP(PROJ_FILE, RENDERS_DIR, client_name="build_active8")

    try:
        # 1. Song settings
        mcp.call("set_song", {
            "title": "Simba d'Ea'alah",
            "genre": "French Cloud Rap",
            "author": "AI-Studio",
            "tempo": BPM,
            "swing": 0.04,
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
        # 1..5 -> BUS Drums (17)
        # 6 -> BUS 808 (18)
        # 7..13 -> BUS Music (19)
        # 20 -> FX Cloud Verb -> Master (0)
        mixer_defs = [
            (1, "Kick Tight", 17, 0.76, 0.0, 0.0),
            (2, "Soft Clap", 17, 0.88, 0.0, 0.0),
            (3, "Andromeda Snare", 17, 0.74, 0.0, 0.0),
            (4, "Cloud Hat", 17, 0.82, 0.08, 0.0),
            (5, "Chrome Open Hat", 17, 0.82, -0.14, 0.0),
            (6, "Deep 808 PNL", 18, 0.84, 0.0, -1.0),
            (7, "Ea'alah Pad Chords", 19, 0.68, -0.05, -0.15),
            (8, "Ea'alah Pad Bed", 19, 0.66, 0.0, -0.12),
            (9, "Ea'alah Guitar Loop", 19, 0.82, 0.04, -0.08),
            (10, "Ea'alah Shimmer +12", 19, 0.74, -0.08, 0.22),
            (11, "Ea'alah Guitar Lead", 19, 0.88, 0.05, -0.06),
            (12, "Ea'alah Vocal Clouds", 19, 0.76, -0.06, 0.16),
            (13, "Ea'alah Counter C5 & FX", 19, 0.76, 0.12, 0.20),
            (17, "BUS Drums", 0, 0.96, 0.0, 0.0),
            (18, "BUS 808", 0, 0.78, 0.0, -1.0),
            (19, "BUS Music", 0, 0.96, 0.0, -0.08),
            (20, "FX Room", 0, 0.36, 0.0, 0.0),
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

        # Sidechains & Lush FX Cloud Verb sends
        sends = [
            (1, 18, 1.0, True),   # Kick -> BUS 808 sidechain
            (1, 19, 0.55, True),  # Kick -> BUS Music micro-ducking
            (2, 20, 0.22, False), # Soft Clap -> lush PNL reverb tail
            (3, 20, 0.16, False), # Snare -> reverb
            (5, 20, 0.10, False), # Open Hat -> reverb
            (7, 20, 0.18, False), # Pad Chords -> reverb
            (8, 20, 0.16, False), # Pad Bed -> reverb
            (9, 20, 0.14, False), # Guitar Loop -> reverb
            (10, 20, 0.22, False),# Guitar Shimmer -> reverb
            (11, 20, 0.18, False),# Guitar Lead -> reverb
            (12, 20, 0.26, False),# Vocal Clouds -> spectral reverb
            (13, 20, 0.28, False),# Counter C5 -> spectral reverb
        ]
        mcp.batch([{"tool": "set_send", "arguments": {"mixer": m, "to": to, "level": lvl, "sidechain": sc}} for m, to, lvl, sc in sends])

        # 3. Channels 1..16 (naming aligned with musicality_metrics._infer_roles)
        channels_spec = [
            # 1..5 Drums
            ("Kick Tight", "gl.sampler", 1, f"{SAMPLES_DIR}/kick_tight.wav", 0.80, 0.0),
            ("Soft Clap", "gl.sampler", 2, f"{SAMPLES_DIR}/clap_soft.wav", 0.86, 0.0),
            ("Andromeda Snare", "gl.sampler", 3, f"{SAMPLES_DIR}/snare_andromeda.wav", 0.74, 0.0),
            ("Cloud Hat", "gl.sampler", 4, f"{SAMPLES_DIR}/hat_cloud.wav", 0.76, 0.06),
            ("Chrome Open Hat", "gl.sampler", 5, f"{SAMPLES_DIR}/openhat_chrome.wav", 0.68, -0.14),
            # 6 Deep 808
            ("Deep 808 PNL", "gl.sampler", 6, f"{SAMPLES_DIR}/808_deep_pnl.wav", 0.86, 0.0),
            # 7..16 Ea'alah Stems & Derived Instruments
            ("Ea'alah Pad Chords", "gl.sampler", 7, f"{SAMPLES_DIR}/eaalah_pad_single.wav", 0.76, -0.05),
            ("Ea'alah Pad Bed FX", "gl.sampler", 8, f"{SAMPLES_DIR}/eaalah_pad_loop.wav", 0.78, 0.0),
            ("Ea'alah Guitar Loop A FX", "gl.sampler", 9, f"{SAMPLES_DIR}/eaalah_guitar_loop.wav", 0.84, 0.04),
            ("Ea'alah Guitar Loop B FX", "gl.sampler", 9, f"{SAMPLES_DIR}/eaalah_guitar_loop_b.wav", 0.84, 0.04),
            ("Ea'alah Shimmer FX", "gl.sampler", 10, f"{SAMPLES_DIR}/eaalah_guitar_shimmer.wav", 0.68, -0.08),
            ("Ea'alah Guitar Lead", "gl.sampler", 11, f"{SAMPLES_DIR}/eaalah_guitar_pluck.wav", 0.86, 0.05),
            ("Ea'alah Vocal Cloud A FX", "gl.sampler", 12, f"{SAMPLES_DIR}/eaalah_vox_cloud_a.wav", 0.78, -0.06),
            ("Ea'alah Vocal Cloud B FX", "gl.sampler", 12, f"{SAMPLES_DIR}/eaalah_vox_cloud_b.wav", 0.78, 0.06),
            ("Ea'alah Counter C5", "gl.sampler", 13, f"{SAMPLES_DIR}/eaalah_vox_note_c5.wav", 0.76, 0.12),
            ("Ea'alah Riser FX", "gl.sampler", 13, f"{SAMPLES_DIR}/eaalah_riser_rev.wav", 0.72, 0.0),
        ]

        mcp.batch([
            {"tool": "add_channel", "arguments": {"name": name, "plugin": plug, "mixer": mixer, "volume": vol, "pan": pan}}
            for name, plug, mixer, _, vol, pan in channels_spec
        ])
        mcp.batch([
            {"tool": "patch", "arguments": {"path": f"channel/{idx}", "set": {"plugin": {"data": spath}}}}
            for idx, (_, _, _, spath, _, _) in enumerate(channels_spec, 1)
        ])

        # 4. Channel tuning, ADSR & monophonic 808 settings
        ch_params = [
            # Ch 1 Kick Tight (decay ~135ms)
            ("channel/1/One shot", 0), ("channel/1/Decay", 0.14), ("channel/1/Sustain", 0.0), ("channel/1/Release", 0.025), ("channel/1/Declick", 2),
            # Ch 2 Clap & Ch 3 Snare
            ("channel/2/One shot", 0), ("channel/2/Release", 0.030), ("channel/2/Declick", 2),
            ("channel/3/One shot", 0), ("channel/3/Release", 0.025), ("channel/3/Declick", 2),
            # Ch 4 Cloud Hat (short 50ms tick!) & Ch 5 Open Hat
            ("channel/4/One shot", 0), ("channel/4/Decay", 0.055), ("channel/4/Sustain", 0.0), ("channel/4/Release", 0.015), ("channel/4/Declick", 2),
            ("channel/5/One shot", 0), ("channel/5/Release", 0.040), ("channel/5/Declick", 2),
            # Ch 6 Deep 808 PNL: Root 24 (C1), Fine -41 ct (measured 33.49 Hz = C1 +41.3ct), Voices=1 Cut Itself!
            ("channel/6/Root note", 24), ("channel/6/Fine tune", -41), ("channel/6/Voices", 1),
            ("channel/6/One shot", 0), ("channel/6/Attack", 0.003), ("channel/6/Decay", 10.0),
            ("channel/6/Sustain", 1.0), ("channel/6/Release", 0.055), ("channel/6/Glide", 0.045),
            ("channel/6/Remove DC", 1), ("channel/6/Declick", 2),
            # Ch 7 Ea'alah Pad Chords: Root 55 (G3), Fine -16 ct (measured 197.76 Hz = G3 +15.5ct)
            ("channel/7/Root note", 55), ("channel/7/Fine tune", -16), ("channel/7/Voices", 12),
            ("channel/7/One shot", 0), ("channel/7/Attack", 0.090), ("channel/7/Decay", 4.0),
            ("channel/7/Sustain", 0.85), ("channel/7/Release", 0.25), ("channel/7/Declick", 2),
            # Ch 8..11 Loops (triggered on B3=59, already tuned to 440.0 Hz)
            ("channel/8/Root note", 59), ("channel/8/One shot", 0), ("channel/8/Attack", 0.040), ("channel/8/Release", 0.20), ("channel/8/Declick", 2),
            ("channel/9/Root note", 59), ("channel/9/One shot", 0), ("channel/9/Attack", 0.015), ("channel/9/Release", 0.14), ("channel/9/Declick", 2),
            ("channel/10/Root note", 59), ("channel/10/One shot", 0), ("channel/10/Attack", 0.015), ("channel/10/Release", 0.14), ("channel/10/Declick", 2),
            ("channel/11/Root note", 59), ("channel/11/One shot", 0), ("channel/11/Attack", 0.030), ("channel/11/Release", 0.18), ("channel/11/Declick", 2),
            # Ch 12 Ea'alah Guitar Lead Pluck: Root 59 (B3), Fine -21 ct (measured 250.02 Hz = B3 +21.4ct)
            ("channel/12/Root note", 59), ("channel/12/Fine tune", -21), ("channel/12/Voices", 6),
            ("channel/12/One shot", 0), ("channel/12/Attack", 0.003), ("channel/12/Decay", 1.2),
            ("channel/12/Sustain", 0.35), ("channel/12/Release", 0.18), ("channel/12/Declick", 2),
            # Ch 13..14 Ea'alah Vocal Cloud Loops (triggered on B3=59)
            ("channel/13/Root note", 59), ("channel/13/Fine tune", -2), ("channel/13/One shot", 0), ("channel/13/Attack", 0.035), ("channel/13/Release", 0.20), ("channel/13/Declick", 2),
            ("channel/14/Root note", 59), ("channel/14/Fine tune", -2), ("channel/14/One shot", 0), ("channel/14/Attack", 0.035), ("channel/14/Release", 0.20), ("channel/14/Declick", 2),
            # Ch 15 Ea'alah Counter C5: Root 72 (C5), Fine -1 ct (measured 523.43 Hz = C5 +0.6ct)
            ("channel/15/Root note", 72), ("channel/15/Fine tune", -1), ("channel/15/Voices", 6),
            ("channel/15/One shot", 0), ("channel/15/Attack", 0.045), ("channel/15/Decay", 2.0),
            ("channel/15/Sustain", 0.72), ("channel/15/Release", 0.28), ("channel/15/Declick", 2),
            # Ch 16 Ea'alah Riser FX
            ("channel/16/Root note", 59), ("channel/16/One shot", 0), ("channel/16/Attack", 0.020), ("channel/16/Release", 0.04), ("channel/16/Declick", 2),
        ]
        mcp.set_params(ch_params)

        # 5. Standard Buses + Selective Insert EQs
        setup_standard_buses(
            mcp,
            bpm=BPM,
            limiter_ceiling_db=-2.45,
            limiter_gain_db=6.2,
            fx_room_vol=0.36,
        )

        insert_fx = [
            {"tool": "add_plugin", "arguments": {"mixer": 1, "plugin": "gl.eq"}},   # Kick
            {"tool": "add_plugin", "arguments": {"mixer": 2, "plugin": "gl.eq"}},   # Clap
            {"tool": "add_plugin", "arguments": {"mixer": 4, "plugin": "gl.eq"}},   # Cloud Hat
            {"tool": "add_plugin", "arguments": {"mixer": 6, "plugin": "gl.dist"}}, # Subtle 808 warm harmonic lift
            {"tool": "add_plugin", "arguments": {"mixer": 7, "plugin": "gl.eq"}},   # Pad Chords
            {"tool": "add_plugin", "arguments": {"mixer": 8, "plugin": "gl.eq"}},   # Pad Bed
            {"tool": "add_plugin", "arguments": {"mixer": 9, "plugin": "gl.eq"}},   # Guitar Loop
            {"tool": "add_plugin", "arguments": {"mixer": 11, "plugin": "gl.eq"}},  # Guitar Lead
            {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.eq"}},  # Vocal Clouds
            {"tool": "add_plugin", "arguments": {"mixer": 13, "plugin": "gl.eq"}},  # Counter C5 & FX
        ]
        mcp.batch(insert_fx)

        fx_changes = [
            # Master: preserve 3-5 dB Cloud Rap dynamic arc
            ("mixer/0/slot/1/Band 7 freq", 17500),
            ("mixer/0/slot/2/Threshold", -9.5), ("mixer/0/slot/2/Ratio", 1.55), ("mixer/0/slot/2/Makeup", 1.3),

            # FX Cloud Verb (Decay = T_noire = 0.415s, warm airy sheen up to 7.8 kHz)
            ("mixer/20/slot/1/Room size", 0.44),
            ("mixer/20/slot/1/Decay time", 0.415),
            ("mixer/20/slot/1/Low cut", 450),
            ("mixer/20/slot/1/High cut", 7800),
            ("mixer/20/slot/1/Wet", 0.52), ("mixer/20/slot/1/Wet (full)", 52),
            ("mixer/20/slot/1/Width", 0.86),

            # BUS Drums: gentle 160 Hz control + clean 10 kHz air shelf
            ("mixer/17/slot/1/Band 3 type", 6), ("mixer/17/slot/1/Band 3 freq", 160), ("mixer/17/slot/1/Band 3 gain", -2.2), ("mixer/17/slot/1/Band 3 Q (legacy)", 1.2),
            ("mixer/17/slot/1/Band 7 type", 7), ("mixer/17/slot/1/Band 7 freq", 10000), ("mixer/17/slot/1/Band 7 gain", 2.4),

            # BUS 808: smooth 63 Hz resonance (-2.2 dB) & carve 160 Hz (-3.2 dB)
            ("mixer/18/slot/3/Band 2 type", 6), ("mixer/18/slot/3/Band 2 freq", 63), ("mixer/18/slot/3/Band 2 gain", -2.2), ("mixer/18/slot/3/Band 2 Q (legacy)", 1.2),
            ("mixer/18/slot/3/Band 3 type", 6), ("mixer/18/slot/3/Band 3 freq", 160), ("mixer/18/slot/3/Band 3 gain", -3.2), ("mixer/18/slot/3/Band 3 Q (legacy)", 1.3),

            # BUS Music: carve low_mids (-3.4 dB @ 330 Hz), add smooth nocturnal air sheen (+2.4 dB @ 8 kHz), soft Glue Threshold -14 dB
            ("mixer/19/slot/1/Band 1 freq", 165),
            ("mixer/19/slot/1/Band 3 type", 6), ("mixer/19/slot/1/Band 3 freq", 330), ("mixer/19/slot/1/Band 3 gain", -3.4), ("mixer/19/slot/1/Band 3 Q (legacy)", 1.0),
            ("mixer/19/slot/1/Band 6 type", 7), ("mixer/19/slot/1/Band 6 freq", 8000), ("mixer/19/slot/1/Band 6 gain", 2.4),
            ("mixer/19/slot/1/Band 7 freq", 16000),
            ("mixer/19/slot/2/Mix", 0.06),
            ("mixer/19/slot/3/Threshold", -14.0), ("mixer/19/slot/3/Makeup", 1.3),

            # Kick (mixer 1): tight 63 Hz control + click @ 2.5 kHz
            ("mixer/1/slot/1/Band 1 type", 1), ("mixer/1/slot/1/Band 1 freq", 34),
            ("mixer/1/slot/1/Band 2 type", 6), ("mixer/1/slot/1/Band 2 freq", 63), ("mixer/1/slot/1/Band 2 gain", -1.6),
            ("mixer/1/slot/1/Band 3 type", 6), ("mixer/1/slot/1/Band 3 freq", 160), ("mixer/1/slot/1/Band 3 gain", -2.6),
            ("mixer/1/slot/1/Band 5 type", 6), ("mixer/1/slot/1/Band 5 freq", 2500), ("mixer/1/slot/1/Band 5 gain", 1.6),

            # Soft Clap (mixer 2): open top air
            ("mixer/2/slot/1/Band 1 type", 1), ("mixer/2/slot/1/Band 1 freq", 210),
            ("mixer/2/slot/1/Band 7 type", 7), ("mixer/2/slot/1/Band 7 freq", 7500), ("mixer/2/slot/1/Band 7 gain", 3.2),

            # Cloud Hat (mixer 4): silky air without sharp HF spike
            ("mixer/4/slot/1/Band 1 type", 1), ("mixer/4/slot/1/Band 1 freq", 650),
            ("mixer/4/slot/1/Band 5 type", 6), ("mixer/4/slot/1/Band 5 freq", 7200), ("mixer/4/slot/1/Band 5 gain", 1.4),
            ("mixer/4/slot/1/Band 7 type", 7), ("mixer/4/slot/1/Band 7 freq", 9500), ("mixer/4/slot/1/Band 7 gain", 1.8),

            # Deep 808 PNL (mixer 6): super gentle 2nd/3rd harmonic warmth
            ("mixer/6/slot/1/Mode", 0), ("mixer/6/slot/1/Drive", 1.8), ("mixer/6/slot/1/Tone", 2400),
            ("mixer/6/slot/1/Mix", 0.09), ("mixer/6/slot/1/Oversampling", 2), ("mixer/6/slot/1/Remove DC", 1),

            # Pad Chords (mixer 7) & Pad Bed (mixer 8): carve 1.8 kHz vocal pocket & 320 Hz mud
            ("mixer/7/slot/1/Band 1 type", 1), ("mixer/7/slot/1/Band 1 freq", 185),
            ("mixer/7/slot/1/Band 3 type", 6), ("mixer/7/slot/1/Band 3 freq", 320), ("mixer/7/slot/1/Band 3 gain", -3.2),
            ("mixer/7/slot/1/Band 5 type", 6), ("mixer/7/slot/1/Band 5 freq", 1800), ("mixer/7/slot/1/Band 5 gain", -2.5),
            ("mixer/7/slot/1/Band 7 type", 3), ("mixer/7/slot/1/Band 7 freq", 3800),

            ("mixer/8/slot/1/Band 1 type", 1), ("mixer/8/slot/1/Band 1 freq", 185),
            ("mixer/8/slot/1/Band 3 type", 6), ("mixer/8/slot/1/Band 3 freq", 320), ("mixer/8/slot/1/Band 3 gain", -3.2),
            ("mixer/8/slot/1/Band 5 type", 6), ("mixer/8/slot/1/Band 5 freq", 1800), ("mixer/8/slot/1/Band 5 gain", -2.2),
            ("mixer/8/slot/1/Band 7 type", 3), ("mixer/8/slot/1/Band 7 freq", 3200),

            # Guitar Loop (mixer 9)
            ("mixer/9/slot/1/Band 1 type", 1), ("mixer/9/slot/1/Band 1 freq", 195),
            ("mixer/9/slot/1/Band 3 type", 6), ("mixer/9/slot/1/Band 3 freq", 340), ("mixer/9/slot/1/Band 3 gain", -2.8),
            ("mixer/9/slot/1/Band 5 type", 6), ("mixer/9/slot/1/Band 5 freq", 1800), ("mixer/9/slot/1/Band 5 gain", -1.8),
            ("mixer/9/slot/1/Band 7 type", 3), ("mixer/9/slot/1/Band 7 freq", 9500),

            # Guitar Lead (mixer 11)
            ("mixer/11/slot/1/Band 1 type", 1), ("mixer/11/slot/1/Band 1 freq", 220),
            ("mixer/11/slot/1/Band 7 type", 3), ("mixer/11/slot/1/Band 7 freq", 9800),

            # Vocal Clouds (mixer 12): spectral underwater band
            ("mixer/12/slot/1/Band 1 type", 1), ("mixer/12/slot/1/Band 1 freq", 360),
            ("mixer/12/slot/1/Band 7 type", 3), ("mixer/12/slot/1/Band 7 freq", 5200),

            # Counter C5 & FX (mixer 13)
            ("mixer/13/slot/1/Band 1 type", 1), ("mixer/13/slot/1/Band 1 freq", 340),
            ("mixer/13/slot/1/Band 7 type", 3), ("mixer/13/slot/1/Band 7 freq", 6800),
        ]
        mcp.set_params(fx_changes)

        # 6. Playlist Tracks 1..10
        track_names = [
            (1, "Deep 808 PNL"),
            (2, "Kick Tight"),
            (3, "Soft Clap & Andromeda Snare"),
            (4, "Cloud Hat & Chrome Open Hat"),
            (5, "Ea'alah Pad Bed (8m)"),
            (6, "Ea'alah Pad Chords (Voice-Led)"),
            (7, "Ea'alah Guitar Loops & Shimmer"),
            (8, "Ea'alah Guitar Lead (A-A'-A''-B)"),
            (9, "Ea'alah Vocal Clouds & Counter C5"),
            (10, "Ea'alah Risers & Automations"),
        ]
        mcp.batch([{"tool": "patch", "arguments": {"path": f"track/{t}", "set": {"name": nm}}} for t, nm in track_names])

        # 7. Patterns & Clips across 64 bars @ 144 BPM
        pat_counter = 1
        queued: list[dict] = [{"tool": "edit_pattern", "arguments": {"pattern": 1, "name": "Ea'alah Pad Bed - Intro 8m", "bars": 8}}]

        def new_pattern(name: str, bars: int) -> int:
            nonlocal pat_counter
            pat_counter += 1
            queued.append({"tool": "add_pattern", "arguments": {"name": name, "bars": bars}})
            return pat_counter

        def add_notes(pat: int, ch: int, notes: list[dict]) -> None:
            queued.append({"tool": "set_notes", "arguments": {"pattern": pat, "channel": ch, "notes": notes}})

        def place(pat: int, track: int, start_bar: int) -> None:
            queued.append({"tool": "add_clip", "arguments": {"pattern": pat, "track": track, "start": (start_bar - 1) * 4.0}})

        # Helpers
        def make_chords(bars: int = 8, vel: int = 70, mute_last_beat: bool = True) -> list[dict]:
            notes = []
            for b in range(bars):
                ln = 2.85 if (mute_last_beat and b == bars - 1) else 3.75
                for vp in voicings[b % 4]:
                    notes.append({"start": b * 4.0, "length": ln, "pitch": vp, "velocity": vel + random.randint(-4, 4)})
            return notes

        phrase_lead = [
            # Bar 0 (Em7)
            (0.0, 0.75, 71, 86), (0.75, 0.75, 67, 80), (1.5, 0.50, 66, 74), (2.0, 1.00, 67, 82),
            # Bar 1 (Cmaj7)
            (4.0, 0.75, 67, 84), (4.75, 0.75, 64, 78), (5.5, 0.50, 62, 72), (6.0, 1.00, 64, 80),
            # Bar 2 (Am7 -> Apex E5=76)
            (8.0, 0.75, 67, 86), (8.75, 0.50, 69, 82), (9.25, 0.75, 72, 88), (10.0, 0.75, 76, 94), (10.75, 0.50, 74, 82),
            # Bar 3 (Bm7)
            (12.0, 0.75, 74, 84), (12.75, 0.75, 71, 80), (13.5, 0.50, 69, 76), (14.0, 0.85, 71, 78),
        ]

        def make_lead(bars: int = 8, vel_scale: float = 1.0, sparse: bool = False, mute_last_beat: bool = True) -> list[dict]:
            notes = []
            cycles = bars // 4
            for cyc in range(cycles):
                for idx, (st, ln, p, v) in enumerate(phrase_lead):
                    bar_in_cyc = int(st // 4.0)
                    # In sparse mode, play full bars 0 & 2 and let bars 1 & 3 breathe (keeps 100% stepwise motion!)
                    if sparse and bar_in_cyc in (1, 3) and idx % 2 == 1:
                        continue
                    abs_st = cyc * 16.0 + st
                    if mute_last_beat and cyc == cycles - 1 and st >= 14.0:
                        continue
                    pitch_adj = 71 if (cyc == cycles - 1 and abs(st - 14.0) < 0.01) else p
                    # Slight length boost in sparse/dreamy sections so Lead fills ~60-65% (rest_ratio ~0.35-0.40)
                    dur = ln * (1.25 if not sparse else 1.15)
                    notes.append({
                        "start": abs_st,
                        "length": min(dur, 1.45),
                        "pitch": pitch_adj,
                        "velocity": max(42, min(120, int(round(v * vel_scale)) + random.randint(-4, 4))),
                    })
            return notes

        def make_808(bars: int = 8, climax: bool = False, laid_back: bool = False) -> list[dict]:
            # E1=28, C2=36, A1=33, B1=35 (all 41.2..65.4 Hz); octave bounces E2=40, A2=45
            roots = [28, 36, 33, 35]
            notes = []
            for b in range(bars):
                r = roots[b % 4]
                is_last = (b == bars - 1)
                if laid_back:
                    # Spacious Cloud Rap Couplet: long notes, gentler level so voice breathes
                    if is_last:
                        notes.append({"start": b * 4.0, "length": 2.60, "pitch": r, "velocity": 94})
                    else:
                        notes.append({"start": b * 4.0, "length": 2.40, "pitch": r, "velocity": 96})
                        if b % 2 == 1:
                            notes.append({"start": b * 4.0 + 2.75, "length": 1.15, "pitch": r, "velocity": 88})
                else:
                    if is_last:
                        notes.append({"start": b * 4.0, "length": 1.40, "pitch": r, "velocity": 110})
                        notes.append({"start": b * 4.0 + 1.5, "length": 1.10, "pitch": r, "velocity": 102})
                    else:
                        notes.append({"start": b * 4.0, "length": 1.40, "pitch": r, "velocity": 112})
                        notes.append({"start": b * 4.0 + 1.5, "length": 1.15, "pitch": r, "velocity": 102})
                        oct_bounce = (climax and b in (3, 7))
                        p_end = r + 12 if oct_bounce else r
                        n = {"start": b * 4.0 + 2.75, "length": 1.15, "pitch": p_end, "velocity": 104 if oct_bounce else 98}
                        if oct_bounce:
                            n["porta"] = True
                        notes.append(n)
            return notes

        def make_kicks(bars: int = 8, laid_back: bool = False) -> list[dict]:
            notes = []
            for b in range(bars):
                is_last = (b == bars - 1)
                if is_last:
                    times = [0.0, 1.5]
                elif laid_back:
                    times = [0.0, 2.75] if b % 2 == 1 else [0.0]
                else:
                    times = [0.0, 1.5, 2.75]
                for t in times:
                    notes.append({"start": b * 4.0 + t, "length": 0.35, "pitch": 60, "velocity": 110 if t == 0.0 else 98})
            return notes

        def make_hats(bars: int = 8, rolls: bool = False) -> list[dict]:
            notes = []
            for b in range(bars):
                for s in range(8):
                    if b == bars - 1 and s >= 6:
                        continue
                    t = b * 4.0 + s * 0.5
                    # Non-frantic PNL Cloud Rap hi-hats: crisp but smooth
                    vel = (94 if s % 2 == 0 else 66) + random.randint(-4, 4)
                    if rolls and b % 4 == 3 and s == 3:
                        for k in range(3):
                            notes.append({"start": round(b * 4.0 + 1.5 + k * (0.5 / 3.0), 4), "length": 0.12, "pitch": 60 - k * 2, "velocity": 82 - k * 6})
                        continue
                    if rolls and b == bars - 1 and s == 5:
                        for k in range(3):
                            notes.append({"start": round(b * 4.0 + 2.5 + k * (0.5 / 3.0), 4), "length": 0.12, "pitch": 60 - k * 2, "velocity": 86 - k * 6})
                        continue
                    notes.append({"start": t, "length": 0.16, "pitch": 60, "velocity": vel})
            return notes

        # ---- PATTERNS & PLACEMENT ----
        # 1. Pad Bed (Track 5, Ch 8, 8-bar loop)
        add_notes(1, 8, [{"start": 0.0, "length": 31.8, "pitch": 59, "velocity": 78}])
        for sb in [1, 33, 57]:
            place(1, 5, sb)

        # 2. Pad Chords Voice-Led (Track 6, Ch 7)
        p_ch_coup = new_pattern("Pad Chords - Couplet 8m", 8)
        add_notes(p_ch_coup, 7, make_chords(8, vel=64, mute_last_beat=True))
        place(p_ch_coup, 6, 9)
        place(p_ch_coup, 6, 17)

        p_ch_ref = new_pattern("Pad Chords - Refrain 8m", 8)
        add_notes(p_ch_ref, 7, make_chords(8, vel=76, mute_last_beat=True))
        place(p_ch_ref, 6, 25)
        place(p_ch_ref, 6, 41)
        place(p_ch_ref, 6, 49)

        p_ch_pont = new_pattern("Pad Chords - Pont 8m", 8)
        add_notes(p_ch_pont, 7, make_chords(8, vel=68, mute_last_beat=True))
        place(p_ch_pont, 6, 33)

        # 3. Guitar Loops & Shimmer (Track 7, Ch 9, 10, 11)
        p_g_loop = new_pattern("Ea'alah Guitar Loop A+B 8m", 8)
        add_notes(p_g_loop, 9, [{"start": 0.0, "length": 15.8, "pitch": 59, "velocity": 82}])
        add_notes(p_g_loop, 10, [{"start": 16.0, "length": 15.8, "pitch": 59, "velocity": 82}])
        for sb in [1, 25, 41, 49, 57]:
            place(p_g_loop, 7, sb)

        p_g_shim = new_pattern("Ea'alah Shimmer +12 8m", 8)
        add_notes(p_g_shim, 11, [
            {"start": 0.0, "length": 15.8, "pitch": 59, "velocity": 76},
            {"start": 16.0, "length": 15.0, "pitch": 59, "velocity": 76},
        ])
        place(p_g_shim, 7, 41)
        place(p_g_shim, 7, 49)

        # 4. Guitar Lead Pluck (Track 8, Ch 12)
        p_lead_intro = new_pattern("Guitar Lead - Intro 4m", 4)
        add_notes(p_lead_intro, 12, make_lead(4, vel_scale=0.68, sparse=False, mute_last_beat=True))
        place(p_lead_intro, 8, 5)

        p_lead_coup = new_pattern("Guitar Lead - Couplet 8m", 8)
        add_notes(p_lead_coup, 12, make_lead(8, vel_scale=0.84, sparse=False, mute_last_beat=True))
        place(p_lead_coup, 8, 9)
        place(p_lead_coup, 8, 17)

        p_lead_ref = new_pattern("Guitar Lead - Refrain 8m", 8)
        add_notes(p_lead_ref, 12, make_lead(8, vel_scale=1.0, sparse=False, mute_last_beat=True))
        place(p_lead_ref, 8, 25)
        place(p_lead_ref, 8, 41)
        place(p_lead_ref, 8, 49)

        p_lead_pont = new_pattern("Guitar Lead - Pont 8m", 8)
        add_notes(p_lead_pont, 12, make_lead(8, vel_scale=0.78, sparse=False, mute_last_beat=True))
        place(p_lead_pont, 8, 33)

        p_lead_outro = new_pattern("Guitar Lead - Outro 4m", 4)
        add_notes(p_lead_outro, 12, make_lead(4, vel_scale=0.62, sparse=False, mute_last_beat=True))
        place(p_lead_outro, 8, 57)

        # 5. Vocal Clouds & Counter Harmony C5 (Track 9, Ch 13, 14, 15)
        # Counter C5 placed ONLY in Lead's rests (beat 3.0..3.85 of bars 1, 3, 5, 7) -> 0 overlap!
        p_vox_intro = new_pattern("Vocal Cloud A - Intro 8m", 8)
        add_notes(p_vox_intro, 13, [{"start": 0.0, "length": 15.8, "pitch": 59, "velocity": 74}])
        add_notes(p_vox_intro, 15, [
            {"start": 19.0, "length": 0.85, "pitch": 71, "velocity": 68},  # B4
            {"start": 27.0, "length": 0.85, "pitch": 74, "velocity": 70},  # D5
        ])
        place(p_vox_intro, 9, 1)

        p_vox_ref = new_pattern("Vocal Cloud B + Counter - Refrain 8m", 8)
        add_notes(p_vox_ref, 14, [{"start": 0.0, "length": 15.8, "pitch": 59, "velocity": 82}])
        add_notes(p_vox_ref, 13, [{"start": 16.0, "length": 15.0, "pitch": 59, "velocity": 80}])
        add_notes(p_vox_ref, 15, [
            {"start": 7.0, "length": 0.85, "pitch": 72, "velocity": 72},   # C5 in Lead rest
            {"start": 15.0, "length": 0.85, "pitch": 71, "velocity": 72},  # B4 in Lead rest
            {"start": 23.0, "length": 0.85, "pitch": 74, "velocity": 76},  # D5 in Lead rest
        ])
        place(p_vox_ref, 9, 25)
        place(p_vox_ref, 9, 41)
        place(p_vox_ref, 9, 49)

        p_vox_pont = new_pattern("Vocal Harmony - Pont 8m", 8)
        add_notes(p_vox_pont, 14, [{"start": 16.0, "length": 15.0, "pitch": 59, "velocity": 74}])
        add_notes(p_vox_pont, 15, [
            {"start": 7.0, "length": 0.85, "pitch": 72, "velocity": 74},
            {"start": 15.0, "length": 0.85, "pitch": 71, "velocity": 76},
            {"start": 23.0, "length": 0.85, "pitch": 76, "velocity": 78},
        ])
        place(p_vox_pont, 9, 33)

        p_vox_outro = new_pattern("Vocal Cloud A - Outro 8m", 8)
        add_notes(p_vox_outro, 13, [{"start": 0.0, "length": 15.8, "pitch": 59, "velocity": 66}])
        add_notes(p_vox_outro, 15, [{"start": 19.0, "length": 1.20, "pitch": 71, "velocity": 60}])
        place(p_vox_outro, 9, 57)

        # 6. Deep 808 PNL (Track 1, Ch 6)
        p_808_coup = new_pattern("808 Deep - Couplet 8m", 8)
        add_notes(p_808_coup, 6, make_808(8, climax=False, laid_back=True))
        place(p_808_coup, 1, 9)
        place(p_808_coup, 1, 17)

        p_808_ref = new_pattern("808 Deep - Refrain 8m", 8)
        add_notes(p_808_ref, 6, make_808(8, climax=False, laid_back=False))
        place(p_808_ref, 1, 25)
        place(p_808_ref, 1, 41)

        p_808_clim = new_pattern("808 Deep - Climax 8m", 8)
        add_notes(p_808_clim, 6, make_808(8, climax=True, laid_back=False))
        place(p_808_clim, 1, 49)

        # 7. Kick Tight (Track 2, Ch 1)
        p_kick_coup = new_pattern("Kick - Couplet 8m", 8)
        add_notes(p_kick_coup, 1, make_kicks(8, laid_back=True))
        place(p_kick_coup, 2, 9)
        place(p_kick_coup, 2, 17)

        p_kick_ref = new_pattern("Kick - Refrain 8m", 8)
        add_notes(p_kick_ref, 1, make_kicks(8, laid_back=False))
        place(p_kick_ref, 2, 25)
        place(p_kick_ref, 2, 41)
        place(p_kick_ref, 2, 49)

        # 8. Soft Clap & Andromeda Snare (Track 3, Ch 2, Ch 3)
        p_cs_coup = new_pattern("Clap - Couplet 8m", 8)
        add_notes(p_cs_coup, 2, [{"start": b * 4.0 + 2.0, "length": 0.45, "pitch": 60, "velocity": 96} for b in range(8)])
        place(p_cs_coup, 3, 9)
        place(p_cs_coup, 3, 17)

        p_cs_ref = new_pattern("Clap + Snare - Refrain 8m", 8)
        add_notes(p_cs_ref, 2, [{"start": b * 4.0 + 2.0, "length": 0.45, "pitch": 60, "velocity": 106} for b in range(8)])
        add_notes(p_cs_ref, 3, [
            {"start": b * 4.0 + 2.0, "length": 0.40, "pitch": 60, "velocity": 78} for b in range(8)
        ] + [
            {"start": 7 * 4.0 + t, "length": 0.22, "pitch": 60, "velocity": v}
            for t, v in [(3.0, 62), (3.5, 74)]
        ])
        place(p_cs_ref, 3, 25)
        place(p_cs_ref, 3, 41)
        place(p_cs_ref, 3, 49)

        p_cs_outro = new_pattern("Clap - Outro Fade 2m", 2)
        add_notes(p_cs_outro, 2, [{"start": b * 4.0 + 2.0, "length": 0.45, "pitch": 60, "velocity": 76 - b * 18} for b in range(2)])
        place(p_cs_outro, 3, 57)

        # 9. Cloud Hat & Chrome Open Hat (Track 4, Ch 4, Ch 5)
        p_hh_coup = new_pattern("Cloud Hat - Couplet 8m", 8)
        add_notes(p_hh_coup, 4, make_hats(8, rolls=False))
        place(p_hh_coup, 4, 9)
        place(p_hh_coup, 4, 17)

        p_hh_ref = new_pattern("Cloud Hat + Open Hat - Refrain 8m", 8)
        add_notes(p_hh_ref, 4, make_hats(8, rolls=True))
        add_notes(p_hh_ref, 5, [{"start": b * 4.0 + 0.5, "length": 0.35, "pitch": 60, "velocity": 72} for b in (0, 4)])
        place(p_hh_ref, 4, 25)
        place(p_hh_ref, 4, 41)
        place(p_hh_ref, 4, 49)

        p_hh_outro = new_pattern("Cloud Hat - Outro 2m", 2)
        add_notes(p_hh_outro, 4, make_hats(2, rolls=False))
        place(p_hh_outro, 4, 57)

        # 10. Risers (Track 10, Ch 16)
        p_riser = new_pattern("Ea'alah Riser FX 1m", 1)
        add_notes(p_riser, 16, [{"start": 0.0, "length": 3.9, "pitch": 59, "velocity": 84}])
        for sb in [8, 24, 40]:
            place(p_riser, 10, sb)

        # 11. Automations: FX Room Pivot Cuts & Dynamic Section Arc (BUS Music + FX Room)
        pts_room = [{"position": 0.0, "value": 0.36, "shape": "hold"}]
        for p_bar in [8, 16, 24, 32, 40, 56]:
            pts_room.append({"position": (p_bar - 1) * 4.0 + 3.0, "value": 0.0, "shape": "hold"})
            pts_room.append({"position": p_bar * 4.0, "value": 0.36, "shape": "hold"})
        pts_room.append({"position": 64 * 4.0, "value": 0.36, "shape": "hold"})
        queued.append({"tool": "add_automation", "arguments": {
            "mixer": 20, "param": "volume", "name": "FX Room Pivot Cuts",
            "points": pts_room, "track": 10, "start": 0.0, "length": 64 * 4.0,
        }})

        # Dynamic section arc on BUS Music (Intro 0.62 -> Couplet 0.74 -> Refrain 0.98 -> Pont 0.56 -> Climax 1.00 -> Outro 0.54 -> fade)
        bus_music_pts = [
            {"position": 0.0, "value": 0.62, "shape": "hold"},
            {"position": 8 * 4.0, "value": 0.74, "shape": "hold"},
            {"position": 24 * 4.0, "value": 0.98, "shape": "hold"},
            {"position": 32 * 4.0, "value": 0.56, "shape": "hold"},
            {"position": 40 * 4.0, "value": 1.00, "shape": "hold"},
            {"position": 56 * 4.0, "value": 0.54, "shape": "hold"},
            {"position": 62 * 4.0, "value": 0.40, "shape": "curve"},
            {"position": 64 * 4.0, "value": 0.10, "shape": "curve"},
        ]
        queued.append({"tool": "add_automation", "arguments": {
            "mixer": 19, "param": "volume", "name": "BUS Music Dynamic Arc",
            "points": bus_music_pts, "track": 10, "start": 0.0, "length": 64 * 4.0,
        }})

        mcp.batch(queued)
        mcp.call("save_project", {})
        mcp.call("export_project", {"format": "midi", "path": ORCH_MIDI})
        song = mcp.call("get_song", {})
        print("Built active8/projet.gridline:", json.dumps({
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

    audit = audit_midi_file(ORCH_MIDI, expected_key="E minor", min_pitch=24, max_pitch=84, strict_musicality=True)
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