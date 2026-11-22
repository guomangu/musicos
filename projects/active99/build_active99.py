#!/usr/bin/env python3
"""
Build full 64-bar French Cloud Rap (PNL *Simba* aesthetic) project:
`projects/active99/projet.gridline` @ 142 BPM (71 BPM felt), C minor.
100% sourced/derived from `projects/active99/samples/`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = "/home/gamo/Bureau/musicos"
PROJ_DIR = f"{ROOT}/projects/active99"
DERIVED_DIR = f"{PROJ_DIR}/samples/derived"
OUT_PROJ = f"{PROJ_DIR}/projet.gridline"
EXPORT_DIR = f"{PROJ_DIR}/exports"

sys.path.insert(0, f"{ROOT}/scripts")
from studio_lib import GridlineMCP, TEMPLATE_EMPTY_MIXER, setup_standard_buses, verify_midi_gate


def main() -> None:
    os.makedirs(EXPORT_DIR, exist_ok=True)
    shutil.copy2(TEMPLATE_EMPTY_MIXER, OUT_PROJ)

    bp = json.load(open(f"{PROJ_DIR}/song_blueprint.json", encoding="utf-8"))
    prog = bp["harmonic_core"]["chord_progression"]

    mcp = GridlineMCP(OUT_PROJ, EXPORT_DIR, client_name="active99_builder")
    try:
        # 1. Song metadata
        mcp.call("set_song", {
            "tempo": 142,
            "swing": 0.04,
            "title": "Lune de Tigre (Simba Cloud)",
            "genre": "French Cloud Rap",
            "comments": "PNL Simba inspired Cloud Rap @ 142 BPM (71 BPM half-time), C minor",
        })

        # 2. Ensure mixer tracks 17..20 exist (BUS Drums, BUS 808, BUS Music, FX Room)
        for name in ["BUS Drums", "BUS 808", "BUS Music", "FX Room"]:
            mcp.call("add_mixer_track", {"name": name})

        setup_standard_buses(
            mcp,
            bpm=142.0,
            bus_drums=17,
            bus_808=18,
            bus_music=19,
            fx_room=20,
            limiter_ceiling_db=-2.50,
            limiter_gain_db=5.5,
            fx_room_vol=0.28,
        )

        # Preserve real dynamic arc between Intro/Pont and Drops
        mcp.set_params([
            ("mixer/0/slot/2/Threshold", -10.8),
            ("mixer/0/slot/2/Ratio", 1.7),
            ("mixer/0/slot/2/Makeup", 2.0),
            ("mixer/0/slot/3/Ceiling", -1.75),
            ("mixer/0/slot/3/Attack (look-ahead)", 0.002),
            ("mixer/0/slot/3/Gain", 11.0),
            ("mixer/18/slot/2/Threshold", -18.5),
            ("mixer/18/slot/2/Ratio", 3.5),
            ("mixer/18/slot/3/Band 2 freq", 56),
            ("mixer/18/slot/3/Band 2 gain", -5.8),
            ("mixer/19/slot/3/Threshold", -14.5),
            ("mixer/19/slot/3/Ratio", 2.0),
            ("mixer/19/slot/3/Makeup", 2.2),
            ("mixer/19/slot/1/Band 4 freq", 2100),
            ("mixer/19/slot/1/Band 4 gain", 2.2),
            ("mixer/17/slot/1/Band 5 freq", 6300),
            ("mixer/17/slot/1/Band 5 gain", -2.2),
            ("mixer/17/slot/1/Band 7 freq", 9200),
            ("mixer/17/slot/1/Band 7 gain", 2.8),
            # Keep FX Room focused above 650 Hz so low_mid_corr >= 0.75
            ("mixer/20/slot/1/Low cut", 650),
            ("mixer/20/slot/1/Low cut (full)", 650),
            ("mixer/20/slot/1/Width", 0.55),
        ])

        # 3. Add 14 channels (exact order 1..14)
        channel_specs = [
            ("Kick Tight", 1, f"{DERIVED_DIR}/kick_tight_clean.wav"),
            ("Clap Soft", 2, f"{DERIVED_DIR}/clap_soft_clean.wav"),
            ("Snare Andromeda", 3, f"{DERIVED_DIR}/snare_andromeda_clean.wav"),
            ("Hi-Hat Cloud", 4, f"{DERIVED_DIR}/hat_cloud_clean.wav"),
            ("Open-Hat Chrome", 5, f"{DERIVED_DIR}/openhat_chrome_clean.wav"),
            ("Perc Ghost", 6, f"{DERIVED_DIR}/perc_ghost_clean.wav"),
            ("808 Deep PNL", 7, f"{DERIVED_DIR}/808_deep_pnl_440.wav"),
            ("Pad Cloud Chords", 8, f"{DERIVED_DIR}/pad_cloud_g3_440.wav"),
            ("Glass Pad Bed FX", 9, f"{DERIVED_DIR}/glass_pad_142.wav"),
            ("PNL AutoTune Lead", 10, f"{DERIVED_DIR}/vox_lead_c5_440.wav"),
            ("Counter Pluck", 11, f"{DERIVED_DIR}/guitar_pluck_b3_440.wav"),
            ("Counter Echo", 12, f"{DERIVED_DIR}/vox_virtual_bb4_440.wav"),
            ("Vocal Swell FX", 13, f"{DERIVED_DIR}/vox_swell_1bar_142.wav"),
            ("Glass Riser FX", 14, f"{DERIVED_DIR}/glass_riser_1bar_142.wav"),
        ]
        for cname, mx, _ in channel_specs:
            mcp.call("add_channel", {"plugin": "gl.sampler", "name": cname, "mixer": mx})

        mcp.batch([
            {"tool": "patch", "arguments": {"path": f"channel/{i+1}", "set": {"plugin": {"data": sp}}}}
            for i, (_, _, sp) in enumerate(channel_specs)
        ])

        # 4. Configure sampler parameters (Root note, Fine tune, ADSR, Loop, Voices, Glide)
        mcp.set_params([
            ("channel/1/1", 60), ("channel/1/10", 1), ("channel/1/46", 1),
            ("channel/2/1", 60), ("channel/2/10", 1),
            ("channel/3/1", 60), ("channel/3/10", 1), ("channel/3/pan", -0.05),
            ("channel/4/1", 60), ("channel/4/10", 1), ("channel/4/6", 0.045), ("channel/4/pan", 0.08),
            ("channel/5/1", 60), ("channel/5/10", 1), ("channel/5/pan", -0.14),
            ("channel/6/1", 60), ("channel/6/10", 1), ("channel/6/pan", 0.16),
            # 808 Deep PNL: C1 +14.2ct -> Root=24, Fine=-14, Mono Cut-Itself + Glide
            ("channel/7/1", 24), ("channel/7/2", -14), ("channel/7/10", 0),
            ("channel/7/3", 0.002), ("channel/7/4", 0.25), ("channel/7/5", 1.0), ("channel/7/6", 0.055),
            ("channel/7/46", 1), ("channel/7/47", 0.060),
            # Pad Cloud Chords: G3 -> Root=55, Fine=+4 (exact A4=440Hz harmonic center)
            ("channel/8/1", 55), ("channel/8/2", 4), ("channel/8/10", 0),
            ("channel/8/3", 0.14), ("channel/8/4", 0.40), ("channel/8/5", 0.90), ("channel/8/6", 0.55),
            ("channel/8/9", 1), ("channel/8/15", 1), ("channel/8/13", 0.22), ("channel/8/14", 0.78), ("channel/8/22", 0.20),
            # Glass Pad Bed FX
            ("channel/9/1", 60), ("channel/9/2", -1), ("channel/9/10", 1),
            # PNL AutoTune Lead: C5 +11ct -> Root=72, Fine=-11, Mono Legato Glide + Loop
            ("channel/10/1", 72), ("channel/10/2", -11), ("channel/10/10", 0),
            ("channel/10/3", 0.035), ("channel/10/4", 0.30), ("channel/10/5", 0.92), ("channel/10/6", 0.22),
            ("channel/10/9", 1), ("channel/10/15", 1), ("channel/10/13", 0.25), ("channel/10/14", 0.75), ("channel/10/22", 0.18),
            ("channel/10/46", 1), ("channel/10/47", 0.065),
            # Counter Pluck: B3 -> Root=59, Fine=-2 (exact A4=440Hz across harmonics)
            ("channel/11/1", 59), ("channel/11/2", -2), ("channel/11/10", 1),
            ("channel/11/3", 0.002), ("channel/11/6", 0.18), ("channel/11/pan", -0.18),
            # Counter Echo: Bb4 -1.3ct -> Root=70, Fine=-1, Mono Glide + Loop
            ("channel/12/1", 70), ("channel/12/2", -1), ("channel/12/10", 0),
            ("channel/12/3", 0.045), ("channel/12/4", 0.30), ("channel/12/5", 0.88), ("channel/12/6", 0.28),
            ("channel/12/9", 1), ("channel/12/15", 1), ("channel/12/13", 0.25), ("channel/12/14", 0.75), ("channel/12/22", 0.18),
            ("channel/12/46", 1), ("channel/12/47", 0.055), ("channel/12/pan", 0.22),
            # Swells/Risers
            ("channel/13/1", 67), ("channel/13/2", 1), ("channel/13/10", 1),
            ("channel/14/1", 72), ("channel/14/2", -1), ("channel/14/10", 1),
        ])

        # 5. Route mixer tracks & initial levels
        mixer_routes = [
            (1, "Kick Tight", 17, 0.94, 0.0),
            (2, "Clap Soft", 17, 0.86, 0.0),
            (3, "Snare Andromeda", 17, 0.70, -0.05),
            (4, "Hi-Hat Cloud", 17, 0.88, 0.06),
            (5, "Open-Hat Chrome", 17, 0.80, -0.10),
            (6, "Perc Ghost", 17, 0.62, 0.12),
            (7, "808 Deep PNL", 18, 0.48, 0.0),
            (8, "Pad Cloud Chords", 19, 0.74, 0.0),
            (9, "Glass Pad Bed FX", 19, 0.34, 0.0),
            (10, "PNL AutoTune Lead", 19, 1.08, 0.0),
            (11, "Counter Pluck", 19, 0.84, -0.14),
            (12, "Counter Echo", 19, 0.78, 0.16),
            (13, "Vocal Swell FX", 19, 0.56, 0.0),
            (14, "Glass Riser FX", 19, 0.50, 0.0),
        ]
        for mx, name, route_to, vol, pan in mixer_routes:
            mcp.call("edit_mixer_track", {"mixer": mx, "name": name, "routeTo": route_to, "volume": vol, "pan": pan})

        mcp.batch([
            {"tool": "set_send", "arguments": {"mixer": 1, "to": 18, "level": 1.0, "preFader": True, "sidechain": True}},
            {"tool": "set_send", "arguments": {"mixer": 1, "to": 19, "level": 1.0, "preFader": True, "sidechain": True}},
            {"tool": "set_send", "arguments": {"mixer": 2, "to": 20, "level": 0.14}},
            {"tool": "set_send", "arguments": {"mixer": 3, "to": 20, "level": 0.18}},
            {"tool": "set_send", "arguments": {"mixer": 4, "to": 20, "level": 0.08}},
            {"tool": "set_send", "arguments": {"mixer": 5, "to": 20, "level": 0.14}},
            {"tool": "set_send", "arguments": {"mixer": 6, "to": 20, "level": 0.18}},
            {"tool": "set_send", "arguments": {"mixer": 8, "to": 20, "level": 0.26}},
            {"tool": "set_send", "arguments": {"mixer": 9, "to": 20, "level": 0.16}},
            {"tool": "set_send", "arguments": {"mixer": 10, "to": 20, "level": 0.18}},
            {"tool": "set_send", "arguments": {"mixer": 11, "to": 20, "level": 0.28}},
            {"tool": "set_send", "arguments": {"mixer": 12, "to": 20, "level": 0.26}},
            {"tool": "set_send", "arguments": {"mixer": 13, "to": 20, "level": 0.22}},
            {"tool": "set_send", "arguments": {"mixer": 14, "to": 20, "level": 0.18}},
        ])

        # 6. Insert FX (PNL AutoTune Vocal Chain + EQ Carving)
        mcp.batch([
            {"tool": "add_plugin", "arguments": {"mixer": 1, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 8, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 9, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.pitcher"}},
            {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.delay"}},
            {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.hall"}},
            {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.stereo"}},
            {"tool": "add_plugin", "arguments": {"mixer": 11, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.pitcher"}},
            {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.hall"}},
            {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.stereo"}},
        ])

        mcp.set_params([
            # Mixer 1 Kick EQ: clean 165 Hz boxiness
            ("mixer/1/slot/1/Band 1 type", 1), ("mixer/1/slot/1/Band 1 freq", 34),
            ("mixer/1/slot/1/Band 3 freq", 165), ("mixer/1/slot/1/Band 3 gain", -2.8),
            ("mixer/1/slot/1/Band 7 type", 3), ("mixer/1/slot/1/Band 7 freq", 4200),
            # Mixer 8 Pad EQ: keep warm 250-400 Hz cushion while carving 1.8k vocal pocket
            ("mixer/8/slot/1/Band 1 type", 1), ("mixer/8/slot/1/Band 1 freq", 145),
            ("mixer/8/slot/1/Band 3 freq", 280), ("mixer/8/slot/1/Band 3 gain", -2.2),
            ("mixer/8/slot/1/Band 5 freq", 1800), ("mixer/8/slot/1/Band 5 gain", -3.0),
            ("mixer/8/slot/1/Band 7 type", 3), ("mixer/8/slot/1/Band 7 freq", 4800),
            # Mixer 9 Glass Pad Bed EQ: carve 260 Hz gently
            ("mixer/9/slot/1/Band 1 type", 1), ("mixer/9/slot/1/Band 1 freq", 200),
            ("mixer/9/slot/1/Band 3 freq", 260), ("mixer/9/slot/1/Band 3 gain", -2.8),
            ("mixer/9/slot/1/Band 7 type", 3), ("mixer/9/slot/1/Band 7 freq", 5200),
            # Mixer 10 Slot 1: gl.pitcher (C Minor, fast 12ms glissando, airy +12st voice only, low spread)
            ("mixer/10/slot/1/1", 0), ("mixer/10/slot/1/2", 2),
            ("mixer/10/slot/1/3", 12.0), ("mixer/10/slot/1/4", 1.0), ("mixer/10/slot/1/8", 1),
            ("mixer/10/slot/1/9", 12), ("mixer/10/slot/1/10", 0.20), ("mixer/10/slot/1/11", -0.15),
            ("mixer/10/slot/1/12", -12), ("mixer/10/slot/1/13", 0.0), ("mixer/10/slot/1/14", 0.15),
            ("mixer/10/slot/1/36", 0.12),
            # Mixer 10 Slot 2: gl.eq (brighter PNL vocal presence + air, clean < 280 Hz)
            ("mixer/10/slot/2/Band 1 type", 1), ("mixer/10/slot/2/Band 1 freq", 280),
            ("mixer/10/slot/2/Band 5 freq", 2600), ("mixer/10/slot/2/Band 5 gain", 2.8),
            ("mixer/10/slot/2/Band 7 type", 3), ("mixer/10/slot/2/Band 7 freq", 9500),
            # Mixer 10 Slot 3: gl.delay (1/8. Normal stereo cloud echo above 850 Hz)
            ("mixer/10/slot/3/1", 1), ("mixer/10/slot/3/2", 8), ("mixer/10/slot/3/4", 0.28),
            ("mixer/10/slot/3/5", 0), ("mixer/10/slot/3/7", 4200), ("mixer/10/slot/3/8", 850),
            ("mixer/10/slot/3/12", 1.0), ("mixer/10/slot/3/13", 0.20),
            # Mixer 10 Slot 4: gl.hall (PNL cavern + shimmer, focused >850 Hz)
            ("mixer/10/slot/4/Input low cut", 850), ("mixer/10/slot/4/Input high cut", 11500),
            ("mixer/10/slot/4/Size", 0.72), ("mixer/10/slot/4/Decay", 2.2),
            ("mixer/10/slot/4/Dry level", 1.0), ("mixer/10/slot/4/Wet level", 0.34), ("mixer/10/slot/4/Width", 0.72),
            ("mixer/10/slot/4/Shimmer on", 1), ("mixer/10/slot/4/Shimmer pitch", 12), ("mixer/10/slot/4/Shimmer gain", 0.22),
            # Mixer 10 Slot 5: gl.stereo (Classic engine so Mono below 780 Hz is enforced!)
            ("mixer/10/slot/5/18", 0), ("mixer/10/slot/5/Width", 0.82), ("mixer/10/slot/5/13", 780),
            # Mixer 11 Pluck EQ
            ("mixer/11/slot/1/Band 1 type", 1), ("mixer/11/slot/1/Band 1 freq", 260),
            ("mixer/11/slot/1/Band 5 freq", 2200), ("mixer/11/slot/1/Band 5 gain", 1.5),
            ("mixer/11/slot/1/Band 7 type", 3), ("mixer/11/slot/1/Band 7 freq", 7200),
            # Mixer 12 Counter Echo: Pitcher (low spread) + EQ + Hall + Stereo (Classic Mono below 780 Hz)
            ("mixer/12/slot/1/1", 0), ("mixer/12/slot/1/2", 2), ("mixer/12/slot/1/3", 18.0), ("mixer/12/slot/1/4", 1.0),
            ("mixer/12/slot/1/36", 0.12),
            ("mixer/12/slot/2/Band 1 type", 1), ("mixer/12/slot/2/Band 1 freq", 300),
            ("mixer/12/slot/2/Band 7 type", 3), ("mixer/12/slot/2/Band 7 freq", 6800),
            ("mixer/12/slot/3/Input low cut", 850), ("mixer/12/slot/3/Input high cut", 6800),
            ("mixer/12/slot/3/Size", 0.75), ("mixer/12/slot/3/Decay", 2.3),
            ("mixer/12/slot/3/Dry level", 0.85), ("mixer/12/slot/3/Wet level", 0.34), ("mixer/12/slot/3/Width", 0.72),
            ("mixer/12/slot/4/18", 0), ("mixer/12/slot/4/Width", 0.82), ("mixer/12/slot/4/13", 780),
        ])

        # 7. Musical note helpers (using `start` per set_notes schema!)
        def mk_pad_notes(vel: int = 76) -> list[dict]:
            notes = []
            for bi, c in enumerate(prog):
                b0 = bi * 4.0
                for p in c["voicing_midi"]:
                    notes.append({"pitch": int(p), "start": b0, "length": 2.85, "velocity": vel})
            return notes

        phrase_bars = [
            [(0.0, 0.88, 75, 92, False), (1.0, 0.52, 74, 86, True), (1.5, 0.52, 72, 88, True), (2.0, 0.90, 70, 84, False)],
            [(0.0, 0.88, 72, 90, True), (1.0, 0.52, 74, 86, False), (1.5, 0.52, 75, 92, True), (2.0, 0.90, 75, 85, False)],
            [(0.0, 0.88, 77, 94, True), (1.0, 0.52, 79, 98, True), (1.5, 0.52, 77, 90, False), (2.0, 0.90, 75, 86, True)],
            [(0.0, 0.88, 74, 90, False), (1.0, 0.48, 72, 85, True), (1.5, 0.48, 70, 82, False), (2.0, 0.48, 72, 84, True), (2.5, 0.48, 74, 88, True)],
        ]

        def mk_lead_notes(vel_scale: float = 1.0, sparse: bool = False) -> list[dict]:
            notes = []
            for bi in range(4):
                b0 = bi * 4.0
                items = phrase_bars[bi]
                if sparse:
                    # Half-note + ending stepwise pickup (keeps 100% stepwise & strong-beat chord tones!)
                    items = [
                        (0.0, 1.35, items[0][2], items[0][3], False),
                        (2.0, 1.10, items[-1][2], items[-1][3], True),
                    ]
                for st, ln, p, v, porta in items:
                    n_obj: dict = {"pitch": p, "start": b0 + st, "length": ln, "velocity": max(45, min(115, int(v * vel_scale)))}
                    if porta:
                        n_obj["porta"] = True
                    notes.append(n_obj)
            return notes

        pluck_resp = [
            [(3.0, 0.45, 67, 74), (3.5, 0.40, 63, 66)],
            [(3.0, 0.45, 68, 72), (3.5, 0.40, 67, 68)],
            [(3.0, 0.45, 65, 74), (3.5, 0.40, 68, 68)],
            [(3.25, 0.35, 67, 72), (3.6, 0.35, 65, 66)],
        ]

        def mk_pluck_notes(vel_scale: float = 1.0, intro_only: bool = False) -> list[dict]:
            notes = []
            for bi in range(4):
                b0 = bi * 4.0
                if intro_only:
                    root_pc = prog[bi]["chord_tones_pc"][0]
                    fifth_pc = prog[bi]["chord_tones_pc"][2] if len(prog[bi]["chord_tones_pc"]) > 2 else root_pc
                    notes.append({"pitch": 60 + fifth_pc, "start": b0 + 1.5, "length": 0.45, "velocity": int(72 * vel_scale)})
                    notes.append({"pitch": 60 + root_pc, "start": b0 + 3.0, "length": 0.45, "velocity": int(64 * vel_scale)})
                else:
                    for st, ln, p, v in pluck_resp[bi]:
                        notes.append({"pitch": p, "start": b0 + st, "length": ln, "velocity": int(v * vel_scale)})
            return notes

        echo_resp = [
            [(3.0, 0.70, 72, 74)],
            [(3.0, 0.70, 75, 72)],
            [(3.0, 0.70, 72, 74)],
            [(3.0, 0.65, 70, 70)],
        ]

        def mk_echo_notes(vel_scale: float = 1.0) -> list[dict]:
            notes = []
            for bi in range(4):
                b0 = bi * 4.0
                for st, ln, p, v in echo_resp[bi]:
                    notes.append({"pitch": p, "start": b0 + st, "length": ln, "velocity": int(v * vel_scale)})
            return notes

        def mk_808_notes(mode: str = "refrain", mute_last_beat: bool = False) -> list[dict]:
            notes = []
            for bi, c in enumerate(prog):
                b0 = bi * 4.0
                bm = int(c["bass_midi"])
                if mode == "couplet":
                    notes.append({"pitch": bm, "start": b0 + 0.0, "length": 0.90, "velocity": 102})
                    notes.append({"pitch": bm, "start": b0 + 2.5, "length": 0.60, "velocity": 92})
                    if bi == 3 and not mute_last_beat:
                        notes.append({"pitch": 34, "start": b0 + 3.5, "length": 0.40, "velocity": 88, "porta": True})
                elif mode == "climax":
                    notes.append({"pitch": bm, "start": b0 + 0.0, "length": 1.35, "velocity": 110})
                    notes.append({"pitch": bm, "start": b0 + 1.5, "length": 0.90, "velocity": 100})
                    if bi == 3:
                        if not mute_last_beat:
                            notes.append({"pitch": bm, "start": b0 + 2.75, "length": 0.45, "velocity": 102})
                            notes.append({"pitch": bm + 12, "start": b0 + 3.25, "length": 0.45, "velocity": 96, "porta": True})
                    else:
                        notes.append({"pitch": bm, "start": b0 + 2.75, "length": 0.95, "velocity": 98})
                else:
                    notes.append({"pitch": bm, "start": b0 + 0.0, "length": 1.35, "velocity": 108})
                    notes.append({"pitch": bm, "start": b0 + 1.5, "length": 0.90, "velocity": 96})
                    if not (bi == 3 and mute_last_beat):
                        notes.append({"pitch": bm, "start": b0 + 2.75, "length": 0.95, "velocity": 98})
                        if bi == 3:
                            notes.append({"pitch": 34, "start": b0 + 3.5, "length": 0.40, "velocity": 90, "porta": True})
            return notes

        def mk_kick_notes(mode: str = "refrain", mute_last_beat: bool = False) -> list[dict]:
            notes = []
            for bi in range(4):
                b0 = bi * 4.0
                if mode == "couplet":
                    notes.append({"pitch": 60, "start": b0 + 0.0, "length": 0.25, "velocity": 104})
                    notes.append({"pitch": 60, "start": b0 + 2.5, "length": 0.25, "velocity": 92})
                else:
                    notes.append({"pitch": 60, "start": b0 + 0.0, "length": 0.25, "velocity": 110})
                    notes.append({"pitch": 60, "start": b0 + 1.5, "length": 0.25, "velocity": 96})
                    if not (bi == 3 and mute_last_beat):
                        notes.append({"pitch": 60, "start": b0 + 2.75, "length": 0.25, "velocity": 100})
            return notes

        def mk_clap_notes() -> list[dict]:
            return [{"pitch": 60, "start": bi * 4.0 + 2.0, "length": 0.25, "velocity": 104 if bi % 2 == 0 else 98} for bi in range(4)]

        def mk_snare_notes(mode: str = "refrain") -> list[dict]:
            notes = []
            for bi in range(4):
                b0 = bi * 4.0
                if mode == "couplet_b":
                    notes.append({"pitch": 60, "start": b0 + 3.5, "length": 0.15, "velocity": 62})
                else:
                    notes.append({"pitch": 60, "start": b0 + 2.007, "length": 0.20, "velocity": 86})
                    if bi in (1, 3):
                        notes.append({"pitch": 60, "start": b0 + 3.5, "length": 0.15, "velocity": 68})
            return notes

        def mk_hat_notes(mode: str = "refrain", mute_last: bool = False) -> list[dict]:
            notes = []
            for bi in range(4):
                b0 = bi * 4.0
                if mode == "couplet_a":
                    for st, v in [(0.0, 88), (1.0, 74), (1.5, 62), (2.0, 86), (3.0, 72)]:
                        if bi == 3 and mute_last and st >= 3.0:
                            continue
                        notes.append({"pitch": 60, "start": b0 + st, "length": 0.12, "velocity": v})
                elif mode == "outro":
                    for st, v in [(0.0, 72), (1.0, 58), (2.0, 68), (3.0, 54)]:
                        notes.append({"pitch": 60, "start": b0 + st, "length": 0.12, "velocity": v})
                else:
                    for k in range(8):
                        st = k * 0.5
                        if bi == 3 and mute_last and st >= 3.0:
                            continue
                        v = 92 if k % 2 == 0 else 66
                        notes.append({"pitch": 60, "start": b0 + st, "length": 0.12, "velocity": v})
                    if bi == 1:
                        for j, p in enumerate([60, 59, 58]):
                            notes.append({"pitch": p, "start": b0 + 3.5 + j * 0.166, "length": 0.10, "velocity": 64 - j * 6})
                    elif bi == 3 and not mute_last:
                        for j, p in enumerate([60, 60, 58, 57]):
                            notes.append({"pitch": p, "start": b0 + 3.25 + j * 0.125, "length": 0.09, "velocity": 70 - j * 5})
            return notes

        def mk_openhat_notes() -> list[dict]:
            return [
                {"pitch": 60, "start": 1.5, "length": 0.40, "velocity": 76},
                {"pitch": 60, "start": 9.5, "length": 0.40, "velocity": 78},
            ]

        def mk_perc_notes() -> list[dict]:
            return [
                {"pitch": 60, "start": 3.75, "length": 0.20, "velocity": 66},
                {"pitch": 60, "start": 7.75, "length": 0.20, "velocity": 70},
                {"pitch": 60, "start": 11.75, "length": 0.20, "velocity": 66},
            ]

        patterns_spec = [
            ("Intro Pad (M1-8)", 4, [(8, mk_pad_notes(60))]),
            ("Intro Pluck (M1-8)", 4, [(11, mk_pluck_notes(0.75, intro_only=True))]),
            ("Pad Refrain (4b)", 4, [(8, mk_pad_notes(82))]),
            ("Pad Couplet (4b)", 4, [(8, mk_pad_notes(68))]),
            ("Pad Outro (4b)", 4, [(8, mk_pad_notes(54))]),
            ("Glass Bed FX (8b)", 8, [(9, [{"pitch": 60, "start": 0.0, "length": 31.8, "velocity": 76}])]),
            ("Vox Lead Intro (4b)", 4, [(10, mk_lead_notes(0.65, sparse=True))]),
            ("Vox Lead Refrain (4b)", 4, [(10, mk_lead_notes(1.0))]),
            ("Vox Lead Couplet Hum (4b)", 4, [(10, mk_lead_notes(0.62, sparse=True))]),
            ("Vox Lead Pont (4b)", 4, [(10, mk_lead_notes(0.84))]),
            ("Vox Lead Outro (4b)", 4, [(10, mk_lead_notes(0.60, sparse=True))]),
            ("Pluck Resp Refrain (4b)", 4, [(11, mk_pluck_notes(0.95))]),
            ("Pluck Resp Couplet (4b)", 4, [(11, mk_pluck_notes(0.76))]),
            ("Echo Vox Pont (4b)", 4, [(12, mk_echo_notes(0.82))]),
            ("Echo Vox Climax (4b)", 4, [(12, mk_echo_notes(0.95))]),
            ("808 Refrain (4b)", 4, [(7, mk_808_notes("refrain"))]),
            ("808 Refrain Mute (4b)", 4, [(7, mk_808_notes("refrain", mute_last_beat=True))]),
            ("808 Couplet (4b)", 4, [(7, mk_808_notes("couplet"))]),
            ("808 Couplet Mute (4b)", 4, [(7, mk_808_notes("couplet", mute_last_beat=True))]),
            ("808 Climax (4b)", 4, [(7, mk_808_notes("climax"))]),
            ("Kick Refrain (4b)", 4, [(1, mk_kick_notes("refrain"))]),
            ("Kick Refrain Mute (4b)", 4, [(1, mk_kick_notes("refrain", mute_last_beat=True))]),
            ("Kick Couplet (4b)", 4, [(1, mk_kick_notes("couplet"))]),
            ("Kick Couplet Mute (4b)", 4, [(1, mk_kick_notes("couplet", mute_last_beat=True))]),
            ("Clap HalfTime (4b)", 4, [(2, mk_clap_notes())]),
            ("Snare Refrain (4b)", 4, [(3, mk_snare_notes("refrain"))]),
            ("Snare Couplet Ghost (4b)", 4, [(3, mk_snare_notes("couplet_b"))]),
            ("Hats Refrain (4b)", 4, [(4, mk_hat_notes("refrain"))]),
            ("Hats Refrain Mute (4b)", 4, [(4, mk_hat_notes("refrain", mute_last=True))]),
            ("Hats Couplet Air (4b)", 4, [(4, mk_hat_notes("couplet_a"))]),
            ("Hats Outro (4b)", 4, [(4, mk_hat_notes("outro"))]),
            ("OpenHat Accent (4b)", 4, [(5, mk_openhat_notes())]),
            ("Perc Ghost (4b)", 4, [(6, mk_perc_notes())]),
            ("Vox Swell FX (1b)", 1, [(13, [{"pitch": 67, "start": 0.0, "length": 3.9, "velocity": 82}])]),
            ("Glass Riser FX (1b)", 1, [(14, [{"pitch": 72, "start": 0.0, "length": 3.9, "velocity": 86}])]),
        ]

        pat_ids: dict[str, int] = {}
        for pname, pbars, ch_notes in patterns_spec:
            res = mcp.call("add_pattern", {"name": pname})
            pid = int(res["pattern"])
            pat_ids[pname] = pid
            mcp.call("edit_pattern", {"pattern": pid, "bars": pbars})
            for ch_idx, nlist in ch_notes:
                mcp.call("set_notes", {"pattern": pid, "channel": ch_idx, "notes": nlist})

        # 8. Arrange Playlist (Tracks 1..14)
        clips: list[tuple[int, str, float]] = [
            # INTRO (M1-8)
            (8, "Intro Pad (M1-8)", 0.0), (8, "Intro Pad (M1-8)", 16.0),
            (11, "Intro Pluck (M1-8)", 0.0), (11, "Intro Pluck (M1-8)", 16.0),
            (10, "Vox Lead Intro (4b)", 16.0),  # M5-8 soft vocal theme preview
            (13, "Vox Swell FX (1b)", 12.0),
            (14, "Glass Riser FX (1b)", 28.0),

            # REFRAIN 1 (M9-24)
            (7, "808 Refrain (4b)", 32.0), (7, "808 Refrain (4b)", 48.0),
            (7, "808 Refrain (4b)", 64.0), (7, "808 Refrain Mute (4b)", 80.0),
            (1, "Kick Refrain (4b)", 32.0), (1, "Kick Refrain (4b)", 48.0),
            (1, "Kick Refrain (4b)", 64.0), (1, "Kick Refrain Mute (4b)", 80.0),
            (2, "Clap HalfTime (4b)", 32.0), (2, "Clap HalfTime (4b)", 48.0),
            (2, "Clap HalfTime (4b)", 64.0), (2, "Clap HalfTime (4b)", 80.0),
            (3, "Snare Refrain (4b)", 48.0), (3, "Snare Refrain (4b)", 64.0), (3, "Snare Refrain (4b)", 80.0),
            (4, "Hats Refrain (4b)", 32.0), (4, "Hats Refrain (4b)", 48.0),
            (4, "Hats Refrain (4b)", 64.0), (4, "Hats Refrain Mute (4b)", 80.0),
            (5, "OpenHat Accent (4b)", 32.0), (5, "OpenHat Accent (4b)", 48.0),
            (5, "OpenHat Accent (4b)", 64.0), (5, "OpenHat Accent (4b)", 80.0),
            (6, "Perc Ghost (4b)", 48.0), (6, "Perc Ghost (4b)", 80.0),
            (8, "Pad Refrain (4b)", 32.0), (8, "Pad Refrain (4b)", 48.0),
            (8, "Pad Refrain (4b)", 64.0), (8, "Pad Refrain (4b)", 80.0),
            (9, "Glass Bed FX (8b)", 32.0), (9, "Glass Bed FX (8b)", 64.0),
            (10, "Vox Lead Refrain (4b)", 32.0), (10, "Vox Lead Refrain (4b)", 48.0),
            (10, "Vox Lead Refrain (4b)", 64.0), (10, "Vox Lead Refrain (4b)", 80.0),
            (11, "Pluck Resp Refrain (4b)", 64.0), (11, "Pluck Resp Refrain (4b)", 80.0),
            (13, "Vox Swell FX (1b)", 92.0),

            # COUPLET 1 (M25-40)
            (7, "808 Couplet (4b)", 96.0), (7, "808 Couplet (4b)", 112.0),
            (7, "808 Couplet (4b)", 128.0), (7, "808 Couplet Mute (4b)", 144.0),
            (1, "Kick Couplet (4b)", 96.0), (1, "Kick Couplet (4b)", 112.0),
            (1, "Kick Couplet (4b)", 128.0), (1, "Kick Couplet Mute (4b)", 144.0),
            (2, "Clap HalfTime (4b)", 96.0), (2, "Clap HalfTime (4b)", 112.0),
            (2, "Clap HalfTime (4b)", 128.0), (2, "Clap HalfTime (4b)", 144.0),
            (3, "Snare Couplet Ghost (4b)", 128.0), (3, "Snare Couplet Ghost (4b)", 144.0),
            (4, "Hats Couplet Air (4b)", 96.0), (4, "Hats Couplet Air (4b)", 112.0),
            (4, "Hats Refrain (4b)", 128.0), (4, "Hats Refrain Mute (4b)", 144.0),
            (8, "Pad Couplet (4b)", 96.0), (8, "Pad Couplet (4b)", 112.0),
            (8, "Pad Couplet (4b)", 128.0), (8, "Pad Couplet (4b)", 144.0),
            (11, "Pluck Resp Couplet (4b)", 112.0), (11, "Pluck Resp Couplet (4b)", 144.0),
            (10, "Vox Lead Couplet Hum (4b)", 128.0), (10, "Vox Lead Couplet Hum (4b)", 144.0),
            (12, "Echo Vox Pont (4b)", 128.0), (12, "Echo Vox Pont (4b)", 144.0),
            (13, "Vox Swell FX (1b)", 156.0),

            # PONT NOCTURNE (M41-48)
            (8, "Pad Couplet (4b)", 160.0), (8, "Pad Couplet (4b)", 176.0),
            (10, "Vox Lead Pont (4b)", 160.0), (10, "Vox Lead Pont (4b)", 176.0),
            (12, "Echo Vox Pont (4b)", 160.0), (12, "Echo Vox Pont (4b)", 176.0),
            (14, "Glass Riser FX (1b)", 188.0),

            # REFRAIN 2 CLIMAX (M49-56)
            (7, "808 Climax (4b)", 192.0), (7, "808 Climax (4b)", 208.0),
            (1, "Kick Refrain (4b)", 192.0), (1, "Kick Refrain (4b)", 208.0),
            (2, "Clap HalfTime (4b)", 192.0), (2, "Clap HalfTime (4b)", 208.0),
            (3, "Snare Refrain (4b)", 192.0), (3, "Snare Refrain (4b)", 208.0),
            (4, "Hats Refrain (4b)", 192.0), (4, "Hats Refrain (4b)", 208.0),
            (5, "OpenHat Accent (4b)", 192.0), (5, "OpenHat Accent (4b)", 208.0),
            (6, "Perc Ghost (4b)", 192.0), (6, "Perc Ghost (4b)", 208.0),
            (8, "Pad Refrain (4b)", 192.0), (8, "Pad Refrain (4b)", 208.0),
            (9, "Glass Bed FX (8b)", 192.0),
            (10, "Vox Lead Refrain (4b)", 192.0), (10, "Vox Lead Refrain (4b)", 208.0),
            (11, "Pluck Resp Refrain (4b)", 192.0), (11, "Pluck Resp Refrain (4b)", 208.0),
            (12, "Echo Vox Climax (4b)", 192.0), (12, "Echo Vox Climax (4b)", 208.0),

            # OUTRO (M57-64)
            (4, "Hats Outro (4b)", 224.0),
            (8, "Pad Outro (4b)", 224.0), (8, "Pad Outro (4b)", 240.0),
            (10, "Vox Lead Outro (4b)", 224.0), (10, "Vox Lead Outro (4b)", 240.0),
            (13, "Vox Swell FX (1b)", 244.0),
        ]
        mcp.batch([
            {"tool": "add_clip", "arguments": {"pattern": pat_ids[pname], "track": trk, "start": sb}}
            for trk, pname, sb in clips
        ])

        # 9. Automations (correct add_automation schema: mixer, slot, param, track, start, length, points[{position, value}])
        # Note: mixer volume range is 0..1.25, so value = target_vol / 1.25
        mcp.call("add_automation", {
            "mixer": 20,
            "param": "volume",
            "name": "FX Room Transition Gate",
            "track": 15,
            "start": 0.0,
            "length": 256.0,
            "points": [
                {"position": 0.0, "value": 0.224},
                {"position": 30.75, "value": 0.224}, {"position": 31.0, "value": 0.0}, {"position": 31.95, "value": 0.0}, {"position": 32.0, "value": 0.224},
                {"position": 94.75, "value": 0.224}, {"position": 95.0, "value": 0.0}, {"position": 95.95, "value": 0.0}, {"position": 96.0, "value": 0.224},
                {"position": 158.75, "value": 0.224}, {"position": 159.0, "value": 0.0}, {"position": 159.95, "value": 0.0}, {"position": 160.0, "value": 0.224},
                {"position": 190.75, "value": 0.224}, {"position": 191.0, "value": 0.0}, {"position": 191.95, "value": 0.0}, {"position": 192.0, "value": 0.224},
                {"position": 252.0, "value": 0.224}, {"position": 256.0, "value": 0.0},
            ],
        })
        mcp.call("add_automation", {
            "mixer": 8,
            "param": "volume",
            "name": "Pad Dynamic Arc",
            "track": 16,
            "start": 0.0,
            "length": 256.0,
            "points": [
                {"position": 0.0, "value": 0.34},
                {"position": 31.5, "value": 0.38}, {"position": 32.0, "value": 0.58},
                {"position": 95.5, "value": 0.58}, {"position": 96.0, "value": 0.46},
                {"position": 159.5, "value": 0.46}, {"position": 160.0, "value": 0.42},
                {"position": 191.5, "value": 0.42}, {"position": 192.0, "value": 0.61},
                {"position": 223.5, "value": 0.61}, {"position": 224.0, "value": 0.40}, {"position": 256.0, "value": 0.20},
            ],
        })
        mcp.call("add_automation", {
            "mixer": 10,
            "slot": 1,
            "param": "10",  # Pitcher Voice 1 (+12st) level
            "name": "AutoTune Harmony Climax",
            "track": 17,
            "start": 0.0,
            "length": 256.0,
            "points": [
                {"position": 0.0, "value": 0.16},
                {"position": 159.5, "value": 0.16}, {"position": 160.0, "value": 0.12},
                {"position": 191.5, "value": 0.12}, {"position": 192.0, "value": 0.28},
                {"position": 224.0, "value": 0.14},
            ],
        })
        mcp.call("add_automation", {
            "mixer": 10,
            "param": "volume",
            "name": "Vox Lead Dynamic Arc",
            "track": 18,
            "start": 0.0,
            "length": 256.0,
            "points": [
                {"position": 0.0, "value": 0.56},
                {"position": 31.5, "value": 0.58}, {"position": 32.0, "value": 0.88},
                {"position": 95.5, "value": 0.88}, {"position": 96.0, "value": 0.52},
                {"position": 159.5, "value": 0.52}, {"position": 160.0, "value": 0.78},
                {"position": 191.5, "value": 0.78}, {"position": 192.0, "value": 0.94},
                {"position": 223.5, "value": 0.94}, {"position": 224.0, "value": 0.54}, {"position": 256.0, "value": 0.22},
            ],
        })

        # 10. Markers & Save
        for b, mname in [(1, "Intro"), (9, "Refrain 1"), (25, "Couplet 1"), (41, "Pont Nocturne"), (49, "Refrain 2 Climax"), (57, "Outro")]:
            mcp.call("add_marker", {"position": (b - 1) * 4.0, "name": mname})

        mcp.call("save_project", {"path": OUT_PROJ})
        exp_mid = mcp.call("export_project", {"format": "midi", "path": f"{EXPORT_DIR}/simba_cloud_full.mid"})
        print("EXPORTED_MIDI:", exp_mid)
    finally:
        mcp.close()

    # Audit full 64-bar orchestrated MIDI against strict 7-metric gate!
    gate = verify_midi_gate(f"{EXPORT_DIR}/simba_cloud_full.mid", expected_key="C minor", max_pitch=84)
    print("FULL_MIDI_GATE_PASSED:", json.dumps({
        "verdict": gate.get("musicality_verdict"),
        "metrics": gate.get("musicality_metrics"),
        "clashes": gate.get("minor_2nd_clashes"),
        "sub_collisions": gate.get("low_end_simultaneous_collisions"),
    }, indent=2))


if __name__ == "__main__":
    main()
