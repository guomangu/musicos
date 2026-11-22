#!/usr/bin/env python3
"""
Build script for projects/active5/projet.gridline
"Obsidian Pulse" — 96 bars @ 180 BPM (128.0 s = 2 min 08 s), E minor.
Top-Down Single-Core Composition:
  1. Computes smooth voice leading (optimize_voice_leading <= 2.5 st) from harmonic_core.
  2. Generates master_sketch.mid (Bass, Chords, Lead A-A'-A''-B) and runs Blocking Gate
     audit_midi_harmony(strict_musicality=True) -> MUST return musicality_verdict == "PASS".
  3. Orchestrates master_sketch.mid into Gridline 0.7.7 using 100% projects/active5 sounds
     (no rapper vocal, unified 440 Hz tuning, unified 180 BPM grid, call-and-response rests).
  4. Applies BUS Music Timbral Glue (EQ LPF 12.5k + Soft Dist 4x 10% + Bus Glue + Kick Micro-Ducking)
     + shared FX Room (Dry=0, Dry full=0) with transition mute automation.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import time

import pretty_midi

ROOT = "/home/gamo/Bureau/musicos"
sys.path.insert(0, f"{ROOT}/mcp-servers/audio-analyzer")
from deep_diagnostics import audit_midi_file
from musicality_metrics import optimize_voice_leading

PROJ_DIR = f"{ROOT}/projects/active5"
PROJ_FILE = f"{PROJ_DIR}/projet.gridline"
SKETCH_MIDI = f"{PROJ_DIR}/master_sketch.mid"
ORCH_MIDI = f"{PROJ_DIR}/obsidian_pulse.mid"
RENDERS_DIR = f"{PROJ_DIR}/renders"
TEMPLATE = "/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline"
SAMPLES_DIR = f"{PROJ_DIR}/samples"

random.seed(42)


def build_master_sketch_and_gate(bpm: float = 180.0) -> tuple[list[list[int]], list[tuple[int, float, float, int]]]:
    """
    Step 1 & 2 (Top-Down): Generate master_sketch.mid (Bass, Chords, Lead)
    and enforce Blocking Gate audit_midi_harmony(..., strict_musicality=True).
    Returns (voicings_4bar, lead_phrase_4bar).
    """
    chord_pcs = [
        [4, 7, 11, 2],   # Bar 1: Em7  (E, G, B, D)
        [0, 4, 7, 2],    # Bar 2: Cadd9 (C, E, G, D) — sans si naturel, zéro cluster C/B
        [9, 0, 4, 7],    # Bar 3: Am7  (A, C, E, G)
        [11, 2, 6, 9],   # Bar 4: Bm7  (B, D, F#, A)
    ]
    voicings = optimize_voice_leading(chord_pcs, low_midi=52, high_midi=67)
    bass_roots = [40, 36, 33, 35]  # E2 -> C2 -> A1 -> B1 (contrary/oblique against Lead!)

    # A - A' - A'' - B singable Trap Top-Line (4 bars = 16 beats)
    # Every transition across bars is stepwise (<= 4 st); single Apex A5 (81) always resolves -2 -> G5 (79).
    lead_phrase = [
        # Bar 1 (A on Em7): E5 -> G5 -> F#5 -> G5 | rest 2.9..4.0
        (0, 0.0, 0.50, 76), (0, 0.50, 0.50, 79), (0, 1.0, 0.75, 78), (0, 2.0, 0.90, 79),
        # Bar 2 (A' on Cadd9): G5 -> E5 -> D5 -> E5 | rest 2.9..4.0
        (1, 0.0, 0.50, 79), (1, 0.50, 0.50, 76), (1, 1.0, 0.75, 74), (1, 2.0, 0.90, 76),
        # Bar 3 (A'' on Am7 -> single Apex A5=81 + gap-fill -2 to G5): E5 -> A5 -> G5 -> E5 | rest 2.9..4.0
        (2, 0.0, 0.50, 76), (2, 0.50, 0.75, 81), (2, 1.25, 0.50, 79), (2, 2.0, 0.90, 76),
        # Bar 4 (B cadence on Bm7): F#5 -> D5 -> B4 -> D5 | rest 2.85..4.0 (D5 -> E5 is +2 stepwise!)
        (3, 0.0, 0.50, 78), (3, 0.50, 0.50, 74), (3, 1.0, 0.75, 71), (3, 2.0, 0.85, 74),
    ]

    beat_s = 60.0 / bpm
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    bass_inst = pretty_midi.Instrument(program=33, name="Bass")
    chords_inst = pretty_midi.Instrument(program=0, name="Chords")
    lead_inst = pretty_midi.Instrument(program=0, name="Lead")

    # Write 8 bars (2 phrases) in master_sketch.mid
    for rep in range(2):
        bar_off = rep * 4
        for b in range(4):
            b0 = (bar_off + b) * 4.0
            bass_inst.notes.append(pretty_midi.Note(108, bass_roots[b], b0 * beat_s, (b0 + 3.5) * beat_s))
            for p in voicings[b]:
                chords_inst.notes.append(pretty_midi.Note(82, p, b0 * beat_s, (b0 + 3.75) * beat_s))
        for b, st, ln, pt in lead_phrase:
            t0 = ((bar_off + b) * 4.0 + st) * beat_s
            lead_inst.notes.append(pretty_midi.Note(96, pt, t0, t0 + ln * beat_s))

    pm.instruments.extend([bass_inst, chords_inst, lead_inst])
    pm.write(SKETCH_MIDI)

    gate = audit_midi_file(SKETCH_MIDI, expected_key="E minor", max_pitch=84, strict_musicality=True)
    print("Master Sketch Gate:", json.dumps({
        "verdict": gate["musicality_verdict"],
        "gate_passed": gate["gate_passed"],
        "detected_key": gate["detected_key"],
        "metrics": gate["musicality_metrics"],
        "failures": gate["musicality_failures"],
    }, indent=2))
    if not gate["gate_passed"]:
        raise RuntimeError(f"BLOCKING GATE FAILED on master_sketch.mid: {gate['musicality_failures']}")

    return voicings, lead_phrase


class GridlineMCP:
    def __init__(self, proj_path: str, export_dir: str):
        os.makedirs(export_dir, exist_ok=True)
        self.proc = subprocess.Popen(
            ["gridline", "mcp", proj_path, "--export-dir", export_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._id = 0
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "build_active5", "version": "3.0"},
        })
        self._notify("notifications/initialized", {})

    def _notify(self, method: str, params: dict):
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        self.proc.stdin.write(msg + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method: str, params: dict):
        self._id += 1
        req_id = self._id
        msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        self.proc.stdin.write(msg + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read()
                raise RuntimeError(f"MCP EOF. stderr={err}")
            resp = json.loads(line)
            if resp.get("id") == req_id:
                if "error" in resp:
                    raise RuntimeError(f"RPC error on {method}: {resp['error']}")
                return resp["result"]

    def call(self, tool: str, arguments: dict):
        for attempt in range(8):
            try:
                res = self._rpc("tools/call", {"name": tool, "arguments": arguments})
                text = res["content"][0]["text"]
                data = json.loads(text) if text.startswith("{") or text.startswith("[") else text
                if res.get("isError"):
                    raise RuntimeError(f"Tool {tool} error: {text}")
                return data
            except RuntimeError as e:
                if "Rate limit" in str(e) and attempt < 7:
                    time.sleep(0.8)
                    continue
                raise

    def batch(self, calls: list):
        results = []
        for i in range(0, len(calls), 120):
            chunk = calls[i:i + 120]
            r = self.call("batch", {"calls": chunk})
            if isinstance(r, dict) and r.get("errors"):
                raise RuntimeError(f"Batch had errors: {r['errors']}")
            results.append(r)
        return results

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=15)


def build():
    # Step 1 & 2: Top-Down Master Sketch + Blocking Gate BEFORE any Gridline generation
    voicings, lead_phrase = build_master_sketch_and_gate(bpm=180.0)

    shutil.copyfile(TEMPLATE, PROJ_FILE)
    mcp = GridlineMCP(PROJ_FILE, RENDERS_DIR)

    try:
        mcp.call("set_song", {
            "title": "Obsidian Pulse",
            "genre": "Dark Melodic Trap",
            "author": "AI-Studio",
            "tempo": 180,
            "swing": 0.05,
        })

        mix_info = mcp.call("get_mixer", {})
        existing_mix = {t["mixer"] for t in mix_info["tracks"]}
        add_mix_calls = [{"tool": "add_mixer_track", "arguments": {}} for m in range(1, 21) if m not in existing_mix]
        if add_mix_calls:
            mcp.batch(add_mix_calls)

        for tr in mix_info["tracks"]:
            if tr["mixer"] == 0:
                for fx in tr["effects"]:
                    mcp.call("remove_plugin", {"mixer": 0, "slot": fx["slot"]})

        # Mixer architecture:
        # 1,4,5,6,7 -> BUS Drums (17)
        # 2,3 -> BUS 808 (18)
        # 8..13 -> BUS Music (19) (timbral glue + Kick micro-ducking)
        # 20 = FX Room -> Master (0)
        mixer_defs = [
            (1, "Kick", 17, 0.78, 0.0, 0.0),
            (2, "808 Apollo", 18, 0.80, 0.0, -0.8),
            (3, "808 Blxckout", 18, 0.78, 0.0, -0.8),
            (4, "Snare", 17, 0.84, 0.0, 0.0),
            (5, "Clap", 17, 0.82, 0.0, 0.0),
            (6, "Hats Trap", 17, 0.76, 0.0, 0.0),
            (7, "Perc / Rim / OpenHat", 17, 0.72, 0.0, 0.0),
            (8, "Chords Pad E", 19, 0.76, 0.0, -0.22),
            (9, "Chords Piano VoiceLed", 19, 0.76, -0.08, -0.25),
            (10, "Chords Stab E", 19, 0.84, 0.0, -0.15),
            (11, "Lead Piano", 19, 0.96, 0.06, -0.10),
            (12, "Counter Pizz E", 19, 0.84, -0.14, 0.0),
            (13, "Chords High +12 & Chants", 19, 0.74, 0.0, -0.15),
            (17, "BUS Drums", 0, 0.94, 0.0, 0.0),
            (18, "BUS 808", 0, 0.81, 0.0, -1.0),
            (19, "BUS Music", 0, 1.00, 0.0, -0.18),
            (20, "FX Room", 0, 0.42, 0.0, 0.0),
        ]
        calls = []
        stereo_changes = []
        for m, name, route, vol, pan, stereo in mixer_defs:
            calls.append({"tool": "edit_mixer_track", "arguments": {"mixer": m, "name": name, "routeTo": route, "volume": vol, "pan": pan}})
            if stereo != 0.0:
                stereo_changes.append({"path": f"mixer/{m}/stereo", "value": stereo})
        mcp.batch(calls)
        if stereo_changes:
            mcp.call("set_parameters", {"changes": stereo_changes})

        # Sidechains: Kick (1) -> BUS 808 (18, 1.0) & Kick (1) -> BUS Music (19, 0.75 micro-ducking)
        # Shared FX Room (20) sends — tight acoustic cement (< 8%) so RT60 <= 2*T_noire
        sends = [
            (1, 18, 1.0, True),
            (1, 19, 0.75, True),
            (4, 20, 0.04, False),
            (5, 20, 0.04, False),
            (7, 20, 0.04, False),
            (8, 20, 0.07, False),
            (9, 20, 0.07, False),
            (10, 20, 0.06, False),
            (11, 20, 0.08, False),
            (12, 20, 0.06, False),
            (13, 20, 0.07, False),
        ]
        mcp.batch([
            {"tool": "set_send", "arguments": {"mixer": m, "to": to, "level": lvl, "sidechain": sc}}
            for m, to, lvl, sc in sends
        ])

        # 3. Channels 1..20 (100% projects/active5 sources, NO rapper vocal!)
        # Note role-aware names so audit_midi_harmony infers bass / chords / lead / counter cleanly!
        channels_spec = [
            ("Kick Native", "gl.kick", 1, None, 0.74, 0.0),
            ("Techno Kick OneShot", "gl.sampler", 1, f"{SAMPLES_DIR}/techno_kick_oneshot.wav", 0.74, 0.0),
            ("808 Apollo", "gl.sampler", 2, f"{ROOT}/soundbanks/JMizzy The Best Drum Kits/!Mike Zara's Stash/808s/MZ 808 [Apollo].wav", 0.82, 0.0),
            ("808 Blxckout", "gl.sampler", 3, f"{ROOT}/soundbanks/JMizzy The Best Drum Kits/!Mike Zara's Stash/808s/MZ 808 [Blxckout].wav", 0.80, 0.0),
            ("Snare Native", "gl.snare", 4, None, 0.76, 0.0),
            ("Clap Native", "gl.clap", 5, None, 0.70, 0.0),
            ("Techno Clap OneShot", "gl.sampler", 5, f"{SAMPLES_DIR}/techno_clap_oneshot.wav", 0.72, -0.05),
            ("Hat Native", "gl.hat", 6, None, 0.66, 0.06),
            ("Techno Hat OneShot", "gl.sampler", 6, f"{SAMPLES_DIR}/techno_hat_oneshot.wav", 0.66, -0.10),
            ("Techno OpenHat OneShot", "gl.sampler", 7, f"{SAMPLES_DIR}/techno_openhat_oneshot.wav", 0.62, 0.14),
            ("Techno Rim OneShot", "gl.sampler", 7, f"{SAMPLES_DIR}/techno_rim_oneshot.wav", 0.64, -0.14),
            ("Techno Shaker OneShot", "gl.sampler", 7, f"{SAMPLES_DIR}/techno_shaker_oneshot.wav", 0.60, 0.12),
            # 13..20 Tuned Melodic/Harmonic Channels (A4=440.0 Hz unified)
            ("Chords Pad E (440Hz)", "gl.sampler", 8, f"{SAMPLES_DIR}/techno_pad_e_440.wav", 0.70, 0.0),
            ("Chords Piano VoiceLed", "gl.sampler", 9, f"{ROOT}/soundbanks/sfz/headroom-piano/Samples/HEADROOM PIANO LEVEL5 CLOSE 93.flac", 0.70, -0.08),
            ("Chords Stab E (440Hz)", "gl.sampler", 10, f"{SAMPLES_DIR}/techno_chord_stab_e_440.wav", 0.74, 0.0),
            ("Lead Piano (440Hz)", "gl.sampler", 11, f"{ROOT}/soundbanks/sfz/headroom-piano/Samples/HEADROOM PIANO LEVEL5 CLOSE 93.flac", 0.86, 0.06),
            ("Counter Pizz E (440Hz)", "gl.sampler", 12, f"{SAMPLES_DIR}/upright_pizz_e3_440.wav", 0.78, -0.14),
            ("Chords High +12 (440Hz)", "gl.sampler", 13, f"{SAMPLES_DIR}/techno_chord_stab_e_440.wav", 0.60, 0.0),
            ("Chords Riser (440Hz)", "gl.sampler", 13, f"{SAMPLES_DIR}/techno_chord_riser_440.wav", 0.74, 0.0),
            ("Chants Accent (SXJ)", "gl.sampler", 13, f"{ROOT}/soundbanks/JMizzy The Best Drum Kits/JUGG DRUM KIT/Chants (Voxs)/(SXJ) Chants (6).wav", 0.64, 0.16),
        ]

        mcp.batch([
            {"tool": "add_channel", "arguments": {"name": name, "plugin": plugin, "mixer": mixer, "volume": vol, "pan": pan}}
            for name, plugin, mixer, _, vol, pan in channels_spec
        ])
        mcp.batch([
            {"tool": "patch", "arguments": {"path": f"channel/{idx}", "set": {"plugin": {"data": spath}}}}
            for idx, (_, _, _, spath, _, _) in enumerate(channels_spec, 1) if spath
        ])

        # 4. Channel tuning & ADSR envelopes (Protocol Section 2 + A4=440.0 Hz)
        ch_params = [
            # Kick Native (125ms)
            ("channel/1/Start pitch", 215), ("channel/1/End pitch", 54),
            ("channel/1/Sweep", 0.045), ("channel/1/Decay", 0.125),
            ("channel/1/Click", 0.36), ("channel/1/Drive", 0.22),
            ("channel/2/One shot", 0), ("channel/2/Release", 0.02), ("channel/2/Declick", 2), ("channel/2/Remove DC", 1),
            # 808 Apollo (Root 24, Fine -14ct, Voices=1 Cut Itself!, Release 45ms, Glide 55ms)
            ("channel/3/Root note", 24), ("channel/3/Fine tune", -14), ("channel/3/Voices", 1),
            ("channel/3/One shot", 0), ("channel/3/Attack", 0.002), ("channel/3/Decay", 10.0),
            ("channel/3/Sustain", 1.0), ("channel/3/Release", 0.045), ("channel/3/Glide", 0.055),
            ("channel/3/Remove DC", 1), ("channel/3/Declick", 2),
            # 808 Blxckout (Root 24, Fine +6ct, Voices=1 Cut Itself!, Release 45ms, Glide 55ms)
            ("channel/4/Root note", 24), ("channel/4/Fine tune", 6), ("channel/4/Voices", 1),
            ("channel/4/One shot", 0), ("channel/4/Attack", 0.002), ("channel/4/Decay", 10.0),
            ("channel/4/Sustain", 1.0), ("channel/4/Release", 0.045), ("channel/4/Glide", 0.055),
            ("channel/4/Remove DC", 1), ("channel/4/Declick", 2),
            # Snare, Clap, Hats
            ("channel/5/Tone", 0.58), ("channel/5/Snappy", 0.74),
            ("channel/5/Body decay", 0.11), ("channel/5/Noise decay", 0.16), ("channel/5/Tune", 198),
            ("channel/6/Decay", 0.18), ("channel/6/Tone", 1800), ("channel/6/Hits", 4), ("channel/6/Spread", 0.011),
            ("channel/7/One shot", 0), ("channel/7/Release", 0.025), ("channel/7/Declick", 2),
            ("channel/8/Decay", 0.045), ("channel/8/Tone", 9200), ("channel/8/Metal", 0.22), ("channel/8/Choke", 1),
            ("channel/9/One shot", 0), ("channel/9/Release", 0.015), ("channel/9/Declick", 2),
            ("channel/10/One shot", 0), ("channel/10/Release", 0.03), ("channel/10/Declick", 2),
            ("channel/11/One shot", 0), ("channel/11/Release", 0.02), ("channel/11/Declick", 2),
            ("channel/12/One shot", 0), ("channel/12/Release", 0.015), ("channel/12/Declick", 2),
            # Ch 13 Chords Pad E (Root 64, Fine -8ct, Attack 40ms, crisp Release 140ms)
            ("channel/13/Root note", 64), ("channel/13/Fine tune", -8),
            ("channel/13/One shot", 0), ("channel/13/Attack", 0.040), ("channel/13/Decay", 1.0),
            ("channel/13/Sustain", 0.78), ("channel/13/Release", 0.14), ("channel/13/Declick", 2), ("channel/13/Remove DC", 1),
            # Ch 14 Chords Piano VoiceLed (Headroom 93, Root 93, Fine -17ct, Attack 22ms, crisp Release 120ms)
            ("channel/14/Root note", 93), ("channel/14/Fine tune", -17),
            ("channel/14/One shot", 0), ("channel/14/Attack", 0.022), ("channel/14/Decay", 1.15),
            ("channel/14/Sustain", 0.72), ("channel/14/Release", 0.12), ("channel/14/Declick", 2), ("channel/14/Remove DC", 1),
            # Ch 15 Chords Stab E (Root 64, Fine -8ct, crisp Attack 4ms)
            ("channel/15/Root note", 64), ("channel/15/Fine tune", -8),
            ("channel/15/One shot", 0), ("channel/15/Attack", 0.004), ("channel/15/Decay", 0.50),
            ("channel/15/Sustain", 0.50), ("channel/15/Release", 0.14), ("channel/15/Declick", 2), ("channel/15/Remove DC", 1),
            # Ch 16 Lead Piano (Headroom 93, Root 93, Fine -17ct, crisp 4ms attack)
            ("channel/16/Root note", 93), ("channel/16/Fine tune", -17),
            ("channel/16/One shot", 0), ("channel/16/Attack", 0.004), ("channel/16/Decay", 1.2),
            ("channel/16/Sustain", 0.72), ("channel/16/Release", 0.22), ("channel/16/Declick", 2), ("channel/16/Remove DC", 1),
            # Ch 17 Counter Pizz E (Root 52, Fine +8ct, Attack 4ms, Release 80ms)
            ("channel/17/Root note", 52), ("channel/17/Fine tune", 8),
            ("channel/17/One shot", 0), ("channel/17/Attack", 0.004), ("channel/17/Decay", 0.45),
            ("channel/17/Sustain", 0.40), ("channel/17/Release", 0.08), ("channel/17/Declick", 2), ("channel/17/Remove DC", 1),
            # Ch 18 Chords High +12 (Root 52, Fine -8ct)
            ("channel/18/Root note", 52), ("channel/18/Fine tune", -8),
            ("channel/18/One shot", 0), ("channel/18/Attack", 0.004), ("channel/18/Decay", 0.50),
            ("channel/18/Sustain", 0.45), ("channel/18/Release", 0.14), ("channel/18/Declick", 2), ("channel/18/Remove DC", 1),
            # Ch 19 Riser, Ch 20 Chants
            ("channel/19/Root note", 64), ("channel/19/Fine tune", -8), ("channel/19/One shot", 0), ("channel/19/Release", 0.03), ("channel/19/Declick", 2),
            ("channel/20/Root note", 64), ("channel/20/One shot", 0), ("channel/20/Release", 0.03), ("channel/20/Declick", 2),
        ]
        mcp.call("set_parameters", {"changes": [{"path": p, "value": v} for p, v in ch_params]})

        # 5. Mixer DSP & BUS Music Timbral Glue Chain (Section 4.2 of cahier des charges)
        fx_adds = [
            # Master (0): EQ (1), Glue Comp (2), Limiter (3)
            {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.comp"}},
            {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.limiter"}},
            # BUS Drums (17): EQ (1), Bus Glue Comp (2)
            {"tool": "add_plugin", "arguments": {"mixer": 17, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 17, "plugin": "gl.comp"}},
            # BUS 808 (18): Stereo Width (1), Sidechain Comp (2), EQ (3)
            {"tool": "add_plugin", "arguments": {"mixer": 18, "plugin": "gl.stereo"}},
            {"tool": "add_plugin", "arguments": {"mixer": 18, "plugin": "gl.comp"}},
            {"tool": "add_plugin", "arguments": {"mixer": 18, "plugin": "gl.eq"}},
            # BUS Music (19):
            #   Slot 1: gl.eq (HPF 135 Hz + LPF 12500 Hz + presence)
            #   Slot 2: gl.dist (Soft, 4x oversampling, Mix 0.10 collective saturation)
            #   Slot 3: gl.comp (Bus Glue 2:1, Attack 20ms, Release 110ms)
            #   Slot 4: gl.comp (Micro-Ducking Sidechain from Kick, Attack 5ms, Release 80ms)
            {"tool": "add_plugin", "arguments": {"mixer": 19, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 19, "plugin": "gl.dist"}},
            {"tool": "add_plugin", "arguments": {"mixer": 19, "plugin": "gl.comp"}},
            {"tool": "add_plugin", "arguments": {"mixer": 19, "plugin": "gl.comp"}},
            # FX Room (20): gl.reverb (1)
            {"tool": "add_plugin", "arguments": {"mixer": 20, "plugin": "gl.reverb"}},
            # Transient/Nyquist LPF smoothing on Snare (4) & Clap (5) + Mirror EQs on harmonic layers:
            {"tool": "add_plugin", "arguments": {"mixer": 4, "plugin": "gl.eq"}},   # Snare LPF 11.5k
            {"tool": "add_plugin", "arguments": {"mixer": 5, "plugin": "gl.eq"}},   # Clap LPF 11.5k
            {"tool": "add_plugin", "arguments": {"mixer": 8, "plugin": "gl.eq"}},   # Chords Pad
            {"tool": "add_plugin", "arguments": {"mixer": 9, "plugin": "gl.eq"}},   # Chords Piano VoiceLed
            {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.eq"}},  # Chords Stab
            {"tool": "add_plugin", "arguments": {"mixer": 11, "plugin": "gl.eq"}},  # Lead Piano
            {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.eq"}},  # Counter Pizz
            {"tool": "add_plugin", "arguments": {"mixer": 13, "plugin": "gl.eq"}},  # Chords High +12
        ]
        mcp.batch(fx_adds)

        fx_changes = [
            # Master (0)
            ("mixer/0/slot/1/Quality", 1),
            ("mixer/0/slot/1/Band 1 type", 1), ("mixer/0/slot/1/Band 1 freq", 28),
            ("mixer/0/slot/1/Band 2 type", 6), ("mixer/0/slot/1/Band 2 freq", 63), ("mixer/0/slot/1/Band 2 gain", -3.9), ("mixer/0/slot/1/Band 2 Q (legacy)", 2.2),
            ("mixer/0/slot/1/Band 7 type", 3), ("mixer/0/slot/1/Band 7 freq", 17500),
            ("mixer/0/slot/2/Threshold", -12.5), ("mixer/0/slot/2/Ratio", 1.8),
            ("mixer/0/slot/2/Attack", 0.020), ("mixer/0/slot/2/Release", 0.11),
            ("mixer/0/slot/2/Makeup", 1.8), ("mixer/0/slot/2/Detector high-pass", 95),
            ("mixer/0/slot/3/Gain", 7.8), ("mixer/0/slot/3/Ceiling", -2.50),
            ("mixer/0/slot/3/Attack (look-ahead)", 0.006), ("mixer/0/slot/3/0.2 dB safety", 1),

            # BUS Drums (17): Attack 12ms preserves snare/clap transients without Nyquist micro-clicks
            ("mixer/17/slot/1/Band 1 type", 1), ("mixer/17/slot/1/Band 1 freq", 32),
            ("mixer/17/slot/1/Band 2 type", 6), ("mixer/17/slot/1/Band 2 freq", 63), ("mixer/17/slot/1/Band 2 gain", -3.2), ("mixer/17/slot/1/Band 2 Q (legacy)", 1.8),
            ("mixer/17/slot/1/Band 7 type", 7), ("mixer/17/slot/1/Band 7 freq", 10500), ("mixer/17/slot/1/Band 7 gain", 1.8),
            ("mixer/17/slot/2/Threshold", -16.5), ("mixer/17/slot/2/Ratio", 2.7),
            ("mixer/17/slot/2/Attack", 0.012), ("mixer/17/slot/2/Release", 0.085), ("mixer/17/slot/2/Makeup", 3.3),

            # BUS 808 (18)
            ("mixer/18/slot/1/Width", 0.0),
            ("mixer/18/slot/2/Detector input", 1),
            ("mixer/18/slot/2/Threshold", -22), ("mixer/18/slot/2/Ratio", 5.5),
            ("mixer/18/slot/2/Attack", 0.002), ("mixer/18/slot/2/Release", 0.065), ("mixer/18/slot/2/Lookahead", 2),
            ("mixer/18/slot/3/Band 1 type", 1), ("mixer/18/slot/3/Band 1 freq", 28),
            ("mixer/18/slot/3/Band 2 type", 6), ("mixer/18/slot/3/Band 2 freq", 63), ("mixer/18/slot/3/Band 2 gain", -7.2), ("mixer/18/slot/3/Band 2 Q (legacy)", 2.0),
            ("mixer/18/slot/3/Band 3 type", 6), ("mixer/18/slot/3/Band 3 freq", 115), ("mixer/18/slot/3/Band 3 gain", 1.6), ("mixer/18/slot/3/Band 3 Q (legacy)", 1.2),

            # BUS Music (19): Timbral Glue Chain
            # Slot 1: gl.eq (HPF 135 Hz + LPF 12500 Hz + subtle presence 1.6 kHz)
            ("mixer/19/slot/1/Band 1 type", 1), ("mixer/19/slot/1/Band 1 freq", 135),
            ("mixer/19/slot/1/Band 5 type", 6), ("mixer/19/slot/1/Band 5 freq", 1600), ("mixer/19/slot/1/Band 5 gain", 1.8), ("mixer/19/slot/1/Band 5 Q (legacy)", 0.85),
            ("mixer/19/slot/1/Band 7 type", 3), ("mixer/19/slot/1/Band 7 freq", 12500),
            # Slot 2: gl.dist (Soft 4x collective saturation, Mix 10%)
            ("mixer/19/slot/2/Mode", 0), ("mixer/19/slot/2/Drive", 6.0),
            ("mixer/19/slot/2/Tone", 12000), ("mixer/19/slot/2/Mix", 0.10),
            ("mixer/19/slot/2/Output", 0.0), ("mixer/19/slot/2/Oversampling", 2), ("mixer/19/slot/2/Remove DC", 1),
            # Slot 3: gl.comp Bus Glue (fills K-weighted melodic density cleanly without ISP clipping)
            ("mixer/19/slot/3/Threshold", -20.5), ("mixer/19/slot/3/Ratio", 2.6),
            ("mixer/19/slot/3/Attack", 0.018), ("mixer/19/slot/3/Release", 0.110), ("mixer/19/slot/3/Makeup", 5.0),
            # Slot 4: gl.comp Micro-Ducking Sidechain from Kick (Attack 5ms, Release 80ms)
            ("mixer/19/slot/4/Detector input", 1),
            ("mixer/19/slot/4/Threshold", -24.0), ("mixer/19/slot/4/Ratio", 3.0),
            ("mixer/19/slot/4/Attack", 0.005), ("mixer/19/slot/4/Release", 0.080), ("mixer/19/slot/4/Lookahead", 1),

            # FX Room (20): shared short acoustic space (Decay 0.250 s <= 1*T_noire, Dry=0, Dry full=0)
            ("mixer/20/slot/1/Model", 1),
            ("mixer/20/slot/1/Pre-delay", 10), ("mixer/20/slot/1/Room size", 0.24),
            ("mixer/20/slot/1/Decay time", 0.210),
            ("mixer/20/slot/1/Low cut", 480), ("mixer/20/slot/1/High cut", 6400), ("mixer/20/slot/1/High damping", 4000),
            ("mixer/20/slot/1/Dry", 0), ("mixer/20/slot/1/Wet", 0.58), ("mixer/20/slot/1/Width", 0.80),
            ("mixer/20/slot/1/Low cut (full)", 480), ("mixer/20/slot/1/Decay (full)", 2),
            ("mixer/20/slot/1/Dry (full)", 0), ("mixer/20/slot/1/Early reflections (full)", 10),
            ("mixer/20/slot/1/Wet (full)", 58),

            # Snare (4) & Clap (5): LPF 11.5 kHz removes raw sample-0 Nyquist transient ringing / ISP spikes
            ("mixer/4/slot/1/Band 1 type", 1), ("mixer/4/slot/1/Band 1 freq", 130),
            ("mixer/4/slot/1/Band 7 type", 3), ("mixer/4/slot/1/Band 7 freq", 11500),
            ("mixer/5/slot/1/Band 1 type", 1), ("mixer/5/slot/1/Band 1 freq", 160),
            ("mixer/5/slot/1/Band 7 type", 3), ("mixer/5/slot/1/Band 7 freq", 11500),

            # Mirror EQs on harmonic layers vs Lead:
            # Pad E (8): HPF 180 Hz, carve -2.5dB @ 320 Hz, carve -2.8dB @ 1.5 kHz for Lead Piano
            ("mixer/8/slot/1/Band 1 type", 1), ("mixer/8/slot/1/Band 1 freq", 180),
            ("mixer/8/slot/1/Band 3 type", 6), ("mixer/8/slot/1/Band 3 freq", 320), ("mixer/8/slot/1/Band 3 gain", -2.5),
            ("mixer/8/slot/1/Band 5 type", 6), ("mixer/8/slot/1/Band 5 freq", 1500), ("mixer/8/slot/1/Band 5 gain", -2.8),
            # Chords Piano VoiceLed (9): HPF 165 Hz, carve -2.5dB @ 1.6 kHz for Lead Piano
            ("mixer/9/slot/1/Band 1 type", 1), ("mixer/9/slot/1/Band 1 freq", 165),
            ("mixer/9/slot/1/Band 5 type", 6), ("mixer/9/slot/1/Band 5 freq", 1600), ("mixer/9/slot/1/Band 5 gain", -2.5),
            # Chords Stab (10): HPF 165 Hz, carve -2.2dB @ 310 Hz
            ("mixer/10/slot/1/Band 1 type", 1), ("mixer/10/slot/1/Band 1 freq", 165),
            ("mixer/10/slot/1/Band 3 type", 6), ("mixer/10/slot/1/Band 3 freq", 310), ("mixer/10/slot/1/Band 3 gain", -2.2),
            # Lead Piano (11): HPF 240 Hz, presence +1.8dB @ 1.8 kHz
            ("mixer/11/slot/1/Band 1 type", 1), ("mixer/11/slot/1/Band 1 freq", 240),
            ("mixer/11/slot/1/Band 5 type", 6), ("mixer/11/slot/1/Band 5 freq", 1800), ("mixer/11/slot/1/Band 5 gain", 1.8),
            # Counter Pizz (12): HPF 175 Hz
            ("mixer/12/slot/1/Band 1 type", 1), ("mixer/12/slot/1/Band 1 freq", 175),
            # Chords High +12 (13): HPF 550 Hz
            ("mixer/13/slot/1/Band 1 type", 1), ("mixer/13/slot/1/Band 1 freq", 550),
        ]
        mcp.call("set_parameters", {"changes": [{"path": p, "value": v} for p, v in fx_changes]})

        # 6. Playlist Tracks 1..13
        track_names = [
            (1, "808 Apollo (Couplets)"),
            (2, "808 Blxckout (Refrains)"),
            (3, "Kick (Composite Lock)"),
            (4, "Snare & Clap"),
            (5, "Hi-Hats Trap (Unified)"),
            (6, "Rim, Shaker & OpenHat"),
            (7, "Chords Pad E (440Hz)"),
            (8, "Chords Piano VoiceLed"),
            (9, "Chords Stab E (Power)"),
            (10, "Chords High +12 & Riser"),
            (11, "Lead Piano (A-A'-A''-B)"),
            (12, "Counter Pizz E (Echoes)"),
            (13, "Chants & FX Room Cuts"),
        ]
        mcp.batch([{"tool": "patch", "arguments": {"path": f"track/{t}", "set": {"name": name}}} for t, name in track_names])

        # 7. Orchestrate master_sketch.mid across 96 bars (Em7 -> Cmaj7 -> Am9 -> Bm7)
        pat_counter = 1
        queued = [{"tool": "edit_pattern", "arguments": {"pattern": 1, "name": "Chords VoiceLed - Intro 8m", "bars": 8}}]

        def new_pattern(name: str, bars: int) -> int:
            nonlocal pat_counter
            pat_counter += 1
            queued.append({"tool": "add_pattern", "arguments": {"name": name, "bars": bars}})
            return pat_counter

        def add_notes(pat: int, ch: int, notes: list):
            queued.append({"tool": "set_notes", "arguments": {"pattern": pat, "channel": ch, "notes": notes}})

        def place(pat: int, track: int, start_bar: int):
            queued.append({"tool": "add_clip", "arguments": {"pattern": pat, "track": track, "start": (start_bar - 1) * 4.0}})

        chord_roots = [64, 60, 57, 59]  # E4, C4, A3, B3

        # ---- CHORDS PIANO VOICE-LED (Track 8, Ch 14) + CHORDS PAD E (Track 7, Ch 13) ----
        # Each bar sustains 2.85 beats so beats 3.0..4.0 stay open for Counter Pizz call-and-response (RT60 + 0 clashes).
        def make_voiceled_chords(bars=8, vel=72, mute_last_beat=True):
            notes = []
            for b in range(bars):
                ln = 2.65 if (mute_last_beat and b == bars - 1) else 2.85
                for pitch in voicings[b % 4]:
                    notes.append({"start": b * 4.0, "length": ln, "pitch": pitch, "velocity": vel})
            return notes

        def make_pad_roots(bars=8, vel=74, mute_last_beat=True):
            notes = []
            for b in range(bars):
                ln = 2.65 if (mute_last_beat and b == bars - 1) else 2.85
                notes.append({"start": b * 4.0, "length": ln, "pitch": chord_roots[b % 4], "velocity": vel})
            return notes

        # Pattern 1 = Intro VoiceLed Piano + Pad (soft contrast >= 3 dB below Drop)
        add_notes(1, 14, make_voiceled_chords(8, vel=48, mute_last_beat=True))
        add_notes(1, 13, make_pad_roots(8, vel=46, mute_last_beat=True))
        place(1, 8, 1)

        p_chords_c1b = new_pattern("Chords VoiceLed - Couplet 1b", 8)
        add_notes(p_chords_c1b, 14, make_voiceled_chords(8, vel=74, mute_last_beat=True))
        add_notes(p_chords_c1b, 13, make_pad_roots(8, vel=70, mute_last_beat=True))
        place(p_chords_c1b, 8, 25)

        p_chords_pont = new_pattern("Chords VoiceLed - Pont", 8)
        add_notes(p_chords_pont, 14, make_voiceled_chords(8, vel=18, mute_last_beat=True))
        add_notes(p_chords_pont, 13, make_pad_roots(8, vel=16, mute_last_beat=True))
        place(p_chords_pont, 8, 33)

        p_chords_outro = new_pattern("Chords VoiceLed - Outro", 8)
        add_notes(p_chords_outro, 14, make_voiceled_chords(8, vel=56, mute_last_beat=False))
        add_notes(p_chords_outro, 13, make_pad_roots(8, vel=54, mute_last_beat=False))
        place(p_chords_outro, 8, 89)

        # ---- CHORDS STAB E (Power Chord E+B, Track 9, Ch 15) & CHORDS HIGH +12 / RISER (Track 10, Ch 18 / 19) ----
        def make_stab_notes(bars=8, vel=94, mode="drop"):
            notes = []
            for b in range(bars):
                r = chord_roots[b % 4]
                is_last = (b == bars - 1)
                if mode == "drop":
                    notes.append({"start": b * 4.0 + 0.0, "length": 1.25, "pitch": r, "velocity": vel})
                    if not is_last:
                        notes.append({"start": b * 4.0 + 1.5, "length": 0.85, "pitch": r, "velocity": vel - 6})
                elif mode == "staccato":
                    notes.append({"start": b * 4.0 + 0.0, "length": 0.45, "pitch": r, "velocity": vel - 4})
                    if not is_last:
                        notes.append({"start": b * 4.0 + 1.5, "length": 0.35, "pitch": r, "velocity": vel - 10})
            return notes

        p_stab_drop = new_pattern("Chords Stab - Drop 8m", 8)
        add_notes(p_stab_drop, 15, make_stab_notes(8, vel=94, mode="drop"))
        add_notes(p_stab_drop, 14, make_voiceled_chords(8, vel=72, mute_last_beat=True))
        for sb in [9, 41, 49, 73, 81]:
            place(p_stab_drop, 9, sb)

        p_stab_c2 = new_pattern("Chords Stab - Couplet 2", 8)
        add_notes(p_stab_c2, 15, make_stab_notes(8, vel=86, mode="staccato"))
        add_notes(p_stab_c2, 14, make_voiceled_chords(8, vel=70, mute_last_beat=True))
        place(p_stab_c2, 9, 57)
        place(p_stab_c2, 9, 65)

        p_stab_hi = new_pattern("Chords High +12 - Climax", 8)
        add_notes(p_stab_hi, 18, make_stab_notes(8, vel=86, mode="drop"))
        place(p_stab_hi, 10, 49)
        place(p_stab_hi, 10, 73)
        place(p_stab_hi, 10, 81)

        p_riser = new_pattern("Chords Riser - 4 beats", 1)
        add_notes(p_riser, 19, [{"start": 0.0, "length": 3.95, "pitch": 64, "velocity": 86}])
        place(p_riser, 10, 8)
        place(p_riser, 10, 72)

        p_riser_pont = new_pattern("Chords Riser - Pont 4 beats", 1)
        add_notes(p_riser_pont, 19, [{"start": 0.0, "length": 3.95, "pitch": 64, "velocity": 56}])
        place(p_riser_pont, 10, 40)

        # ---- LEAD PIANO (Track 11, Ch 16) — Exact Top-Line A-A'-A''-B from master_sketch.mid! ----
        def make_lead_from_sketch(bars=8, vel_base=96, mode="full", oct_shift=0, mute_last_beat=True):
            notes = []
            for rep in range(bars // 4):
                off_b = rep * 4
                for b, st, ln, pt in lead_phrase:
                    abs_b = off_b + b
                    t = abs_b * 4.0 + st
                    if mute_last_beat and abs_b == bars - 1 and t >= (bars * 4.0 - 1.5):
                        continue
                    if mode == "skeleton" and st not in (0.0, 2.0):
                        continue  # Couplet 1a: skeletal strong beats only -> progressive density!
                    jv = max(50, min(115, vel_base + random.randint(-4, 4)))
                    notes.append({"start": t, "length": ln, "pitch": pt + oct_shift, "velocity": jv})
            return notes

        p_lead_intro = new_pattern("Lead Piano - Intro Skeleton", 8)
        add_notes(p_lead_intro, 16, make_lead_from_sketch(8, vel_base=56, mode="skeleton", mute_last_beat=True))
        place(p_lead_intro, 11, 1)

        p_lead_drop = new_pattern("Lead Piano - Drop Full", 8)
        add_notes(p_lead_drop, 16, make_lead_from_sketch(8, vel_base=106, mode="full", mute_last_beat=True))
        for sb in [9, 41, 49, 73, 81]:
            place(p_lead_drop, 11, sb)

        p_lead_c1 = new_pattern("Lead Piano - Couplet 1", 8)
        add_notes(p_lead_c1, 16, make_lead_from_sketch(8, vel_base=84, mode="skeleton", mute_last_beat=True))
        place(p_lead_c1, 11, 17)
        place(p_lead_c1, 11, 25)

        p_lead_pont = new_pattern("Lead Piano - Pont", 8)
        add_notes(p_lead_pont, 16, make_lead_from_sketch(8, vel_base=24, mode="full", mute_last_beat=True))
        place(p_lead_pont, 11, 33)

        p_lead_c2 = new_pattern("Lead Piano - Couplet 2", 8)
        add_notes(p_lead_c2, 16, make_lead_from_sketch(8, vel_base=88, mode="full", mute_last_beat=True))
        place(p_lead_c2, 11, 57)
        place(p_lead_c2, 11, 65)

        p_lead_outro = new_pattern("Lead Piano - Outro", 8)
        add_notes(p_lead_outro, 16, [
            {"start": 0.0, "length": 0.6, "pitch": 76, "velocity": 76},
            {"start": 2.0, "length": 0.9, "pitch": 79, "velocity": 74},
            {"start": 4.0, "length": 0.6, "pitch": 76, "velocity": 70},
            {"start": 8.0, "length": 0.9, "pitch": 76, "velocity": 66},
            {"start": 16.0, "length": 1.2, "pitch": 76, "velocity": 60},
        ])
        place(p_lead_outro, 11, 89)

        # ---- COUNTER PIZZ E (Track 12, Ch 17) — Echo ONLY in Lead's 3.0..3.75 silence & >= 1 octave below! ----
        # Because Lead ends at ~2.9 and Counter hits at 3.25 / 3.5 in E3..B3 (12+ st below Lead E4..A5),
        # lead_overlap_conflict == 0.000!
        def make_counter_echo(bars=8, vel=84):
            notes = []
            # Chord-tone echoes at beat 3.0 & 3.5 strictly after Chords release (2.85) -> zero harmonic/attack clashes!
            echo_bars = {
                0: [(3.0, 52), (3.5, 55)],  # E3 -> G3 (Em7 chord tones)
                1: [(3.0, 55), (3.5, 52)],  # G3 -> E3 (Cadd9 chord tones)
                2: [(3.0, 55), (3.5, 52)],  # G3 -> E3 (Am7 chord tones)
                3: [(3.0, 57), (3.5, 54)],  # A3 -> F#3 (Bm7 chord tones)
            }
            for b in range(bars - 1):  # skip bar 8 for clean pivot mute!
                for st, pt in echo_bars[b % 4]:
                    notes.append({"start": b * 4.0 + st, "length": 0.35, "pitch": pt, "velocity": vel})
            return notes

        p_pizz_echo = new_pattern("Counter Pizz - Echoes 8m", 8)
        add_notes(p_pizz_echo, 17, make_counter_echo(8, vel=84))
        for sb in [9, 25, 41, 49, 57, 65, 73, 81]:
            place(p_pizz_echo, 12, sb)

        # ---- 808 BLXCKOUT (Refrains, Track 2, Ch 4) & 808 APOLLO (Couplets, Track 1, Ch 3) ----
        # Matches bass_roots E -> C -> A -> B (Em7 -> Cmaj7 -> Am9 -> Bm7) with contrary/oblique lock!
        def make_808_blxckout(climax=False):
            notes = []
            for b in range(8):
                is_last = (b == 7)
                m4 = b % 4
                if m4 == 0:
                    hits = [(0.0, 1.45, 28, 112, False), (1.5, 1.20, 28, 104, False), (2.75, 1.15, 40 if climax else 28, 102, True)]
                elif m4 == 1:
                    hits = [(0.0, 1.45, 36, 112, False), (1.5, 1.20, 36, 104, False), (2.75, 1.15, 48 if climax else 36, 100, True)]
                elif m4 == 2:
                    hits = [(0.0, 1.45, 33, 112, False), (1.5, 1.20, 33, 104, False), (2.75, 1.15, 45 if climax else 33, 102, True)]
                else:
                    if is_last:
                        hits = [(0.0, 1.45, 35, 110, False), (1.5, 0.90, 35, 104, False)]
                    else:
                        hits = [(0.0, 1.45, 35, 110, False), (1.5, 1.20, 35, 104, False), (3.0, 0.85, 38 if climax else 35, 100, True)]
                for st, ln, pt, vel, porta in hits:
                    n = {"start": b * 4.0 + st, "length": ln, "pitch": pt, "velocity": vel}
                    if porta:
                        n["porta"] = True
                    notes.append(n)
            return notes

        p_808_ref1 = new_pattern("808 Blxckout - Refrain 1/2a", 8)
        add_notes(p_808_ref1, 4, make_808_blxckout(climax=False))
        place(p_808_ref1, 2, 9)
        place(p_808_ref1, 2, 41)

        p_808_climax = new_pattern("808 Blxckout - Climax", 8)
        add_notes(p_808_climax, 4, make_808_blxckout(climax=True))
        place(p_808_climax, 2, 49)
        place(p_808_climax, 2, 73)
        place(p_808_climax, 2, 81)

        def make_808_apollo(syncopated=False):
            notes = []
            roots = [28, 36, 33, 35]
            for b in range(8):
                root = roots[b % 4]
                if b == 7:
                    hits = [(0.0, 0.85, root, 108, False), (1.5, 0.70, root, 100, False)]
                elif syncopated:
                    hits = [(0.0, 0.80, root, 108, False), (1.5, 0.55, root + 12, 100, True), (2.75, 0.75, root, 102, False)]
                else:
                    hits = [(0.0, 0.85, root, 106, False), (1.5, 0.65, root, 98, False), (2.75, 0.75, root, 98, False)]
                for st, ln, pt, vel, porta in hits:
                    n = {"start": b * 4.0 + st, "length": ln, "pitch": pt, "velocity": vel}
                    if porta:
                        n["porta"] = True
                    notes.append(n)
            return notes

        p_808_c1 = new_pattern("808 Apollo - Couplet 1", 8)
        add_notes(p_808_c1, 3, make_808_apollo(syncopated=False))
        place(p_808_c1, 1, 17)
        place(p_808_c1, 1, 25)

        p_808_c2 = new_pattern("808 Apollo - Couplet 2", 8)
        add_notes(p_808_c2, 3, make_808_apollo(syncopated=True))
        place(p_808_c2, 1, 57)
        place(p_808_c2, 1, 65)

        # ---- KICK (Track 3, Ch 1 + Ch 2) ----
        def make_kick_notes(bars=8, dense=True, mute_last_beat=True):
            notes = []
            for b in range(bars):
                if b == bars - 1 and mute_last_beat:
                    times = [0.0, 1.5]
                elif b % 2 == 0:
                    times = [0.0, 1.5, 2.75]
                else:
                    times = [0.0, 1.5, 2.75, 3.5] if dense else [0.0, 1.5, 2.75]
                for t in times:
                    notes.append({"start": b * 4.0 + t, "length": 0.40, "pitch": 60, "velocity": 112 if t == 0.0 else 102})
            return notes

        p_kick_ref = new_pattern("Kick - Refrain 8m", 8)
        kr = make_kick_notes(8, dense=True, mute_last_beat=True)
        add_notes(p_kick_ref, 1, kr)
        add_notes(p_kick_ref, 2, kr)
        for sb in [9, 41, 49, 73, 81]:
            place(p_kick_ref, 3, sb)

        p_kick_ver = new_pattern("Kick - Couplet 8m", 8)
        kv = make_kick_notes(8, dense=False, mute_last_beat=True)
        add_notes(p_kick_ver, 1, kv)
        add_notes(p_kick_ver, 2, kv)
        for sb in [17, 25, 57, 65]:
            place(p_kick_ver, 3, sb)

        # ---- SNARE & CLAP (Track 4, Ch 5, 6, 7) ----
        def make_sc(bars=8, fill_end=True, mute_last_beat=False):
            sn, cl, tcl = [], [], []
            for b in range(bars):
                sn.append({"start": b * 4.0 + 2.0, "length": 0.42, "pitch": 60, "velocity": 108})
                cl.append({"start": b * 4.0 + 2.012, "length": 0.42, "pitch": 60, "velocity": 100})
                tcl.append({"start": b * 4.0 + 1.994, "length": 0.42, "pitch": 60, "velocity": 98})
                if b in (1, 3, 5):
                    sn.append({"start": b * 4.0 + 3.75, "length": 0.22, "pitch": 60, "velocity": 60})
                if b == bars - 1 and fill_end and not mute_last_beat:
                    for sub, vel in [(3.0, 74), (3.25, 82), (3.5, 94), (3.75, 106)]:
                        sn.append({"start": b * 4.0 + sub, "length": 0.18, "pitch": 60, "velocity": vel})
            return sn, cl, tcl

        p_sc_fill = new_pattern("Snare/Clap - Drop Fill", 8)
        sn, cl, tcl = make_sc(8, fill_end=True, mute_last_beat=False)
        add_notes(p_sc_fill, 5, sn)
        add_notes(p_sc_fill, 6, cl)
        add_notes(p_sc_fill, 7, tcl)
        for sb in [9, 41, 49, 73, 81]:
            place(p_sc_fill, 4, sb)

        p_sc_mute = new_pattern("Snare/Clap - Couplet Mute", 8)
        snm, clm, tclm = make_sc(8, fill_end=False, mute_last_beat=True)
        add_notes(p_sc_mute, 5, snm)
        add_notes(p_sc_mute, 6, clm)
        add_notes(p_sc_mute, 7, tclm)
        for sb in [17, 25, 57, 65]:
            place(p_sc_mute, 4, sb)

        p_sc_pont = new_pattern("Snare - Pont Buildup", 8)
        psn = [{"start": b * 4.0 + 2.0, "length": 0.35, "pitch": 60, "velocity": 26 + (b - 4) * 4} for b in range(4, 7)]
        for step in range(8):
            psn.append({"start": 7 * 4.0 + step * 0.5, "length": 0.18, "pitch": 60, "velocity": 24 + step * 3})
        add_notes(p_sc_pont, 5, psn)
        place(p_sc_pont, 4, 33)

        p_sc_outro = new_pattern("Snare/Clap - Outro 4m", 4)
        sno, clo, _ = make_sc(4, fill_end=False, mute_last_beat=True)
        add_notes(p_sc_outro, 5, sno)
        add_notes(p_sc_outro, 6, clo)
        place(p_sc_outro, 4, 89)

        # ---- UNIFIED HI-HATS (Track 5, Ch 8 + Ch 9) ----
        def make_hats(bars=8, energy="medium", mute_last_beat=True):
            hn, hs = [], []
            for b in range(bars):
                for step in range(8):
                    t = b * 4.0 + step * 0.5
                    if mute_last_beat and t >= bars * 4.0 - 1.0:
                        continue
                    vel = (104 if step % 2 == 0 else 68) + random.randint(-5, 5)
                    if b % 2 == 1 and step == 3:
                        for k in range(3):
                            rt = b * 4.0 + 1.5 + k * (0.5 / 3.0)
                            hn.append({"start": round(rt, 4), "length": 0.11, "pitch": 60 - k * 2, "velocity": 82 + k * 7})
                        continue
                    if energy == "high" and b % 4 == 3 and step in (6, 7) and not (mute_last_beat and b == bars - 1):
                        for k in range(4):
                            rt = b * 4.0 + 3.0 + k * 0.25
                            hn.append({"start": round(rt, 4), "length": 0.11, "pitch": 60 - k * 2, "velocity": 94 - k * 6})
                        continue
                    hn.append({"start": t, "length": 0.18, "pitch": 60, "velocity": vel})
                    if energy in ("medium", "high") and step in (1, 5, 7):
                        hs.append({"start": t + 0.25, "length": 0.16, "pitch": 60, "velocity": 76 if energy == "medium" else 90})
            return hn, hs

        p_hats_c1a = new_pattern("Hats - Couplet 1a", 8)
        hn, _ = make_hats(8, "low", mute_last_beat=True)
        add_notes(p_hats_c1a, 8, hn)
        place(p_hats_c1a, 5, 17)

        p_hats_med = new_pattern("Hats - Medium 8m", 8)
        hn, hs = make_hats(8, "medium", mute_last_beat=True)
        add_notes(p_hats_med, 8, hn)
        add_notes(p_hats_med, 9, hs)
        for sb in [25, 57, 65]:
            place(p_hats_med, 5, sb)

        p_hats_hi = new_pattern("Hats - Drop High 8m", 8)
        hn, hs = make_hats(8, "high", mute_last_beat=True)
        add_notes(p_hats_hi, 8, hn)
        add_notes(p_hats_hi, 9, hs)
        for sb in [9, 41, 49, 73, 81]:
            place(p_hats_hi, 5, sb)

        p_hats_outro = new_pattern("Hats - Outro 4m", 4)
        hn, _ = make_hats(4, "low", mute_last_beat=True)
        add_notes(p_hats_outro, 8, hn)
        place(p_hats_outro, 5, 89)

        # ---- RIM, SHAKER & OPENHAT (Track 6, Ch 10, 11, 12) ----
        p_rim_intro = new_pattern("Rim & Shaker - Intro M5-8", 4)
        add_notes(p_rim_intro, 11, [{"start": b * 4.0 + t, "length": 0.2, "pitch": 60, "velocity": 46} for b in range(3) for t in (1.5, 3.5)])
        add_notes(p_rim_intro, 12, [{"start": b * 4.0 + s * 0.5, "length": 0.15, "pitch": 60, "velocity": 36 if s % 2 == 0 else 24} for b in range(3) for s in range(8)])
        place(p_rim_intro, 6, 5)
        place(p_rim_intro, 6, 33)

        p_perc_c1 = new_pattern("Rim - Couplet 1a", 8)
        add_notes(p_perc_c1, 11, [{"start": b * 4.0 + t, "length": 0.2, "pitch": 60, "velocity": 80} for b in range(7) for t in (1.75, 3.5)])
        place(p_perc_c1, 6, 17)

        p_perc_c_full = new_pattern("Rim & Shaker - Couplet Full", 8)
        add_notes(p_perc_c_full, 11, [{"start": b * 4.0 + t, "length": 0.2, "pitch": 60, "velocity": 82} for b in range(7) for t in (1.75, 3.5)])
        add_notes(p_perc_c_full, 12, [{"start": b * 4.0 + s * 0.5, "length": 0.15, "pitch": 60, "velocity": 72 if s % 2 == 0 else 52} for b in range(7) for s in range(8)])
        for sb in [25, 57, 65]:
            place(p_perc_c_full, 6, sb)

        p_perc_drop2 = new_pattern("OpenHat & Shaker - Drop 8m", 8)
        add_notes(p_perc_drop2, 10, [{"start": b * 4.0 + 0.5, "length": 0.45, "pitch": 60, "velocity": 86} for b in (0, 2, 4, 6)])
        add_notes(p_perc_drop2, 12, [{"start": b * 4.0 + s * 0.5, "length": 0.15, "pitch": 60, "velocity": 76 if s % 2 == 0 else 54} for b in range(7) for s in range(8)])
        for sb in [41, 49]:
            place(p_perc_drop2, 6, sb)

        p_perc_climax = new_pattern("OpenHat, Rim & Shaker - Climax", 8)
        add_notes(p_perc_climax, 10, [{"start": b * 4.0 + 0.5, "length": 0.45, "pitch": 60, "velocity": 90} for b in (0, 2, 4, 6)])
        add_notes(p_perc_climax, 11, [{"start": b * 4.0 + t, "length": 0.2, "pitch": 60, "velocity": 82} for b in range(7) for t in (1.75, 3.5)])
        add_notes(p_perc_climax, 12, [{"start": b * 4.0 + s * 0.5, "length": 0.15, "pitch": 60, "velocity": 80 if s % 2 == 0 else 56} for b in range(7) for s in range(8)])
        for sb in [73, 81]:
            place(p_perc_climax, 6, sb)

        # ---- CHANTS ACCENT (Track 13, Ch 20) ----
        p_chants = new_pattern("Chants - Sparse Accent 8m", 8)
        add_notes(p_chants, 20, [
            {"start": 1 * 4.0 + 3.5, "length": 0.5, "pitch": 64, "velocity": 86},
            {"start": 5 * 4.0 + 3.5, "length": 0.5, "pitch": 64, "velocity": 88},
        ])
        for sb in [9, 41, 49, 73, 81]:
            place(p_chants, 13, sb)

        # ---- FX ROOM TRANSITION CUTS (Track 13, Mixer 20 volume automation) ----
        pts = [{"position": 0.0, "value": 0.42, "shape": "hold"}]
        for pivot_bar in [8, 32, 40, 56, 72]:
            cut_start = (pivot_bar - 1) * 4.0 + 3.0
            cut_end = pivot_bar * 4.0
            pts.append({"position": cut_start, "value": 0.0, "shape": "hold"})
            pts.append({"position": cut_end, "value": 0.42, "shape": "hold"})
        pts.append({"position": 96 * 4.0, "value": 0.42, "shape": "hold"})
        queued.append({"tool": "add_automation", "arguments": {
            "mixer": 20, "param": "volume", "name": "FX Room Transition Cuts",
            "points": pts, "track": 13, "start": 0.0, "length": 96 * 4.0,
        }})

        mcp.batch(queued)
        mcp.call("save_project", {})
        song = mcp.call("get_song", {})
        print("Saved active5/projet.gridline:", json.dumps({
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


if __name__ == "__main__":
    build()
