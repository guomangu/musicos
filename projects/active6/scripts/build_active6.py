#!/usr/bin/env python3
"""
Build script for projects/active6/projet.gridline
"Airey Horizon" — 33 bars @ 132 BPM (60.00 s = 1 min 00 s), G# minor.
100% built from sounds present in projects/active6/newtest.gridline
(plus BPM-locked / DC-cleaned / tuned derivatives in projects/active6/samples/).
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

PROJ_DIR = f"{ROOT}/projects/active6"
PROJ_FILE = f"{PROJ_DIR}/projet.gridline"
SKETCH_MIDI = f"{PROJ_DIR}/master_sketch.mid"
ORCH_MIDI = f"{PROJ_DIR}/airey_horizon.mid"
RENDERS_DIR = f"{PROJ_DIR}/renders"
SAMPLES_DIR = f"{PROJ_DIR}/samples"
TEMPLATE = "/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline"

BPM = 132.0
random.seed(42)


def build_master_sketch_and_gate():
    """Verify master_sketch.mid passes the strict 7-metric musicality gate."""
    chord_pcs = [
        [4, 8, 11],     # Emaj
        [1, 4, 8, 11],  # C#m7
        [11, 3, 6],     # B
        [8, 11, 3, 6],  # G#m7
    ]
    voicings = optimize_voice_leading(chord_pcs, low_midi=51, high_midi=62)

    lead_phrase = [
        # Bar 0 (A on Emaj)
        (0, 0.0, 0.45, 68), (0, 0.5, 0.45, 71), (0, 1.0, 0.45, 68), (0, 1.5, 0.45, 66), (0, 2.0, 0.75, 64),
        # Bar 1 (A' on Emaj -> C#m7)
        (1, 0.0, 0.45, 68), (1, 0.5, 0.45, 71), (1, 1.0, 0.45, 73), (1, 1.5, 0.45, 75), (1, 2.0, 0.75, 73),
        # Bar 2 (A'' Apex F#5=78 on B)
        (2, 0.0, 0.45, 75), (2, 0.5, 0.45, 78), (2, 1.0, 0.75, 75), (2, 2.0, 0.45, 71), (2, 2.5, 0.45, 73),
        # Bar 3 (B cadence on G#m7)
        (3, 0.0, 0.45, 71), (3, 0.5, 0.45, 70), (3, 1.0, 0.75, 68), (3, 2.0, 0.45, 71), (3, 2.5, 0.45, 68),
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
                bass.notes.append(pretty_midi.Note(100, 40, s * beat_s, (s + 3.8) * beat_s))
            elif b == 1:
                bass.notes.append(pretty_midi.Note(100, 40, s * beat_s, (s + 1.9) * beat_s))
                bass.notes.append(pretty_midi.Note(100, 37, (s + 2.0) * beat_s, (s + 3.8) * beat_s))
            elif b == 2:
                bass.notes.append(pretty_midi.Note(100, 35, s * beat_s, (s + 3.8) * beat_s))
            else:
                bass.notes.append(pretty_midi.Note(100, 32, s * beat_s, (s + 2.8) * beat_s))
                bass.notes.append(pretty_midi.Note(95, 30, (s + 3.0) * beat_s, (s + 3.8) * beat_s))

            if b == 1:
                for p in voicings[0]:
                    chords.notes.append(pretty_midi.Note(74, p, s * beat_s, (s + 1.9) * beat_s))
                for p in voicings[1]:
                    chords.notes.append(pretty_midi.Note(74, p, (s + 2.0) * beat_s, (s + 3.5) * beat_s))
            else:
                for p in voicings[b]:
                    chords.notes.append(pretty_midi.Note(74, p, s * beat_s, (s + 3.5) * beat_s))

        for b, st, ln, pt in lead_phrase:
            t0 = ((bar0 + b) * 4.0 + st) * beat_s
            lead.notes.append(pretty_midi.Note(92, pt, t0, t0 + ln * beat_s))

    pm.instruments.extend([bass, chords, lead])
    pm.write(SKETCH_MIDI)

    gate = audit_midi_file(SKETCH_MIDI, expected_key="G# minor", min_pitch=24, max_pitch=84, strict_musicality=True)
    print("Master Sketch Gate:", json.dumps({
        "verdict": gate["musicality_verdict"],
        "gate_passed": gate["gate_passed"],
        "metrics": gate["musicality_metrics"],
        "failures": gate["musicality_failures"],
    }, indent=2))
    if not gate["gate_passed"]:
        raise RuntimeError(f"BLOCKING GATE FAILED: {gate['musicality_failures']}")

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
            "clientInfo": {"name": "build_active6", "version": "1.0"},
        })
        self._notify("notifications/initialized", {})

    def _notify(self, method: str, params: dict):
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method: str, params: dict):
        self._id += 1
        rid = self._id
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP EOF: {self.proc.stderr.read()}")
            resp = json.loads(line)
            if resp.get("id") == rid:
                if "error" in resp:
                    raise RuntimeError(f"RPC error {method}: {resp['error']}")
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
        out = []
        for i in range(0, len(calls), 100):
            r = self.call("batch", {"calls": calls[i:i + 100]})
            if isinstance(r, dict) and r.get("errors"):
                raise RuntimeError(f"Batch errors: {r['errors']}")
            out.append(r)
        return out

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=15)


def build():
    voicings, lead_phrase = build_master_sketch_and_gate()
    shutil.copyfile(TEMPLATE, PROJ_FILE)
    mcp = GridlineMCP(PROJ_FILE, RENDERS_DIR)

    try:
        # 1. Song settings
        mcp.call("set_song", {
            "title": "Airey Horizon",
            "genre": "Melodic Progressive Trap",
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
            (1, "Kick", 17, 0.80, 0.0, 0.0),
            (2, "808 (4) Smooth", 18, 0.82, 0.0, -0.8),
            (3, "808 (5) Climax", 18, 0.80, 0.0, -0.8),
            (4, "Clap (7)", 17, 0.90, 0.0, 0.0),
            (5, "Snare Native", 17, 0.78, 0.0, 0.0),
            (6, "Hi-Hats Closed", 17, 0.74, 0.0, 0.0),
            (7, "OpenHat & Percs", 17, 0.74, 0.0, 0.0),
            (8, "NUAN Pad Open", 19, 0.72, 0.0, -0.25),
            (9, "NUAN Pad Filtered", 19, 0.80, 0.0, -0.28),
            (10, "NUAN High +12 & Riser", 19, 0.80, 0.0, -0.18),
            (11, "Twin String Lead", 19, 1.04, 0.04, -0.12),
            (12, "Synth Counter", 19, 1.06, -0.08, -0.22),
            (13, "Chants & FX", 19, 0.76, 0.0, -0.15),
            (17, "BUS Drums", 0, 0.94, 0.0, 0.0),
            (18, "BUS 808", 0, 0.76, 0.0, -1.0),
            (19, "BUS Music", 0, 1.00, 0.0, -0.18),
            (20, "FX Room", 0, 0.28, 0.0, 0.0),
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

        # Sidechains & FX Room sends
        sends = [
            (1, 18, 1.0, True),   # Kick -> BUS 808 sidechain
            (1, 19, 0.70, True),  # Kick -> BUS Music micro-ducking
            (4, 20, 0.04, False),
            (5, 20, 0.04, False),
            (7, 20, 0.05, False),
            (8, 20, 0.07, False),
            (9, 20, 0.08, False),
            (10, 20, 0.08, False),
            (11, 20, 0.10, False),
            (12, 20, 0.08, False),
            (13, 20, 0.08, False),
        ]
        mcp.batch([{"tool": "set_send", "arguments": {"mixer": m, "to": to, "level": lvl, "sidechain": sc}} for m, to, lvl, sc in sends])

        # 3. Channels 1..23 (100% active6/newtest.gridline sounds + derived active6/samples/)
        SB = f"{ROOT}/soundbanks"
        channels_spec = [
            # 1-2 Kicks
            ("Kick (1)", "gl.sampler", None, 1, f"{SB}/JMizzy The Best Drum Kits/JUGG DRUM KIT/Kicks/(SXJ) Kick (1).wav", 0.80, 0.0),
            ("Kick Native", "gl.kick", None, 1, None, 0.52, 0.0),
            # 3-4 808s (both using tuned 440.0 Hz 808 (4) — mixer 3 adds warm Climax drive!)
            ("808 (4) Smooth", "gl.sampler", None, 2, f"{SAMPLES_DIR}/sxj_808_4_clean.wav", 0.82, 0.0),
            ("808 (4) Climax", "gl.sampler", None, 3, f"{SAMPLES_DIR}/sxj_808_4_clean.wav", 0.82, 0.0),
            # 5-6 Clap & Snare
            ("Clap (7)", "gl.sampler", None, 4, f"{SB}/JMizzy The Best Drum Kits/JUGG DRUM KIT/Claps/(SXJ) Clap (7).wav", 0.90, 0.0),
            ("Snare Native", "gl.snare", None, 5, None, 0.74, 0.0),
            # 7-12 Hats & Percs
            ("Hi-Hat Bat Wing", "gl.sampler", None, 6, f"{SAMPLES_DIR}/hh_batwing_clean.wav", 0.66, 0.08),
            ("Hi-Hat Bankroll", "gl.sampler", None, 6, f"{SAMPLES_DIR}/hh_bankroll_clean.wav", 0.64, -0.08),
            ("Open-Hat Infinite", "gl.sampler", None, 7, f"{SAMPLES_DIR}/hh_infinite_clean.wav", 0.62, -0.14),
            ("Perc Hat Zap", "gl.sampler", None, 7, f"{SB}/CHRISSA SJE & NIKOS - INFINITE (SOUND PACK)/05. Drum Kit ∞/02. Hi-hats/03. hh - dumb (zap).wav", 0.58, 0.16),
            ("Perc 8", "gl.sampler", None, 7, f"{SAMPLES_DIR}/sxj_percs8_clean.wav", 0.68, -0.18),
            ("Perc Triangle", "gl.sampler", None, 7, f"{SB}/JMizzy The Best Drum Kits/DY 808 MAFIA KIT/DY Trap Triangle (1).wav", 0.60, 0.18),
            # 13-19 NUAN SONAR derived layers (Open vs Filtered!)
            ("Chords NUAN A Open", "gl.sampler", None, 8, f"{SAMPLES_DIR}/nuan_4bar_a_132.wav", 0.76, 0.0),
            ("Chords NUAN A Filt", "gl.sampler", None, 9, f"{SAMPLES_DIR}/nuan_4bar_a_132.wav", 0.72, 0.0),
            ("Chords NUAN B Open", "gl.sampler", None, 8, f"{SAMPLES_DIR}/nuan_4bar_b_132.wav", 0.78, 0.0),
            ("Chords NUAN B Filt", "gl.sampler", None, 9, f"{SAMPLES_DIR}/nuan_4bar_b_132.wav", 0.72, 0.0),
            ("Chords NUAN Outro", "gl.sampler", None, 9, f"{SAMPLES_DIR}/nuan_5bar_outro_132.wav", 0.70, 0.0),
            ("Chords NUAN High+12", "gl.sampler", None, 10, f"{SAMPLES_DIR}/nuan_4bar_high12_132.wav", 0.62, 0.0),
            ("Perc NUAN Riser", "gl.sampler", None, 10, f"{SAMPLES_DIR}/nuan_riser_1bar_132.wav", 0.72, 0.0),
            # 20-22 Melodic Instruments already in newtest.gridline
            ("Lead Pluck (Twin)", "gl.twinstring", "Nylon pluck", 11, None, 0.86, 0.04),
            ("Synth Counter", "gl.synth", None, 12, None, 0.72, -0.08),
            ("Chords Synth VoiceLed", "gl.synth", None, 9, None, 0.44, 0.0),
            # 23-24 Chants & FX
            ("Perc Chants (6)", "gl.sampler", None, 13, f"{SAMPLES_DIR}/sxj_chant6_tuned.wav", 0.64, 0.12),
            ("Perc FX (4)_2", "gl.sampler", None, 13, f"{SAMPLES_DIR}/sxj_fx4_tuned.wav", 0.64, -0.12),
        ]

        add_ch_calls = []
        for name, plug, preset, mixer, _, vol, pan in channels_spec:
            arg = {"name": name, "plugin": plug, "mixer": mixer, "volume": vol, "pan": pan}
            if preset:
                arg["preset"] = preset
            add_ch_calls.append({"tool": "add_channel", "arguments": arg})
        mcp.batch(add_ch_calls)

        mcp.batch([
            {"tool": "patch", "arguments": {"path": f"channel/{idx}", "set": {"plugin": {"data": spath}}}}
            for idx, (_, _, _, _, spath, _, _) in enumerate(channels_spec, 1) if spath
        ])

        # 4. Channel tuning, ADSR & sound shaping
        ch_params = [
            # Ch 1 Kick (1)
            ("channel/1/One shot", 0), ("channel/1/Release", 0.02), ("channel/1/Declick", 2), ("channel/1/Remove DC", 1),
            # Ch 2 Kick Native (short 115ms)
            ("channel/2/Start pitch", 195), ("channel/2/End pitch", 56), ("channel/2/Sweep", 0.04),
            ("channel/2/Decay", 0.115), ("channel/2/Click", 0.25), ("channel/2/Drive", 0.15),
            # Ch 3 808 (4) Smooth: Root 26 (D1), Fine -14ct, Voices=1 (Cut Itself!), Release 45ms, Glide 55ms
            ("channel/3/Root note", 26), ("channel/3/Fine tune", -14), ("channel/3/Voices", 1),
            ("channel/3/One shot", 0), ("channel/3/Attack", 0.002), ("channel/3/Decay", 10.0),
            ("channel/3/Sustain", 1.0), ("channel/3/Release", 0.045), ("channel/3/Glide", 0.055),
            ("channel/3/Remove DC", 1), ("channel/3/Declick", 2),
            # Ch 4 808 (4) Climax: Root 26 (D1), Fine -14ct, Voices=1 (Cut Itself!), Release 45ms, Glide 55ms
            ("channel/4/Root note", 26), ("channel/4/Fine tune", -14), ("channel/4/Voices", 1),
            ("channel/4/One shot", 0), ("channel/4/Attack", 0.002), ("channel/4/Decay", 10.0),
            ("channel/4/Sustain", 1.0), ("channel/4/Release", 0.045), ("channel/4/Glide", 0.055),
            ("channel/4/Remove DC", 1), ("channel/4/Declick", 2),
            # Ch 5 Clap (7), Ch 6 Snare Native
            ("channel/5/One shot", 0), ("channel/5/Release", 0.025), ("channel/5/Declick", 2),
            ("channel/6/Tone", 0.56), ("channel/6/Snappy", 0.70), ("channel/6/Body decay", 0.11),
            ("channel/6/Noise decay", 0.16), ("channel/6/Tune", 196),
            # Ch 7-12 Hats & Percs (short crisp gates!)
            ("channel/7/One shot", 0), ("channel/7/Release", 0.015), ("channel/7/Declick", 2),
            ("channel/8/One shot", 0), ("channel/8/Release", 0.015), ("channel/8/Declick", 2),
            ("channel/9/One shot", 0), ("channel/9/Release", 0.030), ("channel/9/Declick", 2),
            ("channel/10/One shot", 0), ("channel/10/Release", 0.015), ("channel/10/Declick", 2),
            ("channel/11/One shot", 0), ("channel/11/Release", 0.025), ("channel/11/Declick", 2),
            ("channel/12/One shot", 0), ("channel/12/Release", 0.040), ("channel/12/Declick", 2),
            # Ch 13-19 NUAN SONAR stems (Root 59 = B3 common tone across all 4 chords -> 0 MIDI clashes!)
            ("channel/13/Root note", 59), ("channel/13/One shot", 0), ("channel/13/Attack", 0.012), ("channel/13/Release", 0.12), ("channel/13/Declick", 2),
            ("channel/14/Root note", 59), ("channel/14/One shot", 0), ("channel/14/Attack", 0.012), ("channel/14/Release", 0.12), ("channel/14/Declick", 2),
            ("channel/15/Root note", 59), ("channel/15/One shot", 0), ("channel/15/Attack", 0.012), ("channel/15/Release", 0.12), ("channel/15/Declick", 2),
            ("channel/16/Root note", 59), ("channel/16/One shot", 0), ("channel/16/Attack", 0.012), ("channel/16/Release", 0.12), ("channel/16/Declick", 2),
            ("channel/17/Root note", 59), ("channel/17/One shot", 0), ("channel/17/Attack", 0.012), ("channel/17/Release", 0.25), ("channel/17/Declick", 2),
            ("channel/18/Root note", 59), ("channel/18/One shot", 0), ("channel/18/Attack", 0.020), ("channel/18/Release", 0.15), ("channel/18/Declick", 2),
            ("channel/19/Root note", 59), ("channel/19/One shot", 0), ("channel/19/Attack", 0.010), ("channel/19/Release", 0.02), ("channel/19/Declick", 2),
            # Ch 20 Twin String Lead ("Nylon pluck", clean Level 0.82 & warm High cut 0.42 -> zero crackle!)
            ("channel/20/Level", 0.82), ("channel/20/Stereo spread", 0.38), ("channel/20/High cut", 0.42),
            # Ch 21 Synth Counter (warm Triangle + Sine bell-pluck, Cutoff 2600 Hz, 0ct detune!)
            ("channel/21/Osc 1 shape", 1), ("channel/21/Osc 1 level", 0.82),
            ("channel/21/Osc 2 shape", 0), ("channel/21/Osc 2 level", 0.50), ("channel/21/Osc 2 coarse", 12), ("channel/21/Osc 2 fine", 0),
            ("channel/21/Osc 3 level", 0.0),
            ("channel/21/Attack", 0.010), ("channel/21/Decay", 0.30), ("channel/21/Sustain", 0.14), ("channel/21/Release", 0.16),
            ("channel/21/Cutoff", 2600), ("channel/21/Resonance", 0.04), ("channel/21/Filter env", 0.30), ("channel/21/Filter decay", 0.22),
            ("channel/21/Volume", 0.84), ("channel/21/Voices", 8),
            # Ch 22 Chords Synth VoiceLed (subtle warm Sine+Triangle support, lower volume 0.44 so low_mids stay clean!)
            ("channel/22/Osc 1 shape", 0), ("channel/22/Osc 1 level", 0.72),
            ("channel/22/Osc 2 shape", 1), ("channel/22/Osc 2 level", 0.40), ("channel/22/Osc 2 coarse", 0), ("channel/22/Osc 2 fine", 0),
            ("channel/22/Osc 3 level", 0.0),
            ("channel/22/Attack", 0.075), ("channel/22/Decay", 1.0), ("channel/22/Sustain", 0.65), ("channel/22/Release", 0.25),
            ("channel/22/Cutoff", 1400), ("channel/22/Resonance", 0.02), ("channel/22/Filter env", 0.12),
            ("channel/22/Volume", 0.44), ("channel/22/Voices", 12),
            # Ch 23-24 Chants & FX (already tuned to A#6, Root 70 triggers original pitch!)
            ("channel/23/Root note", 70), ("channel/23/One shot", 0), ("channel/23/Release", 0.025), ("channel/23/Declick", 2),
            ("channel/24/Root note", 70), ("channel/24/One shot", 0), ("channel/24/Release", 0.025), ("channel/24/Declick", 2),
        ]
        mcp.call("set_parameters", {"changes": [{"path": p, "value": v} for p, v in ch_params]})

        # 5. Mixer DSP & BUS Music Timbral Glue Chain
        fx_adds = [
            # Master (0): EQ, Glue Comp, Limiter
            {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.comp"}},
            {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.limiter"}},
            # BUS Drums (17): EQ, Bus Glue Comp
            {"tool": "add_plugin", "arguments": {"mixer": 17, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 17, "plugin": "gl.comp"}},
            # BUS 808 (18): Stereo Width (Mono), Sidechain Comp, EQ
            {"tool": "add_plugin", "arguments": {"mixer": 18, "plugin": "gl.stereo"}},
            {"tool": "add_plugin", "arguments": {"mixer": 18, "plugin": "gl.comp"}},
            {"tool": "add_plugin", "arguments": {"mixer": 18, "plugin": "gl.eq"}},
            # BUS Music (19): EQ (LPF 12.5k), Dist Soft 4x (Mix 10%), Bus Glue Comp, Kick Micro-Ducking Sidechain Comp
            {"tool": "add_plugin", "arguments": {"mixer": 19, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 19, "plugin": "gl.dist"}},
            {"tool": "add_plugin", "arguments": {"mixer": 19, "plugin": "gl.comp"}},
            {"tool": "add_plugin", "arguments": {"mixer": 19, "plugin": "gl.comp"}},
            # FX Room (20): gl.reverb
            {"tool": "add_plugin", "arguments": {"mixer": 20, "plugin": "gl.reverb"}},
            # Selective insert EQs (no extra distortion on 808 Climax -> clean crackle_vs_clean!)
            {"tool": "add_plugin", "arguments": {"mixer": 4, "plugin": "gl.eq"}},    # Clap LPF 12k
            {"tool": "add_plugin", "arguments": {"mixer": 5, "plugin": "gl.eq"}},    # Snare LPF 12k
            {"tool": "add_plugin", "arguments": {"mixer": 6, "plugin": "gl.eq"}},    # Hi-Hats Closed
            {"tool": "add_plugin", "arguments": {"mixer": 8, "plugin": "gl.eq"}},    # NUAN Pad Open
            {"tool": "add_plugin", "arguments": {"mixer": 9, "plugin": "gl.eq"}},    # NUAN Pad Filtered
            {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.eq"}},   # NUAN High +12
            {"tool": "add_plugin", "arguments": {"mixer": 11, "plugin": "gl.eq"}},   # Twin String Lead
            {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.eq"}},   # Synth Counter
        ]
        mcp.batch(fx_adds)

        fx_changes = [
            # Master (0)
            ("mixer/0/slot/1/Quality", 1),
            ("mixer/0/slot/1/Band 1 type", 1), ("mixer/0/slot/1/Band 1 freq", 28),
            ("mixer/0/slot/1/Band 7 type", 3), ("mixer/0/slot/1/Band 7 freq", 15500),
            ("mixer/0/slot/2/Threshold", -12.0), ("mixer/0/slot/2/Ratio", 1.7),
            ("mixer/0/slot/2/Attack", 0.020), ("mixer/0/slot/2/Release", 0.110),
            ("mixer/0/slot/2/Makeup", 2.4), ("mixer/0/slot/2/Detector high-pass", 95),
            ("mixer/0/slot/3/Gain", 10.1), ("mixer/0/slot/3/Ceiling", -2.15),
            ("mixer/0/slot/3/Attack (look-ahead)", 0.005), ("mixer/0/slot/3/0.2 dB safety", 1),

            # BUS Drums (17) — flatten 63 Hz Kick resonance (-5.2 dB) & control transients cleanly
            ("mixer/17/slot/1/Band 1 type", 1), ("mixer/17/slot/1/Band 1 freq", 34),
            ("mixer/17/slot/1/Band 2 type", 6), ("mixer/17/slot/1/Band 2 freq", 63), ("mixer/17/slot/1/Band 2 gain", -5.2), ("mixer/17/slot/1/Band 2 Q (legacy)", 1.3),
            ("mixer/17/slot/1/Band 3 type", 6), ("mixer/17/slot/1/Band 3 freq", 100), ("mixer/17/slot/1/Band 3 gain", 1.6), ("mixer/17/slot/1/Band 3 Q (legacy)", 1.1),
            ("mixer/17/slot/1/Band 7 type", 7), ("mixer/17/slot/1/Band 7 freq", 9500), ("mixer/17/slot/1/Band 7 gain", 1.4),
            ("mixer/17/slot/2/Threshold", -18.0), ("mixer/17/slot/2/Ratio", 2.8),
            ("mixer/17/slot/2/Attack", 0.012), ("mixer/17/slot/2/Release", 0.085), ("mixer/17/slot/2/Makeup", 3.2),

            # BUS 808 (18) — flatten 63 Hz resonance (-5.0 dB) + warm 45 Hz sub support
            ("mixer/18/slot/1/Width", 0.0),
            ("mixer/18/slot/2/Detector input", 1),
            ("mixer/18/slot/2/Threshold", -22.0), ("mixer/18/slot/2/Ratio", 5.0),
            ("mixer/18/slot/2/Attack", 0.002), ("mixer/18/slot/2/Release", 0.070), ("mixer/18/slot/2/Lookahead", 2),
            ("mixer/18/slot/3/Band 1 type", 1), ("mixer/18/slot/3/Band 1 freq", 28),
            ("mixer/18/slot/3/Band 2 type", 6), ("mixer/18/slot/3/Band 2 freq", 63), ("mixer/18/slot/3/Band 2 gain", -5.0), ("mixer/18/slot/3/Band 2 Q (legacy)", 1.3),
            ("mixer/18/slot/3/Band 3 type", 6), ("mixer/18/slot/3/Band 3 freq", 105), ("mixer/18/slot/3/Band 3 gain", 2.0), ("mixer/18/slot/3/Band 3 Q (legacy)", 1.1),
            ("mixer/18/slot/3/Band 4 type", 6), ("mixer/18/slot/3/Band 4 freq", 46), ("mixer/18/slot/3/Band 4 gain", 1.5), ("mixer/18/slot/3/Band 4 Q (legacy)", 1.2),

            # BUS Music (19) — Timbral Glue
            ("mixer/19/slot/1/Band 1 type", 1), ("mixer/19/slot/1/Band 1 freq", 155),
            ("mixer/19/slot/1/Band 3 type", 6), ("mixer/19/slot/1/Band 3 freq", 260), ("mixer/19/slot/1/Band 3 gain", -2.6), ("mixer/19/slot/1/Band 3 Q (legacy)", 1.0),
            ("mixer/19/slot/1/Band 5 type", 6), ("mixer/19/slot/1/Band 5 freq", 1650), ("mixer/19/slot/1/Band 5 gain", 2.8), ("mixer/19/slot/1/Band 5 Q (legacy)", 0.75),
            ("mixer/19/slot/1/Band 7 type", 3), ("mixer/19/slot/1/Band 7 freq", 12500),
            ("mixer/19/slot/2/Mode", 0), ("mixer/19/slot/2/Drive", 3.5),
            ("mixer/19/slot/2/Tone", 12000), ("mixer/19/slot/2/Mix", 0.06),
            ("mixer/19/slot/2/Output", 0.0), ("mixer/19/slot/2/Oversampling", 2), ("mixer/19/slot/2/Remove DC", 1),
            ("mixer/19/slot/3/Threshold", -19.5), ("mixer/19/slot/3/Ratio", 2.0),
            ("mixer/19/slot/3/Attack", 0.020), ("mixer/19/slot/3/Release", 0.110), ("mixer/19/slot/3/Makeup", 3.6),
            ("mixer/19/slot/4/Detector input", 1),
            ("mixer/19/slot/4/Threshold", -23.0), ("mixer/19/slot/4/Ratio", 2.6),
            ("mixer/19/slot/4/Attack", 0.005), ("mixer/19/slot/4/Release", 0.080), ("mixer/19/slot/4/Lookahead", 1),

            # FX Room (20) — tight shared acoustic space (Dry=0, Dry full=0)
            ("mixer/20/slot/1/Model", 1),
            ("mixer/20/slot/1/Pre-delay", 10), ("mixer/20/slot/1/Room size", 0.22),
            ("mixer/20/slot/1/Decay time", 0.190),
            ("mixer/20/slot/1/Low cut", 520), ("mixer/20/slot/1/High cut", 6200), ("mixer/20/slot/1/High damping", 4000),
            ("mixer/20/slot/1/Dry", 0), ("mixer/20/slot/1/Wet", 0.42), ("mixer/20/slot/1/Width", 0.75),
            ("mixer/20/slot/1/Low cut (full)", 520), ("mixer/20/slot/1/Decay (full)", 2),
            ("mixer/20/slot/1/Dry (full)", 0), ("mixer/20/slot/1/Early reflections (full)", 6),
            ("mixer/20/slot/1/Wet (full)", 42),

            # Insert EQs
            ("mixer/4/slot/1/Band 1 type", 1), ("mixer/4/slot/1/Band 1 freq", 160),
            ("mixer/4/slot/1/Band 5 type", 6), ("mixer/4/slot/1/Band 5 freq", 2500), ("mixer/4/slot/1/Band 5 gain", 2.8),
            ("mixer/4/slot/1/Band 7 type", 3), ("mixer/4/slot/1/Band 7 freq", 12000),
            ("mixer/5/slot/1/Band 1 type", 1), ("mixer/5/slot/1/Band 1 freq", 135),
            ("mixer/5/slot/1/Band 7 type", 3), ("mixer/5/slot/1/Band 7 freq", 12000),
            ("mixer/6/slot/1/Band 1 type", 1), ("mixer/6/slot/1/Band 1 freq", 520),
            ("mixer/6/slot/1/Band 5 type", 6), ("mixer/6/slot/1/Band 5 freq", 7500), ("mixer/6/slot/1/Band 5 gain", -1.4), ("mixer/6/slot/1/Band 5 Q (legacy)", 0.8),
            ("mixer/6/slot/1/Band 7 type", 3), ("mixer/6/slot/1/Band 7 freq", 15000),

            # NUAN Pad Open (8): HPF 185 Hz, strong carve on 245Hz mud (-5.0dB) + Lead pocket (-2.8dB @ 1.5kHz)
            ("mixer/8/slot/1/Band 1 type", 1), ("mixer/8/slot/1/Band 1 freq", 185),
            ("mixer/8/slot/1/Band 3 type", 6), ("mixer/8/slot/1/Band 3 freq", 245), ("mixer/8/slot/1/Band 3 gain", -5.0), ("mixer/8/slot/1/Band 3 Q (legacy)", 1.2),
            ("mixer/8/slot/1/Band 5 type", 6), ("mixer/8/slot/1/Band 5 freq", 1500), ("mixer/8/slot/1/Band 5 gain", -2.8),
            ("mixer/8/slot/1/Band 7 type", 3), ("mixer/8/slot/1/Band 7 freq", 8800),

            # NUAN Pad Filtered (9): HPF 185 Hz, cut -4.8dB @ 245Hz, warm LPF 2000 Hz
            ("mixer/9/slot/1/Band 1 type", 1), ("mixer/9/slot/1/Band 1 freq", 185),
            ("mixer/9/slot/1/Band 3 type", 6), ("mixer/9/slot/1/Band 3 freq", 245), ("mixer/9/slot/1/Band 3 gain", -4.8), ("mixer/9/slot/1/Band 3 Q (legacy)", 1.2),
            ("mixer/9/slot/1/Band 7 type", 3), ("mixer/9/slot/1/Band 7 freq", 2000),

            # NUAN High +12 (10): HPF 550 Hz
            ("mixer/10/slot/1/Band 1 type", 1), ("mixer/10/slot/1/Band 1 freq", 550),

            # Twin String Lead (11): HPF 280 Hz, smooth warm boost (no metallic crackle)
            ("mixer/11/slot/1/Band 1 type", 1), ("mixer/11/slot/1/Band 1 freq", 280),
            ("mixer/11/slot/1/Band 4 type", 6), ("mixer/11/slot/1/Band 4 freq", 1500), ("mixer/11/slot/1/Band 4 gain", 1.5),
            ("mixer/11/slot/1/Band 5 type", 6), ("mixer/11/slot/1/Band 5 freq", 3200), ("mixer/11/slot/1/Band 5 gain", -1.2),

            # Synth Counter (12): HPF 240 Hz, presence +2.2dB @ 1.3kHz, LPF 6000 Hz
            ("mixer/12/slot/1/Band 1 type", 1), ("mixer/12/slot/1/Band 1 freq", 240),
            ("mixer/12/slot/1/Band 5 type", 6), ("mixer/12/slot/1/Band 5 freq", 1300), ("mixer/12/slot/1/Band 5 gain", 2.2),
            ("mixer/12/slot/1/Band 7 type", 3), ("mixer/12/slot/1/Band 7 freq", 6000),
        ]
        mcp.call("set_parameters", {"changes": [{"path": p, "value": v} for p, v in fx_changes]})

        # 6. Playlist Tracks 1..11 (one dedicated track per instrument/role!)
        track_names = [
            (1, "808 (4) Smooth"),
            (2, "808 (5) Climax"),
            (3, "Kick (1) + Native"),
            (4, "Clap (7) & Snare"),
            (5, "Hi-Hats Closed"),
            (6, "OpenHat, Percs & Triangle"),
            (7, "NUAN SONAR Pad (Filtered/Open)"),
            (8, "NUAN High +12 & Riser"),
            (9, "Twin String Lead (A-A'-A''-B)"),
            (10, "Synth Counter (Call & Response)"),
            (11, "Chants, FX & Room Cuts"),
        ]
        mcp.batch([{"tool": "patch", "arguments": {"path": f"track/{t}", "set": {"name": nm}}} for t, nm in track_names])

        # 7. Patterns & Clips across 33 bars (60.00 s @ 132 BPM)
        pat_counter = 1
        queued = [{"tool": "edit_pattern", "arguments": {"pattern": 1, "name": "NUAN Intro 4m (Filtered)", "bars": 4}}]

        def new_pattern(name: str, bars: int) -> int:
            nonlocal pat_counter
            pat_counter += 1
            queued.append({"tool": "add_pattern", "arguments": {"name": name, "bars": bars}})
            return pat_counter

        def add_notes(pat: int, ch: int, notes: list):
            queued.append({"tool": "set_notes", "arguments": {"pattern": pat, "channel": ch, "notes": notes}})

        def place(pat: int, track: int, start_bar: int):
            queued.append({"tool": "add_clip", "arguments": {"pattern": pat, "track": track, "start": (start_bar - 1) * 4.0}})

        # ---- HARMONY: NUAN SONAR + VOICE-LED CHORDS (Tracks 7 & 8) ----
        def make_voiceled_chords(bars=4, vel=68, mute_last_beat=True):
            notes = []
            for b in range(bars):
                ln = 2.60 if (mute_last_beat and b == bars - 1) else 2.85
                m4 = b % 4
                if m4 == 1:
                    for p in voicings[0]:
                        notes.append({"start": b * 4.0, "length": 1.85, "pitch": p, "velocity": vel})
                    for p in voicings[1]:
                        notes.append({"start": b * 4.0 + 2.0, "length": 1.45, "pitch": p, "velocity": vel})
                else:
                    for p in voicings[m4]:
                        notes.append({"start": b * 4.0, "length": ln, "pitch": p, "velocity": vel})
            return notes

        # Pat 1: M1-4 Intro Filtered (Ch 14 + Ch 22)
        add_notes(1, 14, [{"start": 0.0, "length": 15.85, "pitch": 59, "velocity": 106}])
        add_notes(1, 22, make_voiceled_chords(4, vel=74, mute_last_beat=True))
        place(1, 7, 1)

        # M5-12 Refrain 1 Open (Ch 13 x2 = 4+4 bars + Ch 22)
        p_nuan_ref1 = new_pattern("NUAN Refrain 1 Open 8m", 8)
        add_notes(p_nuan_ref1, 13, [
            {"start": 0.0, "length": 15.85, "pitch": 59, "velocity": 100},
            {"start": 16.0, "length": 14.85, "pitch": 59, "velocity": 102},  # cut beat 4 of M12!
        ])
        add_notes(p_nuan_ref1, 22, make_voiceled_chords(8, vel=68, mute_last_beat=True))
        place(p_nuan_ref1, 7, 5)

        # M13-16 Couplet Filtered (Ch 14 + Ch 22)
        p_nuan_coup = new_pattern("NUAN Couplet Filt 4m", 4)
        add_notes(p_nuan_coup, 14, [{"start": 0.0, "length": 15.85, "pitch": 59, "velocity": 86}])
        add_notes(p_nuan_coup, 22, make_voiceled_chords(4, vel=58, mute_last_beat=False))
        place(p_nuan_coup, 7, 13)

        # M17-20 Pont Swell Filtered (Ch 16 + Ch 22)
        p_nuan_pont = new_pattern("NUAN Pont Filt 4m", 4)
        add_notes(p_nuan_pont, 16, [{"start": 0.0, "length": 14.85, "pitch": 59, "velocity": 100}])
        add_notes(p_nuan_pont, 22, make_voiceled_chords(4, vel=68, mute_last_beat=True))
        place(p_nuan_pont, 7, 17)

        # M21-28 Refrain 2 Climax Open (Ch 15 x2 + Ch 22)
        p_nuan_climax = new_pattern("NUAN Climax Open 8m", 8)
        add_notes(p_nuan_climax, 15, [
            {"start": 0.0, "length": 15.85, "pitch": 59, "velocity": 106},
            {"start": 16.0, "length": 14.85, "pitch": 59, "velocity": 108},
        ])
        add_notes(p_nuan_climax, 22, make_voiceled_chords(8, vel=72, mute_last_beat=True))
        place(p_nuan_climax, 7, 21)

        # M29-33 Outro 5m (Ch 17 -> natural G#m7 + tail + Ch 22)
        p_nuan_outro = new_pattern("NUAN Outro 5m", 5)
        add_notes(p_nuan_outro, 17, [{"start": 0.0, "length": 19.8, "pitch": 59, "velocity": 94}])
        add_notes(p_nuan_outro, 22, make_voiceled_chords(4, vel=64, mute_last_beat=False))
        place(p_nuan_outro, 7, 29)

        # NUAN High +12 Shimmer (M21-28) & NUAN Riser (M4, M20) on Track 8
        p_nuan_hi = new_pattern("NUAN High+12 Climax 8m", 8)
        add_notes(p_nuan_hi, 18, [
            {"start": 0.0, "length": 15.85, "pitch": 59, "velocity": 92},
            {"start": 16.0, "length": 14.85, "pitch": 59, "velocity": 95},
        ])
        place(p_nuan_hi, 8, 21)

        p_riser = new_pattern("NUAN Riser 1m", 1)
        add_notes(p_riser, 19, [{"start": 0.0, "length": 3.92, "pitch": 59, "velocity": 96}])
        place(p_riser, 8, 4)
        place(p_riser, 8, 20)

        # ---- LEAD: TWIN STRING (Track 9, Ch 20) ----
        def make_lead(bars=4, vel=96, mode="full", oct_shift=0, mute_last_beat=True):
            notes = []
            for rep in range(bars // 4):
                off = rep * 4
                for b, st, ln, pt in lead_phrase:
                    ab = off + b
                    t = ab * 4.0 + st
                    if mute_last_beat and ab == bars - 1 and t >= bars * 4.0 - 1.25:
                        continue
                    if mode == "skeleton" and st not in (0.0, 2.0):
                        continue
                    jv = max(45, min(115, vel + random.randint(-5, 5)))
                    notes.append({"start": t, "length": ln, "pitch": pt + oct_shift, "velocity": jv})
            return notes

        p_lead_intro = new_pattern("Twin String - Intro 4m", 4)
        add_notes(p_lead_intro, 20, make_lead(4, vel=90, mode="full", mute_last_beat=True))
        place(p_lead_intro, 9, 1)

        p_lead_ref1 = new_pattern("Twin String - Refrain 1 8m", 8)
        add_notes(p_lead_ref1, 20, make_lead(8, vel=106, mode="full", mute_last_beat=True))
        place(p_lead_ref1, 9, 5)

        p_lead_coup = new_pattern("Twin String - Couplet 4m", 4)
        add_notes(p_lead_coup, 20, make_lead(4, vel=88, mode="full", oct_shift=0, mute_last_beat=False))
        place(p_lead_coup, 9, 13)

        p_lead_pont = new_pattern("Twin String - Pont 4m", 4)
        add_notes(p_lead_pont, 20, [
            {"start": 0.0, "length": 1.25, "pitch": 68, "velocity": 78},
            {"start": 2.0, "length": 1.25, "pitch": 71, "velocity": 82},
            {"start": 4.0, "length": 1.25, "pitch": 73, "velocity": 84},
            {"start": 6.0, "length": 1.25, "pitch": 75, "velocity": 86},
            {"start": 8.0, "length": 1.25, "pitch": 75, "velocity": 84},
            {"start": 10.0, "length": 1.25, "pitch": 71, "velocity": 80},
            {"start": 12.0, "length": 1.25, "pitch": 68, "velocity": 76},
        ])
        place(p_lead_pont, 9, 17)

        p_lead_climax = new_pattern("Twin String - Climax 8m", 8)
        add_notes(p_lead_climax, 20, make_lead(8, vel=114, mode="full", mute_last_beat=True))
        place(p_lead_climax, 9, 21)

        p_lead_outro = new_pattern("Twin String - Outro 5m", 5)
        add_notes(p_lead_outro, 20, [
            {"start": 0.0, "length": 1.20, "pitch": 71, "velocity": 78},
            {"start": 1.5, "length": 0.90, "pitch": 70, "velocity": 72},
            {"start": 2.5, "length": 1.40, "pitch": 68, "velocity": 68},
            {"start": 4.0, "length": 1.40, "pitch": 66, "velocity": 62},
            {"start": 8.0, "length": 1.80, "pitch": 68, "velocity": 56},
        ])
        place(p_lead_outro, 9, 29)

        # ---- COUNTER: SYNTH (Track 10, Ch 21) — Call & Response strictly in Lead's resting gaps (3.0, 3.5) ----
        def make_counter(bars=8, vel=82, mute_last_bar=True):
            echo_map = {
                0: [(3.0, 52), (3.5, 59)],  # E3 -> B3 (Emaj)
                1: [(3.0, 56), (3.5, 52)],  # G#3 -> E3 (C#m7)
                2: [(3.0, 54), (3.5, 51)],  # F#3 -> D#3 (B)
                3: [(3.0, 56), (3.5, 51)],  # G#3 -> D#3 (G#m7)
            }
            notes = []
            for b in range(bars):
                if mute_last_bar and b == bars - 1:
                    continue
                for st, pt in echo_map[b % 4]:
                    notes.append({"start": b * 4.0 + st, "length": 0.38, "pitch": pt, "velocity": vel + random.randint(-4, 4)})
            return notes

        p_cnt_ref1 = new_pattern("Synth Counter - Refrain 1", 8)
        add_notes(p_cnt_ref1, 21, make_counter(8, vel=80, mute_last_bar=True))
        place(p_cnt_ref1, 10, 5)

        p_cnt_coup = new_pattern("Synth Counter - Couplet", 4)
        add_notes(p_cnt_coup, 21, make_counter(4, vel=84, mute_last_bar=False))
        place(p_cnt_coup, 10, 13)

        p_cnt_climax = new_pattern("Synth Counter - Climax", 8)
        add_notes(p_cnt_climax, 21, make_counter(8, vel=88, mute_last_bar=True))
        place(p_cnt_climax, 10, 21)

        # ---- 808 (4) SMOOTH (Track 1, Ch 3) & 808 (5) CLIMAX (Track 2, Ch 4) ----
        # Bass roots matching NUAN SONAR bar by bar:
        # b0: E1(28) / b1: E1(28)@0..2 then C#2(37)@2..4 / b2: B1(35) / b3: G#1(32)
        def make_808_ref1():
            notes = []
            for b in range(8):
                m4 = b % 4
                is_last = (b == 7)
                if m4 == 0:
                    hits = [(0.0, 1.45, 28, 110, False), (1.5, 1.20, 28, 102, False), (2.75, 1.15, 40 if b == 4 else 28, 100, True)]
                elif m4 == 1:
                    hits = [(0.0, 1.45, 28, 108, False), (2.0, 1.80, 37, 108, False)]
                elif m4 == 2:
                    hits = [(0.0, 1.45, 35, 110, False), (1.5, 1.20, 35, 102, False), (2.75, 1.15, 35, 100, False)]
                else:
                    if is_last:
                        hits = [(0.0, 1.45, 32, 108, False), (1.5, 1.00, 32, 100, False)]  # cut beat 4!
                    else:
                        hits = [(0.0, 1.45, 32, 108, False), (1.5, 1.20, 32, 102, False), (3.0, 0.85, 30, 98, True)]
                for st, ln, pt, vel, porta in hits:
                    n = {"start": b * 4.0 + st, "length": ln, "pitch": pt, "velocity": vel}
                    if porta:
                        n["porta"] = True
                    notes.append(n)
            return notes

        p_808_ref1 = new_pattern("808 (4) - Refrain 1 8m", 8)
        add_notes(p_808_ref1, 3, make_808_ref1())
        place(p_808_ref1, 1, 5)

        def make_808_bounce():
            notes = []
            for b in range(4):
                if b == 0:
                    hits = [(0.0, 0.70, 28, 106), (1.5, 0.55, 28, 96), (2.75, 0.65, 28, 98)]
                elif b == 1:
                    hits = [(0.0, 0.70, 28, 104), (2.0, 0.65, 37, 106), (3.0, 0.50, 37, 96)]
                elif b == 2:
                    hits = [(0.0, 0.70, 35, 106), (1.5, 0.55, 35, 96), (2.75, 0.65, 35, 98)]
                else:
                    hits = [(0.0, 0.70, 32, 104), (1.5, 0.55, 32, 96), (2.75, 0.60, 30, 96)]
                for st, ln, pt, vel in hits:
                    notes.append({"start": b * 4.0 + st, "length": ln, "pitch": pt, "velocity": vel})
            return notes

        p_808_coup = new_pattern("808 (4) - Couplet Bounce 4m", 4)
        add_notes(p_808_coup, 3, make_808_bounce())
        place(p_808_coup, 1, 13)

        def make_808_climax():
            notes = []
            for b in range(8):
                m4 = b % 4
                is_last = (b == 7)
                if m4 == 0:
                    hits = [(0.0, 1.45, 28, 114, False), (1.5, 1.20, 28, 106, False), (2.75, 1.15, 40, 104, True)]
                elif m4 == 1:
                    hits = [(0.0, 1.45, 28, 112, False), (2.0, 1.00, 37, 112, False), (3.25, 0.65, 49 if b == 5 else 37, 102, True)]
                elif m4 == 2:
                    hits = [(0.0, 1.45, 35, 114, False), (1.5, 1.20, 35, 106, False), (2.75, 1.15, 47 if b == 6 else 35, 104, True)]
                else:
                    if is_last:
                        hits = [(0.0, 1.45, 32, 110, False), (1.5, 0.90, 32, 104, False)]
                    else:
                        hits = [(0.0, 1.45, 32, 112, False), (1.5, 1.15, 32, 104, False), (2.75, 0.60, 44, 102, True), (3.5, 0.45, 30, 98, True)]
                for st, ln, pt, vel, porta in hits:
                    n = {"start": b * 4.0 + st, "length": ln, "pitch": pt, "velocity": vel}
                    if porta:
                        n["porta"] = True
                    notes.append(n)
            return notes

        p_808_climax = new_pattern("808 (5) - Climax 8m", 8)
        add_notes(p_808_climax, 4, make_808_climax())
        place(p_808_climax, 2, 21)

        # ---- KICK (Track 3, Ch 1 + Ch 2) ----
        def make_kicks(bars=8, dense=False, mute_last_beat=True):
            notes = []
            for b in range(bars):
                m4 = b % 4
                if b == bars - 1 and mute_last_beat:
                    times = [0.0, 1.5]
                elif m4 == 1:
                    times = [0.0, 2.0, 3.25] if dense else [0.0, 2.0]
                elif b % 2 == 0:
                    times = [0.0, 1.5, 2.75]
                else:
                    times = [0.0, 1.5, 2.75, 3.5] if dense else [0.0, 1.5, 2.75]
                for t in times:
                    notes.append({"start": b * 4.0 + t, "length": 0.35, "pitch": 60, "velocity": 112 if t == 0.0 else 102})
            return notes

        p_kick_ref1 = new_pattern("Kick (1) - Refrain 1 8m", 8)
        add_notes(p_kick_ref1, 1, make_kicks(8, dense=False, mute_last_beat=True))
        place(p_kick_ref1, 3, 5)

        p_kick_coup = new_pattern("Kick (1) - Couplet 4m", 4)
        add_notes(p_kick_coup, 1, make_kicks(4, dense=False, mute_last_beat=False))
        place(p_kick_coup, 3, 13)

        p_kick_climax = new_pattern("Kick (1)+Native - Climax 8m", 8)
        k_cl = make_kicks(8, dense=True, mute_last_beat=True)
        add_notes(p_kick_climax, 1, k_cl)
        add_notes(p_kick_climax, 2, k_cl)
        place(p_kick_climax, 3, 21)

        # ---- CLAP (7) & SNARE NATIVE (Track 4, Ch 5 & Ch 6) ----
        def make_clap_snare(bars=8, fill_end=True, ghosts=True):
            cl, sn = [], []
            for b in range(bars):
                cl.append({"start": b * 4.0 + 2.0, "length": 0.40, "pitch": 60, "velocity": 106})
                if ghosts and b in (1, 3, 5):
                    sn.append({"start": b * 4.0 + 3.75, "length": 0.22, "pitch": 60, "velocity": 62})
                if fill_end and b == bars - 1:
                    for sub, vel in [(3.0, 72), (3.25, 82), (3.5, 92), (3.75, 102)]:
                        sn.append({"start": b * 4.0 + sub, "length": 0.18, "pitch": 60, "velocity": vel})
            return cl, sn

        p_cs_ref1 = new_pattern("Clap/Snare - Refrain 1 8m", 8)
        cl1, sn1 = make_clap_snare(8, fill_end=True, ghosts=True)
        add_notes(p_cs_ref1, 5, cl1)
        add_notes(p_cs_ref1, 6, sn1)
        place(p_cs_ref1, 4, 5)

        p_cs_coup = new_pattern("Clap/Snare - Couplet 4m", 4)
        cl2, sn2 = make_clap_snare(4, fill_end=False, ghosts=True)
        add_notes(p_cs_coup, 5, cl2)
        add_notes(p_cs_coup, 6, sn2)
        place(p_cs_coup, 4, 13)

        # Pont buildup Snare M17-20
        p_cs_pont = new_pattern("Snare - Pont Buildup 4m", 4)
        psn = [{"start": b * 4.0 + 2.0, "length": 0.30, "pitch": 60, "velocity": 48 + b * 10} for b in range(3)]
        for k in range(6):
            psn.append({"start": 3 * 4.0 + k * 0.5, "length": 0.18, "pitch": 60, "velocity": 52 + k * 9})
        add_notes(p_cs_pont, 6, psn)
        place(p_cs_pont, 4, 17)

        p_cs_climax = new_pattern("Clap/Snare - Climax 8m", 8)
        cl3, sn3 = make_clap_snare(8, fill_end=False, ghosts=True)
        add_notes(p_cs_climax, 5, cl3)
        add_notes(p_cs_climax, 6, sn3)
        place(p_cs_climax, 4, 21)

        # ---- HI-HATS CLOSED (Track 5, Ch 7 & Ch 8) ----
        p_hh_intro = new_pattern("HH Bankroll - Intro M3-4", 2)
        add_notes(p_hh_intro, 8, [
            {"start": b * 4.0 + s * 0.5, "length": 0.14, "pitch": 60, "velocity": 52 if s % 2 == 0 else 36}
            for b in range(2) for s in range(8) if not (b == 1 and s >= 6)
        ])
        place(p_hh_intro, 5, 3)

        def make_hats(bars=8, high=False):
            hn = []
            for b in range(bars):
                for s in range(8):
                    t = b * 4.0 + s * 0.5
                    if b == bars - 1 and s >= 6:
                        continue  # mute last beat for pivot!
                    vel = (102 if s % 2 == 0 else 66) + random.randint(-5, 5)
                    if b % 2 == 1 and s == 3:
                        for k in range(3):
                            hn.append({"start": round(b * 4.0 + 1.5 + k * (0.5 / 3.0), 4), "length": 0.10, "pitch": 60 - k * 2, "velocity": 80 + k * 6})
                        continue
                    if high and b % 2 == 1 and s in (6, 7):
                        for k in range(4):
                            hn.append({"start": round(b * 4.0 + 3.0 + k * 0.25, 4), "length": 0.10, "pitch": 60 - k * 2, "velocity": 92 - k * 7})
                        continue
                    hn.append({"start": t, "length": 0.16, "pitch": 60, "velocity": vel})
                    if high and s in (1, 5):
                        hn.append({"start": t + 0.25, "length": 0.12, "pitch": 60, "velocity": 72})
            return hn

        p_hh_ref1 = new_pattern("HH Bat Wing - Refrain 1 8m", 8)
        add_notes(p_hh_ref1, 7, make_hats(8, high=False))
        place(p_hh_ref1, 5, 5)

        p_hh_coup = new_pattern("HH Bankroll - Couplet 4m", 4)
        add_notes(p_hh_coup, 8, [
            {"start": b * 4.0 + s * 0.5, "length": 0.14, "pitch": 60, "velocity": 90 if s % 2 == 0 else 58}
            for b in range(4) for s in (0, 1, 2, 4, 5, 6, 7)
        ])
        place(p_hh_coup, 5, 13)

        p_hh_climax = new_pattern("HH Bat Wing - Climax 8m", 8)
        add_notes(p_hh_climax, 7, make_hats(8, high=True))
        place(p_hh_climax, 5, 21)

        # ---- OPENHAT, PERCS & TRIANGLE (Track 6, Ch 9, 10, 11, 12) ----
        p_perc_ref1 = new_pattern("Triangle - Refrain 1 8m", 8)
        add_notes(p_perc_ref1, 12, [
            {"start": 0.0, "length": 0.5, "pitch": 60, "velocity": 80},
            {"start": 16.0, "length": 0.5, "pitch": 60, "velocity": 82},
        ])
        place(p_perc_ref1, 6, 5)

        p_perc_coup = new_pattern("Percs (8) & Zap - Couplet 4m", 4)
        add_notes(p_perc_coup, 11, [{"start": b * 4.0 + t, "length": 0.28, "pitch": 60, "velocity": 82} for b in range(4) for t in (1.75, 3.5)])
        add_notes(p_perc_coup, 10, [{"start": b * 4.0 + 3.75, "length": 0.12, "pitch": 60, "velocity": 72} for b in (1, 3)])
        place(p_perc_coup, 6, 13)

        p_perc_pont = new_pattern("Percs (8) - Pont 4m", 4)
        add_notes(p_perc_pont, 11, [{"start": b * 4.0 + 1.5, "length": 0.28, "pitch": 60, "velocity": 68} for b in range(3)])
        place(p_perc_pont, 6, 17)

        p_perc_climax = new_pattern("OpenHat, Triangle & Percs - Climax 8m", 8)
        add_notes(p_perc_climax, 9, [{"start": b * 4.0 + 0.5, "length": 0.26, "pitch": 60, "velocity": 84} for b in (0, 2, 4, 6)])
        add_notes(p_perc_climax, 12, [{"start": 0.0, "length": 0.5, "pitch": 60, "velocity": 84}, {"start": 16.0, "length": 0.5, "pitch": 60, "velocity": 84}])
        add_notes(p_perc_climax, 11, [{"start": b * 4.0 + 1.75, "length": 0.25, "pitch": 60, "velocity": 76} for b in range(7)])
        place(p_perc_climax, 6, 21)

        # ---- CHANTS & FX + FX ROOM CUTS (Track 11, Ch 23, 24) ----
        p_ch_ref1 = new_pattern("Chants & FX - Refrain 1", 8)
        add_notes(p_ch_ref1, 23, [{"start": 0.0, "length": 0.35, "pitch": 70, "velocity": 80}, {"start": 16.0, "length": 0.35, "pitch": 70, "velocity": 82}])
        add_notes(p_ch_ref1, 24, [{"start": 7 * 4.0 + 3.0, "length": 0.45, "pitch": 70, "velocity": 78}])
        place(p_ch_ref1, 11, 5)

        p_ch_climax = new_pattern("Chants & FX - Climax", 8)
        add_notes(p_ch_climax, 23, [{"start": 0.0, "length": 0.35, "pitch": 70, "velocity": 86}, {"start": 16.0, "length": 0.35, "pitch": 70, "velocity": 88}])
        add_notes(p_ch_climax, 24, [{"start": 7 * 4.0 + 3.0, "length": 0.45, "pitch": 70, "velocity": 76}])
        place(p_ch_climax, 11, 21)

        # FX Room transition cuts on M4, M12, M20, M28 beat 4
        pts = [{"position": 0.0, "value": 0.28, "shape": "hold"}]
        for p_bar in [4, 12, 20, 28]:
            pts.append({"position": (p_bar - 1) * 4.0 + 3.0, "value": 0.0, "shape": "hold"})
            pts.append({"position": p_bar * 4.0, "value": 0.28, "shape": "hold"})
        pts.append({"position": 33 * 4.0, "value": 0.28, "shape": "hold"})
        queued.append({"tool": "add_automation", "arguments": {
            "mixer": 20, "param": "volume", "name": "FX Room Pivot Cuts",
            "points": pts, "track": 11, "start": 0.0, "length": 33 * 4.0,
        }})

        mcp.batch(queued)
        mcp.call("save_project", {})
        mcp.call("export_project", {"format": "midi", "path": ORCH_MIDI})
        song = mcp.call("get_song", {})
        print("Built active6/projet.gridline:", json.dumps({
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

    # Audit full orchestrated MIDI
    audit = audit_midi_file(ORCH_MIDI, expected_key="G# minor", min_pitch=24, max_pitch=84, strict_musicality=True)
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
