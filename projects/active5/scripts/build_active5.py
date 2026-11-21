#!/usr/bin/env python3
"""
Build script for projects/active5/projet.gridline
"Obsidian Pulse" — 96 bars @ 180 BPM (128.0 s = 2 min 08 s), E minor.
Pure instrumental Dark Melodic Trap built 100% from projects/active5 sounds (NO rapper vocal).
Applies Vertical & Horizontal Inter-Track Harmonization:
  - Unified A4=440.0 Hz tuning across all tonal samples
  - Single unified 180 BPM Trap grid (zero competing raw techno loops)
  - Shared 4-bar chord progression (Em -> Em7 -> Cmaj7 -> Am/Bm) across 808, Chords, Pad, Pizz, Headroom Piano
  - Vertical Call-and-Response (Chords on strong beats, Pianos in rhythmic gaps)
  - Shared FX Room + Bus Glue + Mirror EQ + Section Dynamic Arc
"""

import json
import os
import random
import shutil
import subprocess
import time

ROOT = "/home/gamo/Bureau/musicos"
PROJ_DIR = f"{ROOT}/projects/active5"
PROJ_FILE = f"{PROJ_DIR}/projet.gridline"
RENDERS_DIR = f"{PROJ_DIR}/renders"
TEMPLATE = "/usr/share/gridline/assets/templates/Basics/Empty with mixer tracks.gridline"
SAMPLES_DIR = f"{PROJ_DIR}/samples"

random.seed(42)


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
            "clientInfo": {"name": "build_active5", "version": "2.0"},
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
    shutil.copyfile(TEMPLATE, PROJ_FILE)
    mcp = GridlineMCP(PROJ_FILE, RENDERS_DIR)

    try:
        # 1. Song settings: 180 BPM, 4/4, swing 0.05
        mcp.call("set_song", {
            "title": "Obsidian Pulse",
            "genre": "Dark Melodic Trap",
            "author": "AI-Studio",
            "tempo": 180,
            "swing": 0.05,
        })

        # 2. Ensure mixer tracks 1..20 exist
        mix_info = mcp.call("get_mixer", {})
        existing_mix = {t["mixer"] for t in mix_info["tracks"]}
        add_mix_calls = [{"tool": "add_mixer_track", "arguments": {}} for m in range(1, 21) if m not in existing_mix]
        if add_mix_calls:
            mcp.batch(add_mix_calls)

        # Clean default Master slot effects if any
        for tr in mix_info["tracks"]:
            if tr["mixer"] == 0:
                for fx in tr["effects"]:
                    mcp.call("remove_plugin", {"mixer": 0, "slot": fx["slot"]})

        # Mixer layout:
        # 1..7 -> BUS Drums (17), 2..3 -> BUS 808 (18), 8..13 -> BUS Music (19), 20 = FX Room -> Master (0)
        mixer_defs = [
            (1, "Kick", 17, 0.78, 0.0, 0.0),
            (2, "808 Apollo", 18, 0.80, 0.0, -0.8),
            (3, "808 Blxckout", 18, 0.78, 0.0, -0.8),
            (4, "Snare", 17, 0.86, 0.0, 0.0),
            (5, "Clap", 17, 0.84, 0.0, 0.0),
            (6, "Hats Trap", 17, 0.76, 0.0, 0.0),
            (7, "Perc / Rim / OpenHat", 17, 0.74, 0.0, 0.0),
            (8, "Techno Pad E", 19, 0.80, 0.0, -0.22),
            (9, "Chord Stab E", 19, 0.86, 0.0, -0.15),
            (10, "Upright Pizz E", 19, 0.88, -0.12, 0.0),
            (11, "Headroom Piano High", 19, 0.88, 0.14, -0.15),
            (12, "Chord High +12 & Riser", 19, 0.78, 0.0, -0.15),
            (13, "Chants Accent", 19, 0.70, 0.14, -0.20),
            (17, "BUS Drums", 0, 0.88, 0.0, 0.0),
            (18, "BUS 808", 0, 0.62, 0.0, -1.0),
            (19, "BUS Music", 0, 0.94, 0.0, -0.15),
            (20, "FX Room", 0, 0.72, 0.0, 0.0),
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

        # Shared FX Room sends (binds all families into ONE studio room!) + Kick -> BUS 808 sidechain
        sends = [
            (1, 18, 1.0, True),    # Kick sidechain -> BUS 808
            (4, 20, 0.05, False),   # Snare -> FX Room
            (5, 20, 0.05, False),   # Clap -> FX Room
            (7, 20, 0.06, False),   # Perc/Rim -> FX Room
            (8, 20, 0.11, False),   # Pad E -> FX Room
            (9, 20, 0.09, False),   # Chord Stab -> FX Room
            (10, 20, 0.09, False),  # Upright Pizz -> FX Room
            (11, 20, 0.11, False),  # Headroom Piano -> FX Room
            (12, 20, 0.10, False),  # Chord High +12 & Riser -> FX Room
            (13, 20, 0.10, False),  # Chants -> FX Room
        ]
        mcp.batch([
            {"tool": "set_send", "arguments": {"mixer": m, "to": to, "level": lvl, "sidechain": sc}}
            for m, to, lvl, sc in sends
        ])

        # 3. Channels 1..19 (100% derived from projects/active5 sounds, NO rapper track!)
        channels_spec = [
            # 1..2 Kick
            ("Kick Native", "gl.kick", 1, None, 0.74, 0.0),
            ("Techno Kick OneShot", "gl.sampler", 1, f"{SAMPLES_DIR}/techno_kick_oneshot.wav", 0.74, 0.0),
            # 3..4 808s
            ("MZ 808 [Apollo]", "gl.sampler", 2, f"{ROOT}/soundbanks/JMizzy The Best Drum Kits/!Mike Zara's Stash/808s/MZ 808 [Apollo].wav", 0.82, 0.0),
            ("MZ 808 [Blxckout]", "gl.sampler", 3, f"{ROOT}/soundbanks/JMizzy The Best Drum Kits/!Mike Zara's Stash/808s/MZ 808 [Blxckout].wav", 0.80, 0.0),
            # 5..12 Drums & Percs on ONE unified grid
            ("Snare Native", "gl.snare", 4, None, 0.76, 0.0),
            ("Clap Native", "gl.clap", 5, None, 0.70, 0.0),
            ("Techno Clap OneShot", "gl.sampler", 5, f"{SAMPLES_DIR}/techno_clap_oneshot.wav", 0.72, -0.05),
            ("Hat Native", "gl.hat", 6, None, 0.66, 0.06),
            ("Techno Hat OneShot", "gl.sampler", 6, f"{SAMPLES_DIR}/techno_hat_oneshot.wav", 0.66, -0.10),
            ("Techno OpenHat OneShot", "gl.sampler", 7, f"{SAMPLES_DIR}/techno_openhat_oneshot.wav", 0.62, 0.14),
            ("Techno Rim OneShot", "gl.sampler", 7, f"{SAMPLES_DIR}/techno_rim_oneshot.wav", 0.66, -0.14),
            ("Techno Shaker OneShot", "gl.sampler", 7, f"{SAMPLES_DIR}/techno_shaker_oneshot.wav", 0.62, 0.12),
            # 13..19 Tuned Melodic / Harmonic Instruments (A4=440.0 Hz unified!)
            ("Techno Pad E (440Hz)", "gl.sampler", 8, f"{SAMPLES_DIR}/techno_pad_e_440.wav", 0.70, 0.0),
            ("Techno Chord Stab (440Hz)", "gl.sampler", 9, f"{SAMPLES_DIR}/techno_chord_stab_e_440.wav", 0.76, 0.0),
            ("Upright Pizz E (440Hz)", "gl.sampler", 10, f"{SAMPLES_DIR}/upright_pizz_e3_440.wav", 0.80, -0.12),
            ("Headroom Piano High (440Hz)", "gl.sampler", 11, f"{ROOT}/soundbanks/sfz/headroom-piano/Samples/HEADROOM PIANO LEVEL5 CLOSE 93.flac", 0.78, 0.16),
            ("Techno Chord High +12 (440Hz)", "gl.sampler", 12, f"{SAMPLES_DIR}/techno_chord_stab_e_440.wav", 0.62, 0.0),
            ("Techno Chord Riser (440Hz)", "gl.sampler", 12, f"{SAMPLES_DIR}/techno_chord_riser_440.wav", 0.76, 0.0),
            ("(SXJ) Chants (6)", "gl.sampler", 13, f"{ROOT}/soundbanks/JMizzy The Best Drum Kits/JUGG DRUM KIT/Chants (Voxs)/(SXJ) Chants (6).wav", 0.66, 0.18),
        ]

        mcp.batch([
            {"tool": "add_channel", "arguments": {"name": name, "plugin": plugin, "mixer": mixer, "volume": vol, "pan": pan}}
            for name, plugin, mixer, _, vol, pan in channels_spec
        ])

        mcp.batch([
            {"tool": "patch", "arguments": {"path": f"channel/{idx}", "set": {"plugin": {"data": spath}}}}
            for idx, (_, _, _, spath, _, _) in enumerate(channels_spec, 1) if spath
        ])

        # 4. Channel tuning & ADSR envelopes (Protocol Section 2 + A4=440 Hz unification)
        ch_params = [
            # Ch 1 Kick Native: 125ms punch, 54Hz end pitch
            ("channel/1/Start pitch", 215), ("channel/1/End pitch", 54),
            ("channel/1/Sweep", 0.045), ("channel/1/Decay", 0.125),
            ("channel/1/Click", 0.36), ("channel/1/Drive", 0.22),
            # Ch 2 Techno Kick OneShot
            ("channel/2/One shot", 0), ("channel/2/Release", 0.02), ("channel/2/Declick", 2), ("channel/2/Remove DC", 1),
            # Ch 3 808 Apollo: Root 24 (C1), Fine -14ct (440 Hz), Voices=1 Cut Itself!, Release 45ms, Glide 55ms
            ("channel/3/Root note", 24), ("channel/3/Fine tune", -14), ("channel/3/Voices", 1),
            ("channel/3/One shot", 0), ("channel/3/Attack", 0.002), ("channel/3/Decay", 10.0),
            ("channel/3/Sustain", 1.0), ("channel/3/Release", 0.045), ("channel/3/Glide", 0.055),
            ("channel/3/Remove DC", 1), ("channel/3/Declick", 2),
            # Ch 4 808 Blxckout: Root 24 (C1), Fine +6ct (440 Hz), Voices=1 Cut Itself!, Release 45ms, Glide 55ms
            ("channel/4/Root note", 24), ("channel/4/Fine tune", 6), ("channel/4/Voices", 1),
            ("channel/4/One shot", 0), ("channel/4/Attack", 0.002), ("channel/4/Decay", 10.0),
            ("channel/4/Sustain", 1.0), ("channel/4/Release", 0.045), ("channel/4/Glide", 0.055),
            ("channel/4/Remove DC", 1), ("channel/4/Declick", 2),
            # Ch 5 Snare Native: 110ms body, 160ms noise
            ("channel/5/Tone", 0.58), ("channel/5/Snappy", 0.74),
            ("channel/5/Body decay", 0.11), ("channel/5/Noise decay", 0.16), ("channel/5/Tune", 198),
            # Ch 6 Clap Native: 180ms decay
            ("channel/6/Decay", 0.18), ("channel/6/Tone", 1800), ("channel/6/Hits", 4), ("channel/6/Spread", 0.011),
            # Ch 7 Techno Clap OneShot
            ("channel/7/One shot", 0), ("channel/7/Release", 0.025), ("channel/7/Declick", 2),
            # Ch 8 Hat Native: crisp 45ms decay (< 1/16 = 83.3ms)
            ("channel/8/Decay", 0.045), ("channel/8/Tone", 9200), ("channel/8/Metal", 0.22), ("channel/8/Choke", 1),
            # Ch 9 Techno Hat OneShot
            ("channel/9/One shot", 0), ("channel/9/Release", 0.015), ("channel/9/Declick", 2),
            # Ch 10 OpenHat, Ch 11 Rim, Ch 12 Shaker
            ("channel/10/One shot", 0), ("channel/10/Release", 0.03), ("channel/10/Declick", 2),
            ("channel/11/One shot", 0), ("channel/11/Release", 0.02), ("channel/11/Declick", 2),
            ("channel/12/One shot", 0), ("channel/12/Release", 0.015), ("channel/12/Declick", 2),
            # Ch 13 Techno Pad E: Root 64 (E4), Fine -4ct (exact 440 Hz!), soft envelope
            ("channel/13/Root note", 64), ("channel/13/Fine tune", -4),
            ("channel/13/One shot", 0), ("channel/13/Attack", 0.035), ("channel/13/Decay", 1.5),
            ("channel/13/Sustain", 0.85), ("channel/13/Release", 0.28), ("channel/13/Declick", 2), ("channel/13/Remove DC", 1),
            # Ch 14 Techno Chord Stab E: Root 64 (E4), Fine -4ct (exact 440 Hz!), smooth 4ms attack + 140ms release
            ("channel/14/Root note", 64), ("channel/14/Fine tune", -4),
            ("channel/14/One shot", 0), ("channel/14/Attack", 0.004), ("channel/14/Decay", 0.55),
            ("channel/14/Sustain", 0.55), ("channel/14/Release", 0.14), ("channel/14/Declick", 2), ("channel/14/Remove DC", 1),
            # Ch 15 Upright Pizz E: Root 52 (E3), Fine +8ct (exact 440 Hz!), 4ms attack + crisp 80ms release
            ("channel/15/Root note", 52), ("channel/15/Fine tune", 8),
            ("channel/15/One shot", 0), ("channel/15/Attack", 0.004), ("channel/15/Decay", 0.45),
            ("channel/15/Sustain", 0.40), ("channel/15/Release", 0.08), ("channel/15/Declick", 2), ("channel/15/Remove DC", 1),
            # Ch 16 Headroom Piano High: Root 93 (A6), Fine -14ct (exact 440 Hz!)
            ("channel/16/Root note", 93), ("channel/16/Fine tune", -14),
            ("channel/16/One shot", 0), ("channel/16/Attack", 0.004), ("channel/16/Decay", 1.2),
            ("channel/16/Sustain", 0.70), ("channel/16/Release", 0.22), ("channel/16/Declick", 2), ("channel/16/Remove DC", 1),
            # Ch 17 Techno Chord High +12: Root 52 (sounds +12 st when played at 64/60/57/59!), Fine -4ct
            ("channel/17/Root note", 52), ("channel/17/Fine tune", -4),
            ("channel/17/One shot", 0), ("channel/17/Attack", 0.004), ("channel/17/Decay", 0.50),
            ("channel/17/Sustain", 0.45), ("channel/17/Release", 0.14), ("channel/17/Declick", 2), ("channel/17/Remove DC", 1),
            # Ch 18 Techno Chord Riser: Root 64, Fine -4ct
            ("channel/18/Root note", 64), ("channel/18/Fine tune", -4),
            ("channel/18/One shot", 0), ("channel/18/Release", 0.03), ("channel/18/Declick", 2),
            # Ch 19 (SXJ) Chants (6): Root 64
            ("channel/19/Root note", 64), ("channel/19/One shot", 0), ("channel/19/Release", 0.03), ("channel/19/Declick", 2),
        ]
        mcp.call("set_parameters", {"changes": [{"path": p, "value": v} for p, v in ch_params]})

        # 5. Mixer DSP & Bus Glue (Quality=0 on inserts, Quality=1 on Master)
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
            # BUS Music (19): EQ (1), Bus Glue Comp (2) -> binds Pad + Chords + Pianos into ONE ensemble!
            {"tool": "add_plugin", "arguments": {"mixer": 19, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 19, "plugin": "gl.comp"}},
            # FX Room (20): gl.reverb (1)
            {"tool": "add_plugin", "arguments": {"mixer": 20, "plugin": "gl.reverb"}},
            # Mirror EQs on harmonic layers:
            # Pad E (8): HPF 165 Hz, dip 320 Hz, dip 1250 Hz (room for Pizz/Piano)
            {"tool": "add_plugin", "arguments": {"mixer": 8, "plugin": "gl.eq"}},
            # Chord Stab (9): HPF 165 Hz, dip 310 Hz
            {"tool": "add_plugin", "arguments": {"mixer": 9, "plugin": "gl.eq"}},
            # Upright Pizz (10): HPF 175 Hz, presence +1.8 dB @ 1250 Hz
            {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.eq"}},
            # Headroom Piano (11): HPF 320 Hz
            {"tool": "add_plugin", "arguments": {"mixer": 11, "plugin": "gl.eq"}},
            # Chord High +12 (12): HPF 550 Hz
            {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.eq"}},
        ]
        mcp.batch(fx_adds)

        fx_changes = [
            # Master (0)
            ("mixer/0/slot/1/Quality", 1),
            ("mixer/0/slot/1/Band 1 type", 1), ("mixer/0/slot/1/Band 1 freq", 28),
            ("mixer/0/slot/1/Band 7 type", 3), ("mixer/0/slot/1/Band 7 freq", 17500),
            ("mixer/0/slot/2/Threshold", -11.5), ("mixer/0/slot/2/Ratio", 1.7),
            ("mixer/0/slot/2/Attack", 0.020), ("mixer/0/slot/2/Release", 0.11),
            ("mixer/0/slot/2/Makeup", 1.8), ("mixer/0/slot/2/Detector high-pass", 95),
            ("mixer/0/slot/3/Gain", 8.2), ("mixer/0/slot/3/Ceiling", -2.75),
            ("mixer/0/slot/3/Attack (look-ahead)", 0.002), ("mixer/0/slot/3/0.2 dB safety", 1),

            # BUS Drums (17)
            ("mixer/17/slot/1/Band 1 type", 1), ("mixer/17/slot/1/Band 1 freq", 32),
            ("mixer/17/slot/1/Band 2 type", 6), ("mixer/17/slot/1/Band 2 freq", 63), ("mixer/17/slot/1/Band 2 gain", -2.0), ("mixer/17/slot/1/Band 2 Q (legacy)", 1.4),
            ("mixer/17/slot/1/Band 7 type", 7), ("mixer/17/slot/1/Band 7 freq", 10500), ("mixer/17/slot/1/Band 7 gain", 1.8),
            ("mixer/17/slot/2/Threshold", -15), ("mixer/17/slot/2/Ratio", 2.5),
            ("mixer/17/slot/2/Attack", 0.012), ("mixer/17/slot/2/Release", 0.085), ("mixer/17/slot/2/Makeup", 1.8),

            # BUS 808 (18)
            ("mixer/18/slot/1/Width", 0.0),
            ("mixer/18/slot/2/Detector input", 1),
            ("mixer/18/slot/2/Threshold", -22), ("mixer/18/slot/2/Ratio", 5.5),
            ("mixer/18/slot/2/Attack", 0.002), ("mixer/18/slot/2/Release", 0.065), ("mixer/18/slot/2/Lookahead", 2),
            ("mixer/18/slot/3/Band 1 type", 1), ("mixer/18/slot/3/Band 1 freq", 28),
            ("mixer/18/slot/3/Band 2 type", 6), ("mixer/18/slot/3/Band 2 freq", 63), ("mixer/18/slot/3/Band 2 gain", -4.5), ("mixer/18/slot/3/Band 2 Q (legacy)", 1.6),
            ("mixer/18/slot/3/Band 3 type", 6), ("mixer/18/slot/3/Band 3 freq", 115), ("mixer/18/slot/3/Band 3 gain", 1.5), ("mixer/18/slot/3/Band 3 Q (legacy)", 1.2),

            # BUS Music (19): HPF 150 Hz + Presence fill (no rapper vocal, so chords/pianos own 1.5-4 kHz!) + Bus Glue Comp
            ("mixer/19/slot/1/Band 1 type", 1), ("mixer/19/slot/1/Band 1 freq", 150),
            ("mixer/19/slot/1/Band 5 type", 6), ("mixer/19/slot/1/Band 5 freq", 1550), ("mixer/19/slot/1/Band 5 gain", 2.4), ("mixer/19/slot/1/Band 5 Q (legacy)", 0.85),
            ("mixer/19/slot/1/Band 6 type", 6), ("mixer/19/slot/1/Band 6 freq", 3600), ("mixer/19/slot/1/Band 6 gain", 2.0), ("mixer/19/slot/1/Band 6 Q (legacy)", 0.90),
            ("mixer/19/slot/2/Threshold", -15), ("mixer/19/slot/2/Ratio", 2.0),
            ("mixer/19/slot/2/Attack", 0.020), ("mixer/19/slot/2/Release", 0.11), ("mixer/19/slot/2/Makeup", 2.2),

            # FX Room (20): shared short acoustic space (0.333 s = 1 * T_noire, Dry=0)
            ("mixer/20/slot/1/Model", 1),
            ("mixer/20/slot/1/Pre-delay", 12), ("mixer/20/slot/1/Room size", 0.28),
            ("mixer/20/slot/1/Decay time", 0.333),
            ("mixer/20/slot/1/Low cut", 420), ("mixer/20/slot/1/High cut", 6800), ("mixer/20/slot/1/High damping", 4200),
            ("mixer/20/slot/1/Dry", 0), ("mixer/20/slot/1/Wet", 0.72), ("mixer/20/slot/1/Width", 0.85),
            ("mixer/20/slot/1/Low cut (full)", 420), ("mixer/20/slot/1/Decay (full)", 3),
            ("mixer/20/slot/1/Dry (full)", 0), ("mixer/20/slot/1/Early reflections (full)", 14),
            ("mixer/20/slot/1/Wet (full)", 72),

            # Mirror EQs on harmonic layers:
            # Pad E (8)
            ("mixer/8/slot/1/Band 1 type", 1), ("mixer/8/slot/1/Band 1 freq", 165),
            ("mixer/8/slot/1/Band 3 type", 6), ("mixer/8/slot/1/Band 3 freq", 320), ("mixer/8/slot/1/Band 3 gain", -2.5),
            ("mixer/8/slot/1/Band 5 type", 6), ("mixer/8/slot/1/Band 5 freq", 1250), ("mixer/8/slot/1/Band 5 gain", -2.2),
            # Chord Stab (9)
            ("mixer/9/slot/1/Band 1 type", 1), ("mixer/9/slot/1/Band 1 freq", 165),
            ("mixer/9/slot/1/Band 3 type", 6), ("mixer/9/slot/1/Band 3 freq", 310), ("mixer/9/slot/1/Band 3 gain", -2.2),
            # Upright Pizz (10)
            ("mixer/10/slot/1/Band 1 type", 1), ("mixer/10/slot/1/Band 1 freq", 175),
            ("mixer/10/slot/1/Band 5 type", 6), ("mixer/10/slot/1/Band 5 freq", 1250), ("mixer/10/slot/1/Band 5 gain", 1.8),
            # Headroom Piano (11)
            ("mixer/11/slot/1/Band 1 type", 1), ("mixer/11/slot/1/Band 1 freq", 320),
            # Chord High +12 (12)
            ("mixer/12/slot/1/Band 1 type", 1), ("mixer/12/slot/1/Band 1 freq", 550),
        ]
        mcp.call("set_parameters", {"changes": [{"path": p, "value": v} for p, v in fx_changes]})

        # 6. Playlist Tracks 1..12 (clean 1-instrument-per-track layout)
        track_names = [
            (1, "808 Apollo (Couplets)"),
            (2, "808 Blxckout (Refrains)"),
            (3, "Kick (Composite Lock)"),
            (4, "Snare & Clap"),
            (5, "Hi-Hats Trap (Unified)"),
            (6, "Rim, Shaker & OpenHat"),
            (7, "Techno Pad E (440 Hz)"),
            (8, "Chord Stab E (440 Hz)"),
            (9, "Chord High +12 & Riser"),
            (10, "Upright Pizz E (440 Hz)"),
            (11, "Headroom Piano High (440 Hz)"),
            (12, "Chants Accent & FX Room Cuts"),
        ]
        mcp.batch([{"tool": "patch", "arguments": {"path": f"track/{t}", "set": {"name": name}}} for t, name in track_names])

        # 7. Shared 4-bar Master Chord Grid: Bar0=Em, Bar1=Em7, Bar2=Cmaj7, Bar3=Am->Bm
        # (Repeated twice per 8-bar block)
        def bar_chord_root(bar_idx: int):
            b = bar_idx % 4
            if b in (0, 1):
                return 64  # E4
            elif b == 2:
                return 60  # C4
            return 57  # A3 (switches to B3=59 at beat 2.0!)

        pat_counter = 1
        queued = [{"tool": "edit_pattern", "arguments": {"pattern": 1, "name": "Pad E - Intro 8m", "bars": 8}}]

        def new_pattern(name: str, bars: int) -> int:
            nonlocal pat_counter
            pat_counter += 1
            queued.append({"tool": "add_pattern", "arguments": {"name": name, "bars": bars}})
            return pat_counter

        def add_notes(pat: int, ch: int, notes: list):
            queued.append({"tool": "set_notes", "arguments": {"pattern": pat, "channel": ch, "notes": notes}})

        def place(pat: int, track: int, start_bar: int):
            queued.append({"tool": "add_clip", "arguments": {"pattern": pat, "track": track, "start": (start_bar - 1) * 4.0}})

        # ---- PAD E (Track 7, Ch 13) ----
        def make_pad_notes(bars=8, vel=82, mute_last_beat=True):
            notes = []
            for b in range(bars):
                r = bar_chord_root(b)
                if b % 4 == 3:
                    # Am (beats 0..2) -> Bm (beats 2..4 or 2..3 if mute)
                    notes.append({"start": b * 4.0, "length": 1.85, "pitch": 57, "velocity": vel})
                    ln_b = 0.9 if (mute_last_beat and b == bars - 1) else 1.85
                    notes.append({"start": b * 4.0 + 2.0, "length": ln_b, "pitch": 59, "velocity": vel - 4})
                else:
                    notes.append({"start": b * 4.0, "length": 3.75, "pitch": r, "velocity": vel})
            return notes

        add_notes(1, 13, make_pad_notes(8, vel=76, mute_last_beat=True))
        place(1, 7, 1)

        p_pad_c1b = new_pattern("Pad E - Couplet 1b", 8)
        add_notes(p_pad_c1b, 13, make_pad_notes(8, vel=84, mute_last_beat=True))
        place(p_pad_c1b, 7, 25)

        p_pad_pont = new_pattern("Pad E - Pont", 8)
        add_notes(p_pad_pont, 13, make_pad_notes(8, vel=76, mute_last_beat=True))
        place(p_pad_pont, 7, 33)

        p_pad_outro = new_pattern("Pad E - Outro", 8)
        add_notes(p_pad_outro, 13, make_pad_notes(8, vel=72, mute_last_beat=False))
        place(p_pad_outro, 7, 89)

        # ---- CHORD STAB E (Track 8, Ch 14) & CHORD HIGH +12 / RISER (Track 9, Ch 17 / Ch 18) ----
        # Locked strictly with Kick & 808 on beats 0.0 and 1.5 (leaving beats 1.0, 2.5, 3.5 for Piano Call-and-Response!)
        def make_chord_stab_notes(bars=8, vel=100, mode="drop"):
            notes = []
            for b in range(bars):
                r = bar_chord_root(b)
                is_last = (b == bars - 1)
                if mode == "drop":
                    if b % 4 == 3:
                        notes.append({"start": b * 4.0 + 0.0, "length": 1.25, "pitch": 57, "velocity": vel})
                        notes.append({"start": b * 4.0 + 1.5, "length": 0.45, "pitch": 57, "velocity": vel - 8})
                        if not is_last:
                            notes.append({"start": b * 4.0 + 2.0, "length": 1.0, "pitch": 59, "velocity": vel - 2})
                        else:
                            notes.append({"start": b * 4.0 + 2.0, "length": 0.85, "pitch": 59, "velocity": vel - 4})
                    else:
                        notes.append({"start": b * 4.0 + 0.0, "length": 1.25, "pitch": r, "velocity": vel})
                        notes.append({"start": b * 4.0 + 1.5, "length": 0.90, "pitch": r, "velocity": vel - 6})
                elif mode == "staccato":
                    # Couplet 2 crisp rhythmic stabs (0.35 beat) leaving maximum gap for Pizz
                    if b % 4 == 3:
                        notes.append({"start": b * 4.0 + 0.0, "length": 0.40, "pitch": 57, "velocity": vel - 6})
                        notes.append({"start": b * 4.0 + 2.0, "length": 0.40, "pitch": 59, "velocity": vel - 8})
                    else:
                        notes.append({"start": b * 4.0 + 0.0, "length": 0.45, "pitch": r, "velocity": vel - 4})
                        notes.append({"start": b * 4.0 + 1.5, "length": 0.35, "pitch": r, "velocity": vel - 10})
            return notes

        p_stab_drop = new_pattern("Chord Stab - Drop 8m", 8)
        add_notes(p_stab_drop, 14, make_chord_stab_notes(8, vel=100, mode="drop"))
        for sb in [9, 41, 49, 73, 81]:
            place(p_stab_drop, 8, sb)

        p_stab_c2 = new_pattern("Chord Stab - Couplet 2", 8)
        add_notes(p_stab_c2, 14, make_chord_stab_notes(8, vel=90, mode="staccato"))
        place(p_stab_c2, 8, 57)
        place(p_stab_c2, 8, 65)

        # High +12 Chord layer (Track 9, Ch 17 Root=52) in Refrain 2b (49) & Refrain 3 (73, 81)
        p_stab_hi = new_pattern("Chord High +12 - Climax 8m", 8)
        add_notes(p_stab_hi, 17, make_chord_stab_notes(8, vel=92, mode="drop"))
        place(p_stab_hi, 9, 49)
        place(p_stab_hi, 9, 73)
        place(p_stab_hi, 9, 81)

        # Riser (Track 9, Ch 18) on M8, M40, M72
        p_riser = new_pattern("Riser - 4 beats", 1)
        add_notes(p_riser, 18, [{"start": 0.0, "length": 3.95, "pitch": 64, "velocity": 100}])
        place(p_riser, 9, 8)
        place(p_riser, 9, 40)
        place(p_riser, 9, 72)

        # ---- UPRIGHT PIZZ E (Track 10, Ch 15, Root=52 E3) ----
        # Plays ONLY chord tones of Em->Em7->Cmaj7->Am/Bm, answering in the rhythmic gaps (1.0, 2.5, 3.5)!
        def make_pizz_notes(bars=8, vel_base=88, mute_last_beat=True):
            notes = []
            # 4-bar chord-locked call-and-response (0.35-0.45 beat crisp notes)
            bar_patterns = {
                0: [(1.0, 55), (2.5, 59), (3.5, 55)],       # Em: G3, B3, G3
                1: [(1.0, 59), (2.5, 62), (3.5, 59)],       # Em7: B3, D4, B3
                2: [(1.0, 55), (2.5, 60), (3.5, 52)],       # Cmaj7: G3, C4, E3
                3: [(1.0, 60), (2.5, 59), (3.5, 54)],       # Am->Bm: C4 -> B3, F#3 (voice-leading back to E3!)
            }
            for b in range(bars):
                for st, pt in bar_patterns[b % 4]:
                    t = b * 4.0 + st
                    if mute_last_beat and t >= bars * 4.0 - 1.0:
                        continue
                    jt = max(0.0, t + random.uniform(-0.008, 0.008))
                    jv = max(48, min(115, vel_base + random.randint(-5, 5)))
                    notes.append({"start": round(jt, 4), "length": 0.42, "pitch": pt, "velocity": jv})
            return notes

        p_pizz_intro = new_pattern("Upright Pizz - Intro", 8)
        add_notes(p_pizz_intro, 15, make_pizz_notes(8, vel_base=78, mute_last_beat=True))
        place(p_pizz_intro, 10, 1)

        p_pizz_c1 = new_pattern("Upright Pizz - Couplet 1", 8)
        add_notes(p_pizz_c1, 15, make_pizz_notes(8, vel_base=92, mute_last_beat=True))
        place(p_pizz_c1, 10, 17)
        place(p_pizz_c1, 10, 25)

        p_pizz_c2 = new_pattern("Upright Pizz - Couplet 2", 8)
        add_notes(p_pizz_c2, 15, make_pizz_notes(8, vel_base=88, mute_last_beat=True))
        place(p_pizz_c2, 10, 57)
        place(p_pizz_c2, 10, 65)

        p_pizz_outro = new_pattern("Upright Pizz - Outro", 8)
        add_notes(p_pizz_outro, 15, [
            {"start": 1.0, "length": 0.5, "pitch": 55, "velocity": 76},
            {"start": 2.5, "length": 0.5, "pitch": 59, "velocity": 72},
            {"start": 5.0, "length": 0.5, "pitch": 55, "velocity": 70},
            {"start": 9.0, "length": 0.5, "pitch": 60, "velocity": 68},
            {"start": 16.0, "length": 0.8, "pitch": 52, "velocity": 66},
            {"start": 24.0, "length": 1.0, "pitch": 52, "velocity": 60},
        ])
        place(p_pizz_outro, 10, 89)

        # ---- HEADROOM PIANO HIGH (Track 11, Ch 16, Root=93 A6) ----
        # Plays strictly upper chord tones in Refrains, Pont, Outro — answering Chord Stab in gaps (1.0, 2.5, 3.5)!
        def make_headroom_notes(bars=8, vel_base=88, mute_last_beat=True):
            notes = []
            bar_patterns = {
                0: [(1.0, 79), (2.5, 83), (3.5, 76)],       # Em: G5, B5, E5
                1: [(1.0, 83), (2.5, 74), (3.5, 79)],       # Em7: B5, D5, G5
                2: [(1.0, 76), (2.5, 79), (3.5, 83)],       # Cmaj7: E5, G5, B5
                3: [(1.0, 81), (2.5, 83), (3.5, 78)],       # Am->Bm: A5 -> B5, F#5
            }
            for b in range(bars):
                for st, pt in bar_patterns[b % 4]:
                    t = b * 4.0 + st
                    if mute_last_beat and t >= bars * 4.0 - 1.0:
                        continue
                    jt = max(0.0, t + random.uniform(-0.008, 0.008))
                    jv = max(48, min(115, vel_base + random.randint(-5, 5)))
                    notes.append({"start": round(jt, 4), "length": 0.45, "pitch": pt, "velocity": jv})
            return notes

        p_hr_drop = new_pattern("Headroom Piano - Drop 8m", 8)
        add_notes(p_hr_drop, 16, make_headroom_notes(8, vel_base=90, mute_last_beat=True))
        for sb in [9, 41, 49, 73, 81]:
            place(p_hr_drop, 11, sb)

        p_hr_pont = new_pattern("Headroom Piano - Pont 8m", 8)
        add_notes(p_hr_pont, 16, make_headroom_notes(8, vel_base=76, mute_last_beat=True))
        place(p_hr_pont, 11, 33)

        p_hr_outro = new_pattern("Headroom Piano - Outro", 8)
        add_notes(p_hr_outro, 16, [
            {"start": 1.0, "length": 0.6, "pitch": 79, "velocity": 74},
            {"start": 2.5, "length": 0.6, "pitch": 83, "velocity": 70},
            {"start": 6.5, "length": 0.6, "pitch": 76, "velocity": 68},
            {"start": 10.5, "length": 0.6, "pitch": 79, "velocity": 64},
            {"start": 16.0, "length": 0.9, "pitch": 76, "velocity": 60},
        ])
        place(p_hr_outro, 11, 89)

        # ---- 808 BLXCKOUT (Refrains, Track 2, Ch 4) & 808 APOLLO (Couplets, Track 1, Ch 3) ----
        # Strictly locked to Em -> Em7 -> Cmaj7 -> Am/Bm AND Kick hits!
        def make_808_blxckout(climax=False):
            notes = []
            for b in range(8):
                is_last = (b == 7)
                m4 = b % 4
                if m4 in (0, 1):
                    # Em
                    hits = [
                        (0.0, 1.45, 28, 114, False),
                        (1.5, 1.20, 28, 106, False),
                        (2.75, 1.15, 40 if (m4 == 1 or climax) else 28, 104, True),
                    ]
                elif m4 == 2:
                    # Cmaj7
                    hits = [
                        (0.0, 1.45, 36, 114, False),
                        (1.5, 1.20, 36, 106, False),
                        (2.75, 1.15, 48 if climax else 36, 102, True),
                    ]
                else:
                    # Am (beats 0..2) -> Bm (beats 2..4)
                    if is_last:
                        hits = [
                            (0.0, 1.45, 33, 112, False),
                            (1.5, 0.45, 33, 104, False),
                            (2.0, 0.90, 35, 108, True),  # cuts cleanly before beat 4!
                        ]
                    else:
                        hits = [
                            (0.0, 1.45, 33, 112, False),
                            (1.5, 0.45, 33, 104, False),
                            (2.0, 1.20, 35, 108, True),
                            (3.5, 0.45, 38 if climax else 35, 100, True),
                        ]
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

        p_808_ref_climax = new_pattern("808 Blxckout - Climax", 8)
        add_notes(p_808_ref_climax, 4, make_808_blxckout(climax=True))
        place(p_808_ref_climax, 2, 49)
        place(p_808_ref_climax, 2, 73)
        place(p_808_ref_climax, 2, 81)

        def make_808_apollo(syncopated=False):
            notes = []
            for b in range(8):
                is_last = (b == 7)
                m4 = b % 4
                if m4 in (0, 1):
                    root = 28
                elif m4 == 2:
                    root = 36
                else:
                    root = 33
                if is_last:
                    hits = [(0.0, 0.85, 33, 110, False), (2.0, 0.85, 35, 106, True)]
                elif m4 == 3:
                    hits = [(0.0, 0.85, 33, 110, False), (2.0, 0.85, 35, 106, True), (3.5, 0.40, 35, 98, False)]
                elif syncopated:
                    hits = [(0.0, 0.80, root, 110, False), (1.5, 0.55, root + 12, 102, True), (2.75, 0.75, root, 104, False)]
                else:
                    hits = [(0.0, 0.85, root, 108, False), (1.5, 0.65, root, 100, False), (2.75, 0.75, root, 100, False)]
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

        # ---- KICK (Track 3, Ch 1 Native + Ch 2 Techno Kick OneShot) ----
        # Composite locked with 808 + Chord Stab!
        def make_kick_notes(bars=8, dense=True, mute_last_beat=True):
            notes = []
            for b in range(bars):
                m4 = b % 4
                if b == bars - 1 and mute_last_beat:
                    times = [0.0, 2.0] if m4 == 3 else [0.0, 1.5]
                elif m4 == 3:
                    times = [0.0, 1.5, 2.0]
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

        # ---- SNARE & CLAP (Track 4, Ch 5 Snare, Ch 6 Clap, Ch 7 Techno Clap) ----
        # Locked on beat 3 (half-time)!
        def make_sc(bars=8, fill_end=True, mute_last_beat=False):
            sn, cl = [], []
            for b in range(bars):
                sn.append({"start": b * 4.0 + 2.0, "length": 0.42, "pitch": 60, "velocity": 112})
                cl.append({"start": b * 4.0 + 2.0, "length": 0.42, "pitch": 60, "velocity": 108})
                if b in (1, 3, 5):
                    sn.append({"start": b * 4.0 + 3.75, "length": 0.22, "pitch": 60, "velocity": 62})
                if b == bars - 1 and fill_end and not mute_last_beat:
                    for sub, vel in [(3.0, 74), (3.25, 82), (3.5, 94), (3.75, 106)]:
                        sn.append({"start": b * 4.0 + sub, "length": 0.18, "pitch": 60, "velocity": vel})
            return sn, cl

        p_sc_fill = new_pattern("Snare/Clap - Drop Fill", 8)
        sn, cl = make_sc(8, fill_end=True, mute_last_beat=False)
        add_notes(p_sc_fill, 5, sn)
        add_notes(p_sc_fill, 6, cl)
        add_notes(p_sc_fill, 7, cl)
        for sb in [9, 41, 49, 73, 81]:
            place(p_sc_fill, 4, sb)

        p_sc_mute = new_pattern("Snare/Clap - Couplet Mute", 8)
        snm, clm = make_sc(8, fill_end=False, mute_last_beat=True)
        add_notes(p_sc_mute, 5, snm)
        add_notes(p_sc_mute, 6, clm)
        add_notes(p_sc_mute, 7, clm)
        for sb in [17, 25, 57, 65]:
            place(p_sc_mute, 4, sb)

        p_sc_pont = new_pattern("Snare - Pont Buildup", 8)
        psn = [{"start": b * 4.0 + 2.0, "length": 0.35, "pitch": 60, "velocity": 84 + (b - 4) * 8} for b in range(4, 7)]
        for step in range(8):
            psn.append({"start": 7 * 4.0 + step * 0.5, "length": 0.18, "pitch": 60, "velocity": 70 + step * 6})
        add_notes(p_sc_pont, 5, psn)
        place(p_sc_pont, 4, 33)

        p_sc_outro = new_pattern("Snare/Clap - Outro 4m", 4)
        sno, clo = make_sc(4, fill_end=False, mute_last_beat=True)
        add_notes(p_sc_outro, 5, sno)
        add_notes(p_sc_outro, 6, clo)
        place(p_sc_outro, 4, 89)

        # ---- UNIFIED TRAP HI-HATS (Track 5, Ch 8 Hat Native + Ch 9 Techno Hat OneShot) ----
        def make_hats(bars=8, energy="medium", mute_last_beat=True):
            hn, hs = [], []
            for b in range(bars):
                for step in range(8):
                    t = b * 4.0 + step * 0.5
                    if mute_last_beat and t >= bars * 4.0 - 1.0:
                        continue
                    vel = (104 if step % 2 == 0 else 68) + random.randint(-5, 5)
                    # Triplet roll before beat 3 on odd bars
                    if b % 2 == 1 and step == 3:
                        for k in range(3):
                            rt = b * 4.0 + 1.5 + k * (0.5 / 3.0)
                            hn.append({"start": round(rt, 4), "length": 0.11, "pitch": 60 - k * 2, "velocity": 82 + k * 7})
                        continue
                    # 1/32 pitch-down roll at end of bar 3/7
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

        # ---- RIM, SHAKER & OPENHAT ON UNIFIED GRID (Track 6, Ch 10 OpenHat, Ch 11 Rim, Ch 12 Shaker) ----
        p_rim_intro = new_pattern("Rim & Shaker - Intro M5-8", 4)
        add_notes(p_rim_intro, 11, [{"start": b * 4.0 + t, "length": 0.2, "pitch": 60, "velocity": 74} for b in range(3) for t in (1.5, 3.5)])
        add_notes(p_rim_intro, 12, [{"start": b * 4.0 + s * 0.5, "length": 0.15, "pitch": 60, "velocity": 68 if s % 2 == 0 else 50} for b in range(3) for s in range(8)])
        place(p_rim_intro, 6, 5)
        place(p_rim_intro, 6, 33)

        p_perc_c1 = new_pattern("Rim - Couplet 1a", 8)
        add_notes(p_perc_c1, 11, [{"start": b * 4.0 + t, "length": 0.2, "pitch": 60, "velocity": 82} for b in range(7) for t in (1.75, 3.5)])
        place(p_perc_c1, 6, 17)

        p_perc_c_full = new_pattern("Rim & Shaker - Couplet Full", 8)
        add_notes(p_perc_c_full, 11, [{"start": b * 4.0 + t, "length": 0.2, "pitch": 60, "velocity": 84} for b in range(7) for t in (1.75, 3.5)])
        add_notes(p_perc_c_full, 12, [{"start": b * 4.0 + s * 0.5, "length": 0.15, "pitch": 60, "velocity": 74 if s % 2 == 0 else 54} for b in range(7) for s in range(8)])
        for sb in [25, 57, 65]:
            place(p_perc_c_full, 6, sb)

        p_perc_drop2 = new_pattern("OpenHat & Shaker - Drop 8m", 8)
        add_notes(p_perc_drop2, 10, [{"start": b * 4.0 + 0.5, "length": 0.45, "pitch": 60, "velocity": 88} for b in (0, 2, 4, 6)])
        add_notes(p_perc_drop2, 12, [{"start": b * 4.0 + s * 0.5, "length": 0.15, "pitch": 60, "velocity": 78 if s % 2 == 0 else 56} for b in range(7) for s in range(8)])
        for sb in [41, 49]:
            place(p_perc_drop2, 6, sb)

        p_perc_climax = new_pattern("OpenHat, Rim & Shaker - Climax", 8)
        add_notes(p_perc_climax, 10, [{"start": b * 4.0 + 0.5, "length": 0.45, "pitch": 60, "velocity": 92} for b in (0, 2, 4, 6)])
        add_notes(p_perc_climax, 11, [{"start": b * 4.0 + t, "length": 0.2, "pitch": 60, "velocity": 84} for b in range(7) for t in (1.75, 3.5)])
        add_notes(p_perc_climax, 12, [{"start": b * 4.0 + s * 0.5, "length": 0.15, "pitch": 60, "velocity": 82 if s % 2 == 0 else 58} for b in range(7) for s in range(8)])
        for sb in [73, 81]:
            place(p_perc_climax, 6, sb)

        # ---- CHANTS ACCENT (Track 12, Ch 19) ----
        p_chants = new_pattern("Chants - Sparse Accent 8m", 8)
        add_notes(p_chants, 19, [
            {"start": 1 * 4.0 + 3.5, "length": 0.6, "pitch": 64, "velocity": 88},
            {"start": 5 * 4.0 + 3.5, "length": 0.6, "pitch": 64, "velocity": 90},
        ])
        for sb in [9, 41, 49, 73, 81]:
            place(p_chants, 12, sb)

        # ---- FX ROOM TRANSITION CUTS (Track 12, Mixer 20 volume automation) ----
        pts = [{"position": 0.0, "value": 0.72, "shape": "hold"}]
        for pivot_bar in [8, 32, 40, 56, 72]:
            cut_start = (pivot_bar - 1) * 4.0 + 3.0
            cut_end = pivot_bar * 4.0
            pts.append({"position": cut_start, "value": 0.0, "shape": "hold"})
            pts.append({"position": cut_end, "value": 0.72, "shape": "hold"})
        pts.append({"position": 96 * 4.0, "value": 0.72, "shape": "hold"})
        queued.append({"tool": "add_automation", "arguments": {
            "mixer": 20, "param": "volume", "name": "FX Room Transition Cuts",
            "points": pts, "track": 12, "start": 0.0, "length": 96 * 4.0,
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
