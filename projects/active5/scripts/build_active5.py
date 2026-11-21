#!/usr/bin/env python3
"""
Build script for projects/active5/projet.gridline
"Stoicville Ignition" — 96 bars @ 180 BPM (128.0 s = 2 min 08 s), E minor.
Uses exclusively the sounds present in projects/active5 (Untitled.gridline + active5/samples).
Implements the Multi-Scale Predictive Protocol + Acoustic Duration & ADSR Sizing (Section 2).
"""

import json
import os
import random
import shutil
import subprocess
import sys
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
            "clientInfo": {"name": "build_active5", "version": "1.0"},
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
        # Split into chunks of <= 120 calls to stay well below 200 limit
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
        # 1. Song settings: 180 BPM, 4/4, swing 0.06
        mcp.call("set_song", {
            "title": "Stoicville Ignition",
            "genre": "Dark Trap / Rap",
            "author": "AI-Studio",
            "tempo": 180,
            "swing": 0.06,
        })

        # 2. Ensure mixer tracks 1..21 exist (template has 1..16 + Master 0)
        mix_info = mcp.call("get_mixer", {})
        existing_mix = {t["mixer"] for t in mix_info["tracks"]}
        calls = []
        for m in range(1, 22):
            if m not in existing_mix:
                calls.append({"tool": "add_mixer_track", "arguments": {}})
        if calls:
            mcp.batch(calls)

        # Rename mixer tracks & set routing
        # Inserts 1..14 -> Buses 17 (Drums), 18 (808), 19 (Music), 21 (Vocal)
        # Bus 20 = FX Room -> Master 0
        mixer_defs = [
            (1, "Kick", 17, 0.80, 0.0, 0.0),
            (2, "808 Apollo", 18, 0.80, 0.0, -0.8),
            (3, "808 Blxckout", 18, 0.78, 0.0, -0.8),
            (4, "Snare", 17, 0.78, 0.0, 0.0),
            (5, "Clap", 17, 0.76, 0.0, 0.0),
            (6, "Hats Trap", 17, 0.74, 0.0, 0.0),
            (7, "Top Loops & OpenHat", 17, 0.64, 0.0, 0.0),
            (8, "Techno Perc", 17, 0.60, 0.0, 0.0),
            (9, "Upright Pizz", 19, 0.74, -0.15, 0.0),
            (10, "Headroom Piano", 19, 0.72, 0.16, -0.15),
            (11, "Chords E", 19, 0.66, 0.0, -0.25),
            (12, "Chords High & Riser", 19, 0.62, 0.0, -0.20),
            (13, "Stoicville Vocals", 21, 0.86, 0.0, -0.20),
            (14, "Vocal FX & Chants", 21, 0.72, 0.0, -0.25),
            (17, "BUS Drums", 0, 0.88, 0.0, 0.0),
            (18, "BUS 808", 0, 0.71, 0.0, -1.0),
            (19, "BUS Music", 0, 0.74, 0.0, -0.18),
            (20, "FX Room", 0, 0.72, 0.0, 0.0),
            (21, "BUS Vocal", 0, 0.88, 0.0, -0.15),
        ]
        calls = []
        stereo_changes = []
        for m, name, route, vol, pan, stereo in mixer_defs:
            calls.append({
                "tool": "edit_mixer_track",
                "arguments": {"mixer": m, "name": name, "routeTo": route, "volume": vol, "pan": pan},
            })
            if stereo != 0.0:
                stereo_changes.append({"path": f"mixer/{m}/stereo", "value": stereo})
        mcp.batch(calls)
        if stereo_changes:
            mcp.call("set_parameters", {"changes": stereo_changes})

        # Sidechain Kick (1) -> BUS 808 (18) + Reverb sends -> FX Room (20)
        sends = [
            (1, 18, 1.0, True),
            (4, 20, 0.04, False),
            (5, 20, 0.04, False),
            (9, 20, 0.08, False),
            (10, 20, 0.09, False),
            (11, 20, 0.07, False),
            (12, 20, 0.07, False),
            (13, 20, 0.07, False),
            (14, 20, 0.10, False),
        ]
        mcp.batch([
            {"tool": "set_send", "arguments": {"mixer": m, "to": to, "level": lvl, "sidechain": sc}}
            for m, to, lvl, sc in sends
        ])

        # 3. Add all channels (1..27) strictly from projects/active5 sources & derived samples
        # Keep index mapping explicit
        channels_spec = [
            # (name, plugin, mixer, sample_path, vol, pan)
            ("Kick Native", "gl.kick", 1, None, 0.72, 0.0),
            ("Techno Kick OneShot", "gl.sampler", 1, f"{SAMPLES_DIR}/techno_kick_oneshot.wav", 0.72, 0.0),
            ("MZ 808 [Apollo]", "gl.sampler", 2, f"{ROOT}/soundbanks/JMizzy The Best Drum Kits/!Mike Zara's Stash/808s/MZ 808 [Apollo].wav", 0.82, 0.0),
            ("MZ 808 [Blxckout]", "gl.sampler", 3, f"{ROOT}/soundbanks/JMizzy The Best Drum Kits/!Mike Zara's Stash/808s/MZ 808 [Blxckout].wav", 0.80, 0.0),
            ("Snare Native", "gl.snare", 4, None, 0.74, 0.0),
            ("Clap Native", "gl.clap", 5, None, 0.68, 0.0),
            ("Techno Clap OneShot", "gl.sampler", 5, f"{SAMPLES_DIR}/techno_clap_oneshot.wav", 0.70, -0.06),
            ("Hat Native", "gl.hat", 6, None, 0.64, 0.08),
            ("Techno Hat OneShot", "gl.sampler", 6, f"{SAMPLES_DIR}/techno_hat_oneshot.wav", 0.64, -0.14),
            ("Techno OpenHat OneShot", "gl.sampler", 7, f"{SAMPLES_DIR}/techno_openhat_oneshot.wav", 0.58, 0.16),
            ("Techno Hats 180BPM", "gl.sampler", 7, f"{SAMPLES_DIR}/techno_hats_180bpm.wav", 0.56, -0.10),
            ("Techno Key Hats 180BPM", "gl.sampler", 7, f"{SAMPLES_DIR}/techno_key_hats_180bpm.wav", 0.44, 0.12),
            ("Techno Perc 180BPM", "gl.sampler", 8, f"{SAMPLES_DIR}/techno_perc_180bpm.wav", 0.58, 0.14),
            ("Upright Piano Pizz E", "gl.sampler", 9, f"{ROOT}/soundbanks/155852__day_tripper13__upright-piano-pizz-e-dull-2.wav", 0.76, -0.16),
            ("Headroom Piano High", "gl.sampler", 10, f"{ROOT}/soundbanks/sfz/headroom-piano/Samples/HEADROOM PIANO LEVEL5 CLOSE 93.flac", 0.74, 0.20),
            ("Chords E Filtered", "gl.sampler", 11, f"{SAMPLES_DIR}/techno_chords_e_180bpm.wav", 0.44, 0.0),
            ("Chords E Open", "gl.sampler", 11, f"{SAMPLES_DIR}/techno_chords_e_180bpm.wav", 0.72, 0.0),
            ("Chords E Chopped", "gl.sampler", 11, f"{SAMPLES_DIR}/techno_chords_e_chop.wav", 0.70, 0.0),
            ("Chords E High +12", "gl.sampler", 12, f"{SAMPLES_DIR}/techno_chords_e_high12.wav", 0.58, 0.0),
            ("Chords E Riser", "gl.sampler", 12, f"{SAMPLES_DIR}/techno_chords_e_rev.wav", 0.74, 0.0),
            ("Stoicville Intro", "gl.sampler", 13, f"{SAMPLES_DIR}/stoicville_intro.wav", 0.62, 0.0),
            ("Stoicville Hook", "gl.sampler", 13, f"{SAMPLES_DIR}/stoicville_hook.wav", 0.86, 0.0),
            ("Stoicville Verse 1", "gl.sampler", 13, f"{SAMPLES_DIR}/stoicville_verse1.wav", 0.82, 0.0),
            ("Stoicville Verse 2", "gl.sampler", 13, f"{SAMPLES_DIR}/stoicville_verse2.wav", 0.82, 0.0),
            ("Stoicville Outro", "gl.sampler", 13, f"{SAMPLES_DIR}/stoicville_outro.wav", 0.60, 0.0),
            ("(SXJ) Chants (6)", "gl.sampler", 14, f"{ROOT}/soundbanks/JMizzy The Best Drum Kits/JUGG DRUM KIT/Chants (Voxs)/(SXJ) Chants (6).wav", 0.68, 0.22),
            ("Bankroll Alright FX", "gl.sampler", 14, f"{SAMPLES_DIR}/bankroll_alright_clean.wav", 0.74, -0.20),
        ]

        mcp.batch([
            {"tool": "add_channel", "arguments": {"name": name, "plugin": plugin, "mixer": mixer, "volume": vol, "pan": pan}}
            for name, plugin, mixer, _, vol, pan in channels_spec
        ])

        # Load sample paths via patch
        patch_calls = []
        for idx, (_, plugin, _, spath, _, _) in enumerate(channels_spec, 1):
            if spath:
                patch_calls.append({
                    "tool": "patch",
                    "arguments": {"path": f"channel/{idx}", "set": {"plugin": {"data": spath}}},
                })
        mcp.batch(patch_calls)

        # 4. Configure channel envelopes, tuning, and ADSR per Protocol Section 2
        ch_params = [
            # Ch 1 Kick Native: short 130ms punch, end pitch 54Hz
            ("channel/1/Start pitch", 215),
            ("channel/1/End pitch", 54),
            ("channel/1/Sweep", 0.045),
            ("channel/1/Decay", 0.13),
            ("channel/1/Click", 0.36),
            ("channel/1/Drive", 0.22),
            # Ch 2 Techno Kick OneShot: Root 60, One shot = 0, Release 0.02s, Declick = 2 (Transient)
            ("channel/2/One shot", 0),
            ("channel/2/Release", 0.02),
            ("channel/2/Declick", 2),
            ("channel/2/Remove DC", 1),
            # Ch 3 MZ 808 [Apollo]: Root 24 (C1), Fine -14ct, Voices=1 (Cut Itself!), One shot=0, Release 0.045s, Glide 0.055s
            ("channel/3/Root note", 24),
            ("channel/3/Fine tune", -14),
            ("channel/3/Voices", 1),
            ("channel/3/One shot", 0),
            ("channel/3/Attack", 0.002),
            ("channel/3/Decay", 10.0),
            ("channel/3/Sustain", 1.0),
            ("channel/3/Release", 0.045),
            ("channel/3/Glide", 0.055),
            ("channel/3/Remove DC", 1),
            ("channel/3/Declick", 2),
            # Ch 4 MZ 808 [Blxckout]: Root 24 (C1), Fine +6ct, Voices=1 (Cut Itself!), One shot=0, Release 0.045s, Glide 0.055s
            ("channel/4/Root note", 24),
            ("channel/4/Fine tune", 6),
            ("channel/4/Voices", 1),
            ("channel/4/One shot", 0),
            ("channel/4/Attack", 0.002),
            ("channel/4/Decay", 10.0),
            ("channel/4/Sustain", 1.0),
            ("channel/4/Release", 0.045),
            ("channel/4/Glide", 0.055),
            ("channel/4/Remove DC", 1),
            ("channel/4/Declick", 2),
            # Ch 5 Snare Native: crisp 110ms body, 160ms noise
            ("channel/5/Tone", 0.58),
            ("channel/5/Snappy", 0.74),
            ("channel/5/Body decay", 0.11),
            ("channel/5/Noise decay", 0.16),
            ("channel/5/Tune", 198),
            # Ch 6 Clap Native: 180ms decay, tight snap
            ("channel/6/Decay", 0.18),
            ("channel/6/Tone", 1800),
            ("channel/6/Hits", 4),
            ("channel/6/Spread", 0.011),
            # Ch 7 Techno Clap OneShot
            ("channel/7/One shot", 0),
            ("channel/7/Release", 0.025),
            ("channel/7/Declick", 2),
            # Ch 8 Hat Native: crisp 45ms decay (< 1/16 = 83.3ms for micro-silences!)
            ("channel/8/Decay", 0.045),
            ("channel/8/Tone", 9200),
            ("channel/8/Metal", 0.22),
            ("channel/8/Choke", 1),
            # Ch 9 Techno Hat OneShot: crisp 65ms sample, One shot=0, Release=0.015s
            ("channel/9/One shot", 0),
            ("channel/9/Release", 0.015),
            ("channel/9/Declick", 2),
            # Ch 10 Techno OpenHat OneShot
            ("channel/10/One shot", 0),
            ("channel/10/Release", 0.03),
            ("channel/10/Declick", 2),
            # Ch 14 Upright Piano Pizz E: Root 52 (E3), Fine +39ct
            ("channel/14/Root note", 52),
            ("channel/14/Fine tune", 39),
            ("channel/14/One shot", 0),
            ("channel/14/Release", 0.18),
            ("channel/14/Declick", 2),
            # Ch 15 Headroom Piano High: Root 93 (A5), Fine -15ct
            ("channel/15/Root note", 93),
            ("channel/15/Fine tune", -15),
            ("channel/15/One shot", 0),
            ("channel/15/Release", 0.22),
            ("channel/15/Declick", 2),
            # Ch 16 Chords E Filtered: Low pass @ 1150 Hz
            ("channel/16/Filter", 1),
            ("channel/16/Cutoff", 1150),
            ("channel/16/Resonance", 0.08),
        ]
        # All long loops & vocal segments (10..13, 16..27): Root=64 (E4), One shot=0, Release=0.035s, Declick=2
        for ch_idx in [10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]:
            ch_params.extend([
                (f"channel/{ch_idx}/Root note", 64),
                (f"channel/{ch_idx}/One shot", 0),
                (f"channel/{ch_idx}/Attack", 0.004),
                (f"channel/{ch_idx}/Decay", 10.0),
                (f"channel/{ch_idx}/Sustain", 1.0),
                (f"channel/{ch_idx}/Release", 0.035),
                (f"channel/{ch_idx}/Declick", 2),
                (f"channel/{ch_idx}/Remove DC", 1),
            ])

        mcp.call("set_parameters", {
            "changes": [{"path": p, "value": v} for p, v in ch_params]
        })

        # 5. Add Mixer DSP Effects (keeping Quality=0 on inserts, Quality=1 on Master only, 1 shared FX Room)
        fx_adds = [
            # Master (0): EQ (slot 1), Comp (slot 2), Limiter (slot 3)
            {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.comp"}},
            {"tool": "add_plugin", "arguments": {"mixer": 0, "plugin": "gl.limiter"}},
            # BUS Drums (17): EQ (slot 1), Comp (slot 2)
            {"tool": "add_plugin", "arguments": {"mixer": 17, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 17, "plugin": "gl.comp"}},
            # BUS 808 (18): Stereo Width (slot 1), Comp Sidechain (slot 2), EQ (slot 3)
            {"tool": "add_plugin", "arguments": {"mixer": 18, "plugin": "gl.stereo"}},
            {"tool": "add_plugin", "arguments": {"mixer": 18, "plugin": "gl.comp"}},
            {"tool": "add_plugin", "arguments": {"mixer": 18, "plugin": "gl.eq"}},
            # BUS Music (19): EQ (slot 1)
            {"tool": "add_plugin", "arguments": {"mixer": 19, "plugin": "gl.eq"}},
            # FX Room (20): gl.reverb (slot 1)
            {"tool": "add_plugin", "arguments": {"mixer": 20, "plugin": "gl.reverb"}},
            # BUS Vocal (21): EQ (slot 1), Comp (slot 2)
            {"tool": "add_plugin", "arguments": {"mixer": 21, "plugin": "gl.eq"}},
            {"tool": "add_plugin", "arguments": {"mixer": 21, "plugin": "gl.comp"}},
            # Individual EQs only where needed:
            # Top Loops (7): HPF 4000 Hz
            {"tool": "add_plugin", "arguments": {"mixer": 7, "plugin": "gl.eq"}},
            # Techno Perc (8): HPF 1600 Hz
            {"tool": "add_plugin", "arguments": {"mixer": 8, "plugin": "gl.eq"}},
            # Upright Pizz (9): HPF 165 Hz
            {"tool": "add_plugin", "arguments": {"mixer": 9, "plugin": "gl.eq"}},
            # Headroom Piano (10): HPF 320 Hz
            {"tool": "add_plugin", "arguments": {"mixer": 10, "plugin": "gl.eq"}},
            # Chords E (11): HPF 165 Hz + dip 310 Hz + vocal pocket dip 1850 Hz
            {"tool": "add_plugin", "arguments": {"mixer": 11, "plugin": "gl.eq"}},
            # Chords High (12): HPF 450 Hz
            {"tool": "add_plugin", "arguments": {"mixer": 12, "plugin": "gl.eq"}},
        ]
        # Note: template Master (0) may already have a Limiter on slot 10; check and remove if present
        m0 = mcp.call("get_mixer", {"mixer": 0}) if False else mcp.call("get_mixer", {})
        for tr in m0["tracks"]:
            if tr["mixer"] == 0:
                for fx in tr["effects"]:
                    mcp.call("remove_plugin", {"mixer": 0, "slot": fx["slot"]})
        mcp.batch(fx_adds)

        # Configure all FX parameters precisely
        fx_changes = [
            # Master (0) slot 1: gl.eq (Quality=1, HPF 28Hz, LPF 17.5kHz)
            ("mixer/0/slot/1/Quality", 1),
            ("mixer/0/slot/1/Band 1 type", 1), ("mixer/0/slot/1/Band 1 freq", 28),
            ("mixer/0/slot/1/Band 7 type", 3), ("mixer/0/slot/1/Band 7 freq", 17500),
            # Master (0) slot 2: gl.comp glue (threshold -10dB so Intro/Pont/Outro keep natural dynamics)
            ("mixer/0/slot/2/Threshold", -10), ("mixer/0/slot/2/Ratio", 1.8),
            ("mixer/0/slot/2/Attack", 0.020), ("mixer/0/slot/2/Release", 0.12),
            ("mixer/0/slot/2/Makeup", 1.0), ("mixer/0/slot/2/Detector high-pass", 95),
            # Master (0) slot 3: gl.limiter (Ceiling -2.9 dB, Gain +5.5 dB -> ~ -11.0 LUFS, <= -1.05 dBTP)
            ("mixer/0/slot/3/Gain", 5.5), ("mixer/0/slot/3/Ceiling", -2.9),
            ("mixer/0/slot/3/Attack (look-ahead)", 0.002), ("mixer/0/slot/3/0.2 dB safety", 1),

            # BUS Drums (17) slot 1: gl.eq (HPF 32Hz, tame 63Hz -2.5dB, Air +1.8dB @ 10.5kHz)
            ("mixer/17/slot/1/Band 1 type", 1), ("mixer/17/slot/1/Band 1 freq", 32),
            ("mixer/17/slot/1/Band 2 type", 6), ("mixer/17/slot/1/Band 2 freq", 63), ("mixer/17/slot/1/Band 2 gain", -2.5), ("mixer/17/slot/1/Band 2 Q (legacy)", 1.4),
            ("mixer/17/slot/1/Band 7 type", 7), ("mixer/17/slot/1/Band 7 freq", 10500), ("mixer/17/slot/1/Band 7 gain", 1.8),
            # BUS Drums (17) slot 2: gl.comp punch
            ("mixer/17/slot/2/Threshold", -15), ("mixer/17/slot/2/Ratio", 2.6),
            ("mixer/17/slot/2/Attack", 0.012), ("mixer/17/slot/2/Release", 0.09), ("mixer/17/slot/2/Makeup", 1.8),

            # BUS 808 (18) slot 1: gl.stereo strict mono
            ("mixer/18/slot/1/Width", 0.0),
            # BUS 808 (18) slot 2: gl.comp Sidechain from Kick
            ("mixer/18/slot/2/Detector input", 1),  # Sidechain
            ("mixer/18/slot/2/Threshold", -22), ("mixer/18/slot/2/Ratio", 5.5),
            ("mixer/18/slot/2/Attack", 0.002), ("mixer/18/slot/2/Release", 0.065),
            ("mixer/18/slot/2/Lookahead", 2),
            # BUS 808 (18) slot 3: gl.eq (HPF 28Hz, tame 63Hz resonance -4.8dB, warmth +1.4dB @ 115Hz)
            ("mixer/18/slot/3/Band 1 type", 1), ("mixer/18/slot/3/Band 1 freq", 28),
            ("mixer/18/slot/3/Band 2 type", 6), ("mixer/18/slot/3/Band 2 freq", 63), ("mixer/18/slot/3/Band 2 gain", -5.6), ("mixer/18/slot/3/Band 2 Q (legacy)", 1.6),
            ("mixer/18/slot/3/Band 3 type", 6), ("mixer/18/slot/3/Band 3 freq", 115), ("mixer/18/slot/3/Band 3 gain", 1.4), ("mixer/18/slot/3/Band 3 Q (legacy)", 1.2),

            # BUS Music (19) slot 1: gl.eq (HPF 145Hz, vocal pocket -2.2dB @ 1850Hz)
            ("mixer/19/slot/1/Band 1 type", 1), ("mixer/19/slot/1/Band 1 freq", 145),
            ("mixer/19/slot/1/Band 5 type", 6), ("mixer/19/slot/1/Band 5 freq", 1850), ("mixer/19/slot/1/Band 5 gain", -2.2), ("mixer/19/slot/1/Band 5 Q (legacy)", 1.1),

            # FX Room (20) slot 1: gl.reverb (Plate, 0.333s = 1 * T_noire @ 180 BPM, Dry=0, HPF=480Hz)
            ("mixer/20/slot/1/Model", 1),
            ("mixer/20/slot/1/Pre-delay", 12),
            ("mixer/20/slot/1/Room size", 0.25),
            ("mixer/20/slot/1/Decay time", 0.333),
            ("mixer/20/slot/1/Low cut", 480),
            ("mixer/20/slot/1/High cut", 6000),
            ("mixer/20/slot/1/High damping", 3800),
            ("mixer/20/slot/1/Dry", 0),
            ("mixer/20/slot/1/Wet", 0.62),
            ("mixer/20/slot/1/Width", 0.80),
            ("mixer/20/slot/1/Low cut (full)", 480),
            ("mixer/20/slot/1/Decay (full)", 3),
            ("mixer/20/slot/1/Dry (full)", 0),
            ("mixer/20/slot/1/Early reflections (full)", 10),
            ("mixer/20/slot/1/Wet (full)", 62),

            # BUS Vocal (21) slot 1: gl.eq (HPF 145Hz, presence +1.8dB @ 2700Hz, tame nasal -1.5dB @ 500Hz)
            ("mixer/21/slot/1/Band 1 type", 1), ("mixer/21/slot/1/Band 1 freq", 145),
            ("mixer/21/slot/1/Band 4 type", 6), ("mixer/21/slot/1/Band 4 freq", 500), ("mixer/21/slot/1/Band 4 gain", -1.5), ("mixer/21/slot/1/Band 4 Q (legacy)", 1.3),
            ("mixer/21/slot/1/Band 6 type", 6), ("mixer/21/slot/1/Band 6 freq", 2700), ("mixer/21/slot/1/Band 6 gain", 1.8), ("mixer/21/slot/1/Band 6 Q (legacy)", 1.1),
            # BUS Vocal (21) slot 2: gl.comp
            ("mixer/21/slot/2/Threshold", -18), ("mixer/21/slot/2/Ratio", 3.2),
            ("mixer/21/slot/2/Attack", 0.008), ("mixer/21/slot/2/Release", 0.09), ("mixer/21/slot/2/Makeup", 2.2),

            # Top Loops (7) slot 1: HPF 4000Hz
            ("mixer/7/slot/1/Band 1 type", 1), ("mixer/7/slot/1/Band 1 freq", 4000),
            # Techno Perc (8) slot 1: HPF 1600Hz
            ("mixer/8/slot/1/Band 1 type", 1), ("mixer/8/slot/1/Band 1 freq", 1600),
            # Upright Pizz (9) slot 1: HPF 165Hz, dip -2.0dB @ 300Hz
            ("mixer/9/slot/1/Band 1 type", 1), ("mixer/9/slot/1/Band 1 freq", 165),
            ("mixer/9/slot/1/Band 3 type", 6), ("mixer/9/slot/1/Band 3 freq", 300), ("mixer/9/slot/1/Band 3 gain", -2.0),
            # Headroom Piano (10) slot 1: HPF 320Hz
            ("mixer/10/slot/1/Band 1 type", 1), ("mixer/10/slot/1/Band 1 freq", 320),
            # Chords E (11) slot 1: HPF 165Hz, dip -2.5dB @ 310Hz, vocal dip -2.8dB @ 1800Hz
            ("mixer/11/slot/1/Band 1 type", 1), ("mixer/11/slot/1/Band 1 freq", 165),
            ("mixer/11/slot/1/Band 3 type", 6), ("mixer/11/slot/1/Band 3 freq", 310), ("mixer/11/slot/1/Band 3 gain", -2.5),
            ("mixer/11/slot/1/Band 5 type", 6), ("mixer/11/slot/1/Band 5 freq", 1800), ("mixer/11/slot/1/Band 5 gain", -2.8),
            # Chords High (12) slot 1: HPF 450Hz
            ("mixer/12/slot/1/Band 1 type", 1), ("mixer/12/slot/1/Band 1 freq", 450),
        ]
        mcp.call("set_parameters", {
            "changes": [{"path": p, "value": v} for p, v in fx_changes]
        })

        # 6. Name Playlist Tracks 1..16 cleanly (1 instrument/version per track)
        track_names = [
            (1, "Stoicville Vocals"),
            (2, "Vocal FX & Chants"),
            (3, "808 Apollo (Couplets)"),
            (4, "808 Blxckout (Refrains)"),
            (5, "Kick (Layered)"),
            (6, "Snare & Clap"),
            (7, "Hi-Hats Trap"),
            (8, "Top Loops & OpenHat"),
            (9, "Techno Perc 180BPM"),
            (10, "Upright Piano Pizz E"),
            (11, "Headroom Piano High (+12)"),
            (12, "Chords E (Filtered)"),
            (13, "Chords E (Open)"),
            (14, "Chords E (Chopped)"),
            (15, "Chords High (+12) & Riser"),
            (16, "FX Room Transition Cuts"),
        ]
        mcp.batch([
            {"tool": "patch", "arguments": {"path": f"track/{t}", "set": {"name": name}}}
            for t, name in track_names
        ])

        # 7. Helper to queue patterns, notes, and clips into batches
        pat_counter = 1
        queued_calls = [
            {"tool": "edit_pattern", "arguments": {"pattern": 1, "name": "Intro - Stoicville", "bars": 8}}
        ]

        def new_pattern(name: str, bars: int) -> int:
            nonlocal pat_counter
            pat_counter += 1
            queued_calls.append({"tool": "add_pattern", "arguments": {"name": name, "bars": bars}})
            return pat_counter

        def add_notes(pat: int, ch: int, notes: list):
            queued_calls.append({"tool": "set_notes", "arguments": {"pattern": pat, "channel": ch, "notes": notes}})

        def place(pat: int, track: int, start_bar: int, length_bars: int = None):
            args = {"pattern": pat, "track": track, "start": (start_bar - 1) * 4.0}
            if length_bars is not None:
                args["length"] = length_bars * 4.0
            queued_calls.append({"tool": "add_clip", "arguments": args})

        # ---- VOCALS ----
        # Ch 21 Intro (8 bars), Ch 22 Hook (8 bars), Ch 23 Verse 1 (16 bars), Ch 24 Verse 2 (16 bars), Ch 25 Outro (8 bars)
        add_notes(1, 21, [{"start": 0.0, "length": 31.8, "pitch": 64, "velocity": 86}])
        place(1, 1, 1)

        p_hook = new_pattern("Hook - Stoicville", 8)
        add_notes(p_hook, 22, [{"start": 0.0, "length": 31.0, "pitch": 64, "velocity": 112}])
        place(p_hook, 1, 9)
        place(p_hook, 1, 41)
        place(p_hook, 1, 49)
        place(p_hook, 1, 73)
        place(p_hook, 1, 81)

        p_v1 = new_pattern("Verse 1 - Stoicville", 16)
        add_notes(p_v1, 23, [{"start": 0.0, "length": 63.0, "pitch": 64, "velocity": 106}])
        place(p_v1, 1, 17)

        p_pont_vox = new_pattern("Pont - Stoicville Chop", 8)
        add_notes(p_pont_vox, 21, [{"start": 0.0, "length": 15.5, "pitch": 64, "velocity": 84}])
        place(p_pont_vox, 1, 33)

        p_v2 = new_pattern("Verse 2 - Stoicville", 16)
        add_notes(p_v2, 24, [{"start": 0.0, "length": 63.0, "pitch": 64, "velocity": 106}])
        place(p_v2, 1, 57)

        p_outro_vox = new_pattern("Outro - Stoicville", 8)
        add_notes(p_outro_vox, 25, [{"start": 0.0, "length": 31.5, "pitch": 64, "velocity": 84}])
        place(p_outro_vox, 1, 89)

        # ---- VOCAL FX & CHANTS (Track 2) ----
        # Ch 26 (SXJ) Chants (6) & Ch 27 Bankroll Alright FX
        p_alright = new_pattern("FX - Bankroll Alright", 4)
        add_notes(p_alright, 27, [{"start": 0.0, "length": 10.0, "pitch": 64, "velocity": 104}])
        place(p_alright, 2, 7)    # Intro pivot into Drop 1
        place(p_alright, 2, 39)   # Pont pivot into Drop 2
        place(p_alright, 2, 71)   # Couplet 2 pivot into Drop 3
        place(p_alright, 2, 93)   # Outro resolution

        p_chants = new_pattern("Chants - SXJ (8 bars)", 8)
        chant_notes = []
        for b in [0, 2, 4, 6]:
            chant_notes.append({"start": b * 4.0 + 3.5, "length": 0.75, "pitch": 64, "velocity": 96})
        add_notes(p_chants, 26, chant_notes)
        place(p_chants, 2, 9)
        place(p_chants, 2, 41)
        place(p_chants, 2, 49)
        place(p_chants, 2, 73)
        place(p_chants, 2, 81)

        # ---- CHORDS E LAYERS & RISER (Tracks 12, 13, 14, 15) ----
        # Ch 16 Chords Filtered, Ch 17 Chords Open, Ch 18 Chords Chopped, Ch 19 Chords High +12, Ch 20 Riser
        p_chords_filt_intro = new_pattern("Chords E Filtered - Intro", 8)
        add_notes(p_chords_filt_intro, 16, [{"start": 0.0, "length": 31.0, "pitch": 64, "velocity": 84}])
        place(p_chords_filt_intro, 12, 1)

        p_chords_filt_pont = new_pattern("Chords E Filtered - Pont", 8)
        add_notes(p_chords_filt_pont, 16, [{"start": 0.0, "length": 30.0, "pitch": 64, "velocity": 86}])
        place(p_chords_filt_pont, 12, 33)

        p_chords_filt_outro = new_pattern("Chords E Filtered - Outro", 8)
        add_notes(p_chords_filt_outro, 16, [{"start": 0.0, "length": 31.5, "pitch": 64, "velocity": 78}])
        place(p_chords_filt_outro, 12, 89)

        p_chords_open8 = new_pattern("Chords E Open - 8 bars", 8)
        add_notes(p_chords_open8, 17, [{"start": 0.0, "length": 31.0, "pitch": 64, "velocity": 102}])
        place(p_chords_open8, 13, 9)

        p_chords_open16 = new_pattern("Chords E Open - 16 bars", 16)
        add_notes(p_chords_open16, 17, [{"start": 0.0, "length": 63.0, "pitch": 64, "velocity": 106}])
        place(p_chords_open16, 13, 41)
        place(p_chords_open16, 13, 73)

        p_chords_chop = new_pattern("Chords E Chopped - Couplet 2", 16)
        add_notes(p_chords_chop, 18, [{"start": 0.0, "length": 63.0, "pitch": 64, "velocity": 96}])
        place(p_chords_chop, 14, 57)

        p_chords_hi8 = new_pattern("Chords E High +12 - 8 bars", 8)
        add_notes(p_chords_hi8, 19, [{"start": 0.0, "length": 31.0, "pitch": 64, "velocity": 96}])
        place(p_chords_hi8, 15, 49)
        place(p_chords_hi8, 15, 73)
        place(p_chords_hi8, 15, 81)

        p_riser = new_pattern("Riser - Chords Reverse", 1)
        add_notes(p_riser, 20, [{"start": 0.0, "length": 4.0, "pitch": 64, "velocity": 104}])
        place(p_riser, 15, 8)
        place(p_riser, 15, 40)
        place(p_riser, 15, 72)

        # ---- UPRIGHT PIANO PIZZ E (Track 10, Ch 14, Root=52 E3) ----
        # Question/Response motif in E minor (E3=52, G3=55, A3=57, B3=59, C4=60, F#3=54)
        def make_pizz_notes(bars=8, mute_last_beat=True):
            notes = []
            # 2-bar antecedent/consequent motif
            motif_2bar = [
                (0.0, 0.75, 52, 92), (1.5, 0.75, 55, 86), (2.5, 0.75, 59, 90), (3.5, 0.5, 57, 82),
                (4.0, 0.75, 60, 94), (5.5, 0.75, 59, 88), (6.5, 0.75, 55, 84), (7.5, 0.5, 54, 80),
            ]
            for rep in range(bars // 2):
                off = rep * 8.0
                for st, ln, pt, vel in motif_2bar:
                    t = off + st
                    if mute_last_beat and t >= (bars * 4.0 - 1.0):
                        continue
                    jitter_t = max(0.0, t + random.uniform(-0.015, 0.015))
                    jitter_v = max(55, min(115, vel + random.randint(-6, 6)))
                    notes.append({"start": round(jitter_t, 4), "length": ln, "pitch": pt, "velocity": jitter_v})
            return notes

        p_pizz_intro = new_pattern("Upright Pizz - Intro", 8)
        pizz_intro_notes = make_pizz_notes(8, mute_last_beat=True)
        for n in pizz_intro_notes:
            n["velocity"] = max(45, n["velocity"] - 18)
        add_notes(p_pizz_intro, 14, pizz_intro_notes)
        place(p_pizz_intro, 10, 1)

        p_pizz_c1b = new_pattern("Upright Pizz - Couplet B", 8)
        add_notes(p_pizz_c1b, 14, make_pizz_notes(8, mute_last_beat=True))
        place(p_pizz_c1b, 10, 25)
        place(p_pizz_c1b, 10, 65)

        p_pizz_outro = new_pattern("Upright Pizz - Outro", 8)
        outro_pizz = [
            {"start": 0.0, "length": 1.2, "pitch": 52, "velocity": 86},
            {"start": 4.0, "length": 1.2, "pitch": 55, "velocity": 80},
            {"start": 8.0, "length": 1.5, "pitch": 59, "velocity": 78},
            {"start": 16.0, "length": 2.0, "pitch": 52, "velocity": 74},
            {"start": 24.0, "length": 2.5, "pitch": 52, "velocity": 68},
        ]
        add_notes(p_pizz_outro, 14, outro_pizz)
        place(p_pizz_outro, 10, 89)

        # ---- HEADROOM PIANO HIGH (Track 11, Ch 15, Root=93 A5, max_pitch<=84) ----
        # Staccato counter-melody in E minor: E5=76, F#5=78, G5=79, A5=81, B5=83, C6=84
        def make_headroom_notes(bars=8, mute_last_beat=True):
            notes = []
            # Sparse rhythmic bell-like piano hits leaving vocal pocket free
            motif_4bar = [
                (0.0, 0.5, 83, 88), (1.5, 0.5, 79, 82), (3.0, 0.5, 81, 84),
                (4.0, 0.5, 76, 90), (6.0, 0.5, 79, 80), (7.5, 0.4, 78, 76),
                (8.0, 0.5, 84, 92), (9.5, 0.5, 83, 85), (11.0, 0.5, 79, 82),
                (12.0, 0.5, 83, 88), (14.5, 0.5, 76, 84),
            ]
            for rep in range(bars // 4):
                off = rep * 16.0
                for st, ln, pt, vel in motif_4bar:
                    t = off + st
                    if mute_last_beat and t >= (bars * 4.0 - 1.0):
                        continue
                    jitter_t = max(0.0, t + random.uniform(-0.012, 0.012))
                    jitter_v = max(55, min(115, vel + random.randint(-6, 6)))
                    notes.append({"start": round(jitter_t, 4), "length": ln, "pitch": pt, "velocity": jitter_v})
            return notes

        p_hr_drop = new_pattern("Headroom High - Drop 8m", 8)
        add_notes(p_hr_drop, 15, make_headroom_notes(8, mute_last_beat=True))
        place(p_hr_drop, 11, 9)
        place(p_hr_drop, 11, 41)
        place(p_hr_drop, 11, 49)
        place(p_hr_drop, 11, 73)
        place(p_hr_drop, 11, 81)

        p_hr_pont = new_pattern("Headroom High - Pont 8m", 8)
        hr_pont_notes = make_headroom_notes(8, mute_last_beat=True)
        for n in hr_pont_notes:
            n["velocity"] = max(45, n["velocity"] - 16)
        add_notes(p_hr_pont, 15, hr_pont_notes)
        place(p_hr_pont, 11, 33)

        p_hr_outro = new_pattern("Headroom High - Outro", 8)
        add_notes(p_hr_outro, 15, [
            {"start": 0.0, "length": 0.6, "pitch": 83, "velocity": 80},
            {"start": 4.0, "length": 0.6, "pitch": 79, "velocity": 76},
            {"start": 8.0, "length": 0.8, "pitch": 76, "velocity": 72},
            {"start": 16.0, "length": 1.0, "pitch": 76, "velocity": 66},
        ])
        place(p_hr_outro, 11, 89)

        # ---- 808 PATTERNS (Tracks 3 & 4) ----
        # E minor fundamentals in golden 41-85 Hz range: E1=28 (41.2Hz), G1=31 (49Hz), A1=33 (55Hz), B1=35 (61.7Hz), C2=36 (65.4Hz), D2=38 (73.4Hz), E2=40 (82.4Hz)
        # Ch 4 = 808 Blxckout (Refrains, sustained + glides)
        def make_808_blxckout(climax=False):
            notes = []
            # 8-bar pattern (bars 1..8) with progression Em -> Em -> C -> G/B | Em -> Em -> C -> D/B -> mute at beat 31.0
            bar_specs = [
                # (bar_idx, [(beat, length, pitch, vel, porta)])
                (0, [(0.0, 1.5, 28, 112, False), (1.5, 1.25, 28, 106, False), (2.75, 1.25, 40, 104, True)]),
                (1, [(0.0, 1.25, 28, 110, False), (1.75, 1.0, 35, 104, False), (3.5, 0.5, 40, 100, True)]),
                (2, [(0.0, 1.5, 36, 112, False), (1.5, 1.25, 36, 106, False), (2.75, 1.25, 48 if climax else 36, 102, True)]),
                (3, [(0.0, 1.5, 31, 110, False), (1.5, 1.25, 35, 106, False), (2.75, 1.0, 38, 102, True)]),
                (4, [(0.0, 1.5, 28, 114, False), (1.5, 1.25, 28, 106, False), (2.75, 1.25, 40, 106, True)]),
                (5, [(0.0, 1.25, 28, 110, False), (1.75, 1.0, 31, 104, False), (3.25, 0.75, 35, 102, True)]),
                (6, [(0.0, 1.5, 36, 112, False), (1.5, 1.25, 36, 106, False), (2.75, 1.25, 38, 104, True)]),
                # Bar 8: pivot mute on beat 4 (t >= 31.0 cut!)
                (7, [(0.0, 1.25, 35, 110, False), (1.5, 1.0, 38 if climax else 35, 104, False), (2.5, 0.45, 28, 100, True)]),
            ]
            for b, hits in bar_specs:
                for st, ln, pt, vel, porta in hits:
                    n = {"start": b * 4.0 + st, "length": ln, "pitch": pt, "velocity": vel}
                    if porta:
                        n["porta"] = True
                    notes.append(n)
            return notes

        p_808_ref1 = new_pattern("808 Blxckout - Refrain 1", 8)
        add_notes(p_808_ref1, 4, make_808_blxckout(climax=False))
        place(p_808_ref1, 4, 9)
        place(p_808_ref1, 4, 41)

        p_808_ref_climax = new_pattern("808 Blxckout - Climax", 8)
        add_notes(p_808_ref_climax, 4, make_808_blxckout(climax=True))
        place(p_808_ref_climax, 4, 49)
        place(p_808_ref_climax, 4, 73)
        place(p_808_ref_climax, 4, 81)

        # Ch 3 = 808 Apollo (Couplets, shorter Bounce lengths 0.5-1.0 beat so rap vocal breathes!)
        def make_808_apollo(syncopated=False):
            notes = []
            for b in range(8):
                root = 28 if b in (0, 1, 4, 5) else (36 if b in (2, 6) else 35)
                if b == 7:
                    # Pivot bar: cut on beat 4
                    hits = [(0.0, 0.75, root, 108, False), (1.5, 0.75, 31, 102, False)]
                elif syncopated:
                    hits = [(0.0, 0.75, root, 110, False), (1.5, 0.5, root + 12, 102, True), (2.75, 0.75, root, 104, False)]
                else:
                    hits = [(0.0, 0.85, root, 108, False), (1.5, 0.65, root, 100, False), (2.75, 0.65, 35 if root == 28 else root, 98, False)]
                for st, ln, pt, vel, porta in hits:
                    n = {"start": b * 4.0 + st, "length": ln, "pitch": pt, "velocity": vel}
                    if porta:
                        n["porta"] = True
                    notes.append(n)
            return notes

        p_808_c1 = new_pattern("808 Apollo - Couplet 1", 8)
        add_notes(p_808_c1, 3, make_808_apollo(syncopated=False))
        place(p_808_c1, 3, 17)
        place(p_808_c1, 3, 25)

        p_808_c2 = new_pattern("808 Apollo - Couplet 2", 8)
        add_notes(p_808_c2, 3, make_808_apollo(syncopated=True))
        place(p_808_c2, 3, 57)
        place(p_808_c2, 3, 65)

        # ---- KICK (Track 5, Ch 1 Native + Ch 2 Techno Kick OneShot) ----
        def make_kick_notes(bars=8, dense=True, mute_last_beat=True):
            notes = []
            for b in range(bars):
                if b == bars - 1 and mute_last_beat:
                    times = [0.0, 1.5]
                elif b % 2 == 0:
                    times = [0.0, 1.5, 2.75]
                else:
                    times = [0.0, 1.25, 2.75, 3.5] if dense else [0.0, 1.5]
                for t in times:
                    notes.append({"start": b * 4.0 + t, "length": 0.45, "pitch": 60, "velocity": 112 if t == 0.0 else 102})
            return notes

        p_kick_ref = new_pattern("Kick - Refrain 8m", 8)
        k_ref = make_kick_notes(8, dense=True, mute_last_beat=True)
        add_notes(p_kick_ref, 1, k_ref)
        add_notes(p_kick_ref, 2, k_ref)
        for sb in [9, 41, 49, 73, 81]:
            place(p_kick_ref, 5, sb)

        p_kick_verse = new_pattern("Kick - Couplet 8m", 8)
        k_ver = make_kick_notes(8, dense=False, mute_last_beat=True)
        add_notes(p_kick_verse, 1, k_ver)
        add_notes(p_kick_verse, 2, k_ver)
        for sb in [17, 25, 57, 65]:
            place(p_kick_verse, 5, sb)

        # ---- SNARE & CLAP (Track 6, Ch 5 Snare, Ch 6 Clap, Ch 7 Techno Clap OneShot) ----
        # Locked strictly on beat 3 (t = 2.0) of each bar!
        def make_snare_clap_notes(bars=8, fill_end=True, mute_last_beat=False):
            sn_notes, cl_notes = [], []
            for b in range(bars):
                # Beat 3 lock
                sn_notes.append({"start": b * 4.0 + 2.0, "length": 0.45, "pitch": 60, "velocity": 112})
                cl_notes.append({"start": b * 4.0 + 2.0, "length": 0.45, "pitch": 60, "velocity": 108})
                # Subtle ghost snare on bar 2, 4, 6
                if b in (1, 3, 5):
                    sn_notes.append({"start": b * 4.0 + 3.75, "length": 0.25, "pitch": 60, "velocity": 62})
                # Transition fill at end of bar 8 (or mute!)
                if b == bars - 1 and fill_end and not mute_last_beat:
                    for sub, vel in [(3.0, 74), (3.25, 82), (3.5, 92), (3.75, 104)]:
                        sn_notes.append({"start": b * 4.0 + sub, "length": 0.2, "pitch": 60, "velocity": vel})
            return sn_notes, cl_notes

        p_sc_fill = new_pattern("Snare/Clap - Drop Fill", 8)
        sn, cl = make_snare_clap_notes(8, fill_end=True, mute_last_beat=False)
        add_notes(p_sc_fill, 5, sn)
        add_notes(p_sc_fill, 6, cl)
        add_notes(p_sc_fill, 7, cl)
        for sb in [9, 41, 49, 73, 81]:
            place(p_sc_fill, 6, sb)

        p_sc_mute = new_pattern("Snare/Clap - Couplet Mute", 8)
        sn_m, cl_m = make_snare_clap_notes(8, fill_end=False, mute_last_beat=True)
        add_notes(p_sc_mute, 5, sn_m)
        add_notes(p_sc_mute, 6, cl_m)
        add_notes(p_sc_mute, 7, cl_m)
        for sb in [17, 25, 57, 65]:
            place(p_sc_mute, 6, sb)

        # Pont buildup snare (M37-40 crescendo)
        p_sc_pont = new_pattern("Snare - Pont Buildup", 8)
        pont_sn = []
        for b in range(4, 7):
            pont_sn.append({"start": b * 4.0 + 2.0, "length": 0.4, "pitch": 60, "velocity": 86 + (b - 4) * 8})
        for step in range(8):
            pont_sn.append({"start": 7 * 4.0 + step * 0.5, "length": 0.2, "pitch": 60, "velocity": 72 + step * 6})
        add_notes(p_sc_pont, 5, pont_sn)
        place(p_sc_pont, 6, 33)

        p_sc_outro = new_pattern("Snare/Clap - Outro 4m", 4)
        sn_o, cl_o = make_snare_clap_notes(4, fill_end=False, mute_last_beat=True)
        add_notes(p_sc_outro, 5, sn_o)
        add_notes(p_sc_outro, 6, cl_o)
        place(p_sc_outro, 6, 89)

        # ---- HI-HATS TRAP (Track 7, Ch 8 Hat Native + Ch 9 Techno Hat OneShot) ----
        def make_trap_hats(bars=8, energy="medium", mute_last_beat=True):
            h_nat, h_samp = [], []
            for b in range(bars):
                # Base 1/8 pulse
                for step in range(8):
                    t = b * 4.0 + step * 0.5
                    if mute_last_beat and t >= bars * 4.0 - 1.0:
                        continue
                    vel = 104 if step % 2 == 0 else 68
                    vel += random.randint(-5, 5)
                    # Skip base steps where rolls occur
                    if b % 2 == 1 and step == 3:  # pre-beat 3 triplet roll!
                        for k in range(3):
                            rt = b * 4.0 + 1.5 + k * (0.5 / 3.0)
                            h_nat.append({"start": round(rt, 4), "length": 0.12, "pitch": 60 - k * 2, "velocity": 82 + k * 8})
                        continue
                    if energy == "high" and b % 4 == 3 and step in (6, 7) and not (mute_last_beat and b == bars - 1):
                        # 1/32 pitch-down roll into next phrase
                        for k in range(4):
                            rt = b * 4.0 + 3.0 + k * 0.25
                            h_nat.append({"start": round(rt, 4), "length": 0.12, "pitch": 60 - k * 2, "velocity": 96 - k * 6})
                        continue
                    h_nat.append({"start": t, "length": 0.2, "pitch": 60, "velocity": vel})
                    # Syncopated Techno Hat OneShot accent on offbeats / high energy 1/16ths
                    if energy in ("medium", "high") and step in (1, 5, 7):
                        h_samp.append({"start": t + 0.25, "length": 0.18, "pitch": 60, "velocity": 76 if energy == "medium" else 90})
            return h_nat, h_samp

        p_hats_verse1 = new_pattern("Hats - Couplet 1a", 8)
        hn, hs = make_trap_hats(8, "low", mute_last_beat=True)
        add_notes(p_hats_verse1, 8, hn)
        place(p_hats_verse1, 7, 17)

        p_hats_verse_b = new_pattern("Hats - Couplet B", 8)
        hn, hs = make_trap_hats(8, "medium", mute_last_beat=True)
        add_notes(p_hats_verse_b, 8, hn)
        add_notes(p_hats_verse_b, 9, hs)
        place(p_hats_verse_b, 7, 25)
        place(p_hats_verse_b, 7, 57)
        place(p_hats_verse_b, 7, 65)

        p_hats_drop = new_pattern("Hats - Drop High", 8)
        hn, hs = make_trap_hats(8, "high", mute_last_beat=True)
        add_notes(p_hats_drop, 8, hn)
        add_notes(p_hats_drop, 9, hs)
        for sb in [9, 41, 49, 73, 81]:
            place(p_hats_drop, 7, sb)

        p_hats_outro = new_pattern("Hats - Outro 4m", 4)
        hn, _ = make_trap_hats(4, "low", mute_last_beat=True)
        add_notes(p_hats_outro, 8, hn)
        place(p_hats_outro, 7, 89)

        # ---- TOP LOOPS & OPENHAT (Track 8, Ch 10 OpenHat, Ch 11 Techno Hats, Ch 12 Key Hats) ----
        p_keyhats_intro = new_pattern("Key Hats - Intro M5-8", 4)
        add_notes(p_keyhats_intro, 12, [{"start": 0.0, "length": 15.0, "pitch": 64, "velocity": 72}])
        place(p_keyhats_intro, 8, 5)
        place(p_keyhats_intro, 8, 33)

        p_toploop_drop = new_pattern("TopLoop & OpenHat - 8m", 8)
        add_notes(p_toploop_drop, 11, [{"start": 0.0, "length": 31.0, "pitch": 64, "velocity": 94}])
        oh_notes = [{"start": b * 4.0 + 0.5, "length": 0.45, "pitch": 64, "velocity": 90} for b in [0, 2, 4, 6]]
        add_notes(p_toploop_drop, 10, oh_notes)
        place(p_toploop_drop, 8, 41)
        place(p_toploop_drop, 8, 49)

        p_toploop_climax = new_pattern("Double TopLoop - Refrain 3", 8)
        add_notes(p_toploop_climax, 11, [{"start": 0.0, "length": 31.0, "pitch": 64, "velocity": 98}])
        add_notes(p_toploop_climax, 12, [{"start": 0.0, "length": 31.0, "pitch": 64, "velocity": 90}])
        add_notes(p_toploop_climax, 10, oh_notes)
        place(p_toploop_climax, 8, 73)
        place(p_toploop_climax, 8, 81)

        # ---- TECHNO PERC 180BPM (Track 9, Ch 13) ----
        p_perc8 = new_pattern("Techno Perc - 8m", 8)
        add_notes(p_perc8, 13, [{"start": 0.0, "length": 31.0, "pitch": 64, "velocity": 92}])
        for sb in [25, 57, 65, 73, 81]:
            place(p_perc8, 9, sb)

        # ---- FX ROOM TRANSITION MUTES (Track 16, Mixer 20 volume automation) ----
        # Cut reverb tail cleanly to 0 during beat 4 of pivot bars (M8, M32, M40, M56, M72)
        # Note: mixer volume 0.78 corresponds to value ~0.78 on 0..1 range
        pts = [{"position": 0.0, "value": 0.78, "shape": "hold"}]
        for pivot_bar in [8, 32, 40, 56, 72]:
            cut_start = (pivot_bar - 1) * 4.0 + 3.0  # beat 4 of pivot bar
            cut_end = pivot_bar * 4.0                # beat 1 of next section
            pts.append({"position": cut_start, "value": 0.0, "shape": "hold"})
            pts.append({"position": cut_end, "value": 0.78, "shape": "hold"})
        pts.append({"position": 96 * 4.0, "value": 0.78, "shape": "hold"})

        queued_calls.append({"tool": "add_automation", "arguments": {
            "mixer": 20,
            "param": "volume",
            "name": "FX Room Transition Cuts",
            "points": pts,
            "track": 16,
            "start": 0.0,
            "length": 96 * 4.0,
        }})

        mcp.batch(queued_calls)

        # Save project!
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
