#!/usr/bin/env python3
"""
End-to-end validation test for Section 7 of the specification:
"Crée une instru sombre et mélancolique en Fa mineur à 132 BPM, 60 secondes,
sans aucune agressivité dans les aigus."
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import pretty_midi
from jsonschema import validate

ROOT = "/home/gamo/Bureau/musicos"
sys.path.extend([f"{ROOT}/scripts", f"{ROOT}/mcp-servers/audio-analyzer"])

from studio_lib import (
    GridlineMCP,
    TEMPLATE_EMPTY_MIXER,
    check_pattern_audibility,
    clean_audio_sample,
    setup_standard_buses,
    verify_midi_gate,
)
from musicality_metrics import optimize_voice_leading

PROJ_DIR = f"{ROOT}/projects/validation_fm_dark"
SAMPLES_OUT = f"{PROJ_DIR}/samples"
os.makedirs(SAMPLES_OUT, exist_ok=True)

BPM = 132.0
BEATS_PER_BAR = 4
TOTAL_BARS = 33  # 33 bars at 132 BPM = 60.00 s exact
KEY_STR = "F minor"
MODE = "aeolian"
PROFILE = "trap_rap"
VALENCE = -0.75
TENSION = 0.55
ENERGY = 0.65
DEPTH = 0.70

# F aeolian (F=5, G=7, Ab=8, Bb=10, C=0, Db=1, Eb=3)
# Progression P4b anchored on i: Fm7 -> Dbmaj7 -> Bbm7 -> Fm
PROG_BARS = [
    {"bar": 1, "symbol": "Fm7",    "bass_midi": 29, "pcs": [5, 8, 0, 3]},
    {"bar": 2, "symbol": "Dbmaj7", "bass_midi": 37, "pcs": [1, 5, 8, 0]},
    {"bar": 3, "symbol": "Bbm7",   "bass_midi": 34, "pcs": [10, 1, 5, 8]},
    {"bar": 4, "symbol": "Fm",     "bass_midi": 29, "pcs": [5, 8, 0]},
]
VOICINGS = optimize_voice_leading([p["pcs"] for p in PROG_BARS], low_midi=48, high_midi=65)

# 1. Build and schema-validate song_blueprint.json
blueprint = {
    "title": "Brume de Minuit (Validation Fm)",
    "genre": "Dark Trap / Cloud Mélancolique",
    "mood": "sombre, mélancolique, feutré, sans agressivité",
    "mix_profile": PROFILE,
    "emotion_vector": {
        "valence": VALENCE,
        "tension": TENSION,
        "energy": ENERGY,
        "depth": DEPTH,
    },
    "bpm": BPM,
    "time_signature": {"numerator": BEATS_PER_BAR, "denominator": 4},
    "key": KEY_STR,
    "swing": 0.08,
    "target_lufs": -11.0,
    "harmonic_core": {
        "key": "F",
        "mode": MODE,
        "progression_bars": 4,
        "chord_progression": [
            {
                "bar": p["bar"],
                "symbol": p["symbol"],
                "bass_midi": p["bass_midi"],
                "voicing_midi": VOICINGS[idx],
                "chord_tones_pc": p["pcs"],
            }
            for idx, p in enumerate(PROG_BARS)
        ],
        "motif_cell": {
            "intervals_from_tonic": [3, 2, 0, -2],  # Ab4 -> G4 -> F4 -> Eb4 (soupir b6 -> 5 -> 1)
            "rhythm_beats": [0.0, 1.0, 1.5, 2.5],
            "apex_midi": 68,
        },
        "phrase_plan": "A-A'-A''-B",
    },
    "structure": [
        {"section": "Intro",    "start_bar": 1,  "bars": 4, "energy": 0.30, "active_roles": ["chords", "lead"]},
        {"section": "Couplet",  "start_bar": 5,  "bars": 8, "energy": 0.60, "active_roles": ["chords", "bass", "drums", "lead"]},
        {"section": "Refrain1", "start_bar": 13, "bars": 8, "energy": 0.90, "active_roles": ["chords", "bass", "drums", "lead", "counter"]},
        {"section": "Pont",     "start_bar": 21, "bars": 4, "energy": 0.35, "active_roles": ["chords", "counter"]},
        {"section": "Climax",   "start_bar": 25, "bars": 4, "energy": 1.00, "active_roles": ["chords", "bass", "drums", "lead", "counter"]},
        {"section": "Outro",    "start_bar": 29, "bars": 5, "energy": 0.25, "active_roles": ["chords", "lead"]},
    ],
    "frequency_plan": {
        "sub_20_60": "808 mono pure (Voices=1, LPF 1.8 kHz, Root=24, Fine=-14ct)",
        "bass_60_250": "Kick tight mono + 808 sidechainée 5:1",
        "low_mids_250_500": "Pad gl.analog LPF 3.2 kHz, creux -2.5 dB @ 450 Hz",
        "mids_500_4000": "Lead gl.keys Soft Muffle 3.6 kHz + LPF 4.8 kHz + creux -2.5 dB @ 3.2 kHz",
        "highs_4000_20000": "Hi-hats doux LPF 11 kHz + micro-variations vélocité/pitch",
    },
    "tracks": [
        {"name": "Kick",        "role": "kick",    "source_type": "sample",          "mixer_track": 1, "pan": 0.0},
        {"name": "Clap",        "role": "snare",   "source_type": "sample",          "mixer_track": 2, "pan": 0.0},
        {"name": "SnareLayer",  "role": "snare",   "source_type": "sample",          "mixer_track": 3, "pan": 0.04},
        {"name": "HiHat",       "role": "hat",     "source_type": "sample",          "mixer_track": 4, "pan": -0.18},
        {"name": "808 Bass",    "role": "bass",    "source_type": "sample",          "mixer_track": 5, "pan": 0.0},
        {"name": "Chords Pad",  "role": "chords",  "source_type": "gridline_native", "plugin": "gl.analog", "preset": "Velvet Pad", "mixer_track": 6, "pan": 0.0},
        {"name": "Lead Keys",   "role": "lead",    "source_type": "gridline_native", "plugin": "gl.keys",   "preset": "Soft",       "mixer_track": 7, "pan": 0.0},
        {"name": "Counter Plk", "role": "counter", "source_type": "gridline_native", "plugin": "gl.pluck",  "preset": "Muted",      "mixer_track": 8, "pan": 0.22},
    ],
}

with open(f"{ROOT}/schemas/song_blueprint.schema.json", encoding="utf-8") as fh:
    schema = json.load(fh)
validate(instance=blueprint, schema=schema)
bp_path = f"{PROJ_DIR}/song_blueprint.json"
with open(bp_path, "w", encoding="utf-8") as fh:
    json.dump(blueprint, fh, indent=2, ensure_ascii=False)

# 2. Prepare cleaned samples into proj/samples/
kick_wav = clean_audio_sample(
    f"{ROOT}/SAMPLES/kick/kick_tight_clean.wav", f"{SAMPLES_OUT}/kick_tight.wav",
    fade_in_ms=1.5, fade_out_ms=12.0, max_peak=0.85,
)["out_path"]
clap_wav = clean_audio_sample(
    f"{ROOT}/SAMPLES/clap/clap_soft_clean.wav", f"{SAMPLES_OUT}/clap_soft.wav",
    lpf_hz=9000.0, fade_in_ms=2.0, fade_out_ms=15.0, max_peak=0.82,
)["out_path"]
snare_wav = clean_audio_sample(
    f"{ROOT}/SAMPLES/snare/snare_andromeda_clean.wav", f"{SAMPLES_OUT}/snare_soft.wav",
    lpf_hz=7500.0, fade_in_ms=2.0, fade_out_ms=15.0, max_peak=0.80,
)["out_path"]
hat_wav = clean_audio_sample(
    f"{ROOT}/SAMPLES/hat/hat_cloud_clean.wav", f"{SAMPLES_OUT}/hat_soft.wav",
    hpf_hz=900.0, lpf_hz=13500.0, fade_in_ms=2.0, fade_out_ms=12.0, max_peak=0.88,
)["out_path"]
sub_wav = clean_audio_sample(
    f"{ROOT}/SAMPLES/808/808_deep_pnl_440.wav", f"{SAMPLES_OUT}/808_deep.wav",
    lpf_hz=1800.0, fade_in_ms=2.0, fade_out_ms=20.0, max_peak=0.85,
)["out_path"]

# 3. Single Source of Truth in BEATS 0-indexed (Plan 2 descending/hypnotic in MIDI 58-68)
# 4-bar phrase cells (beats within a 16-beat block)
LEAD_4BAR_CELL = [
    # Bar 1 (Fm7): Ab4(68) -> G4(67) -> F4(65) -> Eb4(63)
    (0.0, 0.90, 68, 92), (1.0, 0.45, 67, 82), (1.5, 0.90, 65, 78), (2.5, 0.85, 63, 74),
    # Bar 2 (Dbmaj7): Ab4(68) -> F4(65) -> Db4(61) + silence 6.4..8.0
    (4.0, 0.90, 68, 90), (5.0, 0.45, 65, 80), (5.5, 0.90, 61, 76),
    # Bar 3 (Bbm7): Ab4(68) -> G4(67) -> F4(65) -> Db4(61)
    (8.0, 0.90, 68, 94), (9.0, 0.45, 67, 84), (9.5, 0.90, 65, 80), (10.5, 0.85, 61, 74),
    # Bar 4 (Fm): F4(65) -> Eb4(63) -> C4(60) + silence 14.2..16.0
    (12.0, 0.70, 65, 88), (12.75, 0.50, 63, 78), (13.25, 0.95, 60, 72),
]

COUNTER_4BAR_CELL = [
    # Only placed in Lead rests (beats 3.25..3.75 and 6.75..7.75 and 14.5..15.75)
    (6.75, 0.45, 60, 68), (7.25, 0.55, 56, 62),
    (14.5, 0.45, 65, 70), (15.0, 0.65, 60, 62),
]

def build_4bar_sketch() -> str:
    """Validate core 4-bar phrase + full arrangement in master_sketch.mid."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    b2s = 60.0 / BPM
    inst_bass = pretty_midi.Instrument(program=33, name="Bass")
    inst_chords = pretty_midi.Instrument(program=0, name="Chords")
    inst_lead = pretty_midi.Instrument(program=0, name="Lead")
    inst_counter = pretty_midi.Instrument(program=24, name="Counter Harmony")

    for bar_idx in range(4):
        b0 = bar_idx * BEATS_PER_BAR
        root = PROG_BARS[bar_idx]["bass_midi"]
        inst_bass.notes.append(pretty_midi.Note(98, root, b0 * b2s, (b0 + 3.5) * b2s))
        for k, p in enumerate(VOICINGS[bar_idx]):
            st = b0 + k * 0.02
            vel = 72 if k < len(VOICINGS[bar_idx]) - 1 else 80
            inst_chords.notes.append(pretty_midi.Note(vel, p, st * b2s, (b0 + 3.6) * b2s))
    for st, dur, p, vel in LEAD_4BAR_CELL:
        inst_lead.notes.append(pretty_midi.Note(vel, p, st * b2s, (st + dur) * b2s))
    for st, dur, p, vel in COUNTER_4BAR_CELL:
        inst_counter.notes.append(pretty_midi.Note(vel, p, st * b2s, (st + dur) * b2s))

    pm.instruments.extend([inst_bass, inst_chords, inst_lead, inst_counter])
    sketch_path = f"{PROJ_DIR}/master_sketch.mid"
    pm.write(sketch_path)
    gate = verify_midi_gate(
        sketch_path, expected_key=KEY_STR, max_pitch=72, valence=VALENCE, tension=TENSION
    )
    print("MIDI GATE PASS:", json.dumps(gate["musicality_metrics"], ensure_ascii=False))
    return sketch_path

build_4bar_sketch()

# 4. Build Gridline project (33 bars, dynamic Intro/Couplet/Refrain1/Pont/Climax/Outro)
proj_path = f"{PROJ_DIR}/projet.gridline"
shutil.copyfile(TEMPLATE_EMPTY_MIXER, proj_path)
mcp = GridlineMCP(proj_path, f"{PROJ_DIR}/exports")
mcp.call("set_song", {"tempo": BPM, "numerator": BEATS_PER_BAR, "denominator": 4})
setup_standard_buses(mcp, bpm=BPM, profile=PROFILE, valence=VALENCE, limiter_gain_db=8.2, fx_room_vol=0.28)

# Create channels 1..8
chan_defs = [
    ("gl.sampler", None,           "Kick",        1, 0.0),
    ("gl.sampler", None,           "Clap",        2, 0.0),
    ("gl.sampler", None,           "SnareLayer",  3, 0.04),
    ("gl.sampler", None,           "HiHat",       4, -0.18),
    ("gl.sampler", None,           "808 Bass",    5, 0.0),
    ("gl.analog",  "Velvet Pad",   "Chords Pad",  6, 0.0),
    ("gl.keys",    "Classic tine", "Lead Keys",   7, 0.0),
    ("gl.pluck",   "Muted",        "Counter Plk", 8, 0.22),
]
mcp.batch([
    {"tool": "add_channel", "arguments": {"plugin": pl, **({"preset": pr} if pr else {}), "name": nm, "mixer": mx}}
    for pl, pr, nm, mx, _ in chan_defs
])

# Load samples on sampler channels 1..5
for ch_idx, spath in [(1, kick_wav), (2, clap_wav), (3, snare_wav), (4, hat_wav), (5, sub_wav)]:
    mcp.call("patch", {"path": f"channel/{ch_idx}", "set": {"plugin": {"data": spath}}})

# Insert EQs for anti-stridence source filtering on synths (6, 7, 8) and hat (4), plus warm harmonic dist->eq on 808 (5)
mcp.batch([
    {"tool": "add_plugin", "arguments": {"mixer": 4, "plugin": "gl.eq"}},
    {"tool": "add_plugin", "arguments": {"mixer": 5, "plugin": "gl.dist"}},
    {"tool": "add_plugin", "arguments": {"mixer": 5, "plugin": "gl.eq"}},
    {"tool": "add_plugin", "arguments": {"mixer": 6, "plugin": "gl.eq"}},
    {"tool": "add_plugin", "arguments": {"mixer": 7, "plugin": "gl.eq"}},
    {"tool": "add_plugin", "arguments": {"mixer": 8, "plugin": "gl.eq"}},
])

# Configure synth LPFs (<= 7 kHz) + sampler tuning/LPF + mixer routing/pan/levels
param_changes = [
    # Channel 1 Kick
    ("channel/1/One shot", 1), ("channel/1/Volume", 0.72),
    # Channel 2 Clap
    ("channel/2/One shot", 1), ("channel/2/Volume", 0.95),
    ("channel/2/Filter", 1), ("channel/2/Cutoff", 8200),
    # Channel 3 SnareLayer
    ("channel/3/One shot", 1), ("channel/3/Volume", 0.85),
    ("channel/3/Filter", 1), ("channel/3/Cutoff", 6800),
    # Channel 4 HiHat (smooth 1-13.5 kHz air around 1.8-2.5% without spikes)
    ("channel/4/One shot", 1), ("channel/4/Volume", 0.94),
    # Channel 5 808 Bass (Root=24, Fine=-14ct, Voices=1, Release=45ms, LPF=1800Hz)
    ("channel/5/Root note", 24), ("channel/5/Fine tune", -14), ("channel/5/Voices", 1),
    ("channel/5/Attack", 0.002), ("channel/5/Release", 0.045),
    ("channel/5/Filter", 1), ("channel/5/Cutoff", 1800), ("channel/5/Volume", 0.68),
    # Channel 6 Chords Pad (gl.analog Velvet Pad — explicit LPF Cutoff = 3000 Hz, tight stereo)
    ("channel/6/Polyphony", 8), ("channel/6/Filter mode", 1), ("channel/6/Cutoff", 3000),
    ("channel/6/Filter env amount", 0.14), ("channel/6/Amp attack", 0.05), ("channel/6/Amp release", 0.45),
    ("channel/6/Unison spread", 0.35), ("channel/6/Chorus", 0), ("channel/6/Reverb", 0), ("channel/6/Delay", 0), ("channel/6/Volume", 0.80),
    # Channel 7 Lead Keys (gl.keys Classic tine — Muffle = 4400 Hz, Hardness = 0.35, low stereo width)
    ("channel/7/Muffle", 4400), ("channel/7/Hardness", 0.35), ("channel/7/Overdrive", 0.04),
    ("channel/7/Stereo", 0.16), ("channel/7/Tremolo depth", 0.05), ("channel/7/Volume", 0.50),
    # Channel 8 Counter Plk (gl.pluck Muted — Brightness = 0.40, Damping = 0.50)
    ("channel/8/Brightness", 0.40), ("channel/8/Damping", 0.50), ("channel/8/Volume", 0.95),
    # Insert 4 HiHat EQ: HPF 1000 Hz + Peak -3.5dB @ 4000 Hz + gentle -1.5dB @ 6800 Hz + High cut 13500 Hz
    ("mixer/4/slot/1/Band 1 type", 1), ("mixer/4/slot/1/Band 1 freq", 1000),
    ("mixer/4/slot/1/Band 5 type", 6), ("mixer/4/slot/1/Band 5 freq", 6800), ("mixer/4/slot/1/Band 5 gain", -1.5), ("mixer/4/slot/1/Band 5 width", 0.48),
    ("mixer/4/slot/1/Band 6 type", 6), ("mixer/4/slot/1/Band 6 freq", 4000), ("mixer/4/slot/1/Band 6 gain", -3.5), ("mixer/4/slot/1/Band 6 width", 0.35),
    ("mixer/4/slot/1/Band 7 type", 3), ("mixer/4/slot/1/Band 7 freq", 13500),
    # Insert 5 808 Bass: gentle gl.dist (Slot 1) -> gl.eq (Band model=0 Legacy for clean Q)
    ("mixer/5/slot/1/Mode", 0), ("mixer/5/slot/1/Drive", 9.0), ("mixer/5/slot/1/Tone", 1200), ("mixer/5/slot/1/Mix", 0.35),
    ("mixer/5/slot/1/Output", 0.0), ("mixer/5/slot/1/Oversampling", 2), ("mixer/5/slot/1/Remove DC", 1),
    ("mixer/5/slot/2/Band model", 0),
    ("mixer/5/slot/2/Band 1 type", 1), ("mixer/5/slot/2/Band 1 freq", 32),
    ("mixer/5/slot/2/Band 2 type", 6), ("mixer/5/slot/2/Band 2 freq", 44), ("mixer/5/slot/2/Band 2 gain", -2.0), ("mixer/5/slot/2/Band 2 Q (legacy)", 1.6),
    ("mixer/5/slot/2/Band 3 type", 6), ("mixer/5/slot/2/Band 3 freq", 63), ("mixer/5/slot/2/Band 3 gain", -4.0), ("mixer/5/slot/2/Band 3 Q (legacy)", 2.8),
    ("mixer/5/slot/2/Band 7 type", 3), ("mixer/5/slot/2/Band 7 freq", 520),
    # Insert 6 Chords Pad EQ: HPF 135 Hz + anti-340Hz Peak -4.0dB + mirror dip 1.6k + High cut 4200 Hz
    ("mixer/6/slot/1/Band 1 type", 1), ("mixer/6/slot/1/Band 1 freq", 135),
    ("mixer/6/slot/1/Band 4 type", 6), ("mixer/6/slot/1/Band 4 freq", 340), ("mixer/6/slot/1/Band 4 gain", -4.0), ("mixer/6/slot/1/Band 4 width", 0.42),
    ("mixer/6/slot/1/Band 6 type", 6), ("mixer/6/slot/1/Band 6 freq", 1600), ("mixer/6/slot/1/Band 6 gain", -2.5), ("mixer/6/slot/1/Band 6 width", 0.42),
    ("mixer/6/slot/1/Band 7 type", 3), ("mixer/6/slot/1/Band 7 freq", 4200),
    # Insert 7 Lead Keys EQ: HPF 230 Hz + anti-330Hz -4.0dB + Peak -3dB @ 3200 Hz + High cut 5600 Hz
    ("mixer/7/slot/1/Band 1 type", 1), ("mixer/7/slot/1/Band 1 freq", 230),
    ("mixer/7/slot/1/Band 4 type", 6), ("mixer/7/slot/1/Band 4 freq", 330), ("mixer/7/slot/1/Band 4 gain", -4.0), ("mixer/7/slot/1/Band 4 width", 0.42),
    ("mixer/7/slot/1/Band 6 type", 6), ("mixer/7/slot/1/Band 6 freq", 3200), ("mixer/7/slot/1/Band 6 gain", -3.0), ("mixer/7/slot/1/Band 6 width", 0.35),
    ("mixer/7/slot/1/Band 7 type", 3), ("mixer/7/slot/1/Band 7 freq", 5600),
    # Insert 8 Counter Plk EQ: HPF 260 Hz + Peak -2.5dB @ 3200 Hz + High cut 5000 Hz
    ("mixer/8/slot/1/Band 1 type", 1), ("mixer/8/slot/1/Band 1 freq", 260),
    ("mixer/8/slot/1/Band 6 type", 6), ("mixer/8/slot/1/Band 6 freq", 3200), ("mixer/8/slot/1/Band 6 gain", -2.5), ("mixer/8/slot/1/Band 6 width", 0.35),
    ("mixer/8/slot/1/Band 7 type", 3), ("mixer/8/slot/1/Band 7 freq", 5000),
    # BUS Drums (17) + BUS Bass (18): gentle balance 44/63/95 Hz; BUS Music (19): carve 315 Hz
    ("mixer/17/slot/1/Band model", 0),
    ("mixer/17/slot/1/Band 2 type", 6), ("mixer/17/slot/1/Band 2 freq", 95), ("mixer/17/slot/1/Band 2 gain", 2.5), ("mixer/17/slot/1/Band 2 Q (legacy)", 1.8),
    ("mixer/17/slot/1/Band 3 type", 6), ("mixer/17/slot/1/Band 3 freq", 63), ("mixer/17/slot/1/Band 3 gain", -4.0), ("mixer/17/slot/1/Band 3 Q (legacy)", 2.8),
    ("mixer/18/slot/3/Band model", 0),
    ("mixer/18/slot/3/Band 1 freq", 32),
    ("mixer/18/slot/3/Band 2 type", 6), ("mixer/18/slot/3/Band 2 freq", 63), ("mixer/18/slot/3/Band 2 gain", -4.0), ("mixer/18/slot/3/Band 2 Q (legacy)", 2.8),
    ("mixer/18/slot/3/Band 3 type", 6), ("mixer/18/slot/3/Band 3 freq", 92), ("mixer/18/slot/3/Band 3 gain", 2.5), ("mixer/18/slot/3/Band 3 Q (legacy)", 1.8),
    ("mixer/19/slot/2/Band 4 type", 6), ("mixer/19/slot/2/Band 4 freq", 315), ("mixer/19/slot/2/Band 4 gain", -3.0), ("mixer/19/slot/2/Band 4 width", 0.45),
    ("mixer/6/stereo", -0.20),
    ("mixer/7/stereo", -0.25),
    ("mixer/20/slot/1/Width", 0.55),
]
mcp.set_params(param_changes)

# Mixer routing & sends (balanced Kick/808 vs Music/Hats)
mcp.batch([
    {"tool": "edit_mixer_track", "arguments": {"mixer": 1, "name": "Kick",        "routeTo": 17, "volume": 0.62, "pan": 0.0}},
    {"tool": "edit_mixer_track", "arguments": {"mixer": 2, "name": "Clap",        "routeTo": 17, "volume": 1.00, "pan": 0.0}},
    {"tool": "edit_mixer_track", "arguments": {"mixer": 3, "name": "SnareLayer",  "routeTo": 17, "volume": 0.85, "pan": 0.04}},
    {"tool": "edit_mixer_track", "arguments": {"mixer": 4, "name": "HiHat",       "routeTo": 17, "volume": 0.94, "pan": -0.16}},
    {"tool": "edit_mixer_track", "arguments": {"mixer": 5, "name": "808 Bass",    "routeTo": 18, "volume": 0.54, "pan": 0.0}},
    {"tool": "edit_mixer_track", "arguments": {"mixer": 6, "name": "Chords Pad",  "routeTo": 19, "volume": 0.84, "pan": 0.0}},
    {"tool": "edit_mixer_track", "arguments": {"mixer": 7, "name": "Lead Keys",   "routeTo": 19, "volume": 0.50, "pan": 0.0}},
    {"tool": "edit_mixer_track", "arguments": {"mixer": 8, "name": "Counter Plk", "routeTo": 19, "volume": 1.00, "pan": 0.18}},
    # Sidechain Kick -> BUS Bass (18) & BUS Music (19)
    {"tool": "set_send", "arguments": {"mixer": 1, "to": 18, "level": 1.0, "sidechain": True, "preFader": True}},
    {"tool": "set_send", "arguments": {"mixer": 1, "to": 19, "level": 0.8, "sidechain": True, "preFader": True}},
    # FX Room (20) sends (0% on Kick/808/HiHat!)
    {"tool": "set_send", "arguments": {"mixer": 2, "to": 20, "level": 0.06}},
    {"tool": "set_send", "arguments": {"mixer": 6, "to": 20, "level": 0.16}},
    {"tool": "set_send", "arguments": {"mixer": 7, "to": 20, "level": 0.14}},
    {"tool": "set_send", "arguments": {"mixer": 8, "to": 20, "level": 0.18}},
])

# 5. Build dedicated single-sound patterns & dedicated playlist tracks ("1 sonorité = 1 piste")
rng = random.Random(42)

def _build_drum_notes(mode: str) -> dict[int, list[dict]]:
    kick_n, clap_n, snare_n, hat_n = [], [], [], []
    for bar in range(4):
        b0 = bar * 4.0
        k_beats = [0.0, 1.75, 2.75] if bar % 2 == 0 else [0.0, 2.5]
        if mode == "couplet" and bar == 3:
            k_beats = [0.0, 1.75]
        for kb in k_beats:
            kick_n.append({"start": b0 + kb, "length": 0.45, "pitch": 60, "velocity": 112 if kb == 0.0 else 100})

        if not (mode == "couplet" and bar == 3):
            clap_n.append({"start": b0 + 2.0 + 0.008, "length": 0.45, "pitch": 60, "velocity": 102})
        if mode in ("refrain", "climax") and not (bar == 3 and mode == "refrain"):
            snare_n.append({"start": b0 + 2.0 + 0.012, "length": 0.40, "pitch": 60, "velocity": 80})
        if mode in ("refrain", "climax") and bar == 3:
            for idx_f, fb in enumerate([3.25, 3.50, 3.75]):
                snare_n.append({"start": b0 + fb, "length": 0.22, "pitch": 60, "velocity": 72 + idx_f * 10})

        hat_grid = [i * 0.5 for i in range(8)]
        if mode in ("refrain", "climax") and bar in (1, 3):
            hat_grid.extend([1.75, 3.625, 3.75, 3.875])
        elif mode == "couplet":
            hat_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for hb in sorted(set(hat_grid)):
            if mode == "couplet" and bar == 3 and hb >= 3.0:
                continue
            is_roll = hb in (1.75, 3.625, 3.75, 3.875)
            base_v = 78 if abs(hb % 1.0) < 0.01 else (58 if abs(hb % 0.5) < 0.01 else 46)
            vel = max(32, min(95, base_v + rng.randint(-5, 5)))
            p_hat = 58 if is_roll else 60
            shift = -0.005 if not is_roll else 0.0
            hat_n.append({"start": max(0.0, b0 + hb + shift), "length": 0.16, "pitch": p_hat, "velocity": vel})
    return {1: kick_n, 2: clap_n, 3: snare_n, 4: hat_n}

d_couplet = _build_drum_notes("couplet")
d_refrain = _build_drum_notes("refrain")
d_climax  = _build_drum_notes("climax")

def _bass_notes(sparse: bool = False) -> list[dict]:
    out = []
    for bar in range(4):
        b0 = bar * 4.0
        root = PROG_BARS[bar]["bass_midi"]
        out.append({"start": b0 + 0.03, "length": 1.65, "pitch": root, "velocity": 102})
        if not sparse:
            out.append({"start": b0 + 2.50, "length": 1.25, "pitch": root, "velocity": 92})
    return out

def _chords_notes(climax: bool = False, outro_tail: bool = False) -> list[dict]:
    out = []
    if outro_tail:
        for k, p in enumerate(VOICINGS[0]):
            out.append({"start": k * 0.02, "length": 16.0 - k * 0.02, "pitch": p, "velocity": 68})
        return out
    for bar in range(4):
        b0 = bar * 4.0
        for k, p in enumerate(VOICINGS[bar]):
            vel = 68 if k < len(VOICINGS[bar]) - 1 else 74
            if climax:
                vel = min(90, vel + 10)
            out.append({"start": b0 + k * 0.02, "length": 3.6 - k * 0.02, "pitch": p, "velocity": vel})
    return out

def _lead_notes(sparse: bool = False, climax: bool = False, outro_tail: bool = False) -> list[dict]:
    out = []
    for st, dur, p, vel in LEAD_4BAR_CELL:
        if sparse and st >= 8.0 and not outro_tail:
            continue
        if outro_tail and st >= 4.0:
            continue
        out.append({"start": st, "length": dur, "pitch": p, "velocity": vel if not climax else min(106, vel + 8)})
    return out

def _counter_notes() -> list[dict]:
    return [{"start": st, "length": dur, "pitch": p, "velocity": vel} for st, dur, p, vel in COUNTER_4BAR_CELL]

# Each pattern holds notes for ONE channel only: (pat_idx, name, bars, channel, notes)
PATTERN_SPECS = [
    (1,  "Kick_Couplet",  4, 1, d_couplet[1]),
    (2,  "Kick_Refrain",  4, 1, d_refrain[1]),
    (3,  "Clap_Couplet",  4, 2, d_couplet[2]),
    (4,  "Clap_Main",     4, 2, d_refrain[2]),
    (5,  "Snare_Refrain", 4, 3, d_refrain[3]),
    (6,  "Snare_Climax",  4, 3, d_climax[3]),
    (7,  "HiHat_Couplet", 4, 4, d_couplet[4]),
    (8,  "HiHat_Refrain", 4, 4, d_refrain[4]),
    (9,  "808_Couplet",   4, 5, _bass_notes(sparse=True)),
    (10, "808_Refrain",   4, 5, _bass_notes(sparse=False)),
    (11, "Chords_Main",   4, 6, _chords_notes()),
    (12, "Chords_Climax", 4, 6, _chords_notes(climax=True)),
    (13, "Chords_Outro",  5, 6, _chords_notes(outro_tail=True)),
    (14, "Lead_Sparse",   4, 7, _lead_notes(sparse=True)),
    (15, "Lead_Main",     4, 7, _lead_notes()),
    (16, "Lead_Climax",   4, 7, _lead_notes(climax=True)),
    (17, "Lead_Outro",    5, 7, _lead_notes(outro_tail=True)),
    (18, "Counter_Main",  4, 8, _counter_notes()),
]

# Template starts with pattern 1; create patterns 2..18
mcp.batch([{"tool": "add_pattern", "arguments": {"name": nm}} for _, nm, _, _, _ in PATTERN_SPECS[1:]])
mcp.batch([
    {"tool": "edit_pattern", "arguments": {"pattern": pid, "name": nm, "bars": bars}}
    for pid, nm, bars, _, _ in PATTERN_SPECS
])
mcp.batch([
    {"tool": "set_notes", "arguments": {"pattern": pid, "channel": ch, "notes": notes}}
    for pid, _, _, ch, notes in PATTERN_SPECS
])

# 6. Audibility check on core patterns before playlist placement
issues = check_pattern_audibility(mcp, min_lufs=-32.0)
if issues:
    raise RuntimeError(f"Pattern audibility issues: {issues}")

# 7. Arrange playlist with STRICT 1-sound-per-track isolation (Tracks 1..8 = sounds, 9..10 = automations)
playlist_clips = [
    # Track 1: Kick (Couplet 5-12, Refrain1 13-20, Climax 25-28)
    {"tool": "add_clip", "arguments": {"pattern": 1,  "track": 1, "start": 16.0}},
    {"tool": "add_clip", "arguments": {"pattern": 1,  "track": 1, "start": 32.0}},
    {"tool": "add_clip", "arguments": {"pattern": 2,  "track": 1, "start": 48.0}},
    {"tool": "add_clip", "arguments": {"pattern": 2,  "track": 1, "start": 64.0}},
    {"tool": "add_clip", "arguments": {"pattern": 2,  "track": 1, "start": 96.0}},
    # Track 2: Clap
    {"tool": "add_clip", "arguments": {"pattern": 3,  "track": 2, "start": 16.0}},
    {"tool": "add_clip", "arguments": {"pattern": 3,  "track": 2, "start": 32.0}},
    {"tool": "add_clip", "arguments": {"pattern": 4,  "track": 2, "start": 48.0}},
    {"tool": "add_clip", "arguments": {"pattern": 4,  "track": 2, "start": 64.0}},
    {"tool": "add_clip", "arguments": {"pattern": 4,  "track": 2, "start": 96.0}},
    # Track 3: SnareLayer (Refrain1 + Climax)
    {"tool": "add_clip", "arguments": {"pattern": 5,  "track": 3, "start": 48.0}},
    {"tool": "add_clip", "arguments": {"pattern": 5,  "track": 3, "start": 64.0}},
    {"tool": "add_clip", "arguments": {"pattern": 6,  "track": 3, "start": 96.0}},
    # Track 4: HiHat
    {"tool": "add_clip", "arguments": {"pattern": 7,  "track": 4, "start": 16.0}},
    {"tool": "add_clip", "arguments": {"pattern": 7,  "track": 4, "start": 32.0}},
    {"tool": "add_clip", "arguments": {"pattern": 8,  "track": 4, "start": 48.0}},
    {"tool": "add_clip", "arguments": {"pattern": 8,  "track": 4, "start": 64.0}},
    {"tool": "add_clip", "arguments": {"pattern": 8,  "track": 4, "start": 96.0}},
    # Track 5: 808 Bass (Couplet, Refrain1, Climax)
    {"tool": "add_clip", "arguments": {"pattern": 9,  "track": 5, "start": 16.0}},
    {"tool": "add_clip", "arguments": {"pattern": 9,  "track": 5, "start": 32.0}},
    {"tool": "add_clip", "arguments": {"pattern": 10, "track": 5, "start": 48.0}},
    {"tool": "add_clip", "arguments": {"pattern": 10, "track": 5, "start": 64.0}},
    {"tool": "add_clip", "arguments": {"pattern": 10, "track": 5, "start": 96.0}},
    # Track 6: Chords Pad (Intro -> Outro)
    {"tool": "add_clip", "arguments": {"pattern": 11, "track": 6, "start": 0.0}},
    {"tool": "add_clip", "arguments": {"pattern": 11, "track": 6, "start": 16.0}},
    {"tool": "add_clip", "arguments": {"pattern": 11, "track": 6, "start": 32.0}},
    {"tool": "add_clip", "arguments": {"pattern": 11, "track": 6, "start": 48.0}},
    {"tool": "add_clip", "arguments": {"pattern": 11, "track": 6, "start": 64.0}},
    {"tool": "add_clip", "arguments": {"pattern": 11, "track": 6, "start": 80.0}},
    {"tool": "add_clip", "arguments": {"pattern": 12, "track": 6, "start": 96.0}},
    {"tool": "add_clip", "arguments": {"pattern": 13, "track": 6, "start": 112.0}},
    # Track 7: Lead Keys (Intro, Couplet, Refrain1, Climax, Outro)
    {"tool": "add_clip", "arguments": {"pattern": 14, "track": 7, "start": 0.0}},
    {"tool": "add_clip", "arguments": {"pattern": 14, "track": 7, "start": 16.0}},
    {"tool": "add_clip", "arguments": {"pattern": 14, "track": 7, "start": 32.0}},
    {"tool": "add_clip", "arguments": {"pattern": 15, "track": 7, "start": 48.0}},
    {"tool": "add_clip", "arguments": {"pattern": 15, "track": 7, "start": 64.0}},
    {"tool": "add_clip", "arguments": {"pattern": 16, "track": 7, "start": 96.0}},
    {"tool": "add_clip", "arguments": {"pattern": 17, "track": 7, "start": 112.0}},
    # Track 8: Counter Plk (Refrain1, Pont, Climax)
    {"tool": "add_clip", "arguments": {"pattern": 18, "track": 8, "start": 48.0}},
    {"tool": "add_clip", "arguments": {"pattern": 18, "track": 8, "start": 64.0}},
    {"tool": "add_clip", "arguments": {"pattern": 18, "track": 8, "start": 80.0}},
    {"tool": "add_clip", "arguments": {"pattern": 18, "track": 8, "start": 96.0}},
]
mcp.batch(playlist_clips)

# Dynamic arc + clean outro fade-out + Beat-4 FX Room mute on dedicated automation tracks 9 & 10
mcp.batch([
    {
        "tool": "add_automation",
        "arguments": {
            "mixer": 19,
            "param": "volume",
            "track": 9,
            "points": [
                {"position": 0.0,   "value": 0.48},  # Intro intime (~-14.5 LUFS)
                {"position": 15.5,  "value": 0.50},
                {"position": 16.0,  "value": 0.64},  # Couplet
                {"position": 47.5,  "value": 0.66},
                {"position": 48.0,  "value": 0.80},  # Refrain 1 (1.0 gain)
                {"position": 79.5,  "value": 0.80},
                {"position": 80.0,  "value": 0.48},  # Pont intime (~-14.5 LUFS)
                {"position": 95.5,  "value": 0.48},
                {"position": 96.0,  "value": 0.84},  # Climax
                {"position": 111.5, "value": 0.84},
                {"position": 112.0, "value": 0.65},  # Outro start
                {"position": 124.0, "value": 0.38},  # Decay
                {"position": 131.0, "value": 0.0},   # Clean fade-out to zero
            ],
        },
    },
    {
        "tool": "add_automation",
        "arguments": {
            "mixer": 20,
            "param": "volume",
            "track": 10,
            "points": [
                {"position": 0.0,   "value": 0.30},
                {"position": 94.5,  "value": 0.30},
                {"position": 95.0,  "value": 0.0},   # Beat-4 mute before Climax
                {"position": 96.0,  "value": 0.28},
                {"position": 120.0, "value": 0.28},
                {"position": 131.0, "value": 0.0},   # Clean tail fade
            ],
        },
    },
])

# Name all dedicated playlist tracks 1..10 explicitly
mcp.call("run_script", {
    "code": """
function apply()
    local names = {
        "Kick", "Clap", "SnareLayer", "HiHat",
        "808 Bass", "Chords Pad", "Lead Keys", "Counter Plk",
        "Auto BUS Music", "Auto FX Room"
    }
    for i, nm in ipairs(names) do
        project.playlist.setTrackName(i, nm)
    end
end
"""
})

mcp.call("save_project", {})
mcp.close()
print("BUILD COMPLETE:", proj_path)
