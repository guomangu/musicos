#!/usr/bin/env python3
"""
Prepare all Ea'alah-derived audio textures, clean drum/808 samples,
song_blueprint.json, and Top-Down master_sketch.mid for projects/active8.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import jsonschema
import librosa
import mido
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = "/home/gamo/Bureau/musicos"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/mcp-servers/audio-analyzer")

from musicality_metrics import optimize_voice_leading
from studio_lib import clean_audio_sample, verify_midi_gate

PROJ = f"{ROOT}/projects/active8"
STEMS = f"{PROJ}/stems"
SAMPLES = f"{PROJ}/samples"
SR = 48000
TARGET_BPM = 144.0
SRC_BPM = 166.97  # Measured from 5.744s = 16 beats cycle
STRETCH_RATE = TARGET_BPM / SRC_BPM  # ~0.8624 (slows down slightly for dreamy PNL feel)


def ms_stabilize_low_mid(y: np.ndarray, cutoff_hz: float = 650.0, side_gain: float = 0.22) -> np.ndarray:
    """Attenuate Side channel below cutoff_hz so 200-800 Hz correlation stays healthy (>= 0.80)."""
    if y.shape[1] < 2:
        return y
    mid = 0.5 * (y[:, 0] + y[:, 1])
    side = 0.5 * (y[:, 0] - y[:, 1])
    sos_lo = signal.butter(4, cutoff_hz, btype="lowpass", fs=SR, output="sos")
    sos_hi = signal.butter(4, cutoff_hz, btype="highpass", fs=SR, output="sos")
    side_lo = signal.sosfiltfilt(sos_lo, side) * side_gain
    side_hi = signal.sosfiltfilt(sos_hi, side) * 0.90
    side_new = side_lo + side_hi
    out = np.stack([mid + side_new, mid - side_new], axis=1).astype(np.float32)
    return out


def filter_audio(
    y: np.ndarray,
    hpf_hz: float | None = None,
    lpf_hz: float | None = None,
    fade_in_ms: float = 10.0,
    fade_out_ms: float = 25.0,
    target_peak: float = 0.78,
) -> np.ndarray:
    y = y.astype(np.float32).copy()
    y -= np.mean(y, axis=0, keepdims=True)
    if hpf_hz:
        sos = signal.butter(2, hpf_hz, btype="highpass", fs=SR, output="sos")
        y = np.stack([signal.sosfiltfilt(sos, y[:, c]) for c in range(y.shape[1])], axis=1).astype(np.float32)
    if lpf_hz and lpf_hz < SR * 0.48:
        sos = signal.butter(2, lpf_hz, btype="lowpass", fs=SR, output="sos")
        y = np.stack([signal.sosfiltfilt(sos, y[:, c]) for c in range(y.shape[1])], axis=1).astype(np.float32)
    fi = min(len(y) // 4, int(fade_in_ms * SR / 1000.0))
    fo = min(len(y) // 4, int(fade_out_ms * SR / 1000.0))
    if fi > 0:
        y[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
    if fo > 0:
        y[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None]
    pk = float(np.max(np.abs(y)))
    if pk > 1e-6:
        y *= target_peak / pk
    return y


def stretch_stereo(y: np.ndarray, rate: float) -> np.ndarray:
    chans = [librosa.effects.time_stretch(y[:, c], rate=rate) for c in range(y.shape[1])]
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans], axis=1).astype(np.float32)


def pitch_shift_stereo(y: np.ndarray, n_steps: float) -> np.ndarray:
    chans = [librosa.effects.pitch_shift(y[:, c], sr=SR, n_steps=n_steps) for c in range(y.shape[1])]
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans], axis=1).astype(np.float32)


def low_mid_corr(y: np.ndarray) -> float:
    sos = signal.butter(4, [200.0, 800.0], btype="bandpass", fs=SR, output="sos")
    l = signal.sosfiltfilt(sos, y[:, 0])
    r = signal.sosfiltfilt(sos, y[:, 1])
    d = float(np.sqrt(np.sum(l**2) * np.sum(r**2)))
    return round(float(np.sum(l * r) / d), 3) if d > 1e-12 else 1.0


def prepare_samples() -> dict[str, Any]:
    os.makedirs(SAMPLES, exist_ok=True)
    guitar, _ = sf.read(f"{STEMS}/Ea'alah (Family) - Guitar.wav", dtype="float32", always_2d=True)
    other, _ = sf.read(f"{STEMS}/Ea'alah (Family) - Other.wav", dtype="float32", always_2d=True)
    vocals, _ = sf.read(f"{STEMS}/Ea'alah (Family) - Vocals.wav", dtype="float32", always_2d=True)

    bar_len = int(round((4.0 * 60.0 / TARGET_BPM) * SR))  # 1.6667 s = 80000 samples
    four_bar_len = 4 * bar_len  # 6.6667 s
    eight_bar_len = 8 * bar_len  # 13.3333 s

    telemetry: dict[str, Any] = {}

    # 1. Ea'alah Guitar Loop (4 bars @ 144 BPM) from clean intro 1.89s..7.634s (zero drums/bass)
    g_src = guitar[int(1.89 * SR) : int(7.634 * SR)]
    g_st = stretch_stereo(g_src, STRETCH_RATE)
    g_st = pitch_shift_stereo(g_st, -0.04)  # -4 ct -> A4=440 Hz
    g_st = g_st[:four_bar_len] if len(g_st) >= four_bar_len else np.pad(g_st, ((0, four_bar_len - len(g_st)), (0, 0)))
    g_loop = ms_stabilize_low_mid(filter_audio(g_st, hpf_hz=155.0, lpf_hz=5800.0, fade_in_ms=12.0, fade_out_ms=35.0, target_peak=0.72))
    sf.write(f"{SAMPLES}/eaalah_guitar_loop.wav", g_loop, SR, subtype="PCM_24")
    telemetry["eaalah_guitar_loop"] = {"dur_s": round(len(g_loop) / SR, 3), "lm_corr": low_mid_corr(g_loop)}

    # 1b. Ea'alah Guitar Loop B (bars 5-8 of intro: 7.634s..13.378s) for natural 8-bar variation
    g_src_b = guitar[int(7.634 * SR) : int(13.378 * SR)]
    g_st_b = stretch_stereo(g_src_b, STRETCH_RATE)
    g_st_b = pitch_shift_stereo(g_st_b, -0.04)
    g_st_b = g_st_b[:four_bar_len] if len(g_st_b) >= four_bar_len else np.pad(g_st_b, ((0, four_bar_len - len(g_st_b)), (0, 0)))
    g_loop_b = ms_stabilize_low_mid(filter_audio(g_st_b, hpf_hz=155.0, lpf_hz=5800.0, fade_in_ms=12.0, fade_out_ms=35.0, target_peak=0.72))
    sf.write(f"{SAMPLES}/eaalah_guitar_loop_b.wav", g_loop_b, SR, subtype="PCM_24")

    # 2. Ea'alah Guitar Shimmer (+12 st) for Refrain 2 / Climax
    g_shim = pitch_shift_stereo(g_loop, 12.0)
    g_shim = filter_audio(g_shim, hpf_hz=750.0, lpf_hz=8500.0, fade_in_ms=20.0, fade_out_ms=40.0, target_peak=0.58)
    sf.write(f"{SAMPLES}/eaalah_guitar_shimmer.wav", g_shim, SR, subtype="PCM_24")

    # 3. Ea'alah Guitar Pluck (single clean monophonic note B3 from 230.00..230.42s, zero vocals/drums)
    gp = guitar[int(230.00 * SR) : int(230.42 * SR)]
    gp = filter_audio(gp, hpf_hz=160.0, lpf_hz=7500.0, fade_in_ms=2.0, fade_out_ms=55.0, target_peak=0.82)
    f0_gp = float(np.median(librosa.yin(gp.mean(axis=1)[: int(0.25 * SR)], fmin=150, fmax=450, frame_length=4096, sr=SR)))
    midi_gp = float(librosa.hz_to_midi(f0_gp))
    sf.write(f"{SAMPLES}/eaalah_guitar_pluck.wav", gp, SR, subtype="PCM_24")
    telemetry["eaalah_guitar_pluck"] = {"f0_hz": round(f0_gp, 2), "midi": round(midi_gp, 3)}

    # 4. Ea'alah Ambient Pad Loop (8 bars @ 144 BPM) sculpted from Other 102..112s (pure pad/strings, Guitar=0)
    o_src = other[int(102.0 * SR) : int(112.0 * SR)]
    # Time-stretch smoothly to 8 bars (13.333s) and crossfade seamlessly
    o_rate = len(o_src) / float(eight_bar_len)
    o_st = stretch_stereo(o_src, o_rate)
    o_st = pitch_shift_stereo(o_st, -0.06)  # -6 ct -> 440 Hz
    o_st = o_st[:eight_bar_len] if len(o_st) >= eight_bar_len else np.pad(o_st, ((0, eight_bar_len - len(o_st)), (0, 0)))
    o_pad = ms_stabilize_low_mid(filter_audio(o_st, hpf_hz=145.0, lpf_hz=2800.0, fade_in_ms=80.0, fade_out_ms=120.0, target_peak=0.68), side_gain=0.20)
    sf.write(f"{SAMPLES}/eaalah_pad_loop.wav", o_pad, SR, subtype="PCM_24")
    telemetry["eaalah_pad_loop"] = {"dur_s": round(len(o_pad) / SR, 3), "lm_corr": low_mid_corr(o_pad)}

    # 5. Ea'alah Pad Single Note (monophonic G3 from Other 111.00..111.95s, G:1.00!)
    op = other[int(111.00 * SR) : int(111.95 * SR)]
    op_st = stretch_stereo(op, 0.55)  # smooth ~1.7s lush pad sustain
    op_st = ms_stabilize_low_mid(filter_audio(op_st, hpf_hz=150.0, lpf_hz=3200.0, fade_in_ms=70.0, fade_out_ms=220.0, target_peak=0.72), side_gain=0.22)
    f0_op = float(np.median(librosa.yin(op_st.mean(axis=1)[int(0.15 * SR) : int(0.75 * SR)], fmin=120, fmax=360, frame_length=4096, sr=SR)))
    midi_op = float(librosa.hz_to_midi(f0_op))
    sf.write(f"{SAMPLES}/eaalah_pad_single.wav", op_st, SR, subtype="PCM_24")
    telemetry["eaalah_pad_single"] = {"f0_hz": round(f0_op, 2), "midi": round(midi_op, 3), "lm_corr": low_mid_corr(op_st)}

    # 6. Ea'alah Vocal Cloud Phrases A & B (from drumless 0.13..5.87s and 70.40..76.14s, stretched to 4 bars)
    v_a = vocals[int(0.13 * SR) : int(5.874 * SR)]
    v_a = stretch_stereo(v_a, STRETCH_RATE)
    v_a = v_a[:four_bar_len] if len(v_a) >= four_bar_len else np.pad(v_a, ((0, four_bar_len - len(v_a)), (0, 0)))
    v_a = ms_stabilize_low_mid(filter_audio(v_a, hpf_hz=360.0, lpf_hz=4200.0, fade_in_ms=45.0, fade_out_ms=90.0, target_peak=0.68), side_gain=0.30)
    sf.write(f"{SAMPLES}/eaalah_vox_cloud_a.wav", v_a, SR, subtype="PCM_24")

    v_b = vocals[int(70.40 * SR) : int(76.144 * SR)]
    v_b = stretch_stereo(v_b, STRETCH_RATE)
    v_b = v_b[:four_bar_len] if len(v_b) >= four_bar_len else np.pad(v_b, ((0, four_bar_len - len(v_b)), (0, 0)))
    v_b = ms_stabilize_low_mid(filter_audio(v_b, hpf_hz=360.0, lpf_hz=4200.0, fade_in_ms=45.0, fade_out_ms=90.0, target_peak=0.68), side_gain=0.30)
    sf.write(f"{SAMPLES}/eaalah_vox_cloud_b.wav", v_b, SR, subtype="PCM_24")

    # 7. Ea'alah Spectral Vocal Note C5 (from Vocals 5.72..6.22s, stretched smoothly to ~1.0s for harmonies)
    vn = vocals[int(5.72 * SR) : int(6.22 * SR)]
    vn_st = stretch_stereo(vn, 0.50)
    vn_st = ms_stabilize_low_mid(filter_audio(vn_st, hpf_hz=350.0, lpf_hz=5200.0, fade_in_ms=35.0, fade_out_ms=140.0, target_peak=0.72), side_gain=0.28)
    f0_vn = float(np.median(librosa.yin(vn_st.mean(axis=1)[int(0.08 * SR) : int(0.60 * SR)], fmin=350, fmax=700, frame_length=4096, sr=SR)))
    midi_vn = float(librosa.hz_to_midi(f0_vn))
    sf.write(f"{SAMPLES}/eaalah_vox_note_c5.wav", vn_st, SR, subtype="PCM_24")
    telemetry["eaalah_vox_note_c5"] = {"f0_hz": round(f0_vn, 2), "midi": round(midi_vn, 3)}

    # 8. Ea'alah Reversed Pad/Vocal Riser (1 bar = 1.6667s)
    riser_src = (0.7 * other[int(104.0 * SR) : int(105.667 * SR)] + 0.5 * vocals[int(75.5 * SR) : int(77.167 * SR)])
    riser_rev = riser_src[::-1].copy()
    riser_rev = filter_audio(riser_rev, hpf_hz=280.0, lpf_hz=5500.0, fade_in_ms=700.0, fade_out_ms=12.0, target_peak=0.68)
    # Exponential swell curve
    env = np.linspace(0.08, 1.0, len(riser_rev), dtype=np.float32) ** 2.0
    env[-int(0.012 * SR) :] *= np.linspace(1.0, 0.0, int(0.012 * SR), dtype=np.float32)
    riser_rev *= env[:, None]
    sf.write(f"{SAMPLES}/eaalah_riser_rev.wav", riser_rev, SR, subtype="PCM_24")

    # 9. Clean Drum & 808 Samples from soundbanks
    sb = f"{ROOT}/soundbanks"
    telemetry["808_deep_pnl"] = clean_audio_sample(
        f"{sb}/NIKOS - PRODUCER PAX VOL. 01/3. #808s/Deep 808.wav",
        f"{SAMPLES}/808_deep_pnl.wav",
        lpf_hz=2800.0, fade_in_ms=2.0, fade_out_ms=35.0, max_peak=0.85,
    )
    y8, sr8 = sf.read(f"{SAMPLES}/808_deep_pnl.wav", dtype="float32", always_2d=True)
    seg8 = y8.mean(axis=1)[int(0.06 * sr8) : int(0.55 * sr8)]
    f0_8 = float(np.median(librosa.yin(seg8, fmin=25, fmax=120, frame_length=8192, sr=sr8)))
    telemetry["808_deep_pnl"]["f0_hz"] = round(f0_8, 2)
    telemetry["808_deep_pnl"]["midi"] = round(float(librosa.hz_to_midi(f0_8)), 3)

    telemetry["kick_tight"] = clean_audio_sample(
        f"{sb}/TriSamples_-_808_Trapstep_Pack_Vol_2/TriSamples - 808 Trapstep Pack Vol 2/Kick Drums/Kick Drum 001 Tight.wav",
        f"{SAMPLES}/kick_tight.wav",
        hpf_hz=32.0, lpf_hz=9000.0, fade_in_ms=1.0, fade_out_ms=18.0, max_peak=0.88,
    )
    telemetry["clap_soft"] = clean_audio_sample(
        f"{sb}/NIKOS - PRODUCER PAX VOL. 01/4. #Claps/Soft Clap.wav",
        f"{SAMPLES}/clap_soft.wav",
        hpf_hz=180.0, lpf_hz=13500.0, fade_in_ms=1.0, fade_out_ms=15.0, max_peak=0.82,
    )
    telemetry["snare_andromeda"] = clean_audio_sample(
        f"{sb}/NIKOS - PRODUCER PAX VOL. 01/6. #Snares/Andromeda Snare.wav",
        f"{SAMPLES}/snare_andromeda.wav",
        hpf_hz=160.0, lpf_hz=13500.0, fade_in_ms=1.0, fade_out_ms=15.0, max_peak=0.82,
    )
    telemetry["hat_cloud"] = clean_audio_sample(
        f"{sb}/NIKOS - PRODUCER PAX VOL. 01/5. #Hi-hats/Cloud hi-hat.wav",
        f"{SAMPLES}/hat_cloud.wav",
        hpf_hz=3500.0, lpf_hz=12500.0, fade_in_ms=0.5, fade_out_ms=10.0, max_peak=0.72,
    )
    telemetry["openhat_chrome"] = clean_audio_sample(
        f"{sb}/NIKOS - PRODUCER PAX VOL. 01/7. #Open Hats/Chrome Open Hat.wav",
        f"{SAMPLES}/openhat_chrome.wav",
        hpf_hz=2800.0, lpf_hz=9000.0, fade_in_ms=1.5, fade_out_ms=25.0, max_peak=0.68,
    )
    return telemetry


def build_blueprint_and_master_sketch() -> dict[str, Any]:
    # Harmonic Core in E minor (E Aeolian): Em7 -> Cmaj7 -> Am7 -> Bm7
    # Pitch classes: E=4, F#=6, G=7, A=9, B=11, C=0, D=2
    chord_defs = [
        ("Em7", 28, [4, 7, 11, 2], [4, 7, 11, 2]),         # E1 (41.2 Hz), voicing E-G-B-D
        ("Cmaj7", 36, [0, 4, 7, 11], [4, 7, 11, 0]),       # C2 (65.4 Hz), voicing E-G-B-C
        ("Am7", 33, [9, 0, 4, 7], [4, 7, 9, 0]),           # A1 (55.0 Hz), voicing E-G-A-C
        ("Bm7", 35, [11, 2, 6, 9], [6, 9, 11, 2]),         # B1 (61.7 Hz), voicing F#-A-B-D
    ]
    voicings = optimize_voice_leading([c[3] for c in chord_defs], low_midi=48, high_midi=62)

    chord_prog = []
    for i, ((sym, bass_m, tones_pc, _), voic) in enumerate(zip(chord_defs, voicings), 1):
        chord_prog.append({
            "bar": i,
            "symbol": sym,
            "bass_midi": bass_m,
            "voicing_midi": voic,
            "chord_tones_pc": tones_pc,
        })

    blueprint = {
        "title": "Simba d'Ea'alah",
        "genre": "French Cloud Rap / PNL-inspired",
        "mood": "Mélancolie contemplative nocturne, tristesse lumineuse",
        "bpm": 144,
        "time_signature": {"numerator": 4, "denominator": 4},
        "key": "E minor",
        "swing": 0.04,
        "target_lufs": -11.0,
        "harmonic_core": {
            "key": "E",
            "mode": "aeolian",
            "progression_bars": 4,
            "chord_progression": chord_prog,
            "motif_cell": {
                "intervals_from_tonic": [7, 11, 14, 12],
                "rhythm_beats": [0.0, 0.75, 1.5, 2.5],
                "apex_midi": 79,
            },
            "phrase_plan": "A-A'-A''-B",
        },
        "structure": [
            {
                "section": "Intro",
                "start_bar": 1,
                "bars": 8,
                "energy": 0.32,
                "active_roles": ["harmony_filtered", "guitar_loop", "vocal_cloud"],
                "notes": "Exposition sous-marine de Ea'alah (Pad LPF + Guitare originale + Voix spectrale) sans batterie ni 808 ; riser inverse M8.",
            },
            {
                "section": "Couplet 1",
                "start_bar": 9,
                "bars": 16,
                "energy": 0.64,
                "active_roles": ["foundation_808", "harmony_carved", "lead_pluck"],
                "notes": "Entree 808 Deep ronde (fondamentales E1-C2-A1-B1), clap noye dans la reverb sur temps 3, hats 1/8 doux < 12.5 kHz, espace vocal 1.5-2.5 kHz degage ; pivots M16/M24.",
            },
            {
                "section": "Refrain 1",
                "start_bar": 25,
                "bars": 8,
                "energy": 0.88,
                "active_roles": ["foundation_808", "guitar_loop", "vocal_cloud"],
                "notes": "Ouverture du spectre : Guitare d'Ea'alah + Accords Pad + Voix spectrale doublee + couche snare/open-hat + triolets de hats discrets.",
            },
            {
                "section": "Pont",
                "start_bar": 33,
                "bars": 8,
                "energy": 0.40,
                "active_roles": ["harmony_pad", "vocal_harmony", "lead_pluck"],
                "notes": "Apesanteur nocturne : coupure 808/Kick, accords m9 ouverts + harmonie vocale C5 d'Ea'alah + riser M40.",
            },
            {
                "section": "Refrain 2 Climax",
                "start_bar": 41,
                "bars": 16,
                "energy": 0.95,
                "active_roles": ["foundation_808", "guitar_shimmer", "lead_pluck"],
                "notes": "Climax Cloud Rap : Guitare + Shimmer (+12 st filtre) + Motif complet avec Apex G5 + 808 avec octaves de relance M48/M56.",
            },
            {
                "section": "Outro",
                "start_bar": 57,
                "bars": 8,
                "energy": 0.25,
                "active_roles": ["harmony_filtered", "vocal_cloud"],
                "notes": "Soustraction progressive : coupe 808/Kick M57, coupe drums M59, sortie solitaire sur guitare et nappe vocale.",
            },
        ],
        "frequency_plan": {
            "sub_20_60": "Mono strict (Width=0, HPF 28 Hz) reserve a Deep 808 (E1=41.2Hz .. C2=65.4Hz).",
            "bass_60_250": "Kick Tight sidechain -> BUS 808 & BUS Music ; HPF >= 145 Hz sur guitares, pads et voix.",
            "low_mids_250_500": "Nappe Ea'alah Pad dominante, stabilisee M/S (< 650 Hz), coupe douce 300 Hz.",
            "mids_500_4000": "Creux -2.5 dB @ 1.8 kHz sur Pad/Guitare en couplet ; voix spectrale filtrée 360-4200 Hz.",
            "highs_4000_20000": "Cloud Hats coupes > 12.5 kHz pour douceur nocturne ; FX Cloud Verb (HPF 450 Hz).",
        },
        "tracks": [
            {"name": "Kick Tight", "role": "kick", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/kick_tight.wav", "mixer_track": 1, "freq_lane": "55-120 Hz + click"},
            {"name": "Soft Clap", "role": "clap", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/clap_soft.wav", "mixer_track": 2, "freq_lane": "1-6 kHz + FX Verb"},
            {"name": "Andromeda Snare", "role": "snare", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/snare_andromeda.wav", "mixer_track": 3, "freq_lane": "200 Hz - 5 kHz"},
            {"name": "Cloud Hat", "role": "hat", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/hat_cloud.wav", "mixer_track": 4, "freq_lane": "4-12.5 kHz"},
            {"name": "Chrome Open Hat", "role": "hat", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/openhat_chrome.wav", "mixer_track": 5, "freq_lane": "3.5-12 kHz"},
            {"name": "Deep 808 PNL", "role": "bass", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/808_deep_pnl.wav", "mixer_track": 6, "freq_lane": "28-140 Hz mono"},
            {"name": "Ea'alah Pad Chords", "role": "chords", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/eaalah_pad_single.wav", "mixer_track": 7, "freq_lane": "150 Hz - 3.2 kHz"},
            {"name": "Ea'alah Pad Bed", "role": "pad", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/eaalah_pad_loop.wav", "mixer_track": 8, "freq_lane": "150 Hz - 2.8 kHz"},
            {"name": "Ea'alah Guitar Loop A", "role": "guitar", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/eaalah_guitar_loop.wav", "mixer_track": 9, "freq_lane": "160 Hz - 5.8 kHz"},
            {"name": "Ea'alah Guitar Loop B", "role": "guitar", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/eaalah_guitar_loop_b.wav", "mixer_track": 10, "freq_lane": "160 Hz - 5.8 kHz"},
            {"name": "Ea'alah Guitar Shimmer", "role": "guitar", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/eaalah_guitar_shimmer.wav", "mixer_track": 11, "freq_lane": "750 Hz - 8.5 kHz"},
            {"name": "Ea'alah Guitar Pluck", "role": "lead", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/eaalah_guitar_pluck.wav", "mixer_track": 12, "freq_lane": "250 Hz - 7 kHz"},
            {"name": "Ea'alah Vocal Cloud A", "role": "vocal", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/eaalah_vox_cloud_a.wav", "mixer_track": 13, "freq_lane": "360 Hz - 4.2 kHz"},
            {"name": "Ea'alah Vocal Cloud B", "role": "vocal", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/eaalah_vox_cloud_b.wav", "mixer_track": 14, "freq_lane": "360 Hz - 4.2 kHz"},
            {"name": "Ea'alah Vocal Note C5", "role": "vocal", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/eaalah_vox_note_c5.wav", "mixer_track": 15, "freq_lane": "380 Hz - 5.2 kHz"},
            {"name": "Ea'alah Riser Rev", "role": "fx", "source_type": "sample", "plugin": "gl.sampler", "sample_path": f"{SAMPLES}/eaalah_riser_rev.wav", "mixer_track": 16, "freq_lane": "280 Hz - 5.5 kHz"},
        ],
        "vocals": {"enabled": False, "language": "fr", "singer": "", "lyrics_sections": []},
    }

    with open(f"{ROOT}/schemas/song_blueprint.schema.json", "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(blueprint, schema)
    with open(f"{PROJ}/song_blueprint.json", "w", encoding="utf-8") as fh:
        json.dump(blueprint, fh, indent=2, ensure_ascii=False)

    # Generate 8-bar Top-Down master_sketch.mid (Bass, Chords, Lead)
    # 2 phrases of A-A'-A''-B at 144 BPM
    ppq = 480
    mid = mido.MidiFile(ticks_per_beat=ppq)
    tempo = mido.bpm2tempo(TARGET_BPM)

    def make_track(name: str, notes: list[tuple[float, float, int, int]], include_tempo: bool = False) -> mido.MidiTrack:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if include_tempo:
            tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
        evs = []
        for st_b, dur_b, pitch, vel in notes:
            s_t = int(round(st_b * ppq))
            e_t = max(s_t + 30, int(round((st_b + dur_b) * ppq)))
            evs.append((s_t, 1, pitch, vel))
            evs.append((e_t, 0, pitch, 0))
        evs.sort(key=lambda x: (x[0], x[1]))
        prev = 0
        for tick, on, p, v in evs:
            dt = max(0, tick - prev)
            prev = tick
            tr.append(mido.Message("note_on" if on else "note_off", note=p, velocity=v, time=dt))
        return tr

    bass_notes: list[tuple[float, float, int, int]] = []
    chord_notes: list[tuple[float, float, int, int]] = []
    lead_notes: list[tuple[float, float, int, int]] = []

    # 8 bars (2 x 4-bar progressions) in E minor (E4=64 .. E5=76, chords in 48..62)
    # Em7 (E1=28), Cmaj7 (C2=36), Am7 (A1=33), Bm7 (B1=35)
    # Smooth melancholic PNL Cloud Rap motif (A - A' - A'' - B):
    # All intervals <= 4 st (stepwise/small thirds) for vocal singability & zero unresolved leaps!
    phrase_lead = [
        # Bar 0 (Em7: E,G,B,D) -> B4(71) - G4(67) - F#4(66) - G4(67), breath beat 3..4
        (0.0, 0.75, 71, 88), (0.75, 0.75, 67, 82), (1.5, 0.50, 66, 76), (2.0, 1.00, 67, 84),
        # Bar 1 (Cmaj7: C,E,G,B) -> Bass rises to C2(36); Lead steps down G4(67) - E4(64) - D4(62) - E4(64), breath beat 3..4
        (4.0, 0.75, 67, 86), (4.75, 0.75, 64, 80), (5.5, 0.50, 62, 74), (6.0, 1.00, 64, 82),
        # Bar 2 (Am7: A,C,E,G) -> Bass drops to A1(33); Lead ascends smoothly to Apex E5(76)!
        # E4(64)->G4(67,+3)->A4(69,+2)->C5(72,+3)->E5(76,+4)->D5(74,-2)
        (8.0, 0.75, 67, 88), (8.75, 0.50, 69, 84), (9.25, 0.75, 72, 90), (10.0, 0.75, 76, 96), (10.75, 0.50, 74, 84),
        # Bar 3 (Bm7: B,D,F#,A) -> Bass rises to B1(35); Lead resolves down D5(74)->B4(71)->A4(69)->B4(71), breath beat 3..4
        (12.0, 0.75, 74, 86), (12.75, 0.75, 71, 82), (13.5, 0.50, 69, 78), (14.0, 0.85, 71, 80),
    ]

    for cycle in range(2):
        bar_off = cycle * 4
        for idx, cp in enumerate(chord_prog):
            b0 = (bar_off + idx) * 4.0
            bm = cp["bass_midi"]
            bass_notes.append((b0, 3.75, bm, 100))
            for vp in cp["voicing_midi"]:
                chord_notes.append((b0, 3.75, vp, 72))
        for st, ln, p, v in phrase_lead:
            # Final bar 8 ending settles cleanly on B4(71) (root of Bm7 -> start of Em7)
            pitch_adj = 71 if (cycle == 1 and abs(st - 14.0) < 0.01) else p
            lead_notes.append((bar_off * 4.0 + st, ln, pitch_adj, v))

    mid.tracks.append(make_track("Bass", bass_notes, include_tempo=True))
    mid.tracks.append(make_track("Chords", chord_notes))
    mid.tracks.append(make_track("Lead", lead_notes))

    sketch_path = f"{PROJ}/master_sketch.mid"
    mid.save(sketch_path)

    gate_res = verify_midi_gate(sketch_path, expected_key="E minor", max_pitch=84)
    return {
        "blueprint_valid": True,
        "voicings": voicings,
        "midi_gate": gate_res,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-samples", action="store_true")
    args = ap.parse_args()
    tel = {} if args.skip_samples else prepare_samples()
    res = build_blueprint_and_master_sketch()
    print(json.dumps({"samples_telemetry": tel, "blueprint_and_gate": res}, indent=2, ensure_ascii=False))
