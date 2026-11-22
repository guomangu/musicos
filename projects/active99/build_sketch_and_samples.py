#!/usr/bin/env python3
"""
Step 1 & 2 for projects/active99 ("Lune de Tigre - Simba Cloud"):
1. Generate Top-Down `master_sketch.mid` (Bass, Chords, Lead) and pass strict 7-metric gate.
2. Prepare tuned (A4=440 Hz), de-clicked, anti-mud, Nyquist-clean WAV derivatives in `samples/derived/`.
"""

from __future__ import annotations

import json
import os
import sys

import librosa
import mido
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = "/home/gamo/Bureau/musicos"
PROJ_DIR = f"{ROOT}/projects/active99"
RAW_SAMPLES = f"{PROJ_DIR}/samples"
DERIVED_DIR = f"{PROJ_DIR}/samples/derived"

sys.path.insert(0, f"{ROOT}/scripts")
from studio_lib import clean_audio_sample, verify_midi_gate


def generate_master_sketch(out_mid: str, bpm: float = 142.0) -> None:
    ppq = 480
    mid = mido.MidiFile(ticks_per_beat=ppq)
    tempo = mido.bpm2tempo(bpm)

    # Meta track
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    mid.tracks.append(meta)

    def add_track(name: str, channel: int, events: list[tuple[float, float, int, int]]) -> None:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        msgs: list[tuple[int, mido.Message]] = []
        for start_b, dur_b, note, vel in events:
            t0 = int(round(start_b * ppq))
            t1 = int(round((start_b + dur_b) * ppq))
            msgs.append((t0, mido.Message("note_on", channel=channel, note=note, velocity=vel, time=0)))
            msgs.append((t1, mido.Message("note_off", channel=channel, note=note, velocity=0, time=0)))
        msgs.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1))
        cur = 0
        for tick, msg in msgs:
            delta = max(0, tick - cur)
            cur = tick
            tr.append(msg.copy(time=delta))
        tr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr)

    bp = json.load(open(f"{PROJ_DIR}/song_blueprint.json", encoding="utf-8"))
    prog = bp["harmonic_core"]["chord_progression"]

    bass_ev: list[tuple[float, float, int, int]] = []
    chord_ev: list[tuple[float, float, int, int]] = []
    lead_ev: list[tuple[float, float, int, int]] = []

    # 8 bars (two 4-bar phrases)
    for bar_idx in range(8):
        c = prog[bar_idx % 4]
        b0 = bar_idx * 4.0
        bass_midi = int(c["bass_midi"])
        # Bass: held 2.5 beats + turnaround pickup at bar 4 and 8
        if bar_idx in (3, 7):
            bass_ev.append((b0 + 0.0, 2.25, bass_midi, 108))
            bass_ev.append((b0 + 3.0, 0.75, 34, 98))  # Bb1 -> C2
        else:
            bass_ev.append((b0 + 0.0, 1.50, bass_midi, 108))
            bass_ev.append((b0 + 2.5, 1.00, bass_midi, 96))

        # Chords: Smooth Voice Leading, released at 2.85 beats so beat 3-4 breathe
        for note in c["voicing_midi"]:
            chord_ev.append((b0 + 0.0, 2.85, int(note), 82))

    # Lead (A - A' - A'' - B x 2), strict C minor stepwise, rest ~34%, apex G5 (79) in bar 3/7
    phrase_bars = [
        # Bar 1 (Cm7): Eb5 -> D5 -> C5 -> Bb4
        [(0.0, 0.75, 75, 92), (1.0, 0.50, 74, 86), (1.5, 0.50, 72, 88), (2.0, 0.75, 70, 84)],
        # Bar 2 (Abmaj7): C5 -> D5 -> Eb5 -> Eb5
        [(0.0, 0.75, 72, 90), (1.0, 0.50, 74, 86), (1.5, 0.50, 75, 92), (2.0, 0.75, 75, 85)],
        # Bar 3 (Fm7): F5 -> G5 (Apex) -> F5 -> Eb5
        [(0.0, 0.75, 77, 94), (1.0, 0.50, 79, 98), (1.5, 0.50, 77, 90), (2.0, 0.75, 75, 86)],
        # Bar 4 (Gm7): D5 -> C5 -> Bb4 -> C5 -> D5 (stepwise resolution to Eb5)
        [(0.0, 0.75, 74, 90), (1.0, 0.45, 72, 85), (1.5, 0.45, 70, 82), (2.0, 0.45, 72, 84), (2.5, 0.45, 74, 88)],
    ]
    for bar_idx in range(8):
        b0 = bar_idx * 4.0
        for st, dur, n, v in phrase_bars[bar_idx % 4]:
            # Subtle variation on bar 7 (second A'') - stay slightly softer or hold apex
            vel = v if bar_idx < 4 else max(72, v - 2)
            lead_ev.append((b0 + st, dur, n, vel))

    add_track("Bass", 0, bass_ev)
    add_track("Chords", 1, chord_ev)
    add_track("Lead", 2, lead_ev)
    mid.save(out_mid)


def _peaking_eq(y: np.ndarray, sr: int, freq_hz: float, gain_db: float, q: float = 1.5) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * q)
    cosw = np.cos(w0)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cosw
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cosw
    a2 = 1.0 - alpha / a
    sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)
    return np.stack([signal.sosfilt(sos, y[:, c]) for c in range(y.shape[1])], axis=1).astype(np.float32)


def _ms_stabilize(y: np.ndarray, sr: int, split_hz: float = 550.0, low_side_gain: float = 0.55) -> np.ndarray:
    if y.shape[1] != 2:
        return y
    mid = 0.5 * (y[:, 0] + y[:, 1])
    side = 0.5 * (y[:, 0] - y[:, 1])
    sos_lp = signal.butter(2, split_hz, btype="lowpass", fs=sr, output="sos")
    sos_hp = signal.butter(2, split_hz, btype="highpass", fs=sr, output="sos")
    side_lo = signal.sosfilt(sos_lp, side) * low_side_gain
    side_hi = signal.sosfilt(sos_hp, side)
    side_new = side_lo + side_hi
    out = np.stack([mid + side_new, mid - side_new], axis=1)
    return out.astype(np.float32)


def _write_clean(y: np.ndarray, sr: int, out_path: str, fade_in_ms: float = 2.0, fade_out_ms: float = 15.0, max_peak: float = 0.85) -> dict:
    y = y.astype(np.float32)
    y = y - np.mean(y, axis=0, keepdims=True)
    fi = min(len(y) // 4, int(fade_in_ms * sr / 1000.0))
    fo = min(len(y) // 4, int(fade_out_ms * sr / 1000.0))
    if fi > 0:
        y[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
    if fo > 0:
        y[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None]
    pk = float(np.max(np.abs(y)))
    if pk > max_peak and pk > 1e-9:
        y *= float(max_peak / pk)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, y, sr, subtype="PCM_24")
    return {
        "file": os.path.basename(out_path),
        "sr": sr,
        "dur_s": round(len(y) / sr, 3),
        "max_jump": round(float(np.max(np.abs(np.diff(y[:, 0])))), 4),
    }


def _time_stretch_exact(y: np.ndarray, sr: int, target_dur_s: float) -> np.ndarray:
    cur_dur = len(y) / sr
    rate = cur_dur / target_dur_s
    cols = [librosa.effects.time_stretch(y[:, c].astype(np.float32), rate=rate) for c in range(y.shape[1])]
    n = int(round(target_dur_s * sr))
    out = np.zeros((n, y.shape[1]), dtype=np.float32)
    for c in range(y.shape[1]):
        L = min(n, len(cols[c]))
        out[:L, c] = cols[c][:L]
    return out


def prepare_all_samples() -> list[dict]:
    os.makedirs(DERIVED_DIR, exist_ok=True)
    logs = []

    # 1. 808 Deep PNL (keep exact raw waveform sans DC & micro-fade, tune via gl.sampler Root=24 Fine=-14)
    logs.append(clean_audio_sample(
        f"{RAW_SAMPLES}/808_deep_pnl.wav",
        f"{DERIVED_DIR}/808_deep_pnl_440.wav",
        hpf_hz=26.0, fade_in_ms=1.5, fade_out_ms=25.0, max_peak=0.86,
    ))

    # 2. Kick Tight (trim to 160 ms punchy round PNL kick, HPF 32 Hz)
    yk, srk = sf.read(f"{RAW_SAMPLES}/kick_tight.wav", dtype="float32", always_2d=True)
    yk = yk[: int(0.165 * srk)]
    sos = signal.butter(2, 32.0, btype="highpass", fs=srk, output="sos")
    yk = np.stack([signal.sosfilt(sos, yk[:, c]) for c in range(yk.shape[1])], axis=1)
    logs.append(_write_clean(yk, srk, f"{DERIVED_DIR}/kick_tight_clean.wav", fade_in_ms=1.0, fade_out_ms=20.0, max_peak=0.86))

    # 3. Clap Soft & Snare Andromeda (HPF 180 Hz, LPF 11.5 kHz anti-Nyquist)
    logs.append(clean_audio_sample(
        f"{RAW_SAMPLES}/clap_soft.wav",
        f"{DERIVED_DIR}/clap_soft_clean.wav",
        hpf_hz=180.0, lpf_hz=11500.0, fade_in_ms=1.5, fade_out_ms=12.0, max_peak=0.82,
    ))
    logs.append(clean_audio_sample(
        f"{RAW_SAMPLES}/snare_andromeda.wav",
        f"{DERIVED_DIR}/snare_andromeda_clean.wav",
        hpf_hz=180.0, lpf_hz=11500.0, fade_in_ms=1.5, fade_out_ms=12.0, max_peak=0.80,
    ))

    # 4. Hi-Hats & Perc (LPF 12.2 kHz to prevent Nyquist jump / 8kHz shrillness)
    logs.append(clean_audio_sample(
        f"{RAW_SAMPLES}/hat_cloud.wav",
        f"{DERIVED_DIR}/hat_cloud_clean.wav",
        hpf_hz=2400.0, lpf_hz=12200.0, fade_in_ms=1.5, fade_out_ms=10.0, max_peak=0.72,
    ))
    logs.append(clean_audio_sample(
        f"{RAW_SAMPLES}/openhat_chrome.wav",
        f"{DERIVED_DIR}/openhat_chrome_clean.wav",
        hpf_hz=2600.0, lpf_hz=12000.0, fade_in_ms=2.0, fade_out_ms=25.0, max_peak=0.70,
    ))
    yp, srp = sf.read(f"{RAW_SAMPLES}/sxj_percs8_clean.wav", dtype="float32", always_2d=True)
    yp = yp[: int(0.16 * srp)]
    sos_hp = signal.butter(2, 450.0, btype="highpass", fs=srp, output="sos")
    sos_lp = signal.butter(2, 9500.0, btype="lowpass", fs=srp, output="sos")
    yp = np.stack([signal.sosfilt(sos_lp, signal.sosfilt(sos_hp, yp[:, c])) for c in range(yp.shape[1])], axis=1)
    logs.append(_write_clean(yp, srp, f"{DERIVED_DIR}/perc_ghost_clean.wav", fade_in_ms=2.0, fade_out_ms=15.0, max_peak=0.65))

    # 5. Pad Cloud (eaalah_pad_single.wav: G3 +2.6 ct): carve mud 210 Hz, notch faint B4 overtone 491 Hz, LPF 3.8 kHz, HPF 145 Hz, M/S stabilize
    ypad, srpad = sf.read(f"{RAW_SAMPLES}/eaalah_pad_single.wav", dtype="float32", always_2d=True)
    sos_hp = signal.butter(2, 145.0, btype="highpass", fs=srpad, output="sos")
    sos_lp = signal.butter(2, 3800.0, btype="lowpass", fs=srpad, output="sos")
    ypad = np.stack([signal.sosfilt(sos_lp, signal.sosfilt(sos_hp, ypad[:, c])) for c in range(ypad.shape[1])], axis=1)
    ypad = _peaking_eq(ypad, srpad, 205.0, -7.0, q=1.2)
    ypad = _peaking_eq(ypad, srpad, 491.0, -10.0, q=3.5)
    ypad = _ms_stabilize(ypad, srpad, split_hz=550.0, low_side_gain=0.55)
    logs.append(_write_clean(ypad, srpad, f"{DERIVED_DIR}/pad_cloud_g3_440.wav", fade_in_ms=8.0, fade_out_ms=40.0, max_peak=0.80))

    # 6. Glass Pad Bed (glass_waves_pad_a_8bar.wav stretched to 8 bars @ 142 BPM = 13.521127s, HPF 280 Hz, LPF 4.2 kHz)
    ygp, srgp = sf.read(f"{RAW_SAMPLES}/glass_waves_pad_a_8bar.wav", dtype="float32", always_2d=True)
    bar_s = 4.0 * 60.0 / 142.0
    ygp_142 = _time_stretch_exact(ygp, srgp, 8.0 * bar_s)
    sos_hp = signal.butter(2, 280.0, btype="highpass", fs=srgp, output="sos")
    sos_lp = signal.butter(2, 4200.0, btype="lowpass", fs=srgp, output="sos")
    ygp_142 = np.stack([signal.sosfilt(sos_lp, signal.sosfilt(sos_hp, ygp_142[:, c])) for c in range(ygp_142.shape[1])], axis=1)
    ygp_142 = _ms_stabilize(ygp_142, srgp, split_hz=600.0, low_side_gain=0.45)
    logs.append(_write_clean(ygp_142, srgp, f"{DERIVED_DIR}/glass_pad_142.wav", fade_in_ms=12.0, fade_out_ms=25.0, max_peak=0.72))

    # 7. PNL Auto-Tune Vox Lead (vox_-_virtual_-_c_clean.wav: C5 +11 ct): HPF 230 Hz, LPF 7.5 kHz, gentle 1.2k polish
    yvox, srvox = sf.read(f"{RAW_SAMPLES}/vox_-_virtual_-_c_clean.wav", dtype="float32", always_2d=True)
    sos_hp = signal.butter(2, 230.0, btype="highpass", fs=srvox, output="sos")
    sos_lp = signal.butter(2, 7500.0, btype="lowpass", fs=srvox, output="sos")
    yvox = np.stack([signal.sosfilt(sos_lp, signal.sosfilt(sos_hp, yvox[:, c])) for c in range(yvox.shape[1])], axis=1)
    logs.append(_write_clean(yvox, srvox, f"{DERIVED_DIR}/vox_lead_c5_440.wav", fade_in_ms=5.0, fade_out_ms=30.0, max_peak=0.80))

    # 8. Guitar Pluck (eaalah_guitar_pluck.wav: B3 +24.7 ct): HPF 240 Hz, carve 270 Hz, LPF 5.8 kHz
    ygtr, srgtr = sf.read(f"{RAW_SAMPLES}/eaalah_guitar_pluck.wav", dtype="float32", always_2d=True)
    sos_hp = signal.butter(2, 240.0, btype="highpass", fs=srgtr, output="sos")
    sos_lp = signal.butter(2, 5800.0, btype="lowpass", fs=srgtr, output="sos")
    ygtr = np.stack([signal.sosfilt(sos_lp, signal.sosfilt(sos_hp, ygtr[:, c])) for c in range(ygtr.shape[1])], axis=1)
    ygtr = _peaking_eq(ygtr, srgtr, 270.0, -4.5, q=1.3)
    logs.append(_write_clean(ygtr, srgtr, f"{DERIVED_DIR}/guitar_pluck_b3_440.wav", fade_in_ms=2.0, fade_out_ms=20.0, max_peak=0.82))

    # 9. Counter Vox Harm (vox_virtual_bb4.wav: Bb4 -1.3 ct): HPF 280 Hz, LPF 6.0 kHz
    yvh, srvh = sf.read(f"{RAW_SAMPLES}/vox_virtual_bb4.wav", dtype="float32", always_2d=True)
    sos_hp = signal.butter(2, 280.0, btype="highpass", fs=srvh, output="sos")
    sos_lp = signal.butter(2, 6000.0, btype="lowpass", fs=srvh, output="sos")
    yvh = np.stack([signal.sosfilt(sos_lp, signal.sosfilt(sos_hp, yvh[:, c])) for c in range(yvh.shape[1])], axis=1)
    logs.append(_write_clean(yvh, srvh, f"{DERIVED_DIR}/vox_virtual_bb4_440.wav", fade_in_ms=8.0, fade_out_ms=35.0, max_peak=0.76))

    # 10. Swells & Risers (exact 1 bar = 1.690141 s @ 142 BPM)
    yvs, srvs = sf.read(f"{RAW_SAMPLES}/vocal3_swell_rev.wav", dtype="float32", always_2d=True)
    yvs_142 = _time_stretch_exact(yvs, srvs, bar_s)
    sos_hp = signal.butter(2, 350.0, btype="highpass", fs=srvs, output="sos")
    sos_lp = signal.butter(2, 5500.0, btype="lowpass", fs=srvs, output="sos")
    yvs_142 = np.stack([signal.sosfilt(sos_lp, signal.sosfilt(sos_hp, yvs_142[:, c])) for c in range(yvs_142.shape[1])], axis=1)
    yvs_142 = _peaking_eq(yvs_142, srvs, 784.0, -6.0, q=1.6)
    logs.append(_write_clean(yvs_142, srvs, f"{DERIVED_DIR}/vox_swell_1bar_142.wav", fade_in_ms=15.0, fade_out_ms=10.0, max_peak=0.70))

    ygr, srgr = sf.read(f"{RAW_SAMPLES}/glass_waves_riser_1bar.wav", dtype="float32", always_2d=True)
    ygr_142 = _time_stretch_exact(ygr, srgr, bar_s)
    sos_hp = signal.butter(2, 300.0, btype="highpass", fs=srgr, output="sos")
    sos_lp = signal.butter(2, 7000.0, btype="lowpass", fs=srgr, output="sos")
    ygr_142 = np.stack([signal.sosfilt(sos_lp, signal.sosfilt(sos_hp, ygr_142[:, c])) for c in range(ygr_142.shape[1])], axis=1)
    logs.append(_write_clean(ygr_142, srgr, f"{DERIVED_DIR}/glass_riser_1bar_142.wav", fade_in_ms=15.0, fade_out_ms=8.0, max_peak=0.72))

    return logs


def main() -> None:
    sketch_path = f"{PROJ_DIR}/master_sketch.mid"
    generate_master_sketch(sketch_path, bpm=142.0)
    gate_res = verify_midi_gate(sketch_path, expected_key="C minor", max_pitch=84)
    print("GATE_PASSED:", json.dumps({
        "verdict": gate_res.get("musicality_verdict"),
        "metrics": gate_res.get("musicality_metrics"),
        "clashes": gate_res.get("minor_2nd_clashes"),
        "sub_collisions": gate_res.get("low_end_simultaneous_collisions"),
    }, indent=2))
    sample_logs = prepare_all_samples()
    print("PREPARED_SAMPLES:", json.dumps(sample_logs, indent=2))


if __name__ == "__main__":
    main()
