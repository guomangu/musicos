#!/usr/bin/env python3
"""
Centralized deterministic DSP sample preparation & tuning library for AI-Studio.
Eliminates per-song boilerplate in `prepare_samples.py`: provides stereo I/O,
biquad peaking EQ, varispeed resampling, pitch-shift, tanh saturation,
time-stretch, declarative `process_sample_specs()`, FFT + harmonic tuning analysis,
and `manifest.json` generation.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from audio_metrics import analyze_single_file
from studio_lib import clean_audio_sample, stabilize_stereo_low_mid


def read_stereo(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as float32 stereo (N, 2) with DC offset removed."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    y = y - np.mean(y, axis=0, keepdims=True)
    return y, int(sr)


def write_tmp_clean(y: np.ndarray, sr: int, out_path: str, **clean_kw: Any) -> dict[str, Any]:
    """Write temporary array, apply `clean_audio_sample()` hygiene, and remove temp file."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp.wav"
    sf.write(tmp, y.astype(np.float32), sr, subtype="PCM_24")
    try:
        return clean_audio_sample(tmp, out_path, **clean_kw)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def apply_eq_peak(y: np.ndarray, sr: int, freq_hz: float, gain_db: float, q: float = 1.6) -> np.ndarray:
    """Apply an RBJ peaking EQ biquad filter in stereo."""
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * amp
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * amp
    a0 = 1.0 + alpha / amp
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / amp
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    out = np.stack([signal.lfilter(b, a, y[:, c]) for c in range(y.shape[1])], axis=1)
    return out.astype(np.float32)


def apply_soft_saturation(y: np.ndarray, drive: float = 1.5) -> np.ndarray:
    """Apply symmetric tanh harmonic saturation normalized to unity peak scale."""
    denom = float(np.tanh(drive)) or 1.0
    return (np.tanh(y * float(drive)) / denom).astype(np.float32)


def varispeed_resample(y: np.ndarray, ratio_src_over_dst: float) -> np.ndarray:
    """Resample audio by `src_freq / dst_freq` (true varispeed tuning without phase smearing)."""
    n_out = max(64, int(round(len(y) * float(ratio_src_over_dst))))
    return np.stack([signal.resample(y[:, c], n_out) for c in range(y.shape[1])], axis=1).astype(np.float32)


def pitch_shift_stereo(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Time-preserving stereo pitch-shift (avoid on < 80 Hz 808/sub; use varispeed or Fine tune)."""
    chans = [librosa.effects.pitch_shift(y[:, c], sr=sr, n_steps=float(n_steps)) for c in range(y.shape[1])]
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans], axis=1).astype(np.float32)


def time_stretch_stereo(y: np.ndarray, rate: float) -> np.ndarray:
    """Stereo time-stretch (`rate < 1.0` slows/lengthens, `rate > 1.0` speeds/shortens)."""
    chans = [librosa.effects.time_stretch(y[:, c], rate=float(rate)) for c in range(y.shape[1])]
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans], axis=1).astype(np.float32)


def trim_leading_silence(y: np.ndarray, sr: int, thresh: float = 0.015, pre_ms: float = 1.5) -> np.ndarray:
    """Trim leading silence before first transient while keeping `pre_ms` pre-roll."""
    mono = np.max(np.abs(y), axis=1)
    pk = float(np.max(mono)) + 1e-12
    nz = np.where(mono > thresh * pk)[0]
    if len(nz):
        i0 = max(0, int(nz[0] - pre_ms * sr / 1000.0))
        return y[i0:]
    return y


def measure_fft_tuning(path: str, skip_start_s: float = 0.15, fmin: float = 25.0, fmax: float = 2500.0) -> dict[str, Any]:
    """
    Measure steady-state FFT fundamental peak & librosa harmonic tuning offset.
    Note on `gl.sampler` Fine tune sign: positive `+X` shifts pitch UP, negative `-X` shifts DOWN,
    so `gl_fine_tune_ct` is `-round(cents_offset)` to bring the sample to A4=440.0 Hz.
    """
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = y.mean(axis=1)
    i0 = min(len(mono) // 3, int(skip_start_s * sr)) if len(mono) > int(0.4 * sr) else 0
    seg = mono[i0:]
    if len(seg) < 256:
        seg = mono

    win = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    valid = (freqs >= fmin) & (freqs <= fmax)
    fft_hz = None
    fft_midi = None
    fft_ct = None
    if np.any(valid):
        idx_rel = int(np.argmax(spec[valid]))
        fft_hz = float(freqs[valid][idx_rel])
        midi_exact = float(librosa.hz_to_midi(fft_hz))
        fft_midi = int(round(midi_exact))
        fft_ct = round((midi_exact - fft_midi) * 100.0, 1)

    harm_ct = None
    try:
        harm = librosa.effects.harmonic(mono)
        harm_ct = round(float(librosa.estimate_tuning(y=harm, sr=sr)) * 100.0, 1)
    except Exception:
        pass

    return {
        "fft_peak_hz": round(fft_hz, 2) if fft_hz is not None else None,
        "fft_midi_root": fft_midi,
        "fft_cents_offset": fft_ct,
        "harmonic_tuning_ct": harm_ct,
        "recommended_gl_fine_tune_ct": int(-round(fft_ct)) if fft_ct is not None else 0,
    }


def build_multizone_samples(
    src_wav: str,
    out_dir: str,
    base_midi: int,
    zones: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """
    Phase 8 Q6.2 — Build a 3-zone pitch-split sampler set from one tonal one-shot.
    Each zone is pre-transposed to the zone centre so no played note ever exceeds
    +/-4 semitones from its sample; formants are softly preserved and stereo low-mids
    stabilised. Returns one descriptor per zone (`wav_path`, `base_midi`, `zone`,
    `shift_from_source_st`, `tuning_ct`).
    """
    abs_src = os.path.abspath(os.path.expanduser(src_wav))
    y, sr = read_stereo(abs_src)
    eff_zones = zones or [(48, 55), (56, 63), (64, 72)]
    stem = os.path.splitext(os.path.basename(abs_src))[0]
    os.makedirs(out_dir, exist_ok=True)
    out: list[dict[str, Any]] = []
    for i, (lo, hi) in enumerate(eff_zones, start=1):
        if lo > hi:
            raise ValueError(f"Invalid zone {i}: ({lo}, {hi})")
        center = int(round((lo + hi) / 2.0))
        shift_st = float(center - int(base_midi))
        z = pitch_shift_stereo(y, sr, shift_st)
        tmp = os.path.join(out_dir, f"{stem}_zone{i}_tmp.wav")
        os.makedirs(os.path.dirname(os.path.abspath(tmp)), exist_ok=True)
        sf.write(tmp, z.astype(np.float32), sr, subtype="PCM_24")
        tuning: dict[str, Any] = {}
        final_wav = os.path.join(out_dir, f"{stem}_zone{i}_midi{center}.wav")
        try:
            # Gentle stereo low-mid stabilisation (formant-preserving, no mono collapse above 650 Hz)
            mid_p = os.path.join(out_dir, f"{stem}_zone{i}_mid.wav")
            stabilize_stereo_low_mid(tmp, mid_p, cutoff_hz=650.0, side_keep_low=0.18, side_keep_high=0.85)
            tuning = measure_fft_tuning(mid_p)
            ct = tuning.get("fft_cents_offset")
            if ct is not None and abs(float(ct)) > 3.0:
                z_corr = pitch_shift_stereo(y, sr, shift_st - float(ct) / 100.0)
                sf.write(mid_p, z_corr.astype(np.float32), sr, subtype="PCM_24")
                stabilize_stereo_low_mid(mid_p, mid_p, cutoff_hz=650.0, side_keep_low=0.18, side_keep_high=0.85)
                tuning = measure_fft_tuning(mid_p)
            final_wav = os.path.join(out_dir, f"{stem}_zone{i}_midi{center}.wav")
            os.replace(mid_p, final_wav)
        finally:
            for p in (tmp,):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        out.append({
            "wav_path": final_wav,
            "zone": [int(lo), int(hi)],
            "base_midi": center,
            "shift_from_source_st": round(shift_st, 3),
            "tuning_ct": tuning.get("fft_cents_offset"),
        })
    return out


def build_sample_from_spec(src_dir: str, out_dir: str, spec: dict[str, Any]) -> str:
    """
    Transform one source sample into `out_dir/<out>` according to declarative `spec`.
    Supported spec keys:
      - `src` (filename in `src_dir`, or absolute path if `is_out_src=True` reads from `out_dir`)
      - `out` (output filename in `out_dir`)
      - `slice_s`: `(start_s, end_s)`
      - `max_dur_s`: truncate to max seconds
      - `trim_start_ms`: exact milliseconds to drop from start of sample
      - `trim_leading`: `bool | float` (threshold ratio)
      - `invert_phase`: `bool` (invert sample polarity `-y`)
      - `eq_peaks`: `list[(freq_hz, gain_db, q)]`
      - `saturate_drive`: `float`
      - `varispeed_hz`: `(src_hz, dst_hz)` or `varispeed_ratio`: `float`
      - `pitch_shift_st`: `float`
      - `time_stretch_rate`: `float`
      - `reverse`: `bool`
      - `hpf_hz`, `lpf_hz`, `fade_in_ms`, `fade_out_ms`, `max_peak`, `trim_end_below_ratio`
      - `stabilize_low_mid`: `dict(cutoff_hz=680.0, side_keep_low=0.15, side_keep_high=0.75)`
    """
    base_dir = out_dir if spec.get("src_in_out_dir") else src_dir
    src_path = spec["src"] if os.path.isabs(spec["src"]) else os.path.join(base_dir, spec["src"])
    out_path = os.path.join(out_dir, spec["out"])

    y, sr = read_stereo(src_path)

    if spec.get("trim_start_ms"):
        cut_ms = max(0.0, float(spec["trim_start_ms"]))
        cut_samples = min(max(0, len(y) - 64), int(round(cut_ms * sr / 1000.0)))
        if cut_samples > 0:
            y = y[cut_samples:]

    if spec.get("trim_leading"):
        th = 0.015 if spec["trim_leading"] is True else float(spec["trim_leading"])
        y = trim_leading_silence(y, sr, thresh=th)

    if spec.get("invert_phase"):
        y = (-y).astype(np.float32)

    if "slice_s" in spec and spec["slice_s"] is not None:
        s0, s1 = spec["slice_s"]
        i0 = max(0, int(float(s0) * sr))
        i1 = len(y) if s1 is None else min(len(y), int(float(s1) * sr))
        y = y[i0:i1]

    if "max_dur_s" in spec and spec["max_dur_s"] is not None:
        y = y[: min(len(y), int(float(spec["max_dur_s"]) * sr))]

    if spec.get("reverse"):
        y = y[::-1].copy()

    if "varispeed_hz" in spec and spec["varispeed_hz"] is not None:
        src_hz, dst_hz = spec["varispeed_hz"]
        y = varispeed_resample(y, float(src_hz) / float(dst_hz))
    elif "varispeed_ratio" in spec and spec["varispeed_ratio"] is not None:
        y = varispeed_resample(y, float(spec["varispeed_ratio"]))

    if "pitch_shift_st" in spec and spec["pitch_shift_st"]:
        y = pitch_shift_stereo(y, sr, float(spec["pitch_shift_st"]))

    if "time_stretch_rate" in spec and spec["time_stretch_rate"]:
        y = time_stretch_stereo(y, float(spec["time_stretch_rate"]))

    if spec.get("saturate_drive"):
        y = apply_soft_saturation(y, drive=float(spec["saturate_drive"]))

    for eq_item in spec.get("eq_peaks", []):
        freq_hz, gain_db = eq_item[0], eq_item[1]
        q = eq_item[2] if len(eq_item) > 2 else 1.5
        y = apply_eq_peak(y, sr, float(freq_hz), float(gain_db), q=float(q))

    clean_kw = {
        "hpf_hz": spec.get("hpf_hz"),
        "lpf_hz": spec.get("lpf_hz"),
        "fade_in_ms": float(spec.get("fade_in_ms", 2.0)),
        "fade_out_ms": float(spec.get("fade_out_ms", 15.0)),
        "max_peak": spec.get("max_peak", 0.85),
        "trim_end_below_ratio": spec.get("trim_end_below_ratio"),
    }

    if spec.get("stabilize_low_mid"):
        stab = spec["stabilize_low_mid"]
        stab_kw = stab if isinstance(stab, dict) else {}
        tmp_clean = out_path + ".stab_tmp.wav"
        write_tmp_clean(y, sr, tmp_clean, **clean_kw)
        stabilize_stereo_low_mid(
            tmp_clean,
            out_path,
            cutoff_hz=float(stab_kw.get("cutoff_hz", 680.0)),
            side_keep_low=float(stab_kw.get("side_keep_low", 0.15)),
            side_keep_high=float(stab_kw.get("side_keep_high", 0.78)),
        )
        if os.path.exists(tmp_clean):
            os.remove(tmp_clean)
    else:
        write_tmp_clean(y, sr, out_path, **clean_kw)

    return out_path


def audit_samples_manifest(out_dir: str, manifest_name: str = "manifest.json", verbose: bool = True) -> dict[str, Any]:
    """Audit every WAV in `out_dir` (DSP purity + FFT/harmonic tuning) and write `manifest.json`."""
    manifest: dict[str, Any] = {}
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".wav"):
            continue
        p = os.path.join(out_dir, fn)
        entry, _ = analyze_single_file(p, with_clap=False)
        dsp = entry["dsp"]
        fft_info = measure_fft_tuning(p)
        manifest[fn] = {
            "dur_s": entry["duration_sec"],
            "sr": entry["sample_rate"],
            "low_mid_corr": dsp.get("low_mid_corr"),
            "phase_corr": dsp.get("phase_corr"),
            "sample_nature": dsp.get("sample_nature"),
            "playable_chromatically": dsp.get("playable_chromatically"),
            "note": dsp.get("detected_note"),
            "f0_hz": dsp.get("detected_f0_hz"),
            "midi_root": dsp.get("midi_root"),
            "fine_tune_ct": dsp.get("recommended_fine_tune_ct"),
            "fft_peak_hz": fft_info["fft_peak_hz"],
            "fft_midi_root": fft_info["fft_midi_root"],
            "fft_cents_offset": fft_info["fft_cents_offset"],
            "harmonic_tuning_ct": fft_info["harmonic_tuning_ct"],
            "recommended_gl_fine_tune_ct": fft_info["recommended_gl_fine_tune_ct"],
            "centroid_hz": dsp.get("spectral_centroid_hz"),
        }
    manifest_path = os.path.join(out_dir, manifest_name)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if verbose:
        print(f"Prepared {len(manifest)} samples in {out_dir}:")
        for k, v in manifest.items():
            print(
                f"  {k:24s} dur={v['dur_s']:6.3f}s lm={v['low_mid_corr']:.3f} "
                f"nat={str(v['sample_nature']):17s} play={str(v['playable_chromatically']):5s} "
                f"fft={str(v['fft_peak_hz']):>7s}Hz gl_ft={v['recommended_gl_fine_tune_ct']:+4d}ct "
                f"harm_ct={str(v['harmonic_tuning_ct']):>6s}"
            )
    return manifest


def process_sample_specs(
    src_dir: str,
    out_dir: str,
    specs: list[dict[str, Any]],
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute a declarative list of sample derivations in order and emit `manifest.json`."""
    os.makedirs(out_dir, exist_ok=True)
    for spec in specs:
        build_sample_from_spec(src_dir, out_dir, spec)
    return audit_samples_manifest(out_dir, verbose=verbose)


# ---------------------------------------------------------------------------
# v3.0 -- kick tuning guard & loop repitch helpers
# ---------------------------------------------------------------------------

def analyze_kick_tail(y: np.ndarray, sr: int, start_ms: float = 40.0, end_ms: float = 180.0) -> dict[str, Any]:
    """v3.0: analyze the kick tail window [start_ms, end_ms] after the attack click.

    Returns tail RMS (dBFS), dominant narrow peak in [35, 100] Hz (Hz + HNR dB) if any.
    A resonant trap/EDM kick has a long sinusoidal tail there; a dry kick does not.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    i0 = int(start_ms / 1000.0 * sr)
    i1 = min(len(y), int(end_ms / 1000.0 * sr))
    if i1 - i0 < 256:
        return {"tail_rms_dbfs": -120.0, "peak_hz": None, "hnr_db": None, "tonal": False}
    tail = y[i0:i1]
    rms = float(np.sqrt(np.mean(tail ** 2))) + 1e-12
    tail_rms_dbfs = round(20.0 * math.log10(rms), 2)
    n_fft = 32768
    win = tail * np.hanning(len(tail))
    sp = np.abs(np.fft.rfft(win, n=n_fft))
    fr = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (fr >= 35.0) & (fr <= 100.0)
    if not np.any(band):
        return {"tail_rms_dbfs": tail_rms_dbfs, "peak_hz": None, "hnr_db": None, "tonal": False}
    sb, fb = sp[band], fr[band]
    pk = int(np.argmax(sb))
    if float(sb[pk]) <= 1e-12:
        return {"tail_rms_dbfs": tail_rms_dbfs, "peak_hz": None, "hnr_db": None, "tonal": False}
    peak_hz = float(fb[pk])
    floor = max(float(np.median(sb)), 1e-12)
    hnr_db = round(10.0 * math.log10(float(sb[pk]) / floor), 2)
    tonal = hnr_db > 4.0
    return {"tail_rms_dbfs": tail_rms_dbfs, "peak_hz": round(peak_hz, 2), "hnr_db": hnr_db, "tonal": bool(tonal)}


def kick_tuning_guard(y: np.ndarray, sr: int, tonic_pc: int, fifth_pc: int | None = None) -> dict[str, Any]:
    """v3.0: tune a resonant kick to the consonant set {tonic, fifth} of the song key.

    If the tail is loud (> -24 dBFS) and tonal (HNR > 4 dB) in [35, 100] Hz:
      - distance <= 2 st to tonic/fifth -> action "fine_tune_ct" with the exact cents fix;
      - distance > 2 st -> action "tail_gate_95ms" (soft gate at 95 ms) or "swap"
        (a kick that rings a minor 2nd against the 808 is the wrong sample).
    """
    info = analyze_kick_tail(y, sr)
    consonant_pcs = {int(tonic_pc) % 12}
    if fifth_pc is not None:
        consonant_pcs.add(int(fifth_pc) % 12)
    else:
        consonant_pcs.add((int(tonic_pc) + 7) % 12)
    out = {**info, "action": "ok", "cents": 0, "consonant_pcs": sorted(consonant_pcs)}
    if not info["tonal"] or info["tail_rms_dbfs"] <= -24.0 or info["peak_hz"] is None:
        return out
    f = info["peak_hz"]
    best_f, best_st = None, 99
    for pc in consonant_pcs:
        for midi in range(24, 60):
            if midi % 12 != pc:
                continue
            fc = 440.0 * 2.0 ** ((midi - 69) / 12.0)
            if not (30.0 <= fc <= 200.0):
                continue
            st = 1200.0 * math.log2(f / fc) / 100.0
            if abs(st) < abs(best_st):
                best_f, best_st = fc, st
    if best_f is None:
        return out
    cents = int(round(-best_st * 100.0))
    dist_st = abs(best_st)
    if dist_st <= 2.0:
        out.update({"action": "fine_tune_ct", "cents": cents, "target_hz": round(best_f, 2), "dist_st": round(dist_st, 2)})
    else:
        out.update({"action": "tail_gate_95ms", "dist_st": round(dist_st, 2), "peak_hz": round(f, 2),
                    "fix": "kick rings > 2 st away from {tonic, fifth}: apply tail gate at 95 ms or swap for a dry kick"})
    return out


def apply_tail_gate(y: np.ndarray, sr: int, gate_ms: float = 95.0, release_ms: float = 12.0) -> np.ndarray:
    """v3.0: softly gate the tail of a resonant one-shot at `gate_ms` (short release)."""
    y = np.array(y, dtype=np.float64, copy=True)
    g0 = int(gate_ms / 1000.0 * sr)
    r = max(1, int(release_ms / 1000.0 * sr))
    if g0 >= len(y):
        return y
    tail = y[g0:]
    env = np.ones(len(tail))
    k = min(r, len(tail))
    env[:k] = np.linspace(1.0, 0.0, k)
    env[k:] = 0.0
    y[g0:] = tail * env
    return y


def repitch_loop(y: np.ndarray, sr: int, delta_st: float, out_path: str | None = None) -> dict[str, Any]:
    """v3.0: repitch a melodic loop by `delta_st` semitones (phase-vocoder when possible,
    varispeed fallback), then stabilize the stereo low-mid. Optionally writes `out_path`."""
    if abs(delta_st) < 1e-6:
        return {"delta_st": 0.0, "out_path": out_path}
    try:
        y2 = librosa.effects.pitch_shift(np.asarray(y, dtype=np.float32), sr=sr, n_steps=float(delta_st))
    except Exception:
        y2 = varispeed_resample(np.asarray(y, dtype=np.float32), 2.0 ** (float(delta_st) / 12.0))
    if out_path:
        write_tmp_clean(y2, sr, out_path)
        stabilize_stereo_low_mid(out_path, out_path, cutoff_hz=650.0, side_keep_low=0.18, side_keep_high=0.85)
    return {"delta_st": float(delta_st), "out_path": out_path, "method": "pitch_shift"}
