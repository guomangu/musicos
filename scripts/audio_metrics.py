#!/usr/bin/env python3
"""
Shared Psychoacoustic, DSP, Bark-Masking, Pitch & LAION-CLAP Evaluation Library
for AI-Studio Sample Selection & Online API Filtering.

Implements the 4-layer deterministic DSP & psychoacoustic funnel specified in:
docs/04_ANALYSE_SONORE_SAMPLES_APIS_ET_VGM.md
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

try:
    import parselmouth
    _HAS_PARSELMOUTH = True
except ImportError:
    parselmouth = None  # type: ignore[assignment]
    _HAS_PARSELMOUTH = False

# 24 Bark critical band edges (Hz), covering 20 Hz .. 20 kHz
BARK_EDGES_HZ: list[tuple[float, float]] = [
    (20.0, 100.0), (100.0, 200.0), (200.0, 300.0), (300.0, 400.0),
    (400.0, 510.0), (510.0, 630.0), (630.0, 770.0), (770.0, 920.0),
    (920.0, 1080.0), (1080.0, 1270.0), (1270.0, 1480.0), (1480.0, 1720.0),
    (1720.0, 2000.0), (2000.0, 2320.0), (2320.0, 2700.0), (2700.0, 3150.0),
    (3150.0, 3700.0), (3700.0, 4400.0), (4400.0, 5300.0), (5300.0, 6400.0),
    (6400.0, 7700.0), (7700.0, 9500.0), (9500.0, 12000.0), (12000.0, 20000.0),
]

SIX_BANDS_HZ: dict[str, tuple[float, float]] = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "lowmid": (250.0, 800.0),
    "mid": (800.0, 2000.0),
    "harsh": (2000.0, 5000.0),
    "air": (5000.0, 20000.0),
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}

VALID_ROLES = (
    "kick",
    "bass_sub",
    "snare_clap",
    "hihat_perc",
    "lead_melody",
    "pad_atmosphere",
    "chord_keys",
)

POSITIVE_CLAP_PROMPTS = [
    "warm analog studio quality sound, rich harmonics, clean professional mix",
    "pleasant lush musical timbre, high fidelity audio sample",
    "polished modern music production instrument, well balanced tone",
]

NEGATIVE_CLAP_PROMPTS = [
    "cheap plastic midi sound, amateur harsh digital distortion",
    "unpleasant piercing resonant noise, muddy low quality audio",
    "tinny irritating synthetic beep, poor recording quality",
]

HARSH_THRESHOLD_BY_ROLE: dict[str, float] = {
    "kick": 0.16,
    "bass_sub": 0.10,
    "snare_clap": 0.42,
    "hihat_perc": 0.52,
    "lead_melody": 0.32,
    "pad_atmosphere": 0.25,
    "chord_keys": 0.28,
}


def file_md5(path: str, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_and_normalize_audio(
    path: str,
    target_sr: int = 44100,
    max_seconds: float = 30.0,
) -> dict[str, Any]:
    """
    Load audio per Section 2 specification:
      - Preserve stereo signal y_stereo (N, 2) for clipping & phase correlation.
      - Resample mono average to 44100 Hz, remove DC offset, and peak-normalize
        a copy y_mono_norm to -1 dBFS for level-independent spectral metrics.
    """
    abs_path = os.path.abspath(os.path.expanduser(path))
    ext = os.path.splitext(abs_path)[1].lower()
    y: np.ndarray
    sr: int
    total_dur: float

    try:
        info = sf.info(abs_path)
        sr = int(info.samplerate)
        total_dur = float(info.frames / sr) if sr > 0 else 0.0
        max_frames = int(max_seconds * sr) if sr > 0 else -1
        y, sr = sf.read(abs_path, frames=max_frames, dtype="float32", always_2d=True)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = tf.name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-t", str(max_seconds), "-i", abs_path,
                    "-ac", "2", "-ar", str(target_sr), tmp_wav,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            y, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
            total_dur = float(len(y) / sr) if sr > 0 else 0.0
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    if y.shape[1] == 1:
        y_stereo = np.repeat(y, 2, axis=1)
        is_mono_source = True
    else:
        y_stereo = y[:, :2]
        is_mono_source = False

    mono_raw = np.mean(y_stereo, axis=1).astype(np.float32)
    if sr != target_sr and len(mono_raw) > 16:
        mono_44k = librosa.resample(mono_raw, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    else:
        mono_44k = mono_raw.copy()

    mono_44k = mono_44k - float(np.mean(mono_44k))
    peak = float(np.max(np.abs(mono_44k))) if len(mono_44k) else 0.0
    target_peak = 10.0 ** (-1.0 / 20.0)  # -1 dBFS
    mono_norm = (mono_44k * (target_peak / peak)).astype(np.float32) if peak > 1e-9 else mono_44k.copy()

    return {
        "filepath": abs_path,
        "y_stereo": y_stereo,
        "sr_orig": sr,
        "is_mono_source": is_mono_source,
        "mono_raw_44k": mono_44k,
        "mono_norm": mono_norm,
        "sr": target_sr,
        "duration_sec": round(total_dur if ext != ".mp3" else float(len(mono_44k) / target_sr), 4),
    }


def _max_consecutive_true(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return int(np.max(ends - starts)) if len(starts) else 0


def check_hard_rejects(
    y_stereo: np.ndarray,
    mono_raw: np.ndarray,
    sr: int,
    is_mono_source: bool = False,
) -> dict[str, Any]:
    """
    Layer 1 Hard Filters (Section 3.1):
      - Consecutive digital clipping |y| >= 0.99 (>=3) or >0.1% saturated samples
      - Non-zero crossing clicks at start/end (>0.05) -> needs_fade flag
      - Silence / weak signal (< -55 dBFS) or effective duration < 0.03s
      - Out-of-phase stereo (rho_LR < -0.2)
    """
    reasons: list[str] = []
    needs_fade = False

    if len(mono_raw) == 0:
        return {"hard_reject": True, "reject_reasons": ["empty_audio"], "needs_fade": False, "phase_corr": 1.0}

    # 1. Clipping
    clip_mask = np.abs(y_stereo) >= 0.99
    c_max = max(_max_consecutive_true(clip_mask[:, 0]), _max_consecutive_true(clip_mask[:, 1]))
    clip_ratio = float(np.mean(clip_mask))
    if c_max >= 3 or clip_ratio > 0.001:
        reasons.append(f"digital_clipping(max_consec={c_max},ratio={clip_ratio:.4f})")

    # 2. Envelope boundary clicks (5 first / 5 last samples)
    n_edge = min(5, len(mono_raw))
    peak = float(np.max(np.abs(mono_raw))) + 1e-12
    edge_amp = max(float(np.max(np.abs(mono_raw[:n_edge]))), float(np.max(np.abs(mono_raw[-n_edge:]))))
    if edge_amp > 0.05 and (edge_amp / peak) > 0.05:
        needs_fade = True

    # 3. Silence / effective duration
    rms = float(np.sqrt(np.mean(mono_raw ** 2)))
    rms_db = 20.0 * math.log10(max(rms, 1e-12))
    nz = np.where(np.abs(mono_raw) > max(1e-4, peak * 0.01))[0]
    eff_dur = float((nz[-1] - nz[0] + 1) / sr) if len(nz) >= 2 else 0.0
    if rms_db < -55.0:
        reasons.append(f"signal_too_quiet(rms={rms_db:.1f}dBFS)")
    if eff_dur < 0.03:
        reasons.append(f"effective_duration_too_short({eff_dur:.3f}s)")

    # 4. Stereo phase correlation
    if is_mono_source or y_stereo.shape[1] < 2:
        phase_corr = 1.0
    else:
        l_ch = y_stereo[:, 0].astype(np.float64)
        r_ch = y_stereo[:, 1].astype(np.float64)
        denom = float(np.sqrt(np.sum(l_ch ** 2) * np.sum(r_ch ** 2))) + 1e-12
        phase_corr = float(np.sum(l_ch * r_ch) / denom)
        if phase_corr < -0.2:
            reasons.append(f"destructive_phase_inversion(rho={phase_corr:.3f})")

    return {
        "hard_reject": len(reasons) > 0,
        "reject_reasons": reasons,
        "needs_fade": needs_fade,
        "phase_corr": round(phase_corr, 4),
        "rms_db": round(rms_db, 2),
        "eff_dur_sec": round(eff_dur, 4),
    }


def _compute_plomp_levelt_roughness(
    mag_spec: np.ndarray,
    freqs: np.ndarray,
    max_frames: int = 80,
    top_peaks: int = 10,
) -> float:
    """Estimate normalized sensory roughness (Plomp-Levelt beating model) across frames."""
    n_frames = mag_spec.shape[1]
    if n_frames == 0:
        return 0.0
    step = max(1, n_frames // max_frames)
    frame_indices = range(0, n_frames, step)

    valid_mask = (freqs >= 40.0) & (freqs <= 12000.0)
    f_sub = freqs[valid_mask]
    m_sub = mag_spec[valid_mask, :]
    if len(f_sub) < 4:
        return 0.0

    rough_vals: list[float] = []
    for idx in frame_indices:
        col = m_sub[:, idx]
        e_tot = float(np.sum(col ** 2))
        if e_tot < 1e-10:
            continue
        p_idx, _ = signal.find_peaks(col, height=float(np.max(col) * 0.05), distance=2)
        if len(p_idx) < 2:
            rough_vals.append(0.0)
            continue
        if len(p_idx) > top_peaks:
            order = np.argsort(col[p_idx])[::-1][:top_peaks]
            p_idx = p_idx[order]
        pf = f_sub[p_idx]
        pa = col[p_idx]
        norm_e = float(np.sum(pa ** 2)) + 1e-12

        d_frame = 0.0
        n_p = len(pf)
        for i in range(n_p):
            for j in range(i + 1, n_p):
                f_min = min(pf[i], pf[j])
                df = abs(pf[j] - pf[i])
                s = 0.24 / (0.021 * f_min + 19.0)
                x = s * df
                d_frame += float(pa[i] * pa[j]) * (math.exp(-3.5 * x) - math.exp(-5.75 * x))
        rough_vals.append(max(0.0, d_frame / norm_e))

    return round(float(np.mean(rough_vals)) if rough_vals else 0.0, 4)


def _compute_low_mid_corr(y_stereo: np.ndarray, sr: int) -> float:
    """Compute 200-800 Hz stereo phase correlation to flag overly wide pads/samples."""
    if y_stereo.ndim < 2 or y_stereo.shape[1] < 2 or len(y_stereo) < 64:
        return 1.0
    try:
        sos = signal.butter(2, [200.0, min(800.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
        l_lm = signal.sosfilt(sos, y_stereo[:, 0].astype(np.float64))
        r_lm = signal.sosfilt(sos, y_stereo[:, 1].astype(np.float64))
        denom = float(np.sqrt(np.sum(l_lm ** 2) * np.sum(r_lm ** 2))) + 1e-12
        return round(float(np.sum(l_lm * r_lm) / denom), 4)
    except Exception:
        return 1.0


def analyze_pitch_and_purity(
    y_stereo: np.ndarray,
    mono_norm: np.ndarray,
    sr: int,
    bands: dict[str, float],
    r_harm: float = 0.5,
) -> dict[str, Any]:
    """
    Deterministic DSP pitch, stability, polyphony & harmonicity analyzer.
    Distinguishes true single-note `mono_oneshot` samples from `melodic_sequence`
    or `poly_chord_or_dyad` samples to prevent chromatic dissonance in gl.sampler.
    """
    low_mid_corr = _compute_low_mid_corr(y_stereo, sr)
    dur_s = len(mono_norm) / float(sr) if sr > 0 else 0.0
    if len(mono_norm) < 1024:
        return {
            "detected_note": None,
            "detected_f0_hz": None,
            "midi_root": None,
            "cents_offset": None,
            "recommended_fine_tune_ct": 0,
            "f0_std_semitones": 0.0,
            "onset_count": 1,
            "hnr_db": None,
            "polyphony_ratio": 0.0,
            "inharmonic_warnings": [],
            "sample_nature": "percussion",
            "playable_chromatically": False,
            "low_mid_corr": low_mid_corr,
        }

    is_low = (bands.get("sub", 0.0) + bands.get("bass", 0.0)) > 0.60
    is_perc_heavy = (
        ((bands.get("air", 0.0) + bands.get("harsh", 0.0)) > 0.72 and r_harm < 0.35)
        or (dur_s < 0.14 and (bands.get("air", 0.0) + bands.get("harsh", 0.0)) > 0.50)
    )
    fmin = 27.0 if is_low else 65.0
    fmax = 220.0 if is_low else 2400.0
    frame_len = 4096 if is_low else 2048

    sig = np.pad(mono_norm, (0, max(0, frame_len - len(mono_norm)))) if len(mono_norm) < frame_len else mono_norm

    # 1. Onset detection (calibrated delta so vibrato/tremolo on steady tones doesn't spam false note onsets)
    onset_times: list[float] = []
    try:
        o_env = librosa.onset.onset_strength(y=mono_norm, sr=sr, hop_length=512)
        o_frames = librosa.onset.onset_detect(
            onset_envelope=o_env, sr=sr, hop_length=512,
            units="frames", delta=0.22, wait=max(4, int(0.12 * sr / 512)), backtrack=False,
        )
        o_t = librosa.frames_to_time(o_frames, sr=sr, hop_length=512)
        onset_times = [round(float(t), 3) for t in o_t]
    except Exception:
        pass
    distinct_onsets = 0
    last_t = -1.0
    for t in onset_times:
        if t - last_t > 0.14:
            distinct_onsets += 1
            last_t = t
    onset_count = max(1, distinct_onsets) if len(onset_times) > 0 else 1

    # 2. Frame-by-frame YIN pitch & stability
    med_f0: float | None = None
    f0_std_st = 0.0
    detected_note: str | None = None
    midi_root: int | None = None
    cents_offset: float | None = None
    fine_tune_ct = 0
    try:
        f0 = librosa.yin(sig, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_len, hop_length=512)
        rms_frames = librosa.feature.rms(y=sig, frame_length=frame_len, hop_length=512)[0]
        if len(rms_frames) == len(f0):
            thresh = float(np.max(rms_frames)) * 0.14
            valid = f0[(rms_frames >= thresh) & (f0 > fmin * 1.03) & (f0 < fmax * 0.97)]
        else:
            valid = f0[(f0 > fmin * 1.03) & (f0 < fmax * 0.97)]

        if len(valid) >= 2:
            # Measure steady body (skip initial 85ms pitch-drop on 808/kick or 45ms on tonal instruments)
            skip_s = 0.085 if is_low else 0.045
            start_idx = min(len(valid) // 3, int(skip_s * sr / 512)) if dur_s > 0.25 else 0
            body_f0 = valid[start_idx:] if len(valid[start_idx:]) >= 2 else valid
            med_f0 = float(np.median(body_f0))
            midi_vals = 69.0 + 12.0 * np.log2(np.maximum(body_f0, 1e-6) / 440.0)
            f0_std_st = round(float(np.std(midi_vals)), 3)
            # Snap exact FFT peak around med_f0 for sub-cent precision
            n_fft_hi = 65536
            s_body = min(len(sig) - 2048, int(skip_s * sr)) if dur_s > 0.30 else 0
            win = sig[max(0, s_body) : min(len(sig), max(0, s_body) + sr * 2)]
            if len(win) < 512:
                win = sig
            sp = np.abs(np.fft.rfft(win * np.hanning(len(win)), n=n_fft_hi))
            fr = np.fft.rfftfreq(n_fft_hi, 1.0 / sr)
            lo_b, hi_b = med_f0 * 0.88, med_f0 * 1.12
            m_b = (fr >= lo_b) & (fr <= hi_b)
            if np.any(m_b):
                med_f0 = float(fr[m_b][np.argmax(sp[m_b])])
            midi_f = 69.0 + 12.0 * math.log2(med_f0 / 440.0)
            midi_root = int(round(midi_f))
            cents_offset = round((midi_f - midi_root) * 100.0, 1)
            fine_tune_ct = int(round(-cents_offset))
            detected_note = NOTE_NAMES[midi_root % 12]
    except Exception:
        pass

    # 3. Praat HNR (Harmonics-to-Noise Ratio; minimum_pitch=26 Hz for 808 sub support)
    hnr_db: float | None = None
    if _HAS_PARSELMOUTH and dur_s >= 0.10:
        try:
            snd = parselmouth.Sound(mono_norm.astype(np.float64), sampling_frequency=float(sr))
            min_p = 26.0 if is_low else 65.0
            harm = snd.to_harmonicity(minimum_pitch=min_p)
            vals = harm.values[harm.values > -190.0]
            if len(vals) > 0:
                hnr_db = round(float(np.nanmean(vals)), 2)
        except Exception:
            pass

    # 4. Polyphony / Chord detection via spectral peaks outside f0 harmonic series (avoids CQT semitone bleed)
    polyphony_ratio = 0.0
    try:
        if len(sig) >= 2048 and med_f0 is not None and med_f0 > 25.0:
            n_fft_p = 65536
            win = sig[: min(len(sig), sr * 3)]
            sp = np.abs(np.fft.rfft(win * np.hanning(len(win)), n=n_fft_p))
            fr = np.fft.rfftfreq(n_fft_p, 1.0 / sr)
            m_tonal = (fr >= 50.0) & (fr <= 4500.0)
            sp_t = sp[m_tonal]
            fr_t = fr[m_tonal]
            if len(sp_t) > 10:
                p_idx, _ = signal.find_peaks(sp_t, height=float(np.max(sp_t) * 0.15), distance=25)
                if len(p_idx) > 0:
                    dom_amp = float(np.max(sp_t[p_idx])) + 1e-12
                    non_harm_amps = []
                    for idx in p_idx:
                        pf = float(fr_t[idx])
                        # Check if pf is near k*f0 (or f0/2 sub-octave) within +-55 cents
                        ratio = pf / med_f0
                        nearest_k = max(1, round(ratio))
                        is_harm = (abs(1200.0 * math.log2(pf / (nearest_k * med_f0))) <= 55.0) or (
                            abs(1200.0 * math.log2(pf / (0.5 * med_f0))) <= 55.0
                        )
                        if not is_harm:
                            non_harm_amps.append(float(sp_t[idx]) / dom_amp)
                    if non_harm_amps:
                        polyphony_ratio = round(min(1.0, max(non_harm_amps)), 3)
    except Exception:
        pass

    # 5. Harmonicity check (2*f0, 3*f0) for bass_sub / tonal samples
    inharmonic_warnings: list[str] = []
    if med_f0 is not None and med_f0 > 25.0 and len(sig) >= 2048:
        try:
            n_fft_h = 65536
            win = sig[: min(len(sig), sr * 2)]
            sp = np.abs(np.fft.rfft(win * np.hanning(len(win)), n=n_fft_h))
            fr = np.fft.rfftfreq(n_fft_h, 1.0 / sr)
            a1 = float(np.max(sp[(fr >= med_f0 * 0.94) & (fr <= med_f0 * 1.06)])) + 1e-12
            for mult in (2, 3):
                lo, hi = med_f0 * mult * 0.85, med_f0 * mult * 1.15
                mh = (fr >= lo) & (fr <= hi)
                if np.any(mh):
                    fh = float(fr[mh][np.argmax(sp[mh])])
                    ah = float(np.max(sp[mh]))
                    if ah / a1 > 0.12:
                        err_ct = round(1200.0 * math.log2(fh / (med_f0 * mult)), 1)
                        if abs(err_ct) > 45.0:
                            inharmonic_warnings.append(f"h{mult}_deviates_{err_ct:+.0f}ct")
        except Exception:
            pass

    # 6. Classify sample_nature & playable_chromatically
    # Notice: if f0_std_st is super stable (<= 0.25 st) and polyphony_ratio < 0.42, extra onsets from slow reverb/tremolo are NOT a melodic sequence
    is_melodic_seq = (f0_std_st > 0.75 and not is_low) or (onset_count >= 3 and f0_std_st > 0.35)
    if is_perc_heavy or (not is_low and r_harm < 0.30 and (hnr_db is None or hnr_db < 4.5)):
        sample_nature = "percussion"
        playable = False
    elif is_melodic_seq:
        sample_nature = "melodic_sequence"
        playable = False
    elif polyphony_ratio >= 0.45:
        sample_nature = "poly_chord_or_dyad"
        playable = False
    elif inharmonic_warnings and not is_low:
        sample_nature = "inharmonic_bell_or_fx"
        playable = False
    elif dur_s >= 5.0 and f0_std_st > 0.40:
        sample_nature = "ambient_bed"
        playable = False
    elif med_f0 is not None and (f0_std_st <= 0.65 or is_low) and polyphony_ratio < 0.45:
        sample_nature = "mono_oneshot"
        playable = True
    else:
        sample_nature = "ambient_bed" if dur_s >= 4.0 else "percussion"
        playable = False

    return {
        "detected_note": detected_note,
        "detected_f0_hz": round(med_f0, 2) if med_f0 is not None else None,
        "midi_root": midi_root,
        "cents_offset": cents_offset,
        "recommended_fine_tune_ct": fine_tune_ct,
        "f0_std_semitones": f0_std_st,
        "onset_count": onset_count,
        "hnr_db": hnr_db,
        "polyphony_ratio": polyphony_ratio,
        "inharmonic_warnings": inharmonic_warnings,
        "sample_nature": sample_nature,
        "playable_chromatically": playable,
        "low_mid_corr": low_mid_corr,
    }


def analyze_vocal_signature(filepath: str) -> dict[str, Any]:
    """
    Deterministic DSP vocal & timbre analyzer (CPU-only, zero ML required):
    Uses librosa + praat-parselmouth (HNR, F0, Formants F1-F3, Jitter, Shimmer) + optional sox.
    """
    abs_p = os.path.abspath(os.path.expanduser(filepath))
    y, sr = librosa.load(abs_p, sr=None, mono=True)
    dur_total = float(librosa.get_duration(y=y, sr=sr)) if len(y) else 0.0
    if dur_total <= 0.01:
        return {"filepath": abs_p, "duree_totale_s": 0.0, "est_voix_probable": False, "style_vocal": "non_vocal"}

    intervals = librosa.effects.split(y, top_db=30)
    dur_active = float(sum((end - start) / sr for start, end in intervals))
    ratio_presence = dur_active / dur_total if dur_total > 0 else 0.0

    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onsets = librosa.onset.onset_detect(
        onset_envelope=o_env, sr=sr, hop_length=512,
        units="time", delta=0.22, wait=max(4, int(0.12 * sr / 512)),
    )
    densite_syllabique = (len(onsets) / dur_active) if dur_active > 0.05 else 0.0

    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=y)[0]))
    centroid_hz = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))

    hnr_mean = 0.0
    f0_med = 0.0
    f0_std_hz = 0.0
    formants_hz: list[float] = []
    jitter_pct = 0.0
    shimmer_pct = 0.0
    longest_voiced_s = 0.0

    if _HAS_PARSELMOUTH:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=float(sr))
        harm = snd.to_harmonicity()
        valid_h = harm.values[harm.values > -190.0]
        if len(valid_h) > 0:
            hnr_mean = float(np.nanmean(valid_h))

        pitch = snd.to_pitch()
        f0_vals = pitch.selected_array["frequency"]
        dt = float(pitch.dx) if hasattr(pitch, "dx") else 0.01
        f0_voiced = f0_vals[f0_vals > 0]
        if len(f0_voiced) > 0:
            f0_med = float(np.median(f0_voiced))
            f0_std_hz = float(np.std(np.diff(f0_voiced))) if len(f0_voiced) > 1 else 0.0
            # Longest contiguous voiced segment
            run = 0
            best_run = 0
            for v in f0_vals:
                if v > 0:
                    run += 1
                    best_run = max(best_run, run)
                else:
                    run = 0
            longest_voiced_s = best_run * dt

        try:
            formant_obj = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=5500.0)
            t_mid = dur_total / 2.0
            for idx in (1, 2, 3):
                fv = float(formant_obj.get_value_at_time(idx, t_mid))
                if not math.isnan(fv) and fv > 0:
                    formants_hz.append(round(fv, 1))
        except Exception:
            pass

        try:
            pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
            jit = float(parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            shm = float(parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            if not math.isnan(jit):
                jitter_pct = round(jit * 100.0, 3)
            if not math.isnan(shm):
                shimmer_pct = round(shm * 100.0, 3)
        except Exception:
            pass
    else:
        f0 = librosa.yin(y, fmin=70, fmax=550, sr=sr)
        voiced = f0[(f0 > 75) & (f0 < 520)]
        if len(voiced) > 0:
            f0_med = float(np.median(voiced))
            f0_std_hz = float(np.std(np.diff(voiced))) if len(voiced) > 1 else 0.0

    est_voix = bool(80.0 <= f0_med <= 450.0 and hnr_mean > 7.0)
    if not est_voix:
        style_vocal = "non_vocal"
    elif densite_syllabique < 2.0 and (longest_voiced_s >= 0.40 or ratio_presence >= 0.65):
        style_vocal = "chant_lent_tenu"
    elif densite_syllabique < 3.5:
        style_vocal = "chant_rythme"
    else:
        style_vocal = "parole_rapide"

    sox_stats: dict[str, Any] | None = None
    if shutil.which("sox"):
        try:
            proc = subprocess.run(["sox", abs_p, "-n", "stat"], capture_output=True, text=True, timeout=5)
            sox_stats = {"raw_lines": [ln.strip() for ln in proc.stderr.splitlines()[:8] if ln.strip()]}
        except Exception:
            pass

    out = {
        "filepath": abs_p,
        "duree_totale_s": round(dur_total, 2),
        "duree_active_s": round(dur_active, 2),
        "ratio_presence": round(ratio_presence, 2),
        "est_voix_probable": est_voix,
        "style_vocal": style_vocal,
        "hnr_db": round(hnr_mean, 2),
        "frequence_moyenne_hz": round(f0_med, 1),
        "attaques_par_seconde": round(densite_syllabique, 2),
        "instabilite_pitch_hz": round(f0_std_hz, 2),
        "longest_voiced_s": round(longest_voiced_s, 2),
        "formants_hz": formants_hz,
        "jitter_pct": jitter_pct,
        "shimmer_pct": shimmer_pct,
        "zcr_mean": round(zcr_mean, 4),
        "spectral_centroid_hz": round(centroid_hz, 1),
    }
    if sox_stats:
        out["sox_stat"] = sox_stats
    return out


def extract_dsp_metrics(
    y_stereo: np.ndarray,
    mono_norm: np.ndarray,
    sr: int = 44100,
    phase_corr: float = 1.0,
) -> dict[str, Any]:
    """Extract all Layer 1 comfort, envelope, harmonic, 6-band, 24-Bark & pitch metrics."""
    n_fft = 2048
    hop = 512
    if len(mono_norm) < n_fft:
        sig = np.pad(mono_norm, (0, n_fft - len(mono_norm)))
    else:
        sig = mono_norm

    stft = librosa.stft(sig, n_fft=n_fft, hop_length=hop)
    mag = np.abs(stft)
    power = mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    mean_power = np.mean(power, axis=1)
    total_e = float(np.sum(mean_power[(freqs >= 20.0) & (freqs <= 20000.0)])) + 1e-12

    # 6 bands
    bands: dict[str, float] = {}
    for bname, (f_lo, f_hi) in SIX_BANDS_HZ.items():
        m = (freqs >= f_lo) & (freqs < f_hi)
        bands[bname] = round(float(np.sum(mean_power[m])) / total_e, 4)

    # 24 Bark critical bands (normalized sum == 1)
    bark_raw = np.zeros(len(BARK_EDGES_HZ), dtype=np.float64)
    for i, (f_lo, f_hi) in enumerate(BARK_EDGES_HZ):
        m = (freqs >= f_lo) & (freqs < min(f_hi, sr * 0.495))
        if np.any(m):
            bark_raw[i] = float(np.sum(mean_power[m]))
        else:
            # Interpolate nearest bin if narrow low Bark band has 0 FFT bins
            nearest = int(np.argmin(np.abs(freqs - 0.5 * (f_lo + f_hi))))
            bark_raw[i] = float(mean_power[nearest]) * 0.25
    bark_sum = float(np.sum(bark_raw)) + 1e-12
    bark_24 = [round(float(x / bark_sum), 6) for x in bark_raw]

    # Harshness index & resonant peak in 2000-5000 Hz
    r_harsh = bands["harsh"]
    m_harsh = (freqs >= 2000.0) & (freqs <= 5000.0)
    harsh_spec = mean_power[m_harsh]
    if len(harsh_spec):
        med_h = float(np.median(harsh_spec)) + 1e-12
        max_h = float(np.max(harsh_spec)) + 1e-12
        peak_harsh_db = round(10.0 * math.log10(max_h / med_h), 2)
    else:
        peak_harsh_db = 0.0

    # Sensory roughness
    roughness = _compute_plomp_levelt_roughness(mag, freqs)

    # Spectral Flatness Measure (SFM)
    sfm_arr = librosa.feature.spectral_flatness(S=power + 1e-12)[0]
    sfm = round(float(np.mean(sfm_arr)), 4)

    # HPSS harmonic ratio R_harm
    if len(mono_norm) >= int(0.12 * sr):
        y_harm, y_perc = librosa.effects.hpss(mono_norm)
        eh = float(np.sum(y_harm ** 2))
        ep = float(np.sum(y_perc ** 2))
        r_harm = round(eh / (eh + ep + 1e-12), 4)
    else:
        # Ultra-short percussive transient fallback via SFM/tonality
        r_harm = round(max(0.0, min(1.0, 1.0 - sfm * 2.5)) * 0.45, 4)

    # Crest factor (dB)
    pk = float(np.max(np.abs(mono_norm))) + 1e-12
    rms = float(np.sqrt(np.mean(mono_norm ** 2))) + 1e-12
    crest_factor_db = round(20.0 * math.log10(pk / rms), 2)

    # Attack time (ms) from -40 dB crossing to peak RMS envelope
    hop_env = max(32, int(sr * 0.001))  # 1 ms resolution
    win_env = max(64, int(sr * 0.004))
    env = librosa.feature.rms(y=sig, frame_length=win_env, hop_length=hop_env)[0]
    if len(env) >= 2 and np.max(env) > 1e-9:
        env_pk_idx = int(np.argmax(env))
        thresh_40 = float(env[env_pk_idx]) * 0.01  # -40 dB
        above = np.where(env[: env_pk_idx + 1] >= thresh_40)[0]
        start_idx = int(above[0]) if len(above) else 0
        attack_ms = round(float((env_pk_idx - start_idx) * hop_env / sr * 1000.0), 2)
    else:
        attack_ms = 0.0

    # Spectral flux (L2 difference of normalized magnitude frames)
    if mag.shape[1] >= 2:
        col_norms = np.linalg.norm(mag, axis=0, keepdims=True) + 1e-12
        mag_n = mag / col_norms
        diff_m = np.diff(mag_n, axis=1)
        spectral_flux = round(float(np.mean(np.linalg.norm(diff_m, axis=0))), 4)
    else:
        spectral_flux = 0.0

    pitch_purity = analyze_pitch_and_purity(y_stereo, mono_norm, sr, bands, r_harm=r_harm)

    return {
        "phase_corr": round(float(phase_corr), 4),
        "low_mid_corr": pitch_purity["low_mid_corr"],
        "crest_factor_db": crest_factor_db,
        "attack_ms": attack_ms,
        "r_harsh": round(r_harsh, 4),
        "peak_harsh_db": peak_harsh_db,
        "roughness": roughness,
        "sfm": sfm,
        "r_harm": r_harm,
        "spectral_flux": spectral_flux,
        "bands": bands,
        "bark_24": bark_24,
        "detected_note": pitch_purity["detected_note"],
        "detected_f0_hz": pitch_purity["detected_f0_hz"],
        "midi_root": pitch_purity["midi_root"],
        "cents_offset": pitch_purity["cents_offset"],
        "recommended_fine_tune_ct": pitch_purity["recommended_fine_tune_ct"],
        "f0_std_semitones": pitch_purity["f0_std_semitones"],
        "onset_count": pitch_purity["onset_count"],
        "hnr_db": pitch_purity["hnr_db"],
        "polyphony_ratio": pitch_purity["polyphony_ratio"],
        "inharmonic_warnings": pitch_purity["inharmonic_warnings"],
        "sample_nature": pitch_purity["sample_nature"],
        "playable_chromatically": pitch_purity["playable_chromatically"],
    }


def compute_comfort_score(
    dsp: dict[str, Any],
    needs_fade: bool = False,
    role: str | None = None,
) -> float:
    """Compute Auditory Comfort Score S_comfort in [0, 1] (Section 3.3)."""
    score = 1.0
    tau = HARSH_THRESHOLD_BY_ROLE.get(role or "", 0.30)

    r_harsh = float(dsp.get("r_harsh", 0.0))
    if r_harsh > tau:
        score -= min(0.45, ((r_harsh - tau) / max(0.15, 1.0 - tau)) * 1.1)

    peak_harsh_db = float(dsp.get("peak_harsh_db", 0.0))
    if peak_harsh_db > 10.0:
        score -= min(0.35, (peak_harsh_db - 10.0) * 0.04)

    # Roughness penalty (relaxed for hihat_perc metallic cymbals)
    rough = float(dsp.get("roughness", 0.0))
    rough_thresh = 0.38 if role == "hihat_perc" else 0.18
    if rough > rough_thresh:
        score -= min(0.30, (rough - rough_thresh) * 0.9)

    if needs_fade:
        score -= 0.08

    return round(float(max(0.0, min(1.0, score))), 4)


def _range_fit(val: float, lo: float | None, hi: float | None, margin: float) -> float:
    """1.0 inside [lo, hi], linear decay to 0.0 over `margin` outside."""
    if lo is not None and val < lo:
        return max(0.0, 1.0 - (lo - val) / max(margin, 1e-6))
    if hi is not None and val > hi:
        return max(0.0, 1.0 - (val - hi) / max(margin, 1e-6))
    return 1.0


def compute_role_score(dsp: dict[str, Any], duration_sec: float, role_id: str) -> float:
    """
    Compute Layer 2 Role Fitness Score S_role in [0, 1] per Section 4 table.
    Uses dominant frequency-band & harmonicity checks as multiplicative gates so a
    snare/hat can never score high for kick/bass (or vice versa).
    """
    b = dsp["bands"]
    att = float(dsp["attack_ms"])
    cf = float(dsp["crest_factor_db"])
    rh = float(dsp["r_harm"])
    sfm = float(dsp["sfm"])
    rho = float(dsp["phase_corr"])
    harsh = float(dsp["r_harsh"])
    pk_h = float(dsp["peak_harsh_db"])
    flux = float(dsp["spectral_flux"])

    if role_id == "kick":
        gate = _range_fit(b["sub"] + b["bass"], 0.65, None, 0.25) * _range_fit(duration_sec, None, 1.0, 0.6)
        other = [
            _range_fit(att, 0.0, 15.0, 20.0),
            _range_fit(cf, 10.0, 22.0, 6.0),
            _range_fit(rh, 0.10, 0.50, 0.30),
            _range_fit(sfm, None, 0.15, 0.20),
            _range_fit(rho, 0.85, None, 0.35),
        ]
    elif role_id == "bass_sub":
        gate = _range_fit(b["sub"] + b["bass"], 0.70, None, 0.25) * _range_fit(rh, 0.75, None, 0.30)
        other = [
            _range_fit(att, 5.0, 80.0, 60.0),
            _range_fit(cf, 3.0, 12.0, 6.0),
            _range_fit(sfm, None, 0.05, 0.10),
            _range_fit(rho, 0.80, None, 0.35),
            _range_fit(harsh, None, 0.10, 0.15),
        ]
    elif role_id == "snare_clap":
        gate = (
            _range_fit(b["lowmid"] + b["mid"] + b["harsh"], 0.60, None, 0.25)
            * _range_fit(b["sub"], None, 0.05, 0.10)
            * _range_fit(duration_sec, None, 1.2, 0.8)
        )
        other = [
            _range_fit(att, 0.0, 20.0, 25.0),
            _range_fit(cf, 12.0, 24.0, 6.0),
            _range_fit(rh, None, 0.45, 0.35),
            _range_fit(sfm, 0.05, 0.45, 0.20),
        ]
    elif role_id == "hihat_perc":
        gate = _range_fit(b["air"] + b["harsh"], 0.75, None, 0.25) * _range_fit(b["sub"] + b["bass"], None, 0.05, 0.10)
        other = [
            _range_fit(att, 0.0, 15.0, 20.0),
            _range_fit(cf, 10.0, 25.0, 6.0),
            _range_fit(rh, None, 0.30, 0.35),
            _range_fit(sfm, 0.15, None, 0.12),
        ]
    elif role_id == "lead_melody":
        gate = _range_fit(b["lowmid"] + b["mid"] + b["harsh"], 0.65, None, 0.25) * _range_fit(rh, 0.70, None, 0.30)
        other = [
            _range_fit(att, 5.0, 100.0, 80.0),
            _range_fit(cf, 4.0, 14.0, 6.0),
            _range_fit(sfm, None, 0.08, 0.12),
            _range_fit(harsh, None, 0.32, 0.18),
            _range_fit(pk_h, None, 12.0, 6.0),
        ]
    elif role_id == "pad_atmosphere":
        gate = (
            _range_fit(b["lowmid"] + b["mid"], 0.50, None, 0.25)
            * _range_fit(rh, 0.80, None, 0.30)
            * _range_fit(b["sub"], None, 0.10, 0.12)
        )
        other = [
            _range_fit(att, 120.0, None, 100.0),
            _range_fit(cf, 3.0, 10.0, 5.0),
            _range_fit(sfm, None, 0.10, 0.12),
            _range_fit(flux, None, 0.28, 0.25),
            _range_fit(harsh, None, 0.25, 0.18),
        ]
    elif role_id == "chord_keys":
        gate = _range_fit(b["bass"] + b["lowmid"] + b["mid"], 0.70, None, 0.25) * _range_fit(rh, 0.75, None, 0.30)
        other = [
            _range_fit(att, 5.0, 60.0, 60.0),
            _range_fit(cf, 6.0, 16.0, 6.0),
            _range_fit(sfm, None, 0.08, 0.12),
            _range_fit(harsh, None, 0.28, 0.18),
        ]
    else:
        return 0.5

    other_mean = float(np.mean(other)) if other else 1.0
    score = gate * (0.45 + 0.55 * other_mean)
    return round(float(max(0.0, min(1.0, score))), 4)


class ClapEmbedder:
    """Lazy LAION-CLAP wrapper for audio embeddings & zero-shot aesthetic scoring."""

    def __init__(self, device: str = "cpu", model_id: str = "laion/clap-htsat-unfused"):
        self.requested_device = device
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._torch = None
        self._device = "cpu"
        self._pos_prompts: np.ndarray | None = None
        self._neg_prompts: np.ndarray | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import ClapModel, ClapProcessor

        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        self._torch = torch
        if self.requested_device == "cuda" and torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        try:
            self._processor = ClapProcessor.from_pretrained(self.model_id, local_files_only=True)
            self._model = ClapModel.from_pretrained(self.model_id, local_files_only=True)
        except Exception:
            self._processor = ClapProcessor.from_pretrained(self.model_id)
            self._model = ClapModel.from_pretrained(self.model_id)

        self._model.to(self._device)
        self._model.eval()
        self._init_text_centroids()

    def _init_text_centroids(self) -> None:
        assert self._torch is not None and self._model is not None and self._processor is not None
        torch = self._torch
        all_txt = POSITIVE_CLAP_PROMPTS + NEGATIVE_CLAP_PROMPTS
        inputs = self._processor(text=all_txt, return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
            feats = getattr(out, "pooler_output", out) if not isinstance(out, torch.Tensor) else out
            feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = feats.cpu().numpy().astype(np.float32)
        self._pos_prompts = arr[: len(POSITIVE_CLAP_PROMPTS)]
        self._neg_prompts = arr[len(POSITIVE_CLAP_PROMPTS) :]

    def embed_audio(self, mono: np.ndarray, sr: int = 44100) -> np.ndarray:
        self._ensure_loaded()
        assert self._torch is not None and self._model is not None and self._processor is not None
        torch = self._torch
        # CLAP expects 48 kHz mono audio
        if sr != 48000:
            y48 = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=48000)
        else:
            y48 = mono.astype(np.float32)
        # Keep up to 10s for speed & consistency
        if len(y48) > 48000 * 10:
            y48 = y48[: 48000 * 10]
        elif len(y48) < 4800:
            y48 = np.pad(y48, (0, 4800 - len(y48)))

        inputs = self._processor(audio=y48, sampling_rate=48000, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.get_audio_features(**inputs)
            feats = getattr(out, "pooler_output", out) if not isinstance(out, torch.Tensor) else out
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

    def zero_shot_score(self, audio_vec: np.ndarray) -> float:
        self._ensure_loaded()
        assert self._pos_prompts is not None and self._neg_prompts is not None
        sim_pos = float(np.mean(self._pos_prompts @ audio_vec))
        sim_neg = float(np.mean(self._neg_prompts @ audio_vec))
        diff = (sim_pos - sim_neg) / 0.05
        diff = max(-30.0, min(30.0, diff))
        return round(1.0 / (1.0 + math.exp(-diff)), 4)


def zero_shot_from_vec(audio_vec: np.ndarray, embedder: ClapEmbedder) -> float:
    return embedder.zero_shot_score(audio_vec)


def reference_aesthetic_score(audio_vec: np.ndarray, ref_matrix: np.ndarray | None) -> float | None:
    """Compute top-3 cosine similarity S_ref against golden references for a role."""
    if ref_matrix is None or len(ref_matrix) == 0:
        return None
    sims = np.sort(ref_matrix @ audio_vec)[::-1]
    top3_mean = float(np.mean(sims[: min(3, len(sims))]))
    s_ref = (top3_mean - 0.30) / 0.60
    return round(float(max(0.0, min(1.0, s_ref))), 4)


def compute_masking_penalty(bark_cand: list[float], bark_mix: list[float] | None) -> float:
    """Bhattacharyya spectral overlap P_mask in [0, 1] (Section 6.1)."""
    if not bark_mix or len(bark_cand) != len(bark_mix):
        return 0.0
    a = np.maximum(np.asarray(bark_cand, dtype=np.float64), 0.0)
    b = np.maximum(np.asarray(bark_mix, dtype=np.float64), 0.0)
    a /= float(np.sum(a)) + 1e-12
    b /= float(np.sum(b)) + 1e-12
    overlap = float(np.sum(np.sqrt(a * b)))
    return round(float(max(0.0, min(1.0, overlap))), 4)


def parse_key_tonic_pc(target_key: str | None) -> int | None:
    if not target_key or not target_key.strip():
        return None
    m = re.match(r"^\s*([A-G](?:#|b)?)", target_key.strip(), re.IGNORECASE)
    if not m:
        return None
    root = m.group(1)[0].upper() + (m.group(1)[1:] if len(m.group(1)) > 1 else "")
    return NOTE_TO_PC.get(root)


def compute_pitch_shift(detected_note: str | None, target_key: str | None) -> dict[str, Any]:
    """Compute minimal semitone shift [-6, +6] from detected_note to target_key tonic."""
    target_pc = parse_key_tonic_pc(target_key)
    det_pc = NOTE_TO_PC.get(detected_note) if detected_note else None
    if target_pc is None or det_pc is None:
        return {"detected_note": detected_note, "recommended_semitone_shift": 0, "shift_warning": None}
    diff = (target_pc - det_pc) % 12
    if diff > 6:
        diff -= 12
    warn = f"Extreme transposition ({diff:+d} st > +-5 st) may degrade timbre" if abs(diff) > 5 else None
    return {
        "detected_note": detected_note,
        "recommended_semitone_shift": int(diff),
        "shift_warning": warn,
    }


def score_candidate(
    entry: dict[str, Any],
    role_id: str,
    clap_vec: np.ndarray | None = None,
    embedder: ClapEmbedder | None = None,
    ref_matrix: np.ndarray | None = None,
    mix_bark: list[float] | None = None,
    target_key: str | None = None,
    monophonic_only: bool = False,
) -> dict[str, Any]:
    """Run Layers 1-4 and compute final composite score S_final per Section 7."""
    dsp = entry["dsp"]
    warnings: list[str] = []

    if entry.get("needs_fade"):
        warnings.append("needs_fade: apply 5ms linear fade-in/out on load")
    if role_id in ("kick", "bass_sub") and float(dsp.get("phase_corr", 1.0)) < 0.70:
        return {
            "rejected": True,
            "reject_reason": f"low_phase_corr_for_{role_id}({dsp.get('phase_corr')}<0.70)",
        }

    sample_nature = dsp.get("sample_nature", "unknown")
    playable = bool(dsp.get("playable_chromatically", True))
    if monophonic_only and not playable:
        return {
            "rejected": True,
            "reject_reason": f"not_monophonic_oneshot(nature={sample_nature},onsets={dsp.get('onset_count')},std_st={dsp.get('f0_std_semitones')})",
        }

    s_comfort = compute_comfort_score(dsp, needs_fade=bool(entry.get("needs_fade")), role=role_id)
    s_role = compute_role_score(dsp, float(entry.get("duration_sec", 0.0)), role_id)

    s_zs = 0.5
    if clap_vec is not None and embedder is not None:
        s_zs = embedder.zero_shot_score(clap_vec)
    s_ref = reference_aesthetic_score(clap_vec, ref_matrix) if clap_vec is not None else None
    s_aesth = round(0.7 * s_ref + 0.3 * s_zs, 4) if s_ref is not None else s_zs

    p_mask = compute_masking_penalty(dsp.get("bark_24", []), mix_bark)
    if p_mask > 0.65:
        warnings.append(f"high_spectral_masking({p_mask:.2f}): consider EQ carve/HPF or candidate #2")

    is_perc = role_id in ("kick", "snare_clap", "hihat_perc")
    if is_perc:
        w1, w2, w3, w4 = 0.20, 0.40, 0.40, 0.05
    else:
        w1, w2, w3, w4 = 0.25, 0.30, 0.45, 0.20

    penalty = 0.0
    if role_id in ("lead_melody", "chord_keys", "bass_sub") and not playable:
        penalty = 0.28
        warnings.append(
            f"DISSONANCE_RISK: sample is '{sample_nature}' (onsets={dsp.get('onset_count')}, "
            f"f0_std={dsp.get('f0_std_semitones')}st, poly={dsp.get('polyphony_ratio')}); "
            "do NOT play chromatically across MIDI notes without extracting a single mono note!"
        )
    if dsp.get("inharmonic_warnings") and role_id in ("bass_sub", "lead_melody", "chord_keys"):
        warnings.append(f"inharmonic_partials: {dsp.get('inharmonic_warnings')}")
    if not is_perc and float(dsp.get("low_mid_corr", 1.0)) < 0.50:
        warnings.append(f"wide_low_mid_phase(corr={dsp.get('low_mid_corr')}): apply M/S <650Hz stabilization")

    s_final = round(float(max(0.0, min(1.0, w1 * s_comfort + w2 * s_role + w3 * s_aesth - w4 * p_mask - penalty))), 4)

    if s_final >= 0.70:
        tier = "excellent"
    elif s_final >= 0.55:
        tier = "acceptable"
    else:
        tier = "rejected"

    pitch_info = compute_pitch_shift(dsp.get("detected_note"), target_key) if not is_perc else {
        "detected_note": dsp.get("detected_note"),
        "recommended_semitone_shift": 0,
    }
    pitch_info.update({
        "f0_hz": dsp.get("detected_f0_hz"),
        "midi_root": dsp.get("midi_root"),
        "cents_offset": dsp.get("cents_offset"),
        "recommended_fine_tune_ct": dsp.get("recommended_fine_tune_ct"),
        "sample_nature": sample_nature,
        "playable_chromatically": playable,
        "onset_count": dsp.get("onset_count"),
        "f0_std_semitones": dsp.get("f0_std_semitones"),
        "hnr_db": dsp.get("hnr_db"),
        "low_mid_corr": dsp.get("low_mid_corr"),
    })
    if pitch_info.get("shift_warning"):
        warnings.append(pitch_info["shift_warning"])
        pitch_info.pop("shift_warning", None)
    else:
        pitch_info.pop("shift_warning", None)

    return {
        "rejected": False,
        "filepath": entry["filepath"],
        "duration_sec": entry.get("duration_sec"),
        "needs_fade": bool(entry.get("needs_fade")),
        "final_score": s_final,
        "tier": tier,
        "scores": {
            "comfort": s_comfort,
            "role_fit": s_role,
            "aesthetic": s_aesth,
            "masking_penalty": p_mask,
        },
        "pitch_info": pitch_info,
        "warnings": warnings,
    }


def analyze_single_file(
    filepath: str,
    embedder: ClapEmbedder | None = None,
    with_clap: bool = True,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Analyze one audio file and return (index_entry_dict, clap_vec_or_None)."""
    loaded = load_and_normalize_audio(filepath)
    hard = check_hard_rejects(
        loaded["y_stereo"],
        loaded["mono_raw_44k"],
        loaded["sr"],
        is_mono_source=loaded["is_mono_source"],
    )
    dsp = extract_dsp_metrics(
        loaded["y_stereo"],
        loaded["mono_norm"],
        sr=loaded["sr"],
        phase_corr=hard["phase_corr"],
    )
    vec: np.ndarray | None = None
    if with_clap and not hard["hard_reject"]:
        emb = embedder or ClapEmbedder(device="cpu")
        vec = emb.embed_audio(loaded["mono_norm"], sr=loaded["sr"])

    st = os.stat(loaded["filepath"])
    entry = {
        "id": file_md5(loaded["filepath"])[:12],
        "filepath": loaded["filepath"],
        "file_mtime": st.st_mtime,
        "file_size": st.st_size,
        "duration_sec": loaded["duration_sec"],
        "sample_rate": loaded["sr_orig"],
        "hard_reject": hard["hard_reject"],
        "reject_reasons": hard["reject_reasons"],
        "needs_fade": hard["needs_fade"],
        "dsp": dsp,
        "clap_index": None,
    }
    return entry, vec
