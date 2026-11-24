#!/usr/bin/env python3
"""
Unit Track Sound Check in Played Context (`soundcheck_lib.py`, RFC §3).

Never validates a sample or `gl.*` synth in isolation on raw WAV alone:
builds a minimal temporary 1-channel `.gridline` in `<proj>/.tmp_soundcheck/`,
plays a 2-4 bar role-authentic phrase in the blueprint's key & BPM, renders
`<proj>/renders/soundcheck/<track>.wav` (< 1.5s CPU), audits all 6 acoustic
checks, writes full raw telemetry to `<proj>/renders/diagnostics/`, cleans up
temporary files, and returns a concise LLM-friendly JSON verdict via
`ListenerBackend`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import librosa
import numpy as np
import soundfile as sf

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
ANALYZER_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-analyzer")
INDEX_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-index")
for _d in (SCRIPTS_DIR, ANALYZER_DIR, INDEX_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from artifact_cache import compute_e1_track_hash, get_cached_verdict, set_cached_verdict
from audio_metrics import analyze_single_file, extract_dsp_metrics, key_fit_penalty, load_and_normalize_audio, parse_key_tonic_pc
from deep_diagnostics import (
    _scan_glitches_and_crackle,
    _third_octave_profile,
    analyze_audio_pocket_alignment,
    analyze_sustained_spectral_flux,
)
from gl_dsl import SYNTH_REGISTRY, _default_cutoff_hz, eq_band, eq_params, synth_voice
from gain_staging_lib import compute_gain_staging_plan, measure_active_loudness
from groove_lib import apply_anti_machine_gun
from indexer import build_timbre_card_from_dsp
from listener_backend import ListenerBackend, get_listener_backend
from musicality_metrics import optimize_voice_leading
from song_builder import SCALES, ChannelSpec, degree_to_midi
from studio_lib import TEMPLATE_EMPTY_MIXER, GridlineMCP

TONAL_ROLES = {"bass_sub", "chord_keys", "pad_atmosphere", "lead_melody"}
PERC_ROLES = {"kick", "snare_clap", "hihat_perc"}


def _safe_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name.strip()).strip("_") or "track"


def _parse_key_and_mode(key_str: str) -> tuple[int, str]:
    pc = parse_key_tonic_pc(key_str)
    tonic_pc = 0 if pc is None else int(pc)
    low = (key_str or "C minor").lower()
    for mode in sorted(SCALES.keys(), key=len, reverse=True):
        if mode.replace("_", " ") in low or mode in low:
            return tonic_pc, mode
    if "maj" in low:
        return tonic_pc, "major"
    return tonic_pc, "minor"


def _build_role_test_pattern(
    role: str,
    *,
    bpm: float = 132.0,
    beats_per_bar: int = 4,
    key: str = "C minor",
    valence: float = 0.0,
    blueprint: dict[str, Any] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Return `(bars, notes)` for a realistic 2-4 bar played sound check."""
    tonic_pc, mode = _parse_key_and_mode(key)
    hc = (blueprint or {}).get("harmonic_core") or {}
    bpb = max(2, int(beats_per_bar))

    if role == "bass_sub":
        bars = 4
        prog = hc.get("chord_progression") or []
        pitches: list[int] = []
        if prog:
            for ch in prog[:4]:
                p = int(ch.get("bass_midi", 36 + tonic_pc))
                while p < 36:
                    p += 12
                while p > 45:
                    p -= 12
                pitches.append(p)
        if not pitches:
            base = 36 + tonic_pc
            if base > 43:
                base -= 12
            if base < 36:
                base += 12
            pitches = [base, degree_to_midi(base, mode, 5) - (12 if degree_to_midi(base, mode, 5) > 45 else 0),
                       degree_to_midi(base, mode, 3) - (12 if degree_to_midi(base, mode, 3) > 45 else 0),
                       degree_to_midi(base, mode, 4) - (12 if degree_to_midi(base, mode, 4) > 45 else 0)]
        notes = []
        for idx in range(min(3, len(pitches))):
            st = float(idx * bpb)
            notes.append({"start": st, "length": float(bpb) - 0.5, "pitch": int(pitches[idx]), "velocity": 96})
        # Add 4th bar resolution
        notes.append({"start": float(3 * bpb), "length": float(bpb) - 0.5, "pitch": int(pitches[0]), "velocity": 92})
        return bars, notes

    if role in ("chord_keys", "pad_atmosphere"):
        bars = 4
        prog = hc.get("chord_progression") or []
        voicings: list[list[int]] = []
        if prog:
            raw_v = [sorted({int(m) % 12 for m in (ch.get("voicing_midi") or [])}) for ch in prog[:4]]
            raw_v = [v for v in raw_v if len(v) >= 2]
            if raw_v:
                voicings = optimize_voice_leading(raw_v)
        if not voicings:
            base = 48 + tonic_pc
            raw_v = [
                sorted({degree_to_midi(base, mode, 0) % 12, degree_to_midi(base, mode, 2) % 12, degree_to_midi(base, mode, 4) % 12}),
                sorted({degree_to_midi(base, mode, 5) % 12, degree_to_midi(base, mode, 7) % 12, degree_to_midi(base, mode, 9) % 12}),
                sorted({degree_to_midi(base, mode, 3) % 12, degree_to_midi(base, mode, 5) % 12, degree_to_midi(base, mode, 7) % 12}),
                sorted({degree_to_midi(base, mode, 4) % 12, degree_to_midi(base, mode, 6) % 12, degree_to_midi(base, mode, 8) % 12}),
            ]
            voicings = optimize_voice_leading(raw_v)
        notes = []
        for bar_i in range(bars):
            v = voicings[bar_i % len(voicings)]
            for j, pitch in enumerate(v):
                notes.append({
                    "start": round(bar_i * bpb + j * 0.02, 4),
                    "length": float(bpb) - 0.25,
                    "pitch": int(pitch),
                    "velocity": 78 - j * 3,
                })
        return bars, notes

    if role == "lead_melody":
        bars = 4
        max_p = 72 if valence < -0.3 else 74
        base = 60 + tonic_pc
        while base + 7 > max_p:
            base -= 12
        mc = hc.get("motif_cell") or {}
        ivs = mc.get("intervals_from_tonic") or [0, 2, 3, 5, 3, 2, 0]
        rhy = mc.get("rhythm_beats") or [0.75, 0.75, 1.0, 0.5, 1.0, 0.5, 1.5]
        notes = []
        t = 0.0
        total_beats = float(bars * bpb)
        for rep in range(2):
            cur = float(rep * 2 * bpb)
            for iv, dur in zip(ivs, rhy):
                if cur + float(dur) > float((rep + 1) * 2 * bpb) - 0.25:
                    break
                p = base + int(iv)
                while p > max_p:
                    p -= 12
                notes.append({"start": round(cur, 4), "length": max(0.25, float(dur) * 0.85), "pitch": int(p), "velocity": 88})
                cur += float(dur)
            t = cur
        if not notes and total_beats > 0:
            notes = [{"start": 0.0, "length": 1.0, "pitch": base, "velocity": 88}]
        return bars, notes

    if role == "kick":
        bars = 2
        notes = []
        for bar_i in range(bars):
            for off in (0.0, 2.5):
                if off < bpb:
                    notes.append({"start": bar_i * bpb + off, "length": 0.4, "pitch": 60, "velocity": 108})
        return bars, notes

    if role == "snare_clap":
        bars = 2
        notes = []
        for bar_i in range(bars):
            for off in (1.0, 3.0):
                if off < bpb:
                    notes.append({"start": bar_i * bpb + off, "length": 0.4, "pitch": 60, "velocity": 100})
        return bars, notes

    # hihat_perc
    bars = 2
    notes = []
    step = 0.5
    t = 0.0
    while t < bars * bpb - 1e-6:
        notes.append({"start": round(t, 4), "length": 0.22, "pitch": 60, "velocity": 84})
        t += step
    notes = apply_anti_machine_gun(notes, bpm=bpm, seed=42)
    return bars, notes


def _measure_played_tuning_cents(
    mono: np.ndarray,
    sr: int,
    expected_midis: list[int] | None = None,
) -> tuple[float | None, float | None]:
    """
    Measure played pitch deviation in cents vs A4=440.0 Hz (and expected MIDI notes when provided).
    Returns `(cents_error, measured_f0_hz)`. Positive means SHARP -> apply negative `Fine tune`.
    """
    if len(mono) < int(0.15 * sr):
        return None, None

    # 1) Try harmonic librosa tuning estimate (works well on chords & phrases)
    harm_ct: float | None = None
    try:
        harm = librosa.effects.harmonic(mono.astype(np.float32))
        if float(np.sqrt(np.mean(harm**2))) > 1e-4:
            harm_ct = float(librosa.estimate_tuning(y=harm, sr=sr, bins_per_octave=120)) * 100.0
    except Exception:
        harm_ct = None

    # 2) High-resolution FFT peak measurement around expected MIDI fundamentals (sub-cent accuracy!)
    fft_errs: list[float] = []
    measured_f0: float | None = None
    try:
        n_fft = min(len(mono), max(16384, 1 << int(math.ceil(math.log2(min(len(mono), sr * 2))))))
        seg = mono[:n_fft]
        win = np.hanning(len(seg))
        spec = np.abs(np.fft.rfft(seg * win))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)

        if expected_midis:
            uniq_m = sorted(set(int(m) for m in expected_midis))
            for m in uniq_m[:6]:
                target_hz = 440.0 * (2.0 ** ((m - 69) / 12.0))
                lo_hz = target_hz * (2.0 ** (-0.65 / 12.0))
                hi_hz = target_hz * (2.0 ** (0.65 / 12.0))
                mask = (freqs >= lo_hz) & (freqs <= hi_hz)
                if not np.any(mask):
                    continue
                idxs = np.where(mask)[0]
                local_mag = spec[idxs]
                rel_k = int(np.argmax(local_mag))
                k = int(idxs[rel_k])
                if float(spec[k]) < 0.05 * float(np.max(spec)):
                    continue
                # Parabolic interpolation around peak bin k
                if 0 < k < len(spec) - 1:
                    y0, y1, y2 = math.log(spec[k - 1] + 1e-12), math.log(spec[k] + 1e-12), math.log(spec[k + 1] + 1e-12)
                    denom = (y0 - 2.0 * y1 + y2)
                    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
                    delta = max(-0.5, min(0.5, delta))
                else:
                    delta = 0.0
                f_peak = (k + delta) * (sr / float(len(seg)))
                ct = 1200.0 * math.log2(max(f_peak, 1e-6) / target_hz)
                if abs(ct) <= 65.0:
                    fft_errs.append(ct)
                    if measured_f0 is None:
                        measured_f0 = f_peak
    except Exception:
        pass

    if fft_errs:
        med_ct = float(np.median(fft_errs))
        return round(med_ct, 1), (round(measured_f0, 2) if measured_f0 is not None else None)
    if harm_ct is not None:
        return round(harm_ct, 1), measured_f0
    return None, measured_f0


def _estimate_envelope_vs_bpm(
    mono: np.ndarray,
    sr: int,
    role: str,
    bpm: float,
) -> tuple[bool, float | None, str | None, dict[str, Any] | None]:
    """Check track decay / envelope clearance relative to BPM."""
    beat_ms = 60000.0 / max(40.0, float(bpm))
    hop = max(1, int(0.010 * sr))
    n_f = len(mono) // hop
    if n_f < 10:
        return True, None, None, None

    rms_env = np.sqrt(np.mean(mono[: n_f * hop].reshape(n_f, hop) ** 2, axis=1))
    pk = float(np.max(rms_env)) + 1e-9
    p90 = float(np.percentile(rms_env, 90)) + 1e-9
    p15 = float(np.percentile(rms_env, 15)) + 1e-9
    dyn_ratio_db = 20.0 * math.log10(p90 / p15)

    # Estimate median decay from peaks to -18 dB (0.126x)
    peaks, _ = librosa.util.peak_pick(rms_env / pk, pre_max=2, post_max=2, pre_avg=3, post_avg=3, delta=0.15, wait=4), None
    decay_ms_list: list[float] = []
    for p in peaks[:12]:
        lvl = float(rms_env[p])
        thresh = lvl * 0.16
        d_frames = 0
        for q in range(int(p) + 1, min(n_f, int(p) + 120)):
            d_frames += 1
            if rms_env[q] <= thresh:
                break
        decay_ms_list.append(d_frames * 10.0)
    est_decay_ms = round(float(np.median(decay_ms_list)), 1) if decay_ms_list else None

    if role == "hihat_perc":
        t_note_ms = beat_ms * 0.5  # eighth-note reference
        if est_decay_ms is not None and est_decay_ms > 0.8 * t_note_ms and dyn_ratio_db < 6.0:
            target_dec = round(min(180.0, 0.55 * t_note_ms), 1)
            return (
                False,
                est_decay_ms,
                f"queue hi-hat trop longue ({est_decay_ms:.0f} ms > {0.8 * t_note_ms:.0f} ms)",
                {"decay_ms": target_dec, "release_ms": 25.0},
            )
    elif role == "bass_sub":
        # Check for extreme sub mud / zero separation on multi-note phrase
        if est_decay_ms is not None and est_decay_ms > 3.5 * beat_ms and dyn_ratio_db < 1.5:
            return (
                False,
                est_decay_ms,
                "extinction sub trop lente (boue inter-notes)",
                {"release_ms": 45.0, "decay_ms": 320.0},
            )
    elif role in ("lead_melody", "chord_keys"):
        # Detect completely static init organ envelope on active audio (crest < 2.2 dB and near-zero RMS variance)
        rms_total = float(np.sqrt(np.mean(mono**2))) + 1e-9
        crest_db = 20.0 * math.log10((float(np.max(np.abs(mono))) + 1e-9) / rms_total)
        active = rms_env > 0.15 * pk
        if np.sum(active) > 15 and dyn_ratio_db < 1.2 and crest_db < 2.5:
            return (
                False,
                est_decay_ms,
                "enveloppe statique type 'orgue init' sans respiration",
                {"attack_ms": 12.0, "decay_ms": 420.0, "sustain_pct": 55.0, "release_ms": 140.0},
            )

    return True, est_decay_ms, None, None


def audit_soundcheck_wav(
    wav_path: str,
    *,
    role: str = "lead_melody",
    track_name: str = "",
    bpm: float = 132.0,
    valence: float = 0.0,
    expected_midis: list[int] | None = None,
    expected_notes: list[dict[str, Any]] | None = None,
    is_sample: bool = False,
    blueprint: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
    listener: ListenerBackend | None = None,
    diagnostics_dir: str = "",
    current_volume: float = 0.80,
) -> dict[str, Any]:
    """Run all 6 Sound Check verifications + Phase 8 Q1 active loudness staging on a rendered isolated track WAV."""
    abs_wav = os.path.abspath(os.path.expanduser(wav_path))
    if not os.path.isfile(abs_wav):
        raise FileNotFoundError(f"Sound check WAV not found: {abs_wav}")

    t_name = track_name.strip() or role
    data, sr = sf.read(abs_wav, dtype="float64", always_2d=True)
    mono = np.mean(data, axis=1)

    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms_dbfs = round(20.0 * math.log10(max(rms, 1e-12)), 2)
    peak_dbfs = round(20.0 * math.log10(max(peak, 1e-12)), 2)

    active_loud = measure_active_loudness(data, sr)
    mix_profile = str((blueprint or {}).get("mix_profile") or "trap_rap")
    has_drums = bool((blueprint or {}).get("has_drums", mix_profile != "classical_ambient"))
    gs_plan = compute_gain_staging_plan(
        {
            t_name: {
                "role": role,
                "active_lufs": active_loud["active_lufs"],
                "peak_dbfs": active_loud["peak_dbfs"],
                "current_volume": float(current_volume),
                "is_effectively_silent": active_loud["is_effectively_silent"],
            }
        },
        mix_profile=mix_profile,
        has_drums=has_drums,
    )
    gs_entry = gs_plan["tracks"][t_name]

    audibility_ok = (
        (not active_loud["is_effectively_silent"])
        and ((rms_dbfs > -52.0) or (active_loud["active_rms_db"] > -45.0))
        and (peak_dbfs > -24.0)
    )
    needed_db = float(gs_entry["target_active_lufs"]) - float(active_loud["active_lufs"])
    can_compensate_no_clip = (float(active_loud["peak_dbfs"]) + needed_db) <= -1.0
    gain_headroom_ok = True
    if audibility_ok and (float(active_loud["active_lufs"]) < float(gs_entry["target_active_lufs"]) - 10.0) and not can_compensate_no_clip:
        gain_headroom_ok = False

    metrics: dict[str, Any] = {
        "wav_path": abs_wav,
        "duration_s": round(len(mono) / float(sr), 3),
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "active_rms_db": active_loud["active_rms_db"],
        "active_lufs": active_loud["active_lufs"],
        "crest_factor_db": active_loud["crest_factor_db"],
        "active_frame_ratio": active_loud["active_frame_ratio"],
        "is_effectively_silent": active_loud["is_effectively_silent"],
        "target_active_lufs": gs_entry["target_active_lufs"],
        "recommended_gain_db": gs_entry["recommended_gain_db"],
        "recommended_volume_auto": gs_entry["recommended_volume"],
        "gain_staging_status": gs_entry["status"],
        "rendered_volume": float(current_volume),
    }
    checks: dict[str, bool] = {
        "audibility_ok": audibility_ok,
        "gain_headroom_ok": gain_headroom_ok,
        "tuning_ok": True,
        "chromatic_integrity_ok": True,
        "spectral_clean_ok": True,
        "envelope_ok": True,
        "semantic_fit_ok": True,
    }
    timbre_card = ""

    if audibility_ok:
        loaded = load_and_normalize_audio(abs_wav, max_seconds=12.0)
        dsp = extract_dsp_metrics(loaded["y_stereo"], loaded["mono_norm"], sr=loaded["sr"])
        glitches = _scan_glitches_and_crackle(mono.astype(np.float32), sr)
        third_oct = _third_octave_profile(mono.astype(np.float32), sr)

        centroid_hz = float(dsp.get("spectral_centroid_hz") or 0.0)
        r_harsh = float(dsp.get("r_harsh") or 0.0)
        r_harm = float(dsp.get("r_harm") or 0.0)
        n_glitches = int(glitches.get("micro_glitch_or_crackle_events") or 0)
        aliasing_db = glitches.get("ultrasonic_above_16khz_rel_db")

        # 2) Played tuning (tonal roles only)
        tuning_err: float | None = None
        measured_f0: float | None = None
        fine_tune_delta = 0
        root_note_delta = 0
        if role in TONAL_ROLES:
            tuning_err, measured_f0 = _measure_played_tuning_cents(mono, sr, expected_midis=expected_midis)
            if tuning_err is not None:
                try:
                    from mix_config import get_invariant
                    _tmax = float(get_invariant("soundcheck_tuning_max_ct", 8.0))
                except Exception:
                    _tmax = 8.0
                checks["tuning_ok"] = abs(tuning_err) <= _tmax
                fine_tune_delta = int(-round(tuning_err))
                if abs(tuning_err) >= 50.0:
                    # ~un demi-ton ou plus d'écart : le Fine tune (cents) ne suffit pas,
                    # c'est le Root note du sampler qui est faux d'un nombre entier de demi-tons.
                    checks["root_note_mismatch_suspected"] = True
                    root_note_delta = int(round(tuning_err / 100.0))

        # 3) Chromatic integrity (for played samples)
        if is_sample and role in TONAL_ROLES:
            if n_glitches > 2 or r_harm < 0.35 or (aliasing_db is not None and aliasing_db > -18.0):
                checks["chromatic_integrity_ok"] = False

        # 4) Spectral cleanliness: 0 narrow peak > +5 dB in 2-5 kHz (for non-pure-sine/non-hat) & 0 glitch
        peaks_2_5k = [
            b for b in third_oct
            if 2000.0 <= b["freq_hz"] <= 5000.0
            and b.get("is_local_peak", True)
            and b["local_excess_db"] > 5.0
            and b.get("peak_width_bands", 99) <= 2
        ]
        # Avoid false-positive narrow peak on clean tonal fundamentals when total harsh energy is tiny
        if r_harsh < 0.08:
            peaks_2_5k = []
        worst_peak = max(peaks_2_5k, key=lambda b: b["local_excess_db"]) if peaks_2_5k else None
        if n_glitches > 0 or (worst_peak is not None and role != "hihat_perc"):
            checks["spectral_clean_ok"] = False

        # 5) Envelope vs BPM
        env_ok, decay_ms, env_issue, rec_adsr = _estimate_envelope_vs_bpm(mono, sr, role=role, bpm=bpm)
        checks["envelope_ok"] = env_ok

        # 6) Semantic fidelity vs role & valence
        rec_lpf: float | None = None
        bands = dsp.get("bands") or {}
        air_ratio = float(bands.get("air") or 0.0)
        # Active-frame 85% spectral rolloff (catches unfiltered sawtooth harmonics even when f0 dominates power)
        rolloff_85_hz = 0.0
        try:
            ro_frames = librosa.feature.spectral_rolloff(y=loaded["mono_norm"], sr=loaded["sr"], roll_percent=0.85)[0]
            rms_frames = librosa.feature.rms(y=loaded["mono_norm"])[0]
            act_ro = ro_frames[rms_frames >= 0.15 * float(np.max(rms_frames) + 1e-9)]
            if len(act_ro):
                rolloff_85_hz = round(float(np.percentile(act_ro, 75)), 1)
        except Exception:
            rolloff_85_hz = 0.0

        if role in ("lead_melody", "chord_keys", "pad_atmosphere"):
            gl_role = {"lead_melody": "lead", "chord_keys": "chords", "pad_atmosphere": "pad"}[role]
            # Dark valence: strict anti-stridence ceiling on centroid, rolloff & harshness
            max_cent = 2600.0 if valence <= -0.3 else (3800.0 if valence < 0.2 else 5200.0)
            max_ro = 4800.0 if valence <= -0.3 else (7500.0 if valence < 0.2 else 11000.0)
            if role in ("chord_keys", "pad_atmosphere"):
                max_cent *= 0.85
                max_ro *= 0.85
            bright_harmonics = (rolloff_85_hz > max_ro and (air_ratio >= 0.015 or r_harsh >= 0.06))
            if centroid_hz > max_cent or bright_harmonics or (valence <= -0.3 and r_harsh > 0.32):
                checks["semantic_fit_ok"] = False
                rec_lpf = round(_default_cutoff_hz(gl_role, valence), 0)
        elif role == "bass_sub":
            if centroid_hz > 1400.0 or r_harsh > 0.18:
                checks["semantic_fit_ok"] = False
                rec_lpf = round(_default_cutoff_hz("bass", valence), 0)

        timbre_card, _top_tags, _def_tags = build_timbre_card_from_dsp(
            category=role,
            sample_nature=dsp.get("sample_nature"),
            detected_note=dsp.get("detected_note"),
            centroid_hz=centroid_hz,
            attack_ms=dsp.get("attack_ms"),
            duration_sec=loaded["duration_sec"],
            r_harsh=r_harsh,
            r_harm=r_harm,
            roughness=dsp.get("roughness"),
            low_mid_corr=dsp.get("low_mid_corr"),
            playable_chromatically=dsp.get("playable_chromatically"),
        )

        # 7) Phase 6 S1/H: sustained spectral flux & timestretch artifact detection
        s1_cfg = (contract or {}).get("s1_physical") or {}
        s2_cfg = (contract or {}).get("s2_acoustic") or {}
        min_flux = float(s1_cfg.get("min_sustained_spectral_flux", 0.015))
        flux_res = analyze_sustained_spectral_flux(mono, sr=sr, min_flux=min_flux, role=role)
        static_warn = flux_res.get("warning")

        # 8) Phase 6 S2/H: high-res onset lag vs expected MIDI notes
        pocket_res: dict[str, Any] | None = None
        onset_lag_ms: float | None = None
        onset_jitter_ms: float | None = None
        if expected_notes:
            pocket_res = analyze_audio_pocket_alignment(
                {t_name: mono},
                {t_name: expected_notes},
                bpm=bpm,
                haas_targets_ms=s2_cfg.get("haas_targets_ms"),
                tolerance_ms=float(s2_cfg.get("onset_lag_tolerance_ms", 7.0)),
                sr=sr,
                track_is_sampler={t_name: is_sample},
            )
            r_info = (pocket_res.get("roles") or {}).get(t_name)
            if r_info:
                onset_lag_ms = r_info.get("median_lag_ms")
                onset_jitter_ms = r_info.get("jitter_iqr_ms")

        metrics.update({
            "spectral_centroid_hz": round(centroid_hz, 1),
            "spectral_rolloff_85_hz": rolloff_85_hz,
            "r_harsh": round(r_harsh, 4),
            "r_harm": round(r_harm, 4),
            "low_mid_corr": dsp.get("low_mid_corr"),
            "micro_glitches": n_glitches,
            "ultrasonic_above_16khz_rel_db": aliasing_db,
            "tuning_error_cents": tuning_err,
            "measured_f0_hz": measured_f0,
            "recommended_fine_tune_delta_ct": fine_tune_delta,
            "recommended_root_note_delta_st": root_note_delta,
            "worst_2_5khz_peak": worst_peak,
            "estimated_decay_ms": decay_ms,
            "envelope_issue": env_issue,
            "recommended_adsr_params": rec_adsr,
            "recommended_lpf_hz": rec_lpf,
            "sustained_spectral_flux": flux_res.get("spectral_flux_norm"),
            "flux_kurtosis": flux_res.get("flux_kurtosis"),
            "static_timbre_warning": static_warn,
            "onset_lag_ms": onset_lag_ms,
            "onset_jitter_ms": onset_jitter_ms,
            "pocket_alignment": pocket_res,
        })

    dsp_report = {
        "track": t_name,
        "role": role,
        "checks": checks,
        "metrics": metrics,
        "timbre_card": timbre_card,
    }

    backend = listener or get_listener_backend()
    result = backend.evaluate_track(
        wav_path=abs_wav,
        track_spec={"name": t_name, "role": role, "valence": valence},
        dsp_report=dsp_report,
        blueprint=blueprint,
    )
    result["role"] = role
    result["wav_path"] = abs_wav
    result["timbre_card"] = timbre_card
    result["checks"] = checks
    result["s1_h_flux"] = metrics.get("sustained_spectral_flux")
    result["static_timbre_warning"] = metrics.get("static_timbre_warning")
    result["s2_h_onset_lag_ms"] = metrics.get("onset_lag_ms")
    result["active_rms_db"] = metrics.get("active_rms_db")
    result["active_lufs"] = metrics.get("active_lufs")
    result["crest_factor_db"] = metrics.get("crest_factor_db")
    result["target_active_lufs"] = metrics.get("target_active_lufs")
    result["recommended_gain_db"] = metrics.get("recommended_gain_db")
    result["recommended_volume_auto"] = metrics.get("recommended_volume_auto")
    result["metrics_summary"] = {
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "active_rms_db": metrics.get("active_rms_db"),
        "active_lufs": metrics.get("active_lufs"),
        "crest_factor_db": metrics.get("crest_factor_db"),
        "target_active_lufs": metrics.get("target_active_lufs"),
        "recommended_gain_db": metrics.get("recommended_gain_db"),
        "recommended_volume_auto": metrics.get("recommended_volume_auto"),
        "spectral_centroid_hz": metrics.get("spectral_centroid_hz"),
        "tuning_error_cents": metrics.get("tuning_error_cents"),
        "recommended_root_note_delta_st": metrics.get("recommended_root_note_delta_st", 0),
        "micro_glitches": metrics.get("micro_glitches", 0),
        "estimated_decay_ms": metrics.get("estimated_decay_ms"),
        "sustained_spectral_flux": metrics.get("sustained_spectral_flux"),
        "static_timbre_warning": metrics.get("static_timbre_warning"),
        "onset_lag_ms": metrics.get("onset_lag_ms"),
    }

    if diagnostics_dir:
        os.makedirs(diagnostics_dir, exist_ok=True)
        diag_path = os.path.join(diagnostics_dir, f"soundcheck_{_safe_slug(t_name)}.json")
        with open(diag_path, "w", encoding="utf-8") as fh:
            json.dump({"result": result, "raw_dsp": dsp_report}, fh, indent=2, ensure_ascii=False)
        result["diagnostics_path"] = diag_path

        metrics_path = os.path.join(diagnostics_dir, "soundcheck_metrics.json")
        agg: dict[str, Any] = {}
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, encoding="utf-8") as mfh:
                    agg = json.load(mfh) or {}
            except Exception:
                agg = {}
        agg[t_name] = {
            "track": t_name,
            "slug": _safe_slug(t_name),
            "role": role,
            "wav_path": abs_wav,
            "active_rms_db": metrics.get("active_rms_db"),
            "active_lufs": metrics.get("active_lufs"),
            "peak_dbfs": peak_dbfs,
            "crest_factor_db": metrics.get("crest_factor_db"),
            "is_effectively_silent": metrics.get("is_effectively_silent"),
            "target_active_lufs": metrics.get("target_active_lufs"),
            "recommended_gain_db": metrics.get("recommended_gain_db"),
            "recommended_volume_auto": metrics.get("recommended_volume_auto"),
            "rendered_volume": float(current_volume),
            "verdict": result.get("verdict"),
        }
        tmp_mp = metrics_path + ".tmp"
        with open(tmp_mp, "w", encoding="utf-8") as mfh:
            json.dump(agg, mfh, indent=2, ensure_ascii=False)
        os.replace(tmp_mp, metrics_path)

    return result


def audit_track_soundcheck_fast(
    *,
    track_spec: dict[str, Any] | ChannelSpec | None = None,
    wav_path: str = "",
    role: str = "lead_melody",
    track_name: str = "",
    bpm: float = 132.0,
    key: str = "C minor",
    valence: float = 0.0,
    blueprint: dict[str, Any] | None = None,
    expected_midis: list[int] | None = None,
    audition: str = "dsp",  # "dsp" (no Gridline at all) | "session" (live MCP probe, no wav) | "render" (legacy full wav)
    mcp: Any = None,
    pattern: int = 1,
    channel: int = 1,
    current_volume: float = 0.80,
) -> dict[str, Any]:
    """Fast per-track soundcheck WITHOUT compiling a .wav.

    - `audition="dsp"`: pure DSP on the sample file + music-theory checks on expected notes
      (~50 ms, no Gridline process at all).
    - `audition="session"`: same DSP checks + in-session loudness probe on the live project
      (`mcp.analyze_loudness(scope="pattern")` with solo), still no .wav export.
    - `audition="render"`: delegates to the legacy `audit_track_soundcheck` (wav render).

    Returns the same verdict shape (`verdict` PASS/FIX/SWAP, `checks`, `metrics`).
    """
    if audition == "render":
        return audit_track_soundcheck(
            track_spec=track_spec, wav_path=wav_path, role=role, track_name=track_name,
            bpm=bpm, key=key, valence=valence, blueprint=blueprint,
            expected_midis=expected_midis,
        )

    spec = dict(track_spec or {})
    eff_role = str(spec.get("role") or role)
    t_name = track_name or str(spec.get("name") or eff_role)
    plugin = str(spec.get("plugin") or "gl.sampler")
    is_sample = plugin == "gl.sampler"

    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {"track": t_name, "role": eff_role, "audition": audition}
    warnings: list[str] = []

    dsp: dict[str, Any] = {}
    sample_file = str(spec.get("sample_file") or "")
    if is_sample and sample_file and os.path.isfile(sample_file):
        try:
            entry, _ = analyze_single_file(sample_file, embedder=None, with_clap=False, max_seconds=12.0)
            dsp = entry.get("dsp") or {}
            metrics.update({
                "spectral_centroid_hz": dsp.get("spectral_centroid_hz"),
                "r_harsh": dsp.get("r_harsh"),
                "r_harm": dsp.get("r_harm"),
                "attack_ms": dsp.get("attack_ms"),
                "detected_note": dsp.get("detected_note"),
                "cents_offset": dsp.get("cents_offset"),
                "playable_chromatically": bool(dsp.get("playable_chromatically")),
            })
            if entry.get("hard_reject"):
                checks["hard_reject_ok"] = False
                warnings.extend(entry.get("reject_reasons") or [])
            else:
                checks["hard_reject_ok"] = True
            if eff_role in TONAL_ROLES:
                checks["playable_chromatically_ok"] = bool(dsp.get("playable_chromatically"))
                co = dsp.get("cents_offset")
                try:
                    from mix_config import get_invariant
                    _tmax = float(get_invariant("soundcheck_tuning_max_ct", 8.0))
                except Exception:
                    _tmax = 8.0
                checks["tuning_ok"] = co is None or abs(float(co)) <= _tmax
                if co is not None and abs(float(co)) >= 50.0:
                    metrics["recommended_root_note_delta_st"] = int(round(-float(co) / 100.0))
                mr = dsp.get("midi_root")
                if mr is not None and key:
                    pen = key_fit_penalty(mr, key, eff_role)
                    checks["key_fit_ok"] = pen < 0.12
                    metrics["key_fit_penalty"] = pen
        except Exception as exc:
            checks["hard_reject_ok"] = False
            warnings.append(f"analyze_failed: {exc}")
    else:
        checks["hard_reject_ok"] = True
        if eff_role in TONAL_ROLES and not is_sample:
            checks["playable_chromatically_ok"] = True  # native gl.* synths are chromatic

    # In-session loudness probe (no wav): solo the channel, measure its pattern live.
    if audition == "session" and mcp is not None:
        try:
            all_ch = [int(c.get("channel", i)) for i, c in enumerate(mcp.call("list_channels", {}).get("channels", []), start=1)]
            muted = mcp.solo_channel(int(channel), all_ch)
            try:
                ld = mcp.analyze_loudness(scope="pattern", pattern=int(pattern))
            finally:
                mcp.unsolo(muted, keep_solo=int(channel))
            metrics["active_lufs"] = ld.get("lufs")
            metrics["peak_dbfs"] = ld.get("peak")
            metrics["is_effectively_silent"] = bool(ld.get("silent"))
            checks["audibility_ok"] = not bool(ld.get("silent")) and (ld.get("lufs") is None or float(ld["lufs"]) >= -24.0)
        except Exception as exc:
            warnings.append(f"session_probe_failed: {exc}")
            checks["audibility_ok"] = True

    # Verdict
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        verdict = "SWAP" if any(k in ("hard_reject_ok", "playable_chromatically_ok") for k in failed) else "FIX"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "track": t_name,
        "role": eff_role,
        "checks": checks,
        "metrics": metrics,
        "warnings": warnings,
        "audition": audition,
        "fast": True,
    }


def audit_track_soundcheck(
    *,
    project_dir: str = "",
    track_spec: dict[str, Any] | ChannelSpec | None = None,
    wav_path: str = "",
    role: str = "lead_melody",
    track_name: str = "",
    bpm: float = 132.0,
    beats_per_bar: int = 4,
    key: str = "C minor",
    valence: float = 0.0,
    blueprint: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
    expected_midis: list[int] | None = None,
    expected_notes: list[dict[str, Any]] | None = None,
    listener_config: dict[str, Any] | None = None,
    use_cache: bool = True,
    keep_tmp: bool = False,
) -> dict[str, Any]:
    """
    Full Sound Check entrypoint:
      - If `wav_path` is provided: directly audits that rendered track WAV.
      - Otherwise: builds a temporary 1-channel `.gridline` in `<project_dir>/.tmp_soundcheck/`,
        plays a 2-4 bar test phrase via `GridlineMCP`, renders `<project_dir>/renders/soundcheck/<track>.wav`,
        audits all 6 criteria, and cleans up `.tmp_soundcheck/`.
    """
    bp = blueprint
    if bp is None and project_dir:
        bp_file = os.path.join(project_dir, "song_blueprint.json")
        if os.path.isfile(bp_file):
            with open(bp_file, encoding="utf-8") as fh:
                bp = json.load(fh)
    if bp:
        bpm = float(bp.get("bpm", bpm))
        ts = bp.get("time_signature") or {}
        beats_per_bar = int(ts.get("numerator", beats_per_bar))
        key = str(bp.get("key", key))
        ev = bp.get("emotion_vector") or {}
        if "valence" in ev and valence == 0.0:
            valence = float(ev["valence"])

    eff_contract = contract
    if eff_contract is None and project_dir:
        c_file = os.path.join(project_dir, "multi_scale_contract.json")
        if os.path.isfile(c_file):
            try:
                with open(c_file, encoding="utf-8") as fh:
                    eff_contract = json.load(fh)
            except Exception:
                eff_contract = None

    listener = get_listener_backend(listener_config)
    diag_dir = os.path.join(project_dir, "renders", "diagnostics") if project_dir else ""

    if wav_path and os.path.isfile(os.path.expanduser(wav_path)):
        is_samp = False
        if isinstance(track_spec, ChannelSpec):
            is_samp = track_spec.plugin == "gl.sampler"
        elif isinstance(track_spec, dict):
            is_samp = track_spec.get("plugin") == "gl.sampler" or bool(track_spec.get("sample_file"))
        eff_name = track_name or (track_spec.name if isinstance(track_spec, ChannelSpec) else (track_spec or {}).get("name", role))
        e1_key = f"e1:{eff_name}"
        e1_hash = ""
        if use_cache and project_dir:
            e1_hash = compute_e1_track_hash(
                wav_path=wav_path,
                notes_slice=expected_notes,
                bpm=bpm,
                role=role,
                key=key,
                valence=valence,
                project_dir=project_dir,
            )
            cached = get_cached_verdict(project_dir, e1_key, e1_hash)
            if cached is not None:
                return cached
        cur_vol_wav = float(track_spec.volume if isinstance(track_spec, ChannelSpec) else (track_spec or {}).get("volume", 0.80))
        res_wav = audit_soundcheck_wav(
            wav_path,
            role=role,
            track_name=eff_name,
            bpm=bpm,
            valence=valence,
            expected_midis=expected_midis,
            expected_notes=expected_notes,
            is_sample=is_samp,
            blueprint=bp,
            contract=eff_contract,
            listener=listener,
            diagnostics_dir=diag_dir,
            current_volume=cur_vol_wav,
        )
        if use_cache and project_dir and e1_hash:
            set_cached_verdict(project_dir, e1_key, e1_hash, res_wav)
        return res_wav

    if track_spec is None:
        raise ValueError("Either `wav_path` or `track_spec` must be provided to audit_track_soundcheck.")

    # Normalize track_spec to dict
    if isinstance(track_spec, ChannelSpec):
        spec = {
            "name": track_spec.name,
            "role": track_spec.role or role,
            "plugin": track_spec.plugin,
            "volume": track_spec.volume,
            "pan": track_spec.pan,
            "sample_file": track_spec.sample_file,
            "root_note": track_spec.root_note,
            "fine_tune": track_spec.fine_tune or 0,
            "one_shot": track_spec.one_shot,
            "voices": track_spec.voices,
            "attack": track_spec.attack,
            "release": track_spec.release,
            "lpf_hz": track_spec.lpf_hz if track_spec.lpf_hz is not None else track_spec.cutoff_hz,
            "plugin_params": dict(track_spec.plugin_params),
        }
    else:
        spec = dict(track_spec)

    eff_role = str(spec.get("role") or role)
    t_name = track_name or str(spec.get("name") or eff_role)
    slug = _safe_slug(t_name)
    base_dir = os.path.abspath(project_dir) if project_dir else tempfile.mkdtemp(prefix="gl_soundcheck_")
    tmp_dir = os.path.join(base_dir, ".tmp_soundcheck")
    render_dir = os.path.join(base_dir, "renders", "soundcheck")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    tmp_proj = os.path.join(tmp_dir, f"{slug}.gridline")
    out_wav = os.path.join(render_dir, f"{slug}.wav")

    bars, notes = _build_role_test_pattern(
        eff_role,
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        key=key,
        valence=valence,
        blueprint=bp,
    )
    e1_key = f"e1:{t_name}"
    e1_hash = ""
    if use_cache and project_dir:
        e1_hash = compute_e1_track_hash(
            sample_path=str(spec.get("sample_file") or ""),
            plugin_params=spec.get("plugin_params") or {},
            notes_slice=notes,
            bpm=bpm,
            track_spec=spec,
            role=eff_role,
            key=key,
            valence=valence,
            project_dir=project_dir,
        )
        cached = get_cached_verdict(project_dir, e1_key, e1_hash)
        if cached is not None:
            return cached

    shutil.copyfile(TEMPLATE_EMPTY_MIXER, tmp_proj)
    plugin = str(spec.get("plugin") or "gl.sampler")
    is_sample = plugin == "gl.sampler"

    mcp = GridlineMCP(tmp_proj, tmp_dir, client_name="soundcheck")
    try:
        mcp.call("set_song", {"tempo": bpm, "numerator": beats_per_bar, "denominator": 4})
        mcp.call("add_channel", {
            "name": t_name,
            "plugin": plugin,
            "mixer": 1,
            "volume": float(spec.get("volume", 0.80)),
            "pan": float(spec.get("pan", 0.0)),
        })
        mcp.call("edit_mixer_track", {"mixer": 1, "name": t_name, "routeTo": 0, "volume": 1.0, "pan": 0.0})

        p_changes: list[tuple[str, Any]] = []
        insert_eq_tuples: list[tuple[str, Any]] = []
        if is_sample:
            spath = str(spec.get("sample_file") or "")
            if spath and not os.path.isabs(spath) and project_dir:
                spath = os.path.join(project_dir, "samples", spath)
            if spath:
                mcp.call("patch", {"patches": [{"path": "channel/1", "set": {"plugin": {"data": spath}}}]})
            p_changes.extend([
                ("channel/1/Root note", int(spec.get("root_note", 60))),
                ("channel/1/Fine tune", int(spec.get("fine_tune", 0))),
                ("channel/1/One shot", int(spec.get("one_shot", 1 if eff_role in PERC_ROLES else 0))),
                ("channel/1/Voices", int(spec.get("voices", 6))),
                ("channel/1/Attack", float(spec.get("attack", 0.002))),
                ("channel/1/Release", float(spec.get("release", 0.08))),
                ("channel/1/Declick", 2),
                ("channel/1/Remove DC", 1),
            ])
            if spec.get("lpf_hz") is not None:
                p_changes.extend([
                    ("channel/1/Filter", 1),
                    ("channel/1/Cutoff", float(spec["lpf_hz"])),
                    ("channel/1/Resonance", float(spec.get("resonance", 0.08))),
                ])
        elif plugin in SYNTH_REGISTRY:
            gl_role_map = {
                "bass_sub": "bass",
                "chord_keys": "chords",
                "pad_atmosphere": "pad",
                "lead_melody": "lead",
            }
            gl_role = str(spec.get("gl_role") or gl_role_map.get(eff_role, "lead"))
            if gl_role not in ("pad", "chords", "lead", "pluck", "bell", "bass"):
                gl_role = "lead"
            if not spec.get("raw_init_patch"):
                vcfg = synth_voice(
                    plugin,
                    gl_role,  # type: ignore[arg-type]
                    valence=valence,
                    cutoff_hz=spec.get("lpf_hz"),
                )
                p_changes.extend(vcfg.to_channel_tuples(1))
                if vcfg.needs_insert_lpf:
                    insert_eq_tuples.extend(vcfg.to_insert_eq_tuples(1, slot=1))

        for k, v in (spec.get("plugin_params") or {}).items():
            p_changes.append((f"channel/1/{k}", v))

        if p_changes:
            mcp.set_params(p_changes)

        extra_eq = list(spec.get("insert_eq_bands") or [])
        if extra_eq:
            insert_eq_tuples.extend(eq_params(1, extra_eq, slot=1, quality=0))
        if insert_eq_tuples:
            mcp.call("add_plugin", {"mixer": 1, "plugin": "gl.eq"})
            mcp.set_params(insert_eq_tuples)

        mcp.call("edit_pattern", {"pattern": 1, "name": f"SC {t_name}", "bars": bars})
        mcp.call("set_notes", {"pattern": 1, "channel": 1, "notes": notes})
        mcp.call("add_clip", {"pattern": 1, "track": 1, "start": 0.0, "length": float(bars * beats_per_bar)})
        mcp.call("save_project", {})
    finally:
        mcp.close()

    cmd = ["gridline", "render", tmp_proj, "-o", out_wav, "--bits", "24", "--scope", f"range:1-{bars}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"Sound check gridline render failed ({proc.returncode}): {proc.stderr or proc.stdout}")

    played_midis = expected_midis if expected_midis is not None else [int(n["pitch"]) for n in notes]
    res = audit_soundcheck_wav(
        out_wav,
        role=eff_role,
        track_name=t_name,
        bpm=bpm,
        valence=valence,
        expected_midis=played_midis if eff_role in TONAL_ROLES else None,
        expected_notes=expected_notes or notes,
        is_sample=is_sample,
        blueprint=bp,
        contract=eff_contract,
        listener=listener,
        diagnostics_dir=diag_dir,
        current_volume=float(spec.get("volume", 0.80)),
    )

    if use_cache and project_dir and e1_hash:
        set_cached_verdict(project_dir, e1_key, e1_hash, res)

    if not keep_tmp and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Unit track sound check in played context.")
    ap.add_argument("--wav", default="", help="Existing isolated track WAV to audit directly.")
    ap.add_argument("--project-dir", default="")
    ap.add_argument("--role", default="lead_melody")
    ap.add_argument("--track-name", default="")
    ap.add_argument("--plugin", default="gl.sampler")
    ap.add_argument("--sample-file", default="")
    ap.add_argument("--bpm", type=float, default=132.0)
    ap.add_argument("--key", default="C minor")
    ap.add_argument("--valence", type=float, default=0.0)
    ap.add_argument("--lpf-hz", type=float, default=None)
    args = ap.parse_args()

    spec = None if args.wav else {
        "name": args.track_name or args.role,
        "role": args.role,
        "plugin": args.plugin,
        "sample_file": args.sample_file,
        "lpf_hz": args.lpf_hz,
    }
    res = audit_track_soundcheck(
        project_dir=args.project_dir,
        track_spec=spec,
        wav_path=args.wav,
        role=args.role,
        track_name=args.track_name,
        bpm=args.bpm,
        key=args.key,
        valence=args.valence,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
