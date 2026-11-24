"""Modules avancés de diagnostic psychoacoustique, spectral, harmonique et perceptuel pour audio-analyzer."""

from __future__ import annotations

import contextlib
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

TARGETS_DIR = Path(__file__).resolve().parent / "targets"

IEC_THIRD_OCTAVE_HZ = [
    25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0,
    250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0,
    2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0,
]

_CLAP_CACHE: dict[str, Any] = {}
_AESTHETICS_CACHE: Any = None


def _lin_to_db(val: float, floor: float = -120.0) -> float:
    return round(float(max(floor, 20.0 * math.log10(max(val, 1e-12)))), 2)


def _third_octave_profile(mono: np.ndarray, sr: int) -> list[dict[str, float]]:
    """Calcule la densité spectrale par bandes au tiers d'octave et détecte les pics/creux locaux."""
    nperseg = min(len(mono), 16384)
    freqs, psd = signal.welch(mono, fs=sr, nperseg=nperseg, scaling="spectrum")
    factor = 2.0 ** (1.0 / 6.0)
    bands: list[dict[str, float]] = []
    for fc in IEC_THIRD_OCTAVE_HZ:
        lo = fc / factor
        hi = min(fc * factor, sr * 0.49)
        if lo >= sr * 0.48:
            continue
        mask = (freqs >= lo) & (freqs < hi)
        energy = float(np.sum(psd[mask])) if np.any(mask) else 1e-12
        bands.append({"freq_hz": fc, "db": round(10.0 * math.log10(max(energy, 1e-12)), 2)})

    dbs = np.array([b["db"] for b in bands], dtype=np.float64)
    for i, b in enumerate(bands):
        w_lo = max(0, i - 3)
        w_hi = min(len(dbs), i + 4)
        neighbors = np.delete(dbs[w_lo:w_hi], i - w_lo)
        baseline = float(np.median(neighbors)) if len(neighbors) else dbs[i]
        b["local_excess_db"] = round(float(dbs[i] - baseline), 2)
        left_ok = i == 0 or dbs[i] >= dbs[i - 1] - 0.5
        right_ok = i == len(dbs) - 1 or dbs[i] >= dbs[i + 1] - 0.5
        b["is_local_peak"] = bool(left_ok and right_ok)
    return bands


def _analyze_transients_and_roominess(data: np.ndarray, mono: np.ndarray, sr: int) -> dict[str, Any]:
    """Détecte la réverbération excessive / effet grande pièce vide et l'étalement des transitoires."""
    rms_total = float(np.sqrt(np.mean(mono**2)))
    peak_total = float(np.max(np.abs(mono)))
    crest_db = round(_lin_to_db(peak_total) - _lin_to_db(rms_total), 2)

    # Énergie RMS par trames de 20 ms
    hop = max(1, int(sr * 0.02))
    frame_rms = librosa.feature.rms(y=mono, frame_length=hop * 2, hop_length=hop)[0]
    p90 = float(np.percentile(frame_rms, 90)) + 1e-9
    p25 = float(np.percentile(frame_rms, 25)) + 1e-9
    p10 = float(np.percentile(frame_rms, 10)) + 1e-9
    tail_floor_ratio_db = round(_lin_to_db(p25 / p90), 2)
    silence_clearance_db = round(_lin_to_db(p90 / p10), 2)

    # Estimation RT60 approximative sur les plus fortes chutes d'énergie (pyroomacoustics si possible)
    rt60_estimates: list[float] = []
    try:
        import pyroomacoustics as pra

        onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=512)
        peaks = librosa.util.peak_pick(onset_env, pre_max=4, post_max=4, pre_avg=6, post_avg=6, delta=0.8, wait=12)
        win = int(sr * 0.85)
        for p in peaks[:12]:
            s_idx = int(p * 512)
            e_idx = min(len(mono), s_idx + win)
            seg = mono[s_idx:e_idx]
            if len(seg) > int(sr * 0.3):
                rt = float(pra.experimental.rt60.measure_rt60(seg, fs=sr, decay_db=20, rt60_tgt=None))
                if 0.04 < rt < 4.5:
                    rt60_estimates.append(rt)
    except Exception:
        pass

    # Décorrélation stéréo dans 200-800 Hz (réverbération/chorus excessif dans les bas-médiums)
    low_mid_corr = 1.0
    if data.shape[1] >= 2:
        sos = signal.butter(4, [200.0, 800.0], btype="bandpass", fs=sr, output="sos")
        l_lm = signal.sosfiltfilt(sos, data[:, 0])
        r_lm = signal.sosfiltfilt(sos, data[:, 1])
        denom = float(np.sqrt(np.sum(l_lm**2) * np.sum(r_lm**2)))
        if denom > 1e-12:
            low_mid_corr = round(float(np.sum(l_lm * r_lm) / denom), 3)

    # Détection de délai court / slapback / résonance de boîte (autocorrélation enveloppe 15-120 ms)
    slapback_score = 0.0
    if len(mono) > sr:
        env = librosa.onset.onset_strength(y=mono[: min(len(mono), sr * 12)], sr=sr, hop_length=128)
        env = env - np.mean(env)
        if np.std(env) > 1e-6:
            ac = np.correlate(env, env, mode="full")[len(env) - 1 :]
            ac = ac / (ac[0] + 1e-12)
            lag_min = max(2, int(0.02 * sr / 128))
            lag_max = min(len(ac) - 1, int(0.14 * sr / 128))
            if lag_max > lag_min:
                slapback_score = round(float(np.max(ac[lag_min:lag_max])), 3)

    rt60_median_ms = round(float(np.median(rt60_estimates) * 1000.0), 1) if rt60_estimates else None
    return {
        "crest_factor_db": crest_db,
        "tail_floor_ratio_db": tail_floor_ratio_db,
        "dynamic_clearance_db": silence_clearance_db,
        "low_mid_200_800hz_correlation": low_mid_corr,
        "slapback_echo_score": slapback_score,
        "estimated_rt60_ms": rt60_median_ms,
    }


def _scan_glitches_and_crackle(mono: np.ndarray, sr: int) -> dict[str, Any]:
    """Détecte les grésillements, clics numériques, aliasing ultrasonique et défauts de phase/clipping."""
    dc_offset = round(float(np.mean(mono)), 6)
    near_full_scale = int(np.sum(np.abs(mono) >= 0.988))

    # Discontinuités brutales (xruns / buffer drops / clics numériques : saut > 0.45 sur 1 sample ou |d2| > 0.55)
    diff1 = np.abs(np.diff(mono, prepend=mono[0]))
    diff2 = np.abs(np.diff(mono, n=2, prepend=mono[0], append=mono[-1]))
    win = max(64, int(sr * 0.05))
    local_rms = np.sqrt(np.convolve(mono**2, np.ones(win) / win, mode="same") + 1e-10)
    active = local_rms > 0.002
    glitch_mask = active & (diff1 > 0.52) & (diff2 > 0.85) & ((diff2 / local_rms) > 12.0)
    glitch_events = int(np.sum(glitch_mask & (~np.roll(glitch_mask, 1))))

    # Énergie ultrasonique > 16 kHz (repliement d'aliasing / synthés numériques agressifs)
    aliasing_db = None
    if sr >= 44100:
        freqs, psd = signal.welch(mono, fs=sr, nperseg=min(len(mono), 8192), scaling="spectrum")
        total_e = float(np.sum(psd[(freqs >= 20) & (freqs < sr * 0.49)])) + 1e-12
        ultra_e = float(np.sum(psd[(freqs >= 16000) & (freqs < sr * 0.49)])) + 1e-12
        aliasing_db = round(10.0 * math.log10(ultra_e / total_e), 2)

    return {
        "dc_offset": dc_offset,
        "near_full_scale_samples": near_full_scale,
        "micro_glitch_or_crackle_events": glitch_events,
        "ultrasonic_above_16khz_rel_db": aliasing_db,
    }


def _compute_psychoacoustics(mono: np.ndarray, sr: int) -> dict[str, Any]:
    """Calcule l'acuité (Sharpness DIN 45692) via MoSQITo et les descripteurs Essentia."""
    res: dict[str, Any] = {}
    # Sélectionne un extrait représentatif de max 12s autour du pic RMS
    max_samples = min(len(mono), sr * 12)
    if len(mono) > max_samples:
        step = sr
        best_i, best_e = 0, -1.0
        for i in range(0, len(mono) - max_samples + 1, step):
            e = float(np.mean(mono[i : i + max_samples] ** 2))
            if e > best_e:
                best_e = e
                best_i = i
        excerpt = mono[best_i : best_i + max_samples]
    else:
        excerpt = mono

    try:
        import mosqito

        with contextlib.redirect_stdout(sys.stderr):
            sharp = float(mosqito.sharpness_din_st(excerpt.astype(np.float64), fs=sr, weighting="din"))
        res["sharpness_din_acum"] = round(sharp, 3)
    except Exception as exc:
        res["sharpness_error"] = str(exc)

    try:
        import essentia.standard as es

        if sr != 44100:
            ex44 = es.Resample(inputSampleRate=float(sr), outputSampleRate=44100.0)(excerpt.astype(np.float32))
        else:
            ex44 = excerpt.astype(np.float32)
        key, scale, strength = es.KeyExtractor(sampleRate=44100.0)(ex44)
        res["detected_key"] = f"{key} {scale}"
        res["key_confidence"] = round(float(strength), 3)
        # Mesure du diapason sur la composante harmonique complète (évite le biais des transitoires de kick/hi-hats)
        harm_ex = librosa.effects.harmonic(ex44)
        tuning_st = float(librosa.estimate_tuning(y=harm_ex, sr=44100))
        tuning_hz = 440.0 * (2.0 ** (tuning_st / 12.0))
        res["tuning_a4_hz"] = round(tuning_hz, 2)
        res["tuning_cents_offset"] = round(tuning_st * 100.0, 1)
    except Exception:
        pass

    return res


def _simulate_headphone_translation(third_oct: list[dict[str, float]], target_id: str) -> dict[str, Any] | None:
    """Simule comment le mix sonne à travers la réponse en fréquence d'un casque cible AutoEq."""
    aliases = {
        "sony_xm6": "Sony_WH-1000XM6.csv",
        "sony_xm5": "Sony_WH-1000XM5.csv",
        "sony_xm4": "Sony_WH-1000XM4.csv",
    }
    fname = aliases.get(target_id.lower())
    if not fname:
        return None
    csv_path = TARGETS_DIR / fname
    if not csv_path.is_file():
        return None

    freqs_hp: list[float] = []
    err_hp: list[float] = []
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            freqs_hp.append(float(row["frequency"]))
            err_hp.append(float(row["error_smoothed"]))

    mix_freqs = np.array([b["freq_hz"] for b in third_oct], dtype=np.float64)
    mix_excess = np.array([b["local_excess_db"] for b in third_oct], dtype=np.float64)
    hp_dev = np.interp(np.log10(mix_freqs), np.log10(np.array(freqs_hp)), np.array(err_hp))
    combined = mix_excess + hp_dev

    hotspots = []
    for f, m_ex, h_dev, comb in zip(mix_freqs, mix_excess, hp_dev, combined):
        if comb >= 4.5 or h_dev >= 2.8 and m_ex >= 1.5:
            hotspots.append(
                {
                    "freq_hz": float(f),
                    "mix_local_db": round(float(m_ex), 2),
                    "headphone_boost_db": round(float(h_dev), 2),
                    "perceived_total_db": round(float(comb), 2),
                }
            )
    return {
        "target_profile": target_id,
        "low_mid_100_315hz_avg_boost_db": round(
            float(np.mean([h for f, h in zip(mix_freqs, hp_dev) if 100 <= f <= 315])), 2
        ),
        "upper_mid_2000_3150hz_boost_db": round(
            float(np.mean([h for f, h in zip(mix_freqs, hp_dev) if 2000 <= f <= 3150])), 2
        ),
        "translation_hotspots": hotspots[:6],
    }


def _load_audiobox_predictor():
    global _AESTHETICS_CACHE
    if _AESTHETICS_CACHE is not None:
        return _AESTHETICS_CACHE
    from audiobox_aesthetics.infer import initialize_predictor

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    with contextlib.redirect_stdout(sys.stderr):
        _AESTHETICS_CACHE = initialize_predictor()
    return _AESTHETICS_CACHE


def _load_clap():
    if "model" in _CLAP_CACHE:
        return _CLAP_CACHE["processor"], _CLAP_CACHE["model"]
    from transformers import ClapModel, ClapProcessor

    model_id = "laion/clap-htsat-unfused"
    with contextlib.redirect_stdout(sys.stderr):
        try:
            proc = ClapProcessor.from_pretrained(model_id, local_files_only=True)
            mod = ClapModel.from_pretrained(model_id, local_files_only=True)
        except Exception:
            proc = ClapProcessor.from_pretrained(model_id)
            mod = ClapModel.from_pretrained(model_id)
        mod.eval()
    _CLAP_CACHE["processor"] = proc
    _CLAP_CACHE["model"] = mod
    return proc, mod


def _run_neural_perceptual_audit(audio_path: str, data: np.ndarray, mono: np.ndarray, sr: int) -> dict[str, Any]:
    """Scores perceptuels neural (Audiobox Aesthetics + sondes CLAP opposées)."""
    import torch

    out: dict[str, Any] = {}
    try:
        pred = _load_audiobox_predictor()
        wav_tensor = torch.from_numpy(data.T.astype(np.float32))
        with contextlib.redirect_stdout(sys.stderr):
            scores = pred.forward([{"path": wav_tensor, "sample_rate": int(sr)}])[0]
        out["audiobox_aesthetics"] = {k: round(float(v), 3) for k, v in scores.items()}
    except Exception as exc:
        out["audiobox_error"] = str(exc)

    try:
        proc, mod = _load_clap()
        # Resample vers 48kHz pour CLAP et découpe en max 2 segments de 10s
        mono48 = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=48000) if sr != 48000 else mono.astype(np.float32)
        seg_len = 48000 * 10
        segs = [mono48[:seg_len]]
        if len(mono48) > seg_len * 2:
            mid = len(mono48) // 2 - seg_len // 2
            segs.append(mono48[mid : mid + seg_len])
        with torch.no_grad():
            ain = proc(audio=segs, sampling_rate=48000, return_tensors="pt", padding=True)
            af = mod.get_audio_features(**ain)
            if not isinstance(af, torch.Tensor):
                af = getattr(af, "pooler_output", getattr(af, "last_hidden_state", af[0]))
            af = af / af.norm(dim=-1, keepdim=True)
            audio_vec = af.mean(dim=0, keepdim=True)
            audio_vec = audio_vec / audio_vec.norm(dim=-1, keepdim=True)

            probe_pairs = {
                "reverb_vs_dry": (
                    "heavy cavern hall reverb hollow empty room echo",
                    "tight dry punchy modern studio recording zero room echo",
                ),
                "crackle_vs_clean": (
                    "crackling distorted clipping glitch audio artifacts",
                    "clean smooth high fidelity pristine audio master",
                ),
                "shrill_vs_warm": (
                    "piercing painful harsh shrill high pitched treble synth",
                    "warm dark balanced smooth deep bass and keys",
                ),
                "muddy_vs_clear": (
                    "muddy boxy muffled congested low midrange resonance",
                    "clear crisp transparent well separated stereo mix",
                ),
            }
            all_texts = [t for pair in probe_pairs.values() for t in pair]
            tin = proc(text=all_texts, return_tensors="pt", padding=True)
            tf = mod.get_text_features(**tin)
            if not isinstance(tf, torch.Tensor):
                tf = getattr(tf, "pooler_output", getattr(tf, "last_hidden_state", tf[0]))
            tf = tf / tf.norm(dim=-1, keepdim=True)
            sims = (audio_vec @ tf.T).squeeze(0).cpu().numpy()

            probes: dict[str, dict[str, float]] = {}
            for idx, (name, _) in enumerate(probe_pairs.items()):
                bad_s = float(sims[idx * 2])
                good_s = float(sims[idx * 2 + 1])
                probes[name] = {
                    "defect_sim": round(bad_s, 4),
                    "healthy_sim": round(good_s, 4),
                    "defect_margin": round(bad_s - good_s, 4),
                }
            out["clap_perceptual_probes"] = probes
    except Exception as exc:
        out["clap_error"] = str(exc)

    return out


def run_deep_audit(
    audio_path: str,
    base_metrics: dict[str, Any],
    data: np.ndarray,
    sr: int,
    target_genre: str = "trap",
    use_neural: bool = True,
    headphone_target: str = "sony_xm6",
) -> dict[str, Any]:
    """Assemble le diagnostic complet multi-domaines d'un rendu audio."""
    mono = np.mean(data, axis=1).astype(np.float32)
    third_oct = _third_octave_profile(mono, sr)
    resonances = [b for b in third_oct if b["local_excess_db"] >= 5.0 and b.get("is_local_peak", True)]
    notches = [b for b in third_oct if b["local_excess_db"] <= -6.0]
    room = _analyze_transients_and_roominess(data, mono, sr)
    glitches = _scan_glitches_and_crackle(mono, sr)
    psycho = _compute_psychoacoustics(mono, sr)
    hp_sim = _simulate_headphone_translation(third_oct, headphone_target) if headphone_target else None
    neural = _run_neural_perceptual_audit(audio_path, data, mono, sr) if use_neural else {}

    issues: list[dict[str, Any]] = []
    # 1. Résonances 1/3 octave (boîte/studio 200-500 Hz ou sifflement 2-6 kHz)
    boxy = [r for r in resonances if 160 <= r["freq_hz"] <= 500]
    if boxy:
        issues.append(
            {
                "category": "boxy_resonance",
                "severity": "high",
                "detail": f"Pics bas-médiums détectés à {[r['freq_hz'] for r in boxy]} Hz (+{max(r['local_excess_db'] for r in boxy)} dB).",
                "fix": "Appliquer un coupe-cloche gl.eq Peak=6 (Band N width = 0.20..0.25, ≈ Q 2.5..4 ; -3 à -5 dB) sur les pistes mélodiques/pads dans cette plage et passe-haut >= 140 Hz.",
            }
        )
    shrill = [r for r in resonances if 1800 <= r["freq_hz"] <= 7000]
    sharp_val = psycho.get("sharpness_din_acum", 0.0)
    sharp_warn_thresh = 1.85 if target_genre in ("lofi_chill", "classical_ambient", "jazz_acoustic") else 1.95
    if shrill or sharp_val >= sharp_warn_thresh:
        issues.append(
            {
                "category": "harsh_shrill_treble",
                "severity": "high" if sharp_val >= 2.1 else "medium",
                "detail": f"Acuité DIN={sharp_val} acum (seuil {sharp_warn_thresh}), pics aigus={[(r['freq_hz'], r['local_excess_db']) for r in shrill]}.",
                "fix": "Vérifier Lead <= MIDI 74 (sinon -12 st), abaisser le LPF du synthé/insert (<= 5-7 kHz) et creuser gl.eq Peak = 6 (-3.5 dB @ 3200 Hz).",
            }
        )

    # 2. Réverbération / Grande pièce vide
    if (
        (room.get("estimated_rt60_ms") or 0) > 650
        or room["low_mid_200_800hz_correlation"] < 0.55
        or room["slapback_echo_score"] > 0.42
    ):
        issues.append(
            {
                "category": "excessive_room_reverb",
                "severity": "medium",
                "detail": f"RT60={room.get('estimated_rt60_ms')}ms, corrélation 200-800Hz={room['low_mid_200_800hz_correlation']}, slapback={room['slapback_echo_score']}.",
                "fix": "Réduire le Wet de gl.hall/gl.reverb (< 12-18%), mettre un passe-haut à 250 Hz sur le send reverb, et resserrer l'unison stéréo des bas-médiums.",
            }
        )

    # 3. Grésillements / glitches
    if (
        glitches["micro_glitch_or_crackle_events"] > 15
        or base_metrics.get("clipped_samples", 0) > 0
        or base_metrics.get("true_peak_dbtp", -10) > -0.8
    ):
        issues.append(
            {
                "category": "crackle_or_clipping",
                "severity": "high",
                "detail": f"micro-clics={glitches['micro_glitch_or_crackle_events']}, clipped={base_metrics.get('clipped_samples')}, TP={base_metrics.get('true_peak_dbtp')} dBTP.",
                "fix": "Vérifier attaque/release (> 5 ms), réduire saturation/plafond limiteur (-1.0 dBTP), et forcer quantum PipeWire >= 1024 si Bluetooth XM6 actif.",
            }
        )

    return {
        "base_mix": base_metrics,
        "target_genre": target_genre,
        "psychoacoustics": psycho,
        "room_and_transients": room,
        "glitches_and_artifacts": glitches,
        "third_octave_resonances": resonances,
        "third_octave_notches": notches,
        "headphone_translation": hp_sim,
        "neural_perception": neural,
        "issues_detected": issues,
        "third_octave_bands": third_oct,
    }


def resolve_allowed_pcs(ref_key_str: str) -> set[int]:
    """Résout l'ensemble des pitch classes (0..11) autorisées pour une tonalité donnée."""
    import music21

    parts = (ref_key_str or "C minor").strip().split()
    tonic_raw = parts[0].replace("b", "-") if parts else "C"
    mode_raw = " ".join(parts[1:]).lower().replace("-", "_") if len(parts) > 1 else "minor"
    tonic_pc = music21.pitch.Pitch(tonic_raw).pitchClass
    if "phrygian_dominant" in mode_raw or "phrygian dominant" in mode_raw:
        return {(tonic_pc + iv) % 12 for iv in (0, 1, 4, 5, 7, 8, 10)}
    if "blues" in mode_raw:
        return {(tonic_pc + iv) % 12 for iv in (0, 3, 5, 6, 7, 10)}
    if "pentatonic" in mode_raw and ("min" in mode_raw or "aeolian" in mode_raw):
        return {(tonic_pc + iv) % 12 for iv in (0, 3, 5, 7, 10)}
    if "pentatonic" in mode_raw:
        return {(tonic_pc + iv) % 12 for iv in (0, 2, 4, 7, 9)}
    if "harmonic" in mode_raw:
        return {p.pitchClass for p in music21.scale.HarmonicMinorScale(tonic_raw).getPitches()}
    if "melodic" in mode_raw:
        return {p.pitchClass for p in music21.scale.MelodicMinorScale(tonic_raw).getPitches()}
    if "dorian" in mode_raw:
        return {p.pitchClass for p in music21.scale.DorianScale(tonic_raw).getPitches()}
    if "phrygian" in mode_raw:
        return {p.pitchClass for p in music21.scale.PhrygianScale(tonic_raw).getPitches()}
    if "lydian" in mode_raw:
        return {p.pitchClass for p in music21.scale.LydianScale(tonic_raw).getPitches()}
    if "mixolydian" in mode_raw:
        return {p.pitchClass for p in music21.scale.MixolydianScale(tonic_raw).getPitches()}
    if "locrian" in mode_raw:
        return {p.pitchClass for p in music21.scale.LocrianScale(tonic_raw).getPitches()}
    if "min" in mode_raw or "aeolian" in mode_raw:
        return {p.pitchClass for p in music21.scale.MinorScale(tonic_raw).getPitches()}
    return {p.pitchClass for p in music21.scale.MajorScale(tonic_raw).getPitches()}


def audit_midi_file(
    midi_path: str,
    expected_key: str = "",
    min_pitch: int = 24,
    max_pitch: int = 74,
    strict_musicality: bool = True,
    valence: float = 0.0,
    tension: float = 0.3,
    time_signature: tuple[int, int] | str | None = None,
    phrase_plan: str = "",
) -> dict[str, Any]:
    """Audite la justesse harmonique, la tessiture (notes trop aiguës), les clashs MIDI et les métriques de musicalité."""
    import music21
    import pretty_midi
    from musicality_metrics import _PERC_TRACK_RE, evaluate_midi_musicality

    abs_path = os.path.abspath(os.path.expanduser(midi_path))
    if not os.path.isfile(abs_path):
        return {"error": f"Fichier MIDI introuvable : {abs_path}"}

    pm = pretty_midi.PrettyMIDI(abs_path)
    score = music21.converter.parse(abs_path)
    est_key = score.analyze("key")
    detected_key_str = f"{est_key.tonic.name.replace('-', 'b')} {est_key.mode}"

    ref_key_str = expected_key.strip() or detected_key_str
    allowed_pcs = resolve_allowed_pcs(ref_key_str)

    track_reports: list[dict[str, Any]] = []
    melodic_notes: list[tuple[float, float, int, str]] = []
    for inst in pm.instruments:
        name = inst.name or f"Track_{inst.program}"
        if not inst.notes:
            continue
        is_perc = bool(inst.is_drum) or bool(_PERC_TRACK_RE.search(name.lower()))
        pitches = [n.pitch for n in inst.notes]
        report: dict[str, Any] = {
            "name": name,
            "is_drum": is_perc,
            "note_count": len(inst.notes),
            "min_pitch": min(pitches),
            "max_pitch": max(pitches),
        }
        if not is_perc:
            out_of_scale = [
                {"pitch": n.pitch, "note": pretty_midi.note_number_to_name(n.pitch), "start_s": round(n.start, 3)}
                for n in inst.notes
                if (n.pitch % 12) not in allowed_pcs
            ]
            too_high = [n.pitch for n in inst.notes if n.pitch > max_pitch]
            too_low = [n.pitch for n in inst.notes if n.pitch < min_pitch]
            report["out_of_scale_count"] = len(out_of_scale)
            report["out_of_scale_samples"] = out_of_scale[:8]
            report["too_high_count"] = len(too_high)
            report["too_low_count"] = len(too_low)
            for n in inst.notes:
                melodic_notes.append((n.start, n.end, n.pitch, name))
        track_reports.append(report)

    # Collisions sub-grave (< MIDI 48 = C3, soit <= 123.5 Hz) inter-pistes
    sub_collisions: list[dict[str, Any]] = []
    melodic_notes.sort(key=lambda x: x[0])
    for i, (s1, e1, p1, tr1) in enumerate(melodic_notes):
        for s2, e2, p2, tr2 in melodic_notes[i + 1 :]:
            if s2 >= e1 - 0.01:
                break
            if tr1 == tr2:
                continue
            if p1 < 48 and p2 < 48 and p1 != p2 and len(sub_collisions) < 12:
                sub_collisions.append(
                    {
                        "time_s": round(float(s2), 3),
                        "tracks": [tr1, tr2],
                        "notes": [pretty_midi.note_number_to_name(p1), pretty_midi.note_number_to_name(p2)],
                    }
                )

    musicality = evaluate_midi_musicality(
        abs_path,
        valence=valence,
        tension=tension,
        allowed_pcs=allowed_pcs,
        time_signature=time_signature,
        phrase_plan=phrase_plan,
    )
    clashes = musicality.get("minor_2nd_clashes", [])
    blocking_clashes = musicality.get("blocking_minor_2nd_clashes", clashes)
    out_of_scale_total = sum(tr.get("out_of_scale_count", 0) for tr in track_reports if not tr["is_drum"])
    too_high_total = sum(tr.get("too_high_count", 0) for tr in track_reports if not tr["is_drum"])
    too_low_total = sum(tr.get("too_low_count", 0) for tr in track_reports if not tr["is_drum"])
    pitch_range_ok = (too_high_total == 0) and (too_low_total == 0)
    gate_ok = (
        musicality["musicality_verdict"] == "PASS"
        and out_of_scale_total == 0
        and pitch_range_ok
        and len(blocking_clashes) == 0
        and len(sub_collisions) == 0
    )

    return {
        "midi_path": abs_path,
        "detected_key": detected_key_str,
        "expected_key": ref_key_str,
        "key_matches_expected": detected_key_str.lower() == ref_key_str.lower(),
        "time_signature": musicality.get("time_signature", "4/4"),
        "phrase_plan": musicality.get("phrase_plan", phrase_plan or "standard"),
        "strict_musicality": strict_musicality,
        "emotion_mode": musicality.get("emotion_mode", "consonant"),
        "valence": valence,
        "tension": tension,
        "max_pitch_limit": max_pitch,
        "out_of_scale_notes": out_of_scale_total,
        "too_high_count": too_high_total,
        "too_low_count": too_low_total,
        "musicality_verdict": musicality["musicality_verdict"],
        "gate_passed": gate_ok if strict_musicality else (out_of_scale_total == 0 and pitch_range_ok and len(blocking_clashes) == 0),
        "musicality_metrics": musicality["metrics"],
        "musicality_detected_roles": musicality["detected_roles"],
        "musicality_failures": musicality["failures"],
        "tracks": track_reports,
        "minor_2nd_clashes": clashes[:24],
        "blocking_minor_2nd_clashes": blocking_clashes[:24],
        "low_end_simultaneous_collisions": sub_collisions,
    }


def match_with_reference(
    mix_path: str,
    reference_path: str,
    out_matched_wav: str = "",
) -> dict[str, Any]:
    """Compare la courbe spectrale à une référence pro et peut générer un master auto-matché via Matchering."""
    abs_mix = os.path.abspath(os.path.expanduser(mix_path))
    abs_ref = os.path.abspath(os.path.expanduser(reference_path))
    for p in (abs_mix, abs_ref):
        if not os.path.isfile(p):
            return {"error": f"Fichier introuvable : {p}"}

    y_m, sr_m = sf.read(abs_mix, always_2d=True, dtype="float32")
    y_r, sr_r = sf.read(abs_ref, always_2d=True, dtype="float32")
    mono_m = np.mean(y_m, axis=1)
    mono_r = np.mean(y_r, axis=1)
    if sr_r != sr_m:
        mono_r = librosa.resample(mono_r, orig_sr=sr_r, target_sr=sr_m)

    f_m, p_m = signal.welch(mono_m, fs=sr_m, nperseg=16384, scaling="spectrum")
    f_r, p_r = signal.welch(mono_r, fs=sr_m, nperseg=16384, scaling="spectrum")
    p_m = p_m / (np.sum(p_m) + 1e-12)
    p_r = p_r / (np.sum(p_r) + 1e-12)

    bands = [
        ("sub_20_60", 20, 60),
        ("bass_60_150", 60, 150),
        ("low_mid_150_400", 150, 400),
        ("mid_400_1200", 400, 1200),
        ("upper_mid_1200_3500", 1200, 3500),
        ("presence_3500_7000", 3500, 7000),
        ("air_7000_16000", 7000, 16000),
    ]
    deltas: dict[str, float] = {}
    for name, lo, hi in bands:
        mask = (f_m >= lo) & (f_m < hi)
        em = float(np.sum(p_m[mask])) + 1e-12
        er = float(np.sum(p_r[mask])) + 1e-12
        deltas[name] = round(10.0 * math.log10(em / er), 2)

    matched_path = None
    if out_matched_wav.strip():
        import matchering as mg

        matched_path = os.path.abspath(os.path.expanduser(out_matched_wav))
        os.makedirs(os.path.dirname(matched_path) or ".", exist_ok=True)
        with contextlib.redirect_stdout(sys.stderr):
            mg.process(
                target=abs_mix,
                reference=abs_ref,
                results=[mg.pcm24(matched_path)],
            )

    return {
        "mix_path": abs_mix,
        "reference_path": abs_ref,
        "spectral_delta_mix_minus_ref_db": deltas,
        "matched_wav_generated": matched_path,
        "interpretation": (
            "Valeurs positives = ton mix a TROP d'énergie dans cette bande par rapport à la référence ; "
            "valeurs négatives = ton mix en MANQUE."
        ),
    }


def render_stems_and_audit(project_path: str, output_dir: str = "", mode: str = "tracks") -> dict[str, Any]:
    """Rend toutes les pistes/stems d'un projet .gridline en WAV puis audite chaque stem séparément."""
    abs_proj = os.path.abspath(os.path.expanduser(project_path))
    if not os.path.isfile(abs_proj):
        return {"error": f"Projet introuvable : {abs_proj}"}
    out_dir = (
        os.path.abspath(os.path.expanduser(output_dir))
        if output_dir.strip()
        else os.path.join(os.path.dirname(abs_proj), "renders", f"stems_{mode}")
    )
    os.makedirs(out_dir, exist_ok=True)
    flag = "--stems" if mode == "stems" else "--tracks"
    proc = subprocess.run(
        ["gridline", "render", abs_proj, "-o", out_dir.rstrip("/") + "/", flag],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        return {"error": f"Échec gridline render ({proc.returncode}): {proc.stderr or proc.stdout}"}

    stem_reports: list[dict[str, Any]] = []
    for wav in sorted(Path(out_dir).glob("*.wav")):
        data, sr = sf.read(str(wav), always_2d=True, dtype="float32")
        mono = np.mean(data, axis=1)
        rms = float(np.sqrt(np.mean(mono**2)))
        peak = float(np.max(np.abs(mono)))
        if rms < 1e-5:
            stem_reports.append({"stem": wav.name, "path": str(wav), "silent": True})
            continue
        f, p = signal.welch(mono, fs=sr, nperseg=min(len(mono), 8192), scaling="spectrum")
        tot = float(np.sum(p)) + 1e-12
        centroid = float(np.sum(f * p) / tot)
        boxy_share = float(np.sum(p[(f >= 200) & (f <= 500)]) / tot)
        harsh_share = float(np.sum(p[(f >= 2000) & (f <= 6000)]) / tot)
        stem_reports.append(
            {
                "stem": wav.name,
                "path": str(wav),
                "silent": False,
                "peak_db": _lin_to_db(peak),
                "rms_db": _lin_to_db(rms),
                "spectral_centroid_hz": round(centroid, 1),
                "boxy_200_500hz_ratio": round(boxy_share, 3),
                "harsh_2_6khz_ratio": round(harsh_share, 3),
            }
        )

    return {"project_path": abs_proj, "output_dir": out_dir, "mode": mode, "stems": stem_reports}


def compute_pink_slope_db_per_oct(mono: np.ndarray, sr: int) -> float:
    """Calcule la pente spectrale moyenne (dB/octave) entre 60 Hz et 12 kHz par régression log2."""
    if len(mono) < 256 or sr < 16000:
        return 0.0
    nperseg = min(len(mono), 16384)
    freqs, psd = signal.welch(mono.astype(np.float64), fs=sr, nperseg=nperseg, scaling="density")
    hi_hz = min(12000.0, 0.45 * float(sr))
    mask = (freqs >= 60.0) & (freqs <= hi_hz) & (psd > 0.0)
    if np.sum(mask) < 4:
        return 0.0
    f_sub = freqs[mask]
    p_sub = psd[mask]
    log_grid = np.linspace(np.log2(60.0), np.log2(hi_hz), 96)
    f_grid = np.power(2.0, log_grid)
    p_grid = np.interp(f_grid, f_sub, p_sub)
    psd_db = 10.0 * np.log10(np.maximum(p_grid, 1e-18))
    slope, _ = np.polyfit(log_grid, psd_db, 1)
    return round(float(slope), 3)


def _load_and_sum_stems(paths: list[Path], target_sr: int | None = None, target_len: int | None = None) -> tuple[np.ndarray, int]:
    acc: np.ndarray | None = None
    out_sr = int(target_sr or 44100)
    for p in paths:
        data, sr = sf.read(str(p), always_2d=True, dtype="float64")
        mono = np.mean(data, axis=1)
        if target_sr is None:
            out_sr = sr
        elif sr != target_sr:
            g = math.gcd(int(sr), int(target_sr))
            mono = signal.resample_poly(mono, int(target_sr) // g, int(sr) // g)
            out_sr = int(target_sr)
        if acc is None:
            n_target = int(target_len or len(mono))
            acc = np.zeros(n_target, dtype=np.float64)
        n_copy = min(len(acc), len(mono))
        acc[:n_copy] += mono[:n_copy]
    if acc is None:
        return np.zeros(int(target_len or 0), dtype=np.float64), out_sr
    return acc, out_sr


def analyze_stem_hierarchy(
    project_path: str = "",
    range_bars: str = "",
    stems_dir: str = "",
    has_drums: bool = True,
    out_dir: str = "",
) -> dict[str, Any]:
    """
    Audit hiérarchique inter-stems (Axe 4 §1) :
      - `lead_vs_harmony_db` (800 Hz - 3.5 kHz sur fenêtres Lead actives, cible >= +2.5 dB)
      - `kick_bass_transient_overlap` (corrélation d'enveloppe < 150 Hz aux attaques Kick)
      - `pink_slope_db_per_oct` (régression log-freq 60 Hz - 12 kHz, cible -3.0 .. -5.0 dB/oct)
    """
    abs_proj = os.path.abspath(os.path.expanduser(project_path)) if project_path.strip() else ""
    target_dir = (
        os.path.abspath(os.path.expanduser(stems_dir))
        if stems_dir.strip()
        else (
            os.path.abspath(os.path.expanduser(out_dir))
            if out_dir.strip()
            else os.path.join(os.path.dirname(abs_proj), "renders", "stems_hierarchy")
        )
    )

    wavs = sorted(Path(target_dir).glob("*.wav")) if os.path.isdir(target_dir) else []
    if not wavs:
        if not abs_proj or not os.path.isfile(abs_proj):
            return {"error": f"Projet ou dossier stems introuvable : proj={abs_proj!r} stems={target_dir!r}"}
        os.makedirs(target_dir, exist_ok=True)
        cmd = ["gridline", "render", abs_proj, "-o", target_dir.rstrip("/") + "/", "--stems"]
        if range_bars.strip():
            rng = range_bars.strip()
            if not rng.startswith("range:"):
                rng = f"range:{rng}"
            cmd.extend(["--scope", rng])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            return {"error": f"Échec gridline render --stems ({proc.returncode}): {proc.stderr or proc.stdout}"}
        wavs = sorted(Path(target_dir).glob("*.wav"))

    if not wavs:
        return {"error": f"Aucun WAV généré dans {target_dir}"}

    master_paths: list[Path] = []
    bus_drums_paths: list[Path] = []
    kick_paths: list[Path] = []
    bus_bass_paths: list[Path] = []
    bass_track_paths: list[Path] = []
    bus_music_paths: list[Path] = []
    fx_room_paths: list[Path] = []
    lead_paths: list[Path] = []
    harmony_paths: list[Path] = []

    for w in wavs:
        stem = w.stem
        if " - " not in stem:
            master_paths.append(w)
            continue
        label = stem.split(" - ", 1)[1].lower()
        if "bus" in label and "drum" in label:
            bus_drums_paths.append(w)
        elif "bus" in label and ("bass" in label or "808" in label):
            bus_bass_paths.append(w)
        elif "bus" in label and "music" in label:
            bus_music_paths.append(w)
        elif "fx room" in label or "reverb" in label or "fx_room" in label:
            fx_room_paths.append(w)
        elif "kick" in label:
            kick_paths.append(w)
        elif any(k in label for k in ("lead", "melody", "topline", "vox", "vocal", "solo")):
            lead_paths.append(w)
        elif any(k in label for k in ("808", "bass", "sub")):
            bass_track_paths.append(w)
        elif any(k in label for k in ("drum", "clap", "snare", "hat", "perc", "crash", "rim")):
            bus_drums_paths.append(w)
        elif any(k in label for k in ("chord", "pad", "harmony", "string", "rhodes", "arp", "counter", "keys", "piano", "guitar", "organ", "synth")):
            harmony_paths.append(w)

    ref_wav = master_paths[0] if master_paths else wavs[0]
    ref_data, sr = sf.read(str(ref_wav), always_2d=True, dtype="float64")
    ref_len = len(ref_data)
    if master_paths:
        master_mono = np.mean(ref_data, axis=1)
    else:
        sum_src = (bus_drums_paths or []) + (bus_bass_paths or bass_track_paths) + (bus_music_paths or harmony_paths + lead_paths) + fx_room_paths
        master_mono, sr = _load_and_sum_stems(sum_src or wavs, target_sr=sr, target_len=ref_len)

    pink_slope = compute_pink_slope_db_per_oct(master_mono, sr)

    # 1) Lead vs Harmony (800 Hz - 3.5 kHz) on active Lead windows
    lead_vs_harmony_db: float | None = None
    lead_active_pct: float | None = None
    if lead_paths and (harmony_paths or bus_music_paths):
        lead_mono, _ = _load_and_sum_stems(lead_paths, target_sr=sr, target_len=ref_len)
        if harmony_paths:
            harm_mono, _ = _load_and_sum_stems(harmony_paths, target_sr=sr, target_len=ref_len)
        else:
            bm_mono, _ = _load_and_sum_stems(bus_music_paths, target_sr=sr, target_len=ref_len)
            harm_mono = bm_mono - lead_mono

        lead_rms_all = float(np.sqrt(np.mean(lead_mono**2)))
        harm_rms_all = float(np.sqrt(np.mean(harm_mono**2)))
        if lead_rms_all > 1e-5 and harm_rms_all > 1e-6:
            hi_bp = min(3500.0, 0.45 * float(sr))
            sos_bp = signal.butter(4, [800.0, hi_bp], btype="bandpass", fs=sr, output="sos")
            lead_bp = signal.sosfiltfilt(sos_bp, lead_mono)
            harm_bp = signal.sosfiltfilt(sos_bp, harm_mono)
            win = max(1, int(0.05 * sr))
            n_frames = len(lead_mono) // win
            if n_frames >= 2:
                l_raw_f = lead_mono[: n_frames * win].reshape(n_frames, win)
                l_bp_f = lead_bp[: n_frames * win].reshape(n_frames, win)
                h_bp_f = harm_bp[: n_frames * win].reshape(n_frames, win)
                f_rms = np.sqrt(np.mean(l_raw_f**2, axis=1))
                thresh = max(1e-4, 0.15 * float(np.max(f_rms)))
                active = f_rms >= thresh
                if np.any(active):
                    l_band = float(np.sqrt(np.mean(l_bp_f[active] ** 2)))
                    h_band = float(np.sqrt(np.mean(h_bp_f[active] ** 2)))
                    lead_vs_harmony_db = round(float(20.0 * np.log10((l_band + 1e-9) / (h_band + 1e-9))), 2)
                    lead_active_pct = round(100.0 * float(np.mean(active)), 1)

    # 2) Kick vs Bass transient overlap (< 150 Hz) during Kick attacks
    kick_bass_overlap: float | None = None
    kick_bass_duck_ratio: float | None = None
    kick_src = kick_paths or bus_drums_paths
    bass_src = bus_bass_paths or bass_track_paths
    if has_drums and kick_src and bass_src:
        kick_mono, _ = _load_and_sum_stems(kick_src, target_sr=sr, target_len=ref_len)
        bass_mono, _ = _load_and_sum_stems(bass_src, target_sr=sr, target_len=ref_len)
        if float(np.sqrt(np.mean(kick_mono**2))) > 1e-5 and float(np.sqrt(np.mean(bass_mono**2))) > 1e-5:
            sos_lp = signal.butter(4, 150.0, btype="lowpass", fs=sr, output="sos")
            k_sub = signal.sosfiltfilt(sos_lp, kick_mono)
            b_sub = signal.sosfiltfilt(sos_lp, bass_mono)
            hop = max(1, int(0.010 * sr))
            n_f = len(k_sub) // hop
            if n_f >= 8:
                env_k = np.sqrt(np.mean(k_sub[: n_f * hop].reshape(n_f, hop) ** 2, axis=1))
                env_b = np.sqrt(np.mean(b_sub[: n_f * hop].reshape(n_f, hop) ** 2, axis=1))
                pk_thresh = max(1e-4, 0.25 * float(np.max(env_k)))
                peaks, _ = signal.find_peaks(env_k, height=pk_thresh, distance=max(1, int(0.12 / 0.010)))
                mask = np.zeros(n_f, dtype=bool)
                for pk in peaks:
                    lo = max(0, int(pk) - 1)
                    hi = min(n_f, int(pk) + 7)
                    mask[lo:hi] = True
                if np.sum(mask) >= 4:
                    ek = env_k[mask]
                    eb = env_b[mask]
                    if float(np.std(ek)) > 1e-9 and float(np.std(eb)) > 1e-9:
                        kick_bass_overlap = round(float(np.corrcoef(ek, eb)[0, 1]), 3)
                    else:
                        kick_bass_overlap = 0.0
                    kick_bass_duck_ratio = round(float(np.mean(eb) / (float(np.mean(env_b)) + 1e-9)), 3)

    diagnostics: list[str] = []
    if lead_vs_harmony_db is not None and lead_vs_harmony_db < 2.5:
        diagnostics.append(
            f"Lead masqué par l'harmonie dans 800Hz-3.5kHz ({lead_vs_harmony_db:+.2f} dB < +2.5 dB cible) -> creuser 1.5-2.5kHz sur BUS Music ou monter Lead."
        )
    if kick_bass_overlap is not None and kick_bass_overlap > 0.35:
        diagnostics.append(
            f"Chevauchement Kick/Bass <150Hz détecté (corr={kick_bass_overlap:+.2f} > 0.35, duck_ratio={kick_bass_duck_ratio}) -> vérifier routage sidechain Kick -> BUS Bass gl.comp."
        )
    if pink_slope < -5.0 or pink_slope > -3.0:
        diagnostics.append(
            f"Pente spectrale hors cible rose ({pink_slope:.2f} dB/oct hors [-5.0, -3.0]) -> {'trop sombre/boomy' if pink_slope < -5.0 else 'trop brillant/harsh'}."
        )

    return {
        "project_path": abs_proj or None,
        "stems_dir": target_dir,
        "range": range_bars or "full",
        "metrics": {
            "lead_vs_harmony_db": lead_vs_harmony_db,
            "lead_active_pct": lead_active_pct,
            "kick_bass_transient_overlap": kick_bass_overlap,
            "kick_bass_ducking_ratio": kick_bass_duck_ratio,
            "pink_slope_db_per_oct": pink_slope,
        },
        "checks": {
            "lead_hierarchy_ok": True if lead_vs_harmony_db is None else (lead_vs_harmony_db >= 2.5),
            "kick_bass_separation_ok": True if kick_bass_overlap is None else (kick_bass_overlap <= 0.35),
            "pink_slope_ok": (-5.0 <= pink_slope <= -3.0),
        },
        "stems_detected": {
            "master": master_paths[0].name if master_paths else None,
            "kick_ref": [p.name for p in kick_src],
            "bass_ref": [p.name for p in bass_src],
            "lead": [p.name for p in lead_paths],
            "harmony": [p.name for p in harmony_paths],
        },
        "diagnostics": diagnostics,
    }


VERTICAL_STAGES = ("V1_LOW_END", "V2_GROOVE", "V3_HARMONY", "V4_LEAD")

_VERTICAL_PROFILE_LIMITS: dict[str, dict[str, float]] = {
    "trap_rap":          {"low_mids_max": 24.0, "sharp_max": 1.90, "dark_sharp_max": 1.85, "pink_min": -5.5, "pink_max": -2.8},
    "pop_rnb":           {"low_mids_max": 26.0, "sharp_max": 1.95, "dark_sharp_max": 1.85, "pink_min": -5.5, "pink_max": -2.8},
    "edm_club":          {"low_mids_max": 24.0, "sharp_max": 2.05, "dark_sharp_max": 1.90, "pink_min": -5.5, "pink_max": -2.5},
    "rock_metal":        {"low_mids_max": 28.0, "sharp_max": 2.10, "dark_sharp_max": 1.90, "pink_min": -5.5, "pink_max": -2.5},
    "lofi_chill":        {"low_mids_max": 34.0, "sharp_max": 1.75, "dark_sharp_max": 1.75, "pink_min": -6.2, "pink_max": -3.0},
    "jazz_acoustic":     {"low_mids_max": 36.0, "sharp_max": 1.85, "dark_sharp_max": 1.80, "pink_min": -6.0, "pink_max": -3.0},
    "classical_ambient": {"low_mids_max": 40.0, "sharp_max": 1.75, "dark_sharp_max": 1.75, "pink_min": -6.5, "pink_max": -3.0},
}


def _detect_bass_harmony_chroma_clash(mono: np.ndarray, sr: int) -> dict[str, Any]:
    """
    Detect persistent bass (< 220 Hz) vs harmony (220-1800 Hz) minor-2nd chroma clashes
    without false-positiving on legitimate maj7/m9 voicings (requires bass pitch class to be
    weak in harmony while its semitone neighbor dominates).
    """
    if len(mono) < sr // 2:
        return {"clash_ratio": 0.0, "active_frames": 0, "clash_detected": False}
    try:
        sos_b = signal.butter(4, [42.0, 220.0], btype="bandpass", fs=sr, output="sos")
        sos_h = signal.butter(4, [220.0, min(1800.0, 0.45 * sr)], btype="bandpass", fs=sr, output="sos")
        y_b = signal.sosfiltfilt(sos_b, mono.astype(np.float64)).astype(np.float32)
        y_h = signal.sosfiltfilt(sos_h, mono.astype(np.float64)).astype(np.float32)

        hop = 2048
        # Use Constant-Q chroma with fixed tuning=0.0 so C2 (65.4Hz) and C#2 (69.3Hz)
        # are cleanly resolved (STFT bins at 44.1kHz are too coarse below 150Hz).
        c_b = librosa.feature.chroma_cqt(
            y=y_b, sr=sr, hop_length=hop, fmin=32.703, n_octaves=3, bins_per_octave=24, tuning=0.0
        )
        c_h = librosa.feature.chroma_cqt(
            y=y_h, sr=sr, hop_length=hop, fmin=130.813, n_octaves=4, bins_per_octave=24, tuning=0.0
        )
        r_b = librosa.feature.rms(y=y_b, frame_length=4096, hop_length=hop)[0]
        r_h = librosa.feature.rms(y=y_h, frame_length=4096, hop_length=hop)[0]

        n_f = min(c_b.shape[1], c_h.shape[1], len(r_b), len(r_h))
        if n_f < 3:
            return {"clash_ratio": 0.0, "active_frames": 0, "clash_detected": False}

        th_b = max(1e-4, 0.20 * float(np.max(r_b)))
        th_h = max(1e-4, 0.20 * float(np.max(r_h)))
        active = 0
        clashes = 0
        for i in range(n_f):
            if r_b[i] < th_b or r_h[i] < th_h:
                continue
            vb = c_b[:, i]
            vh = c_h[:, i]
            mb = float(np.max(vb))
            mh = float(np.max(vh))
            if mb < 1e-6 or mh < 1e-6:
                continue
            active += 1
            b_pc = int(np.argmax(vb))
            h_norm = vh / mh
            semitone_hi = float(h_norm[(b_pc + 1) % 12])
            semitone_lo = float(h_norm[(b_pc - 1) % 12])
            bass_in_harm = float(h_norm[b_pc])
            # True tonal clash: harmony strongly emphasizes semitone neighbor while bass root is absent
            if max(semitone_hi, semitone_lo) >= 0.72 and bass_in_harm <= 0.35:
                clashes += 1

        ratio = round(clashes / float(active), 3) if active > 0 else 0.0
        return {
            "clash_ratio": ratio,
            "active_frames": active,
            "clash_detected": bool(active >= 4 and ratio >= 0.35),
        }
    except Exception:
        return {"clash_ratio": 0.0, "active_frames": 0, "clash_detected": False}


def analyze_vertical_stage(
    wav_path: str,
    stage: str,
    profile: str = "trap_rap",
    blueprint: dict[str, Any] | None = None,
    stems: dict[str, str] | str | None = None,
) -> dict[str, Any]:
    """
    Incremental vertical stage audit (RFC §4.2):
      - `V1_LOW_END`: Kick + Bass/808 separation, low_end_corr < 150 Hz, sub inertia
      - `V2_GROOVE`: V1 + Snare/Clap/Hats sharpness, 2.5-4.5 kHz peaks, anti-machine-gun variance
      - `V3_HARMONY`: V2 + Chords/Pad chroma conflict vs bass, low_mids 250-500 Hz, low_mid_corr
      - `V4_LEAD`: V3 + Lead/Vocals hierarchy, breathing silence ratio, pink slope, spectral match
    Returns concise uniform JSON contract (`stage`, `verdict`, `score`, `human_diagnosis`, `next_action`).
    """
    st_norm = str(stage).strip().upper()
    if st_norm not in VERTICAL_STAGES:
        raise ValueError(f"Invalid vertical stage {stage!r}; expected one of {VERTICAL_STAGES}")

    prof = profile if profile in _VERTICAL_PROFILE_LIMITS else "trap_rap"
    limits = _VERTICAL_PROFILE_LIMITS[prof]
    bp = blueprint or {}
    ev = bp.get("emotion_vector") or {}
    valence = float(ev.get("valence", 0.0))
    is_dark = valence < -0.2
    has_drums = bool(bp["has_drums"]) if "has_drums" in bp else (prof != "classical_ambient")

    # V2_GROOVE automatically skipped when has_drums=False
    if st_norm == "V2_GROOVE" and not has_drums:
        return {
            "stage": st_norm,
            "verdict": "PASS",
            "score": 1.0,
            "skipped": True,
            "human_diagnosis": [
                f"Palier V2_GROOVE ignoré (has_drums=false pour profil '{prof}').",
            ],
            "next_action": {"code": "ADVANCE_TO_V3_HARMONY", "params": {"next_stage": "V3_HARMONY"}},
            "checks": {"skipped_no_drums": True},
            "metrics": {},
        }

    abs_wav = os.path.abspath(os.path.expanduser(wav_path))
    if not os.path.isfile(abs_wav):
        raise FileNotFoundError(f"Vertical stage WAV not found: {abs_wav}")

    data, sr = sf.read(abs_wav, dtype="float64", always_2d=True)
    mono = np.mean(data, axis=1)
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
    rms_db = _lin_to_db(rms)
    if rms_db < -48.0:
        return {
            "stage": st_norm,
            "verdict": "FAIL",
            "score": 0.0,
            "skipped": False,
            "human_diagnosis": [f"Rendu {st_norm} silencieux (RMS {rms_db} dBFS)."],
            "next_action": {"code": "CHECK_LAYER_MUTING_OR_LEVEL", "params": {"stage": st_norm}},
            "checks": {"audible": False},
            "metrics": {"rms_dbfs": rms_db},
        }

    # Shared spectral & stereo calculations
    nperseg = min(16384, max(256, len(mono)))
    freqs, psd = signal.welch(mono, fs=sr, nperseg=nperseg, scaling="spectrum")
    tot_e = float(np.sum(psd)) + 1e-18

    def _band_pct(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < min(hi, sr * 0.495))
        return round(100.0 * float(np.sum(psd[m])) / tot_e, 2) if np.any(m) else 0.0

    # Low-end stereo correlation (< 150 Hz) & low-mid (200-800 Hz)
    low_end_corr = 1.0
    low_mid_corr = 1.0
    if data.shape[1] >= 2:
        sos_le = signal.butter(4, min(150.0, sr * 0.45), btype="lowpass", fs=sr, output="sos")
        l_le, r_le = signal.sosfiltfilt(sos_le, data[:, 0]), signal.sosfiltfilt(sos_le, data[:, 1])
        d_le = float(np.sqrt(np.sum(l_le**2) * np.sum(r_le**2))) + 1e-18
        low_end_corr = round(float(np.sum(l_le * r_le) / d_le), 3)

        sos_lm = signal.butter(4, [200.0, 800.0], btype="bandpass", fs=sr, output="sos")
        l_lm, r_lm = signal.sosfiltfilt(sos_lm, data[:, 0]), signal.sosfiltfilt(sos_lm, data[:, 1])
        d_lm = float(np.sqrt(np.sum(l_lm**2) * np.sum(r_lm**2))) + 1e-18
        low_mid_corr = round(float(np.sum(l_lm * r_lm) / d_lm), 3)

    stems_dir = stems if isinstance(stems, str) and os.path.isdir(stems) else ""
    stems_map = stems if isinstance(stems, dict) else {}

    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {"rms_dbfs": rms_db}
    diag: list[str] = []
    next_action: dict[str, Any] = {"code": "ADVANCE_NEXT_STAGE", "params": {}}
    score = 1.0

    if st_norm == "V1_LOW_END":
        sub_20_45 = _band_pct(20.0, 45.0)
        low_50_100 = _band_pct(50.0, 100.0)
        low_total = max(0.01, _band_pct(20.0, 150.0))
        infra_ratio = round(sub_20_45 / low_total, 3)

        overlap: float | None = None
        if stems_dir:
            hier = analyze_stem_hierarchy(stems_dir=stems_dir, has_drums=has_drums)
            overlap = (hier.get("metrics") or {}).get("kick_bass_transient_overlap")
        elif stems_map.get("kick") and stems_map.get("bass"):
            k_d, k_sr = sf.read(stems_map["kick"], always_2d=True, dtype="float64")
            b_d, b_sr = sf.read(stems_map["bass"], always_2d=True, dtype="float64")
            km = np.mean(k_d, axis=1)
            bm = np.mean(b_d, axis=1)
            n_min = min(len(km), len(bm))
            if n_min > sr // 4:
                sos_lp = signal.butter(4, 150.0, btype="lowpass", fs=k_sr, output="sos")
                ek = np.abs(signal.sosfiltfilt(sos_lp, km[:n_min]))
                eb = np.abs(signal.sosfiltfilt(sos_lp, bm[:n_min]))
                hop = max(1, int(0.01 * k_sr))
                nf = n_min // hop
                ek_f = np.sqrt(np.mean(ek[: nf * hop].reshape(nf, hop) ** 2, axis=1))
                eb_f = np.sqrt(np.mean(eb[: nf * hop].reshape(nf, hop) ** 2, axis=1))
                pks, _ = signal.find_peaks(ek_f, height=max(1e-4, 0.25 * float(np.max(ek_f))), distance=12)
                msk = np.zeros(nf, dtype=bool)
                for pk in pks:
                    msk[max(0, pk - 1) : min(nf, pk + 7)] = True
                if np.sum(msk) >= 4 and np.std(ek_f[msk]) > 1e-9 and np.std(eb_f[msk]) > 1e-9:
                    overlap = round(float(np.corrcoef(ek_f[msk], eb_f[msk])[0, 1]), 3)

        corr_thresh = 0.70 if has_drums else 0.40
        corr_ok = (low_end_corr >= corr_thresh) or (low_total < 5.0 and not has_drums)
        overlap_ok = True if overlap is None else (overlap <= 0.30)
        infra_ok = (infra_ratio <= 0.42) or (low_total < 8.0)

        checks.update({"low_end_corr_ok": corr_ok, "transient_overlap_ok": overlap_ok, "infra_sub_ok": infra_ok})
        metrics.update({
            "low_end_corr": low_end_corr,
            "kick_bass_transient_overlap": overlap,
            "energy_50_100hz_pct": low_50_100,
            "infra_sub_ratio_lt45hz": infra_ratio,
        })

        if not corr_ok:
            score -= 0.35
            diag.append(f"Phase grave < 150 Hz trop large (corr={low_end_corr} < {corr_thresh}) : forcer mono sur BUS Bass.")
            next_action = {"code": "MONO_LOW_END", "params": {"mixer": 18, "plugin": "gl.stereo", "Width": 0.0}}
        if not overlap_ok:
            score -= 0.35
            diag.append(f"Masquage Kick/Basse < 150 Hz (overlap={overlap} > 0.30) : renforcer sidechain ou apply_haas_pocket +12 ms.")
            next_action = {"code": "ADJUST_SIDECHAIN_OR_POCKET", "params": {"sidechain_level": 1.0, "haas_bass_ms": 12.0}}
        if not infra_ok:
            score -= 0.25
            diag.append(f"Infra-sub excessif < 45 Hz ({infra_ratio*100:.0f}% du grave) : monter Root note +12 st ou HPF 32 Hz.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = {"code": "RAISE_SUB_OCTAVE_OR_HPF", "params": {"root_note_delta_st": 12, "hpf_hz": 32}}
        if not diag:
            diag.append(f"Palier V1_LOW_END propre : phase grave {low_end_corr} (>= {corr_thresh}), bas du spectre maîtrisé.")
            next_action = {"code": "ADVANCE_TO_V2_GROOVE", "params": {"next_stage": "V2_GROOVE"}}

    elif st_norm == "V2_GROOVE":
        third_oct = _third_octave_profile(mono.astype(np.float32), sr)
        peaks_25_45 = [
            b for b in third_oct
            if 2500.0 <= b["freq_hz"] <= 4500.0 and b.get("is_local_peak", True) and b["local_excess_db"] >= 5.5
        ]
        sharp_max = limits["dark_sharp_max"] if is_dark else limits["sharp_max"]
        sharp_val = 0.0
        try:
            import mosqito
            with contextlib.redirect_stdout(sys.stderr):
                sharp_val = round(float(mosqito.sharpness_din_st(mono[: min(len(mono), sr * 8)].astype(np.float64), fs=sr, weighting="din")), 3)
        except Exception:
            sharp_val = 0.0

        # Hi-hat band (5-14 kHz) anti-machine-gun variance check
        hat_cv: float | None = None
        hat_ok = True
        try:
            sos_hh = signal.butter(4, [5000.0, min(14000.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
            y_hh = signal.sosfiltfilt(sos_hh, mono)
            if float(np.sqrt(np.mean(y_hh**2))) > 1e-4:
                o_env = librosa.onset.onset_strength(y=y_hh.astype(np.float32), sr=sr, hop_length=256)
                pks, _ = signal.find_peaks(o_env, height=0.25 * float(np.max(o_env)), distance=max(2, int(0.06 * sr / 256)))
                if len(pks) >= 6:
                    vals = o_env[pks]
                    hat_cv = round(float(np.std(vals) / (float(np.mean(vals)) + 1e-9)), 3)
                    hat_ok = hat_cv >= 0.03
        except Exception:
            pass

        sharp_ok = sharp_val < sharp_max
        peaks_ok = len(peaks_25_45) == 0
        checks.update({"sharpness_ok": sharp_ok, "no_harsh_drum_peaks": peaks_ok, "hat_groove_variance_ok": hat_ok})
        metrics.update({
            "sharpness_din_acum": sharp_val,
            "sharp_max": sharp_max,
            "peaks_2500_4500hz": peaks_25_45[:3],
            "hat_velocity_cv": hat_cv,
            "crest_factor_db": round(_lin_to_db(float(np.max(np.abs(mono)))) - rms_db, 2),
        })

        if not sharp_ok:
            score -= 0.35
            diag.append(f"Batterie trop aiguë/agressive (sharpness={sharp_val} >= {sharp_max}) : baisser hats -2 dB ou creuser 3.2 kHz.")
            next_action = {"code": "TAME_DRUMS_OR_HATS", "params": {"eq_cut_hz": 3200.0, "eq_cut_db": -3.0, "hat_volume_delta_db": -2.0}}
        if not peaks_ok:
            score -= 0.30
            f_pk = peaks_25_45[0]["freq_hz"]
            diag.append(f"Pic étroit batterie à {f_pk:.0f} Hz (+{peaks_25_45[0]['local_excess_db']} dB) : EQ Peak -3 dB.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = {"code": "TAME_DRUMS_OR_HATS", "params": {"eq_cut_hz": f_pk, "eq_cut_db": -3.0, "q": 2.8}}
        if not hat_ok:
            score -= 0.25
            diag.append(f"Hi-hats trop mécaniques ('mitraillette', CV={hat_cv}) : appliquer groove_lib.apply_anti_machine_gun.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = {"code": "APPLY_ANTI_MACHINE_GUN", "params": {"target": "hihat_perc"}}
        if not diag:
            diag.append(f"Palier V2_GROOVE validé : sharpness={sharp_val} (< {sharp_max}), 0 pic 2.5-4.5 kHz, groove vivant.")
            next_action = {"code": "ADVANCE_TO_V3_HARMONY", "params": {"next_stage": "V3_HARMONY"}}

    elif st_norm == "V3_HARMONY":
        low_mids_pct = _band_pct(250.0, 500.0)
        lm_max = limits["low_mids_max"]
        corr_min = 0.55 if prof == "classical_ambient" else 0.65
        chroma_info = _detect_bass_harmony_chroma_clash(mono, sr)

        lm_ok = low_mids_pct <= lm_max
        corr_ok = low_mid_corr >= corr_min
        chroma_ok = not chroma_info["clash_detected"]

        checks.update({"low_mids_ok": lm_ok, "low_mid_corr_ok": corr_ok, "chroma_consonance_ok": chroma_ok})
        metrics.update({
            "low_mids_pct": low_mids_pct,
            "low_mids_max": lm_max,
            "low_mid_corr": low_mid_corr,
            "low_mid_corr_min": corr_min,
            "bass_harmony_chroma_clash": chroma_info,
        })

        if not chroma_ok:
            score -= 0.45
            diag.append(
                f"Conflit tonal audio détecté entre basse et accords (clash_ratio={chroma_info['clash_ratio']}) : "
                "vérifier Root note / Fine tune ou accordage de la couche Chords."
            )
            next_action = {"code": "FIX_HARMONY_CHROMA_CLASH", "params": {"target_layer": "Chords", "clash_ratio": chroma_info["clash_ratio"]}}
        if not lm_ok:
            score -= 0.30
            diag.append(f"Bas-médiums chargés (250-500 Hz = {low_mids_pct}% > {lm_max}%) : HPF 180 Hz + Peak -3.5 dB @ 300 Hz sur Chords.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = {"code": "CARVE_HARMONY_LOW_MIDS", "params": {"hpf_hz": 180.0, "peak_hz": 300.0, "peak_db": -3.5}}
        if not corr_ok:
            score -= 0.25
            diag.append(f"Phase bas-médiums trop large (corr={low_mid_corr} < {corr_min}) : appliquer stabilize_stereo_low_mid.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = {"code": "STABILIZE_HARMONY_PHASE", "params": {"cutoff_hz": 650.0, "side_keep_low": 0.18}}
        if not diag:
            diag.append(f"Palier V3_HARMONY validé : bas-médiums {low_mids_pct}% (<= {lm_max}%), phase {low_mid_corr}, 0 conflit tonal.")
            next_action = {"code": "ADVANCE_TO_V4_LEAD", "params": {"next_stage": "V4_LEAD"}}

    else:  # V4_LEAD
        pink_slope = compute_pink_slope_db_per_oct(mono, sr)
        lead_vs_harm: float | None = None
        lead_silence_ratio: float | None = None

        if stems_dir:
            hier = analyze_stem_hierarchy(stems_dir=stems_dir, has_drums=has_drums)
            hm = hier.get("metrics") or {}
            lead_vs_harm = hm.get("lead_vs_harmony_db")
            if hm.get("lead_active_pct") is not None:
                lead_silence_ratio = round(1.0 - float(hm["lead_active_pct"]) / 100.0, 3)
        elif stems_map.get("lead") and stems_map.get("harmony"):
            ld, lsr = sf.read(stems_map["lead"], always_2d=True, dtype="float64")
            hd, _ = sf.read(stems_map["harmony"], always_2d=True, dtype="float64")
            lm = np.mean(ld, axis=1)
            hm_s = np.mean(hd, axis=1)
            n_min = min(len(lm), len(hm_s))
            if n_min > lsr // 4:
                sos_bp = signal.butter(4, [800.0, min(3500.0, 0.45 * lsr)], btype="bandpass", fs=lsr, output="sos")
                l_bp = signal.sosfiltfilt(sos_bp, lm[:n_min])
                h_bp = signal.sosfiltfilt(sos_bp, hm_s[:n_min])
                win = max(1, int(0.05 * lsr))
                nf = n_min // win
                l_raw_f = np.sqrt(np.mean(lm[: nf * win].reshape(nf, win) ** 2, axis=1))
                act = l_raw_f >= max(1e-4, 0.15 * float(np.max(l_raw_f)))
                if np.any(act):
                    lb = float(np.sqrt(np.mean(l_bp[: nf * win].reshape(nf, win)[act] ** 2)))
                    hb = float(np.sqrt(np.mean(h_bp[: nf * win].reshape(nf, win)[act] ** 2)))
                    lead_vs_harm = round(float(20.0 * math.log10((lb + 1e-9) / (hb + 1e-9))), 2)
                    lead_silence_ratio = round(1.0 - float(np.mean(act)), 3)

        spec_match = None
        try:
            scripts_d = str(Path(__file__).resolve().parents[2] / "scripts")
            if scripts_d not in sys.path:
                sys.path.insert(0, scripts_d)
            from bootstrap_golden_refs import compute_spectral_match
            sm = compute_spectral_match(mono=mono, sr=sr, profile=prof)
            spec_match = sm.get("spectral_match_score")
        except Exception:
            spec_match = None

        lead_ok = True if lead_vs_harm is None else (lead_vs_harm >= 2.0)
        slope_ok = (limits["pink_min"] <= pink_slope <= limits["pink_max"])
        mid_1_3k_pct = _band_pct(1000.0, 3000.0)
        masking_1_3k_ok = mid_1_3k_pct <= 48.0

        checks.update({"lead_hierarchy_ok": lead_ok, "pink_slope_ok": slope_ok, "no_mid_congestion_1_3k": masking_1_3k_ok})
        metrics.update({
            "lead_vs_harmony_db": lead_vs_harm,
            "lead_silence_ratio": lead_silence_ratio,
            "pink_slope_db_per_oct": pink_slope,
            "pink_slope_range": [limits["pink_min"], limits["pink_max"]],
            "spectral_match_score": spec_match,
            "band_1_3khz_pct": mid_1_3k_pct,
        })

        if not lead_ok:
            score -= 0.40
            diag.append(f"Lead masqué par l'harmonie ({lead_vs_harm:+.2f} dB < +2.0 dB) : EQ miroir -3 dB @ 1.8 kHz sur Chords ou +2 dB Lead.")
            next_action = {"code": "CARVE_LEAD_POCKET", "params": {"chords_eq_hz": 1800.0, "chords_eq_db": -3.0, "lead_gain_db": 2.0}}
        if not slope_ok:
            score -= 0.25
            direction = "trop sombre/boomy" if pink_slope < limits["pink_min"] else "trop brillant/agressif"
            diag.append(f"Pente rose hors cible ({pink_slope:.2f} dB/oct, {direction}) : ajuster équilibre aigu/grave de la couche Lead.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = {
                    "code": "ADJUST_LEAD_TONAL_BALANCE",
                    "params": {"pink_slope_db_per_oct": pink_slope, "direction": direction},
                }
        if not masking_1_3k_ok:
            score -= 0.20
            diag.append(f"Congestion 1-3 kHz ({mid_1_3k_pct}%) : creuser Chords -3 dB @ 2 kHz.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = {"code": "CARVE_LEAD_POCKET", "params": {"chords_eq_hz": 2000.0, "chords_eq_db": -3.0}}
        if not diag:
            diag.append(
                f"Palier V4_LEAD validé : pente rose {pink_slope:.2f} dB/oct"
                + (f", Lead/Harmonie {lead_vs_harm:+.1f} dB" if lead_vs_harm is not None else "")
                + (f", spectral_match={spec_match}" if spec_match is not None else "")
                + " -> noyau verrouillable."
            )
            next_action = {"code": "LOCK_CORE_LOOP", "params": {"core_loop_locked": True}}

    all_ok = all(checks.values())
    verdict = "PASS" if all_ok else ("SWAP" if score < 0.45 else "FIX")
    return {
        "stage": st_norm,
        "verdict": verdict,
        "score": round(max(0.0, min(1.0, score)), 4),
        "skipped": False,
        "human_diagnosis": diag[:3],
        "next_action": next_action,
        "checks": checks,
        "metrics": metrics,
    }


def inspect_pipewire() -> dict[str, Any]:
    """Vérifie le quantum PipeWire, la présence d'un casque Bluetooth et les xruns temps réel."""
    try:
        proc = subprocess.run(["pw-top", "-b", "-n", "2"], capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        return {"error": "pw-top introuvable (PipeWire non installé ?)"}

    if proc.returncode != 0:
        return {"error": proc.stderr or f"pw-top code {proc.returncode}"}

    # Garde le deuxième instantané s'il y en a plusieurs
    sections = proc.stdout.strip().split("\nS ")
    nodes: list[dict[str, Any]] = []
    min_quantum = None
    xrun_nodes: list[str] = []
    bluetooth_active = False

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("S "):
            continue
        m = re.match(r"^([RIA])\s+(\d+)\s+(\d+)\s+(\d+)\s+\S+\s+\S+\s+\S+\s+\S+\s+(\d+)\s+(.+)$", line)
        if not m:
            continue
        state, node_id, quant, rate, err_cnt, name = (
            m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6).strip()
        )
        if state == "R" and quant > 0:
            min_quantum = quant if min_quantum is None else min(min_quantum, quant)
        if "bluez" in name.lower():
            bluetooth_active = True
        if err_cnt > 0:
            xrun_nodes.append(f"{name} (ERR={err_cnt})")
        nodes.append({"state": state, "id": node_id, "quantum": quant, "rate": rate, "err": err_cnt, "name": name})

    warnings: list[str] = []
    if bluetooth_active and min_quantum is not None and min_quantum <= 512:
        warnings.append(
            f"Casque Bluetooth détecté avec quantum faible ({min_quantum} samples @ 48kHz ≈ {min_quantum/48:.1f} ms) -> risque de grésillements/craquements sous Gridline."
        )
    if xrun_nodes:
        warnings.append(f"Xruns audio actifs sur : {', '.join(sorted(set(xrun_nodes)))}")

    return {
        "active_quantum": min_quantum,
        "bluetooth_sink_detected": bluetooth_active,
        "xrun_nodes": sorted(set(xrun_nodes)),
        "warnings": warnings,
        "recommended_command": "pw-metadata -n settings 0 clock.force-quantum 1024" if warnings else None,
        "node_count": len(nodes),
    }
