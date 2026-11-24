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
    strict_s3: bool = False,
    s3_contract: dict[str, Any] | None = None,
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
        strict_s3=strict_s3,
        s3_contract=s3_contract,
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
        "strict_s3": strict_s3,
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
        "s3_diagnostics": musicality.get("s3_diagnostics"),
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


def _load_mono_audio_input(wav_or_arr: Any, default_sr: int = 44100) -> tuple[np.ndarray, int]:
    if isinstance(wav_or_arr, str):
        data, sr = sf.read(os.path.abspath(os.path.expanduser(wav_or_arr)), dtype="float64", always_2d=True)
        return np.mean(data, axis=1), int(sr)
    if isinstance(wav_or_arr, (tuple, list)) and len(wav_or_arr) == 2 and isinstance(wav_or_arr[0], np.ndarray):
        arr = np.asarray(wav_or_arr[0], dtype=np.float64)
        if arr.ndim == 2:
            arr = np.mean(arr, axis=1)
        return arr, int(wav_or_arr[1])
    arr = np.asarray(wav_or_arr, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1)
    return arr, int(default_sr)


def analyze_kick_bass_transient_phase(
    kick_wav: Any,
    bass_wav: Any,
    sr: int = 44100,
    min_corr: float = 0.25,
) -> dict[str, Any]:
    """
    Échelle S1 / Axe V (Phase 6 §3.1) :
    Alignement de phase par cycle sous 120 Hz sur les 35 ms d'attaque commune Kick/Basse
    pour tau in [-10 ms, +10 ms].
    """
    k_mono, k_sr = _load_mono_audio_input(kick_wav, default_sr=sr)
    b_mono, b_sr = _load_mono_audio_input(bass_wav, default_sr=sr)
    if b_sr != k_sr:
        g = math.gcd(k_sr, b_sr)
        b_mono = signal.resample_poly(b_mono, k_sr // g, b_sr // g)
    eff_sr = k_sr
    n_min = min(len(k_mono), len(b_mono))
    if n_min < max(256, int(0.05 * eff_sr)):
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "r_0": None,
            "r_max": None,
            "r_inv": None,
            "tau_star_ms": 0.0,
            "shared_transients": 0,
            "next_action": {"code": "KEEP", "params": {}},
        }
    k_mono = k_mono[:n_min]
    b_mono = b_mono[:n_min]
    # Cap long stems (> 30 s) to first active 24s window containing kick hits (< 0.1s CPU)
    if len(k_mono) > int(30.0 * eff_sr):
        k_pk = float(np.max(np.abs(k_mono))) + 1e-12
        hits = np.where(np.abs(k_mono) >= 0.25 * k_pk)[0]
        w_start = max(0, int(hits[0]) - int(0.05 * eff_sr)) if len(hits) else 0
        w_end = min(len(k_mono), w_start + int(24.0 * eff_sr))
        k_mono = k_mono[w_start:w_end]
        b_mono = b_mono[w_start:w_end]
        n_min = len(k_mono)

    # Decimate low-end analysis to ~11.025 kHz (Nyquist 5.5 kHz >> 120 Hz, 4x faster on CPU)
    if eff_sr >= 22050:
        d_pre = max(1, int(eff_sr // 11025))
        k_mono = k_mono[::d_pre]
        b_mono = b_mono[::d_pre]
        eff_sr = max(1, int(round(float(eff_sr) / float(d_pre))))
        n_min = len(k_mono)

    sos_lp = signal.butter(4, min(120.0, 0.45 * eff_sr), btype="lowpass", fs=eff_sr, output="sos")
    k_lp = signal.sosfilt(sos_lp, k_mono)
    b_lp = signal.sosfilt(sos_lp, b_mono)

    # Detect Kick attacks purely via fast <120 Hz + fullband RMS envelope rise (<2 ms CPU, 0 JIT)
    onsets_s: list[int] = []
    hop_rms = max(1, int(0.003 * eff_sr))
    nf = len(k_mono) // hop_rms
    if nf >= 4:
        k_env = np.sqrt(np.mean(k_mono[: nf * hop_rms].reshape(nf, hop_rms) ** 2, axis=1))
        pk_max = float(np.max(k_env))
        if pk_max > 0.01:
            pk_h = max(1e-4, 0.28 * pk_max)
            pks, _ = signal.find_peaks(k_env, height=pk_h, distance=max(2, int(0.08 * eff_sr / hop_rms)))
            if len(pks) == 0 or (int(pks[0]) > 4 and float(np.max(k_env[:3])) >= pk_h):
                nz = np.where(k_env[: max(4, int(0.02 * eff_sr / hop_rms))] >= 0.15 * pk_max)[0]
                if len(nz):
                    onsets_s.append(max(0, int(nz[0]) * hop_rms))
            for pk in pks:
                loc_h = float(k_env[int(pk)])
                bt = int(pk)
                while bt > 0 and k_env[bt - 1] >= 0.15 * loc_h and (int(pk) - bt) < 8:
                    bt -= 1
                s_cand = max(0, bt * hop_rms)
                if not any(abs(s_cand - o) < int(0.06 * eff_sr) for o in onsets_s):
                    onsets_s.append(s_cand)

    onsets_s.sort()
    # Decimate already low-passed (<120 Hz) signals to ~5.5 kHz for O(1) vectorized C cross-corr (<5 ms)
    dec = max(1, int(round(float(eff_sr) / 5512.0)))
    sr_sub = float(eff_sr) / float(dec)
    k_sub = k_lp[::dec]
    b_sub = b_lp[::dec]
    n_sub = len(k_sub)

    win_n = max(16, int(round(0.035 * sr_sub)))
    max_lag = max(2, int(round(0.010 * sr_sub)))
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    bass_thresh_rms = 10.0 ** (-30.0 / 20.0)  # -30 dBFS = 0.03162

    corr_curves: list[np.ndarray] = []
    for t_k in onsets_s[:24]:
        tk_sub = t_k // dec
        end_k = min(n_sub, tk_sub + win_n)
        seg_len = end_k - tk_sub
        if seg_len < max(12, win_n // 2):
            continue
        b_rms = float(np.sqrt(np.mean(b_sub[tk_sub:end_k] ** 2)))
        if b_rms < bass_thresh_rms:
            continue
        xk = k_sub[tk_sub:end_k]
        xk_norm = float(np.sum(xk ** 2))
        if xk_norm < 1e-10:
            continue

        lo_b = tk_sub - max_lag
        hi_b = tk_sub + seg_len + max_lag
        pad_l = max(0, -lo_b)
        pad_r = max(0, hi_b - n_sub)
        b_slice = b_sub[max(0, lo_b) : min(n_sub, hi_b)]
        if pad_l or pad_r:
            b_slice = np.pad(b_slice, (pad_l, pad_r))
        win_mat = np.lib.stride_tricks.sliding_window_view(b_slice, seg_len)[: len(lags)]
        dots = win_mat @ xk
        denoms = np.sqrt(xk_norm * np.sum(win_mat ** 2, axis=1)) + 1e-12
        corr_curves.append(dots / denoms)

    if not corr_curves:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "r_0": None,
            "r_max": None,
            "r_inv": None,
            "tau_star_ms": 0.0,
            "shared_transients": 0,
            "next_action": {"code": "KEEP", "params": {}},
        }

    mean_curve = np.median(np.vstack(corr_curves), axis=0)
    zero_idx = int(max_lag)
    r_0 = round(float(mean_curve[zero_idx]), 4)
    best_idx = int(np.argmax(mean_curve))
    r_max = round(float(mean_curve[best_idx]), 4)
    tau_star_ms = round(float(lags[best_idx]) * 1000.0 / float(sr_sub), 2)
    r_inv = round(float(np.max(-mean_curve)), 4)

    if r_0 >= float(min_corr):
        verdict = "PASS"
        status = "PASS"
        next_action: dict[str, Any] = {"code": "KEEP", "params": {}}
        score = round(min(1.0, 0.70 + 0.30 * max(0.0, r_0)), 3)
    elif r_0 < -0.15 and (-r_0) > 0.50:
        verdict = "FIX"
        status = "FAIL"
        next_action = {"code": "INVERT_808_POLARITY", "params": {"r_0": r_0}}
        score = round(max(0.10, 0.50 + r_0 * 0.4), 3)
    elif r_0 < float(min_corr) and r_max >= 0.55 and abs(tau_star_ms) <= 10.0:
        verdict = "FIX"
        status = "FAIL"
        delta_ms = round(-tau_star_ms, 2)
        next_action = {
            "code": "SHIFT_BASS_MICRO_MS",
            "params": {"delta_ms": delta_ms, "tau_star_ms": tau_star_ms, "r_0": r_0, "r_max": r_max},
        }
        score = round(max(0.25, 0.55 + r_0 * 0.3), 3)
    elif r_0 < -0.15 and r_inv > 0.50:
        verdict = "FIX"
        status = "FAIL"
        next_action = {"code": "INVERT_808_POLARITY", "params": {"r_0": r_0, "r_inv": r_inv}}
        score = round(max(0.15, 0.45 + r_0 * 0.3), 3)
    else:
        verdict = "FIX"
        status = "FAIL"
        next_action = {"code": "ADJUST_SIDECHAIN_OR_POCKET", "params": {"r_0": r_0, "r_max": r_max}}
        score = round(max(0.20, 0.50 + r_0 * 0.3), 3)

    return {
        "status": status,
        "verdict": verdict,
        "score": score,
        "r_0": r_0,
        "r_max": r_max,
        "r_inv": r_inv,
        "tau_star_ms": tau_star_ms,
        "shared_transients": len(corr_curves),
        "next_action": next_action,
    }


def analyze_sustained_spectral_flux(
    wav_or_arr: Any,
    sr: int = 44100,
    min_flux: float = 0.015,
    role: str = "pad",
) -> dict[str, Any]:
    """
    Échelle S1 / Axe H (Phase 6 §3.1) :
    Sur notes tenues > 0.4 s (hors percussions), mesure le flux spectral normalisé
    trame à trame sur la portion centrale [20%, 80%] + kurtosis (TIMESTRETCH_ARTIFACT).
    """
    low_role = (role or "").lower()
    if any(k in low_role for k in ("kick", "snare", "clap", "hat", "perc", "drum")):
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "spectral_flux_norm": None,
            "flux_kurtosis": None,
            "warning": None,
            "next_action": {"code": "KEEP", "params": {}},
        }

    mono, eff_sr = _load_mono_audio_input(wav_or_arr, default_sr=sr)
    if len(mono) > int(24.0 * eff_sr):
        # Analyze centered 24s window so long 2-min stems run in <0.1s CPU
        mid = len(mono) // 2
        half = int(12.0 * eff_sr)
        mono = mono[max(0, mid - half) : min(len(mono), mid + half)]
    if len(mono) < int(0.42 * eff_sr):
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "spectral_flux_norm": None,
            "flux_kurtosis": None,
            "warning": None,
            "next_action": {"code": "KEEP", "params": {}},
        }

    hop = 256
    n_fft = 1024
    rms_frames = librosa.feature.rms(y=mono.astype(np.float32), frame_length=n_fft, hop_length=hop)[0]
    pk_rms = float(np.max(rms_frames)) + 1e-12
    active = rms_frames > (0.15 * pk_rms)

    # Find contiguous sustained segments > 0.4 s
    min_frames = max(4, int(round(0.40 * eff_sr / hop)))
    segments: list[tuple[int, int]] = []
    cur_s: int | None = None
    for idx, flag in enumerate(active):
        if flag and cur_s is None:
            cur_s = idx
        elif not flag and cur_s is not None:
            if idx - cur_s >= min_frames:
                segments.append((cur_s, idx))
            cur_s = None
    if cur_s is not None and len(active) - cur_s >= min_frames:
        segments.append((cur_s, len(active)))

    if not segments:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "spectral_flux_norm": None,
            "flux_kurtosis": None,
            "warning": None,
            "next_action": {"code": "KEEP", "params": {}},
        }

    stft_mag = np.abs(librosa.stft(mono.astype(np.float32), n_fft=n_fft, hop_length=hop))
    flux_vals: list[np.ndarray] = []
    for f0, f1 in segments:
        seg_len = f1 - f0
        c0 = f0 + int(round(0.20 * seg_len))
        c1 = min(stft_mag.shape[1], f0 + int(round(0.80 * seg_len)))
        if c1 - c0 < 3:
            continue
        sub = stft_mag[:, c0:c1]
        norms = np.linalg.norm(sub[:, 1:], axis=0) + 1e-9
        diffs = np.linalg.norm(sub[:, 1:] - sub[:, :-1], axis=0)
        flux_vals.append(diffs / norms)

    if not flux_vals:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "spectral_flux_norm": None,
            "flux_kurtosis": None,
            "warning": None,
            "next_action": {"code": "KEEP", "params": {}},
        }

    all_flux = np.concatenate(flux_vals)
    f_med = round(float(np.median(all_flux)), 5)
    from scipy.stats import kurtosis as _scipy_kurtosis

    flux_kurt = (
        round(float(_scipy_kurtosis(all_flux, fisher=True)), 2)
        if len(all_flux) >= 8 and float(np.std(all_flux)) > 1e-9
        else 0.0
    )

    static_timbre = f_med < float(min_flux)
    timestretch_art = flux_kurt > 25.0
    warning: str | None = None
    next_action: dict[str, Any] = {"code": "KEEP", "params": {}}
    score = 1.0
    if static_timbre:
        warning = "STATIC_TIMBRE"
        next_action = {
            "code": "ADD_SUBTLE_MODULATION_OR_LPF_MOVEMENT",
            "params": {"spectral_flux_norm": f_med, "min_flux": min_flux},
        }
        score = round(max(0.35, min(0.64, f_med / max(1e-5, float(min_flux)) * 0.65)), 3)
    elif timestretch_art:
        warning = "TIMESTRETCH_ARTIFACT"
        next_action = {
            "code": "FIX_TIMESTRETCH_ARTIFACT",
            "params": {"flux_kurtosis": flux_kurt},
        }
        score = 0.50

    return {
        "status": "WARN" if warning else "PASS",
        "verdict": "FIX" if warning else "PASS",
        "score": score,
        "spectral_flux_norm": f_med,
        "flux_kurtosis": flux_kurt,
        "sustained_segments": len(segments),
        "warning": warning,
        "next_action": next_action,
    }


def _detect_highres_onsets_s(mono: np.ndarray, sr: int) -> list[float]:
    if len(mono) < 256 or float(np.max(np.abs(mono))) < 1e-4:
        return []
    hop = 64
    o_env = librosa.onset.onset_strength(
        y=mono.astype(np.float32),
        sr=sr,
        hop_length=hop,
        n_fft=512,
        n_mels=48,
        fmin=30.0,
        fmax=min(16000.0, 0.45 * float(sr)),
    )
    pre_m = max(1, int(round(0.030 * sr / hop)))
    avg_m = max(2, int(round(0.080 * sr / hop)))
    wait_m = max(2, int(round(0.060 * sr / hop)))
    frames = librosa.onset.onset_detect(
        onset_envelope=o_env,
        sr=sr,
        hop_length=hop,
        pre_max=pre_m,
        post_max=pre_m,
        pre_avg=avg_m,
        post_avg=avg_m,
        delta=0.07,
        wait=wait_m,
        backtrack=True,
        units="frames",
    )
    samples = [int(f * hop) for f in frames]
    abs_m = np.abs(mono)
    pk = float(np.max(abs_m))
    if not samples:
        nz = np.where(abs_m >= 0.12 * pk)[0]
        if len(nz):
            samples = [int(nz[0])]
    elif int(samples[0]) > int(0.025 * sr) and float(np.max(abs_m[: int(0.015 * sr)])) >= 0.25 * pk:
        # Handle initial hit right near sample 0 missed by peak-picking window
        nz0 = np.where(abs_m[: int(0.015 * sr)] >= 0.08 * pk)[0]
        if len(nz0):
            samples.insert(0, int(nz0[0]))

    refined_s: list[float] = []
    look_back = int(0.025 * sr)
    look_fwd = int(0.025 * sr)
    win_cmp = int(0.012 * sr)
    for s_idx in samples:
        lo = max(0, s_idx - look_back)
        hi = min(len(mono), s_idx + look_fwd)
        seg = abs_m[lo:hi]
        if len(seg) < 16:
            refined_s.append(float(s_idx) / float(sr))
            continue
        loc_pk = float(np.max(seg))
        if loc_pk < 1e-4:
            continue
        # Snap to exact threshold crossing where signal rises above 8% of local peak
        nz = np.where(seg >= 0.08 * loc_pk)[0]
        exact_idx = lo + int(nz[0]) if len(nz) else s_idx
        if exact_idx > int(0.015 * sr) and (exact_idx + win_cmp) < len(mono):
            pre_rms = float(np.sqrt(np.mean(mono[max(0, exact_idx - win_cmp) : exact_idx] ** 2)))
            post_rms = float(np.sqrt(np.mean(mono[exact_idx : exact_idx + win_cmp] ** 2)))
            if pre_rms >= post_rms * 0.95:
                continue
        t_s = float(exact_idx) / float(sr)
        if not refined_s or (t_s - refined_s[-1]) >= 0.050:
            refined_s.append(t_s)
    return refined_s


def _lookup_haas_target_ms(role_name: str, haas_targets_ms: dict[str, float] | None) -> float:
    if not haas_targets_ms:
        return 0.0
    low = role_name.lower()
    if low in haas_targets_ms:
        return float(haas_targets_ms[low])
    for kw, val in haas_targets_ms.items():
        if kw.lower() in low or low in kw.lower():
            return float(val)
    return 0.0


def analyze_audio_pocket_alignment(
    stems_by_role: dict[str, Any],
    midi_notes_by_role: dict[str, list[Any]],
    bpm: float = 132.0,
    haas_targets_ms: dict[str, float] | None = None,
    tolerance_ms: float = 7.0,
    sr: int = 44100,
    track_is_sampler: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """
    Échelle S2 / Axe H (Phase 6 §3.2) :
    Détecte les onsets audio haute résolution (hop_length=64, backtrack=True),
    apparie à la note MIDI attendue (±60 ms) et compare le lag médian à la cible Haas.
    """
    beat_s = 60.0 / max(40.0, float(bpm))
    role_reports: dict[str, Any] = {}
    failing_roles: list[dict[str, Any]] = []
    scores: list[float] = []

    for role_key, wav_src in (stems_by_role or {}).items():
        notes_raw = (midi_notes_by_role or {}).get(role_key)
        if not notes_raw:
            continue
        midi_times_s: list[float] = []
        for item in notes_raw:
            if isinstance(item, dict):
                midi_times_s.append(float(item.get("start", 0.0)) * beat_s)
            else:
                midi_times_s.append(float(item))
        midi_times_s = sorted(set(round(t, 5) for t in midi_times_s))
        if not midi_times_s:
            continue

        mono, eff_sr = _load_mono_audio_input(wav_src, default_sr=sr)
        offset_s = 0.0
        if len(mono) > int(30.0 * eff_sr):
            m_pk = float(np.max(np.abs(mono))) + 1e-12
            nz_h = np.where(np.abs(mono) >= 0.15 * m_pk)[0]
            w_s = max(0, int(nz_h[0]) - int(0.05 * eff_sr)) if len(nz_h) else 0
            w_e = min(len(mono), w_s + int(24.0 * eff_sr))
            offset_s = float(w_s) / float(eff_sr)
            mono = mono[w_s:w_e]
        audio_onsets = [t_a + offset_s for t_a in _detect_highres_onsets_s(mono, eff_sr)]
        if not audio_onsets:
            continue

        lags_ms: list[float] = []
        used_audio_idx: set[int] = set()
        for t_m in midi_times_s:
            best_idx: int | None = None
            best_abs_diff = 1e9
            for idx, t_a in enumerate(audio_onsets):
                if idx in used_audio_idx:
                    continue
                diff_s = t_a - t_m
                if -0.045 <= diff_s <= 0.060 and abs(diff_s) < best_abs_diff:
                    best_abs_diff = abs(diff_s)
                    best_idx = idx
            if best_idx is not None:
                used_audio_idx.add(best_idx)
                lags_ms.append((audio_onsets[best_idx] - t_m) * 1000.0)

        if not lags_ms:
            continue

        med_lag = round(float(np.median(lags_ms)), 2)
        q75, q25 = np.percentile(lags_ms, [75, 25])
        iqr_ms = round(float(q75 - q25), 2)
        target_ms = _lookup_haas_target_ms(role_key, haas_targets_ms)
        lag_err = round(med_lag - target_ms, 2)
        abs_err = abs(lag_err)

        is_sampler = bool((track_is_sampler or {}).get(role_key, True))
        ok = abs_err <= float(tolerance_ms)
        if ok:
            r_score = 1.0
            act = {"code": "KEEP", "params": {}}
        else:
            r_score = round(max(0.20, 1.0 - (abs_err - float(tolerance_ms)) / 25.0), 3)
            if lag_err > 0 and is_sampler:
                act = {
                    "code": "TRIM_SAMPLE_START_MS",
                    "params": {"track": role_key, "ms": round(lag_err, 2)},
                }
            else:
                act = {
                    "code": "ADJUST_SYNTH_ATTACK_MS",
                    "params": {"track": role_key, "delta_ms": round(-lag_err, 2)},
                }
            failing_roles.append({
                "role": role_key,
                "lag_ms": med_lag,
                "target_ms": target_ms,
                "lag_error_ms": lag_err,
                "next_action": act,
            })

        scores.append(r_score)
        role_reports[role_key] = {
            "median_lag_ms": med_lag,
            "jitter_iqr_ms": iqr_ms,
            "target_haas_ms": target_ms,
            "lag_error_ms": lag_err,
            "matched_onsets": len(lags_ms),
            "passed": ok,
            "next_action": act,
        }

    if not role_reports:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "roles": {},
            "failing_roles": [],
            "next_action": {"code": "KEEP", "params": {}},
        }

    overall_score = round(float(min(scores)), 3) if scores else 1.0
    top_fail = max(failing_roles, key=lambda x: abs(x["lag_error_ms"])) if failing_roles else None
    return {
        "status": "FAIL" if failing_roles else "PASS",
        "verdict": "FIX" if failing_roles else "PASS",
        "score": overall_score,
        "roles": role_reports,
        "failing_roles": failing_roles,
        "next_action": top_fail["next_action"] if top_fail else {"code": "KEEP", "params": {}},
    }


def analyze_transient_stacking(
    stem_wavs: dict[str, Any] | list[Any],
    bpm: float = 132.0,
    sr: int = 44100,
    max_simultaneous: int = 2,
    max_collision_ratio: float = 0.15,
) -> dict[str, Any]:
    """
    Échelle S2 / Axe V (Phase 6 §3.2) :
    Matrice d'attaques Bark x Grille 1/16 sur pistes non-percussives :
    détecte les pas 1/16 où > 2 pistes non-percussives déclenchent simultanément (+6 dB, > -24 dBFS)
    dans Presence (800-4000 Hz) ou LowMid (120-800 Hz).
    """
    perc_bus_kw = (
        "kick", "snare", "clap", "hat", "hh", "perc", "drum", "rim",
        "crash", "cymbal", "shaker", "tom", "master", "bus ", "fx room", "reverb",
    )
    items: list[tuple[str, Any]] = []
    if isinstance(stem_wavs, dict):
        for k, v in stem_wavs.items():
            if not any(kw in k.lower() for kw in perc_bus_kw):
                items.append((k, v))
    else:
        for idx, v in enumerate(stem_wavs):
            nm = os.path.basename(v) if isinstance(v, str) else f"stem_{idx}"
            if not any(kw in nm.lower() for kw in perc_bus_kw):
                items.append((nm, v))

    if len(items) <= int(max_simultaneous):
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "stems_analyzed": [nm for nm, _ in items],
            "collision_steps": 0,
            "active_steps": 0,
            "collision_ratio": 0.0,
            "next_action": {"code": "KEEP", "params": {}},
        }

    loaded: list[tuple[str, np.ndarray, int]] = []
    max_len = 0
    eff_sr = sr
    for nm, src in items:
        mono, s_sr = _load_mono_audio_input(src, default_sr=sr)
        if loaded and s_sr != eff_sr:
            g = math.gcd(eff_sr, s_sr)
            mono = signal.resample_poly(mono, eff_sr // g, s_sr // g)
        else:
            eff_sr = s_sr
        max_len = max(max_len, len(mono))
        loaded.append((nm, mono, eff_sr))

    step_samples = max(64, int(round((60.0 / max(40.0, float(bpm)) / 4.0) * eff_sr)))
    n_steps = max_len // step_samples
    if n_steps < 2:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "stems_analyzed": [nm for nm, _ in items],
            "collision_steps": 0,
            "active_steps": 0,
            "collision_ratio": 0.0,
            "next_action": {"code": "KEEP", "params": {}},
        }

    bands = {
        "Sub": (20.0, 120.0),
        "LowMid": (120.0, 800.0),
        "Presence": (800.0, 4000.0),
        "Air": (4000.0, min(16000.0, 0.48 * eff_sr)),
    }
    sos_bands: dict[str, np.ndarray] = {}
    for b_name, (lo, hi) in bands.items():
        if hi > lo * 1.1:
            sos_bands[b_name] = signal.butter(4, [lo, hi], btype="bandpass", fs=eff_sr, output="sos")

    # attacks[band][step_k] = list of stem names with strong attack on step_k
    attack_counts: dict[str, list[list[str]]] = {
        b: [[] for _ in range(n_steps)] for b in sos_bands
    }

    usable_len = n_steps * step_samples
    max_band_steps = int((24.0 * float(eff_sr)) // step_samples)
    if n_steps > max_band_steps >= 8:
        # Center on active 24s slice for long stems while preserving 1/16 step precision
        st_lo = (n_steps - max_band_steps) // 2
        st_hi = st_lo + max_band_steps
        s_off = st_lo * step_samples
        e_off = st_hi * step_samples
        n_steps = max_band_steps
        usable_len = n_steps * step_samples
        attack_counts = {b: [[] for _ in range(n_steps)] for b in sos_bands}
    else:
        s_off = 0
        e_off = usable_len

    for nm, mono, _ in loaded:
        buf = np.zeros(usable_len, dtype=np.float64)
        seg = mono[s_off:] if s_off < len(mono) else np.zeros(0, dtype=np.float64)
        n_cp = min(len(seg), usable_len)
        buf[:n_cp] = seg[:n_cp]
        for b_name, sos in sos_bands.items():
            yb = signal.sosfilt(sos, buf)
            pk_steps = np.max(np.abs(yb.reshape(n_steps, step_samples)), axis=1)
            db_steps = 20.0 * np.log10(np.maximum(pk_steps, 1e-12))
            for k in range(n_steps):
                lvl = float(db_steps[k])
                if lvl <= -24.0:
                    continue
                if k == 0 or (lvl - float(db_steps[k - 1])) > 6.0:
                    attack_counts[b_name][k].append(nm)

    collision_steps = 0
    active_steps = 0
    collision_examples: list[dict[str, Any]] = []
    for k in range(n_steps):
        pres_hits = attack_counts.get("Presence", [[] for _ in range(n_steps)])[k]
        lm_hits = attack_counts.get("LowMid", [[] for _ in range(n_steps)])[k]
        if pres_hits or lm_hits:
            active_steps += 1
        pres_col = len(pres_hits) > int(max_simultaneous)
        lm_col = len(lm_hits) > int(max_simultaneous)
        if pres_col or lm_col:
            collision_steps += 1
            if len(collision_examples) < 6:
                collision_examples.append({
                    "step_16": k,
                    "presence_tracks": pres_hits,
                    "lowmid_tracks": lm_hits,
                })

    collision_ratio = round(collision_steps / float(max(1, active_steps)), 4) if active_steps > 0 else 0.0
    passed = collision_ratio <= float(max_collision_ratio)
    score = 1.0 if passed else round(max(0.20, 1.0 - (collision_ratio - float(max_collision_ratio)) * 1.5), 3)

    target_tracks = sorted({
        t for ex in collision_examples for t in (ex["presence_tracks"] + ex["lowmid_tracks"])
        if any(kw in t.lower() for kw in ("chord", "pad", "keys", "counter", "arp"))
    })
    next_act: dict[str, Any] = (
        {
            "code": "STAGGER_ATTACK_OR_STRUM",
            "params": {
                "tracks": target_tracks or [nm for nm, _ in items],
                "collision_ratio": collision_ratio,
            },
        }
        if not passed
        else {"code": "KEEP", "params": {}}
    )

    return {
        "status": "FAIL" if not passed else "PASS",
        "verdict": "FIX" if not passed else "PASS",
        "score": score,
        "stems_analyzed": [nm for nm, _ in items],
        "active_steps": active_steps,
        "collision_steps": collision_steps,
        "collision_ratio": collision_ratio,
        "collision_examples": collision_examples,
        "next_action": next_act,
    }


def _find_role_stems_in_dir(stems_dir: str) -> dict[str, Any]:
    """Classify stem WAV files inside `stems_dir` into kick, bass, and non-percussive track maps."""
    kick_wav: str | None = None
    bass_wav: str | None = None
    bus_bass_wav: str | None = None
    non_perc: dict[str, str] = {}
    all_tracks: dict[str, str] = {}
    for w in sorted(Path(stems_dir).glob("*.wav")):
        stem = w.stem
        label = stem.split(" - ", 1)[1] if " - " in stem else stem
        low = label.lower()
        if "master" in low:
            continue
        if "bus " in low or "fx room" in low or "reverb" in low:
            if "bus" in low and ("bass" in low or "808" in low):
                bus_bass_wav = str(w)
            continue
        all_tracks[label] = str(w)
        if "kick" in low and kick_wav is None:
            kick_wav = str(w)
        elif any(k in low for k in ("808", "bass", "sub")):
            if bass_wav is None:
                bass_wav = str(w)
            non_perc[label] = str(w)
        elif not any(k in low for k in ("snare", "clap", "hat", "hh", "perc", "drum", "rim", "crash", "cymbal", "shaker", "tom")):
            non_perc[label] = str(w)
    return {
        "kick": kick_wav,
        "bass": bass_wav or bus_bass_wav,
        "non_perc": non_perc,
        "all_tracks": all_tracks,
    }


def analyze_vertical_stage(
    wav_path: str,
    stage: str,
    profile: str = "trap_rap",
    blueprint: dict[str, Any] | None = None,
    stems: dict[str, str] | str | None = None,
    expected_onsets_by_role: dict[str, list[Any]] | None = None,
    bpm: float | None = None,
    contract: dict[str, Any] | None = None,
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
    dir_stems = _find_role_stems_in_dir(stems_dir) if stems_dir else {"kick": None, "bass": None, "non_perc": {}, "all_tracks": {}}
    eff_bpm = float(bpm or bp.get("bpm") or 132.0)
    s1_cfg = (contract or {}).get("s1_physical") or {}
    s2_cfg = (contract or {}).get("s2_acoustic") or {}
    min_kb_corr = float(s1_cfg.get("min_kick_bass_transient_corr", 0.25))
    haas_targets = s2_cfg.get("haas_targets_ms")
    lag_tol_ms = float(s2_cfg.get("onset_lag_tolerance_ms", 7.0))
    max_sim_trans = int(s2_cfg.get("max_simultaneous_bark_transients", 2))

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

        # Phase 6 S1/V: Kick/Bass transient phase alignment (< 120 Hz, 35 ms window)
        kb_phase: dict[str, Any] | None = None
        k_src = stems_map.get("kick") or dir_stems.get("kick")
        b_src = stems_map.get("bass") or dir_stems.get("bass")
        if has_drums and k_src and b_src:
            kb_phase = analyze_kick_bass_transient_phase(k_src, b_src, sr=sr, min_corr=min_kb_corr)

        corr_thresh = 0.70 if has_drums else 0.40
        corr_ok = (low_end_corr >= corr_thresh) or (low_total < 5.0 and not has_drums)
        overlap_ok = True if overlap is None else (overlap <= 0.30)
        infra_ok = (infra_ratio <= 0.42) or (low_total < 8.0)
        phase_ok = True if (kb_phase is None or kb_phase.get("status") == "SKIP") else (kb_phase.get("verdict") == "PASS")

        checks.update({
            "low_end_corr_ok": corr_ok,
            "transient_overlap_ok": overlap_ok,
            "infra_sub_ok": infra_ok,
            "kick_bass_transient_phase_ok": phase_ok,
        })
        metrics.update({
            "low_end_corr": low_end_corr,
            "kick_bass_transient_overlap": overlap,
            "kick_bass_transient_phase": kb_phase,
            "energy_50_100hz_pct": low_50_100,
            "infra_sub_ratio_lt45hz": infra_ratio,
        })

        if not phase_ok and kb_phase is not None:
            score -= 0.40
            diag.append(
                f"Désalignement de phase transitoire Kick/Basse <120Hz (r0={kb_phase['r_0']}, tau*={kb_phase['tau_star_ms']} ms)."
            )
            next_action = kb_phase["next_action"]
        if not corr_ok:
            score -= 0.35
            diag.append(f"Phase grave < 150 Hz trop large (corr={low_end_corr} < {corr_thresh}) : forcer mono sur BUS Bass.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = {"code": "MONO_LOW_END", "params": {"mixer": 18, "plugin": "gl.stereo", "Width": 0.0}}
        if not overlap_ok:
            score -= 0.35
            diag.append(f"Masquage Kick/Basse < 150 Hz (overlap={overlap} > 0.30) : renforcer sidechain ou apply_haas_pocket +12 ms.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
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

        # Phase 6 S2/H: audio pocket alignment when stems + expected onsets provided
        pocket_res: dict[str, Any] | None = None
        if expected_onsets_by_role and (dir_stems.get("all_tracks") or stems_map):
            st_in = stems_map if stems_map else dir_stems["all_tracks"]
            pocket_res = analyze_audio_pocket_alignment(
                st_in,
                expected_onsets_by_role,
                bpm=eff_bpm,
                haas_targets_ms=haas_targets,
                tolerance_ms=lag_tol_ms,
                sr=sr,
            )
        pocket_ok = True if (pocket_res is None or pocket_res.get("status") == "SKIP") else (pocket_res.get("verdict") == "PASS")

        checks.update({
            "sharpness_ok": sharp_ok,
            "no_harsh_drum_peaks": peaks_ok,
            "hat_groove_variance_ok": hat_ok,
            "audio_pocket_alignment_ok": pocket_ok,
        })
        metrics.update({
            "sharpness_din_acum": sharp_val,
            "sharp_max": sharp_max,
            "peaks_2500_4500hz": peaks_25_45[:3],
            "hat_velocity_cv": hat_cv,
            "audio_pocket_alignment": pocket_res,
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
        if not pocket_ok and pocket_res is not None:
            score -= 0.25
            diag.append(f"Décalage micro-timing audio vs grille Haas détecté : {pocket_res['next_action']['code']}.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = pocket_res["next_action"]
        if not diag:
            diag.append(f"Palier V2_GROOVE validé : sharpness={sharp_val} (< {sharp_max}), 0 pic 2.5-4.5 kHz, groove vivant.")
            next_action = {"code": "ADVANCE_TO_V3_HARMONY", "params": {"next_stage": "V3_HARMONY"}}

    elif st_norm == "V3_HARMONY":
        low_mids_pct = _band_pct(250.0, 500.0)
        lm_max = limits["low_mids_max"]
        corr_min = 0.55 if prof == "classical_ambient" else 0.65
        chroma_info = _detect_bass_harmony_chroma_clash(mono, sr)

        # Phase 6 S2/V: envelope collision matrix (Bark x 1/16 grid) on non-percussive stems
        non_perc_stems = dict(dir_stems.get("non_perc") or {})
        if not non_perc_stems and stems_map:
            non_perc_stems = dict(stems_map)
        stack_res: dict[str, Any] | None = None
        if len(non_perc_stems) > max_sim_trans:
            stack_res = analyze_transient_stacking(
                non_perc_stems, bpm=eff_bpm, sr=sr, max_simultaneous=max_sim_trans
            )
        stack_ok = True if (stack_res is None or stack_res.get("status") == "SKIP") else (stack_res.get("verdict") == "PASS")

        lm_ok = low_mids_pct <= lm_max
        corr_ok = low_mid_corr >= corr_min
        chroma_ok = not chroma_info["clash_detected"]

        checks.update({
            "low_mids_ok": lm_ok,
            "low_mid_corr_ok": corr_ok,
            "chroma_consonance_ok": chroma_ok,
            "transient_stacking_ok": stack_ok,
        })
        metrics.update({
            "low_mids_pct": low_mids_pct,
            "low_mids_max": lm_max,
            "low_mid_corr": low_mid_corr,
            "low_mid_corr_min": corr_min,
            "bass_harmony_chroma_clash": chroma_info,
            "transient_stacking": stack_res,
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
        if not stack_ok and stack_res is not None:
            score -= 0.25
            diag.append(
                f"Collision d'enveloppes non-percussives ({stack_res['collision_ratio']*100:.0f}% des pas 1/16) : étaler attaques/strum."
            )
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = stack_res["next_action"]
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

        non_perc_stems_v4 = dict(dir_stems.get("non_perc") or {})
        if not non_perc_stems_v4 and stems_map:
            non_perc_stems_v4 = dict(stems_map)
        stack_v4: dict[str, Any] | None = None
        if len(non_perc_stems_v4) > max_sim_trans:
            stack_v4 = analyze_transient_stacking(
                non_perc_stems_v4, bpm=eff_bpm, sr=sr, max_simultaneous=max_sim_trans
            )
        stack_v4_ok = True if (stack_v4 is None or stack_v4.get("status") == "SKIP") else (stack_v4.get("verdict") == "PASS")

        pocket_v4: dict[str, Any] | None = None
        if expected_onsets_by_role and (dir_stems.get("all_tracks") or stems_map):
            st_in = stems_map if stems_map else dir_stems["all_tracks"]
            pocket_v4 = analyze_audio_pocket_alignment(
                st_in,
                expected_onsets_by_role,
                bpm=eff_bpm,
                haas_targets_ms=haas_targets,
                tolerance_ms=lag_tol_ms,
                sr=sr,
            )
        pocket_v4_ok = True if (pocket_v4 is None or pocket_v4.get("status") == "SKIP") else (pocket_v4.get("verdict") == "PASS")

        lead_ok = True if lead_vs_harm is None else (lead_vs_harm >= 2.0)
        slope_ok = (limits["pink_min"] <= pink_slope <= limits["pink_max"])
        mid_1_3k_pct = _band_pct(1000.0, 3000.0)
        masking_1_3k_ok = mid_1_3k_pct <= 48.0

        checks.update({
            "lead_hierarchy_ok": lead_ok,
            "pink_slope_ok": slope_ok,
            "no_mid_congestion_1_3k": masking_1_3k_ok,
            "transient_stacking_ok": stack_v4_ok,
            "audio_pocket_alignment_ok": pocket_v4_ok,
        })
        metrics.update({
            "lead_vs_harmony_db": lead_vs_harm,
            "lead_silence_ratio": lead_silence_ratio,
            "pink_slope_db_per_oct": pink_slope,
            "pink_slope_range": [limits["pink_min"], limits["pink_max"]],
            "spectral_match_score": spec_match,
            "band_1_3khz_pct": mid_1_3k_pct,
            "transient_stacking": stack_v4,
            "audio_pocket_alignment": pocket_v4,
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
        if not stack_v4_ok and stack_v4 is not None:
            score -= 0.20
            diag.append(f"Empilement de transitoires non-percussifs ({stack_v4['collision_ratio']*100:.0f}% > 15%) : STAGGER_ATTACK_OR_STRUM.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = stack_v4["next_action"]
        if not pocket_v4_ok and pocket_v4 is not None:
            score -= 0.20
            diag.append(f"Décalage d'attaque Lead vs grille Haas : {pocket_v4['next_action']['code']}.")
            if next_action["code"] == "ADVANCE_NEXT_STAGE":
                next_action = pocket_v4["next_action"]
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
