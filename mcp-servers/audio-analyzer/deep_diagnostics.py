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
    resonances = [b for b in third_oct if b["local_excess_db"] >= 5.0]
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
                "fix": "Appliquer un coupe-cloche Q=2.5..4 (-3 à -5 dB) sur les pistes mélodiques/pads dans cette plage et passe-haut >= 140 Hz.",
            }
        )
    shrill = [r for r in resonances if 1800 <= r["freq_hz"] <= 7000]
    sharp_val = psycho.get("sharpness_din_acum", 0.0)
    if shrill or sharp_val >= 2.25:
        issues.append(
            {
                "category": "harsh_shrill_treble",
                "severity": "high" if sharp_val >= 2.5 else "medium",
                "detail": f"Acuité DIN={sharp_val} acum, pics aigus={[(r['freq_hz'], r['local_excess_db']) for r in shrill]}.",
                "fix": "Baisser l'octave du synthé/piano s'il est > C5, ou adoucir avec un passe-bas/shelf -3..-6 dB entre 2.5 et 6 kHz.",
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


def audit_midi_file(
    midi_path: str,
    expected_key: str = "",
    min_pitch: int = 24,
    max_pitch: int = 91,
    strict_musicality: bool = True,
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
    parts = ref_key_str.split()
    tonic_raw = parts[0].replace("b", "-") if parts else "C"
    mode_raw = parts[1].lower() if len(parts) > 1 else "minor"
    if "harmonic" in mode_raw:
        target_scale = music21.scale.HarmonicMinorScale(tonic_raw)
    elif "melodic" in mode_raw:
        target_scale = music21.scale.MelodicMinorScale(tonic_raw)
    elif "dorian" in mode_raw:
        target_scale = music21.scale.DorianScale(tonic_raw)
    elif "phrygian" in mode_raw:
        target_scale = music21.scale.PhrygianScale(tonic_raw)
    elif "min" in mode_raw:
        target_scale = music21.scale.MinorScale(tonic_raw)
    else:
        target_scale = music21.scale.MajorScale(tonic_raw)
    allowed_pcs = {p.pitchClass for p in target_scale.getPitches()}

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

    # Détection de frottements dissonants (seconde mineure directe = 1 demi-ton, ou neuvième mineure montante = 13 demi-tons)
    clashes: list[dict[str, Any]] = []
    sub_collisions: list[dict[str, Any]] = []
    melodic_notes.sort(key=lambda x: x[0])
    for i, (s1, e1, p1, tr1) in enumerate(melodic_notes):
        for s2, e2, p2, tr2 in melodic_notes[i + 1 :]:
            if s2 >= e1 - 0.01:
                break
            if tr1 == tr2:
                continue
            diff = abs(p1 - p2)
            if diff in (1, 13) and len(clashes) < 12:
                clashes.append(
                    {
                        "time_s": round(s2, 3),
                        "tracks": [tr1, tr2],
                        "notes": [pretty_midi.note_number_to_name(p1), pretty_midi.note_number_to_name(p2)],
                        "interval": "minor_2nd_cluster" if diff == 1 else "minor_9th_clash",
                    }
                )
            if p1 < 50 and p2 < 50 and p1 != p2 and len(sub_collisions) < 12:
                sub_collisions.append(
                    {
                        "time_s": round(s2, 3),
                        "tracks": [tr1, tr2],
                        "notes": [pretty_midi.note_number_to_name(p1), pretty_midi.note_number_to_name(p2)],
                    }
                )

    musicality = evaluate_midi_musicality(abs_path)
    out_of_scale_total = sum(tr.get("out_of_scale_count", 0) for tr in track_reports if not tr["is_drum"])
    gate_ok = (
        musicality["musicality_verdict"] == "PASS"
        and out_of_scale_total == 0
        and len(clashes) == 0
        and len(sub_collisions) == 0
    )

    return {
        "midi_path": abs_path,
        "detected_key": detected_key_str,
        "expected_key": ref_key_str,
        "key_matches_expected": detected_key_str.lower() == ref_key_str.lower(),
        "strict_musicality": strict_musicality,
        "musicality_verdict": musicality["musicality_verdict"],
        "gate_passed": gate_ok if strict_musicality else (out_of_scale_total == 0 and len(clashes) == 0),
        "musicality_metrics": musicality["metrics"],
        "musicality_detected_roles": musicality["detected_roles"],
        "musicality_failures": musicality["failures"],
        "tracks": track_reports,
        "minor_2nd_clashes": clashes,
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
