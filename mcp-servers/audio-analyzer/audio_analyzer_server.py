"""Serveur MCP d'analyse audio et boucle d'auto-mixage (pyloudnorm + DSP spectral)."""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_SERVER_DIR))
_SCRIPTS_DIR = os.path.join(_ROOT_DIR, "scripts")
for _d in (_SERVER_DIR, _SCRIPTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from mcp.server.mcpserver import MCPServer
from scipy import signal

from deep_diagnostics import (
    analyze_stem_hierarchy as _analyze_stem_hierarchy_impl,
    analyze_vertical_stage as _analyze_vertical_stage_impl,
    audit_midi_file,
    inspect_pipewire,
    match_with_reference,
    render_stems_and_audit,
    run_deep_audit,
)

mcp = MCPServer(
    name="audio-analyzer",
    instructions=(
        "Analyse mathématique et perceptuelle de mixage pour Gridline : LUFS (ITU-R BS.1770), "
        "True Peak 4x, clipping, résonances 1/3 octave, acuité DIN MoSQITo, RT60/réverb, "
        "détection de grésillements/xruns, simulation Sony WH-1000XM6 AutoEq, "
        "Audiobox Aesthetics & sondes CLAP, audit harmonique MIDI, Matchering et rendu stems."
    ),
)

BANDS_HZ: dict[str, tuple[float, float]] = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "low_mids": (250.0, 500.0),
    "high_mids": (500.0, 6000.0),
    "air": (6000.0, 20000.0),
}

# Référence équilibrée approximative d'un mix moderne (en part d'énergie spectrale)
TARGET_BAND_SHARES: dict[str, tuple[float, float]] = {
    "sub": (0.04, 0.25),
    "bass": (0.15, 0.42),
    "low_mids": (0.08, 0.24),
    "high_mids": (0.18, 0.48),
    "air": (0.03, 0.20),
}


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Charge un fichier audio en float64 2D (samples, channels)."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Fichier audio introuvable : {abs_path}")
    data, sr = sf.read(abs_path, dtype="float64", always_2d=True)
    if data.shape[0] == 0:
        raise ValueError(f"Fichier audio vide : {abs_path}")
    return data, int(sr)


def _linear_to_db(val: float, floor: float = -120.0) -> float:
    if val <= 1e-12:
        return floor
    return float(20.0 * math.log10(val))


def _compute_true_peak_db(data: np.ndarray) -> float:
    """True Peak (dBTP) par suréchantillonnage polyphase 4x (ITU-R BS.1770)."""
    upsampled = signal.resample_poly(data, 4, 1, axis=0)
    peak = float(np.max(np.abs(upsampled)))
    return round(_linear_to_db(peak), 2)


def _compute_lufs(data: np.ndarray, sr: int) -> float | None:
    """Calcule le loudness intégré LUFS; gère les fichiers < 400 ms par padding."""
    min_samples = int(0.45 * sr)
    buf = data
    if buf.shape[0] < min_samples:
        pad = np.zeros((min_samples - buf.shape[0], buf.shape[1]), dtype=np.float64)
        buf = np.vstack([buf, pad])
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(buf))
    if math.isinf(lufs) or math.isnan(lufs):
        return None
    return round(lufs, 2)


def _compute_spectral_bands(mono: np.ndarray, sr: int) -> tuple[dict[str, Any], float]:
    """Calcule la répartition d'énergie sur 5 bandes et le centroïde spectral."""
    nperseg = min(16384, max(256, len(mono)))
    freqs, psd = signal.welch(mono, fs=sr, nperseg=nperseg, scaling="spectrum")
    total_energy = float(np.sum(psd)) + 1e-18

    bands_out: dict[str, Any] = {}
    for name, (f_lo, f_hi) in BANDS_HZ.items():
        hi = min(f_hi, sr * 0.495)
        mask = (freqs >= f_lo) & (freqs < hi)
        energy = float(np.sum(psd[mask])) if np.any(mask) else 0.0
        share = energy / total_energy
        bands_out[name] = {
            "range_hz": [f_lo, f_hi],
            "energy_pct": round(share * 100.0, 2),
            "relative_db": round(10.0 * math.log10(max(share, 1e-12)), 2),
        }

    centroid = float(np.sum(freqs * psd) / total_energy)
    return bands_out, round(centroid, 1)


def _compute_stereo_phase(data: np.ndarray, sr: int) -> dict[str, Any]:
    """Calcule la corrélation de phase globale, sous 150 Hz et la largeur Mid/Side."""
    if data.shape[1] < 2:
        return {
            "channels": 1,
            "correlation_full": 1.0,
            "correlation_low_end_150hz": 1.0,
            "stereo_width_ratio": 0.0,
            "mono_compatible": True,
        }

    left, right = data[:, 0], data[:, 1]
    denom_full = float(np.sqrt(np.sum(left**2) * np.sum(right**2))) + 1e-18
    corr_full = float(np.sum(left * right) / denom_full)

    sos = signal.butter(4, min(150.0, sr * 0.45), btype="lowpass", fs=sr, output="sos")
    l_low = signal.sosfiltfilt(sos, left)
    r_low = signal.sosfiltfilt(sos, right)
    denom_low = float(np.sqrt(np.sum(l_low**2) * np.sum(r_low**2))) + 1e-18
    corr_low = float(np.sum(l_low * r_low) / denom_low)

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_rms = float(np.sqrt(np.mean(mid**2))) + 1e-12
    side_rms = float(np.sqrt(np.mean(side**2)))

    return {
        "channels": data.shape[1],
        "correlation_full": round(corr_full, 3),
        "correlation_low_end_150hz": round(corr_low, 3),
        "stereo_width_ratio": round(side_rms / mid_rms, 3),
        "mono_compatible": corr_full >= 0.2 and corr_low >= 0.5,
    }


def _build_recommendations(
    lufs: float | None,
    target_lufs: float,
    true_peak_db: float,
    clipped_samples: int,
    bands: dict[str, Any],
    stereo: dict[str, Any],
) -> list[str]:
    """Génère des recommandations concrètes pour Gridline (EQ, faders, limiteur)."""
    recs: list[str] = []

    if clipped_samples > 0 or true_peak_db > -0.3:
        recs.append(
            f"CLIPPING / PEAK ÉLEVÉ ({true_peak_db} dBTP, {clipped_samples} samples clippés) : "
            "baisser le fader Master ou régler gl.limiter ('Brickwall -1 dB') avec ceiling à -1.0 dB."
        )
    elif true_peak_db > -1.0:
        recs.append(
            f"True Peak à {true_peak_db} dBTP (marge < 1 dB pour encodage streaming) : "
            "abaisser le plafond du limiteur à -1.0 dBTP."
        )

    if lufs is not None:
        delta = round(target_lufs - lufs, 2)
        if abs(delta) > 1.2:
            direction = "augmenter" if delta > 0 else "réduire"
            recs.append(
                f"Loudness à {lufs} LUFS (cible {target_lufs} LUFS, écart {delta:+.1f} dB) : "
                f"{direction} le gain pré-limiteur / bus master d'environ {abs(delta):.1f} dB."
            )

    low_mid_pct = bands["low_mids"]["energy_pct"] / 100.0
    if low_mid_pct > TARGET_BAND_SHARES["low_mids"][1]:
        recs.append(
            f"Bas-médiums chargés (250-500 Hz = {bands['low_mids']['energy_pct']}%) : "
            "ajouter un gl.eq en coupe douce (-2 à -4 dB vers 300 Hz) sur les pads/accords et passe-haut >= 150 Hz."
        )

    sub_pct = bands["sub"]["energy_pct"] / 100.0
    if sub_pct > TARGET_BAND_SHARES["sub"][1]:
        recs.append(
            f"Sub très présent (20-60 Hz = {bands['sub']['energy_pct']}%) : "
            "vérifier le passe-haut < 30 Hz sur le master et le niveau de la 808/Sub."
        )

    air_pct = bands["air"]["energy_pct"] / 100.0
    if air_pct < TARGET_BAND_SHARES["air"][0]:
        recs.append(
            f"Manque d'air dans les aigus (> 6 kHz = {bands['air']['energy_pct']}%) : "
            "ajouter un léger high-shelf +1.5 à +2.5 dB vers 10 kHz (gl.eq preset 'Air') ou ouvrir les hi-hats."
        )

    if stereo["channels"] >= 2 and stereo["correlation_low_end_150hz"] < 0.5:
        recs.append(
            f"Phase basse fréquence faible (< 150 Hz corr = {stereo['correlation_low_end_150hz']}) : "
            "passer la basse/kick en mono ou ajouter gl.stereo pour centrer le bas du spectre."
        )

    if not recs:
        recs.append("Mix équilibré : loudness, True Peak, spectre et phase dans les cibles.")
    return recs


def _analyze_file(audio_path: str, target_lufs: float = -14.0) -> dict[str, Any]:
    data, sr = _load_audio(audio_path)
    mono = np.mean(data, axis=1)
    duration_sec = round(data.shape[0] / sr, 3)
    sample_peak = float(np.max(np.abs(data)))
    sample_peak_db = round(_linear_to_db(sample_peak), 2)
    rms = float(np.sqrt(np.mean(mono**2)))
    rms_db = round(_linear_to_db(rms), 2)
    clipped_samples = int(np.sum(np.abs(data) >= 0.9999))
    true_peak_db = _compute_true_peak_db(data)
    lufs = _compute_lufs(data, sr)
    bands, centroid_hz = _compute_spectral_bands(mono, sr)
    stereo = _compute_stereo_phase(data, sr)
    recs = _build_recommendations(lufs, target_lufs, true_peak_db, clipped_samples, bands, stereo)

    return {
        "file": os.path.abspath(os.path.expanduser(audio_path)),
        "duration_seconds": duration_sec,
        "sample_rate": sr,
        "target_lufs": target_lufs,
        "integrated_lufs": lufs,
        "true_peak_dbtp": true_peak_db,
        "sample_peak_dbfs": sample_peak_db,
        "rms_dbfs": rms_db,
        "crest_factor_db": round(sample_peak_db - rms_db, 2),
        "clipped_samples": clipped_samples,
        "spectral_centroid_hz": centroid_hz,
        "spectral_bands": bands,
        "stereo": stereo,
        "recommendations": recs,
    }


@mcp.tool()
def analyze_mix(audio_path: str, target_lufs: float = -14.0) -> str:
    """Analyse un rendu audio (.wav/.flac/.ogg) : LUFS, True Peak, clipping, 5 bandes, phase et conseils Gridline."""
    return json.dumps(_analyze_file(audio_path, target_lufs=target_lufs), ensure_ascii=False, indent=2)


@mcp.tool()
def analyze_low_end_masking(kick_or_mix_path: str, bass_path: str | None = None) -> str:
    """Analyse le masquage Kick/Basse entre 30 et 180 Hz (deux stems séparés ou un mix complet)."""
    d1, sr1 = _load_audio(kick_or_mix_path)
    m1 = np.mean(d1, axis=1)

    if bass_path is None:
        freqs, psd = signal.welch(m1, fs=sr1, nperseg=min(16384, len(m1)), scaling="spectrum")
        sub_mask = (freqs >= 30.0) & (freqs < 80.0)
        punch_mask = (freqs >= 80.0) & (freqs < 180.0)
        mud_mask = (freqs >= 180.0) & (freqs < 350.0)
        total = float(np.sum(psd)) + 1e-18
        sub_e = float(np.sum(psd[sub_mask])) / total
        punch_e = float(np.sum(psd[punch_mask])) / total
        mud_e = float(np.sum(psd[mud_mask])) / total
        peak_idx = int(np.argmax(psd[(freqs >= 30.0) & (freqs <= 180.0)]))
        low_freqs = freqs[(freqs >= 30.0) & (freqs <= 180.0)]
        dominant_hz = round(float(low_freqs[peak_idx]), 1) if len(low_freqs) else 0.0
        return json.dumps(
            {
                "mode": "single_mix",
                "dominant_low_hz": dominant_hz,
                "energy_30_80hz_pct": round(sub_e * 100, 2),
                "energy_80_180hz_pct": round(punch_e * 100, 2),
                "energy_180_350hz_pct": round(mud_e * 100, 2),
                "suggestion": (
                    f"Pic basse fréquence à {dominant_hz} Hz. Si kick et 808 se masquent, "
                    "configurer un sidechain Kick -> 808 (set_send avec sidechain=true + gl.comp sur la 808) "
                    "et creuser légèrement la 808 autour de la fondamentale du kick (60-90 Hz)."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    d2, sr2 = _load_audio(bass_path)
    m2 = np.mean(d2, axis=1)
    if sr2 != sr1:
        g = math.gcd(sr1, sr2)
        m2 = signal.resample_poly(m2, sr1 // g, sr2 // g)

    nperseg = min(16384, len(m1), len(m2))
    f1, p1 = signal.welch(m1, fs=sr1, nperseg=nperseg, scaling="spectrum")
    _, p2 = signal.welch(m2, fs=sr1, nperseg=nperseg, scaling="spectrum")
    mask = (f1 >= 30.0) & (f1 <= 180.0)
    f_low = f1[mask]
    p1_n = p1[mask] / (float(np.sum(p1[mask])) + 1e-18)
    p2_n = p2[mask] / (float(np.sum(p2[mask])) + 1e-18)
    overlap = float(np.sum(np.minimum(p1_n, p2_n)))
    clash_hz = round(float(f_low[int(np.argmax(np.minimum(p1_n, p2_n)))]), 1) if len(f_low) else 0.0

    return json.dumps(
        {
            "mode": "dual_stems",
            "overlap_score_0_to_1": round(overlap, 3),
            "peak_clash_hz": clash_hz,
            "suggestion": (
                f"Chevauchement Kick/Basse = {overlap:.2f}, conflit maximal vers {clash_hz} Hz. "
                + (
                    "Sidechain recommandé (set_send sidechain=true Kick -> Bass + gl.comp) "
                    f"et coupe EQ -3 dB sur la basse à {clash_hz} Hz."
                    if overlap > 0.45
                    else "Séparation correcte."
                )
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def compare_renders(before_path: str, after_path: str, target_lufs: float = -14.0) -> str:
    """Compare deux rendus audio avant/après correction de mixage."""
    b = _analyze_file(before_path, target_lufs=target_lufs)
    a = _analyze_file(after_path, target_lufs=target_lufs)

    lufs_delta = (
        round(a["integrated_lufs"] - b["integrated_lufs"], 2)
        if a["integrated_lufs"] is not None and b["integrated_lufs"] is not None
        else None
    )
    band_deltas = {
        band: round(
            a["spectral_bands"][band]["energy_pct"] - b["spectral_bands"][band]["energy_pct"],
            2,
        )
        for band in BANDS_HZ
    }
    return json.dumps(
        {
            "before": b,
            "after": a,
            "delta": {
                "lufs": lufs_delta,
                "true_peak_dbtp": round(a["true_peak_dbtp"] - b["true_peak_dbtp"], 2),
                "clipped_samples": a["clipped_samples"] - b["clipped_samples"],
                "low_end_correlation": round(
                    a["stereo"]["correlation_low_end_150hz"] - b["stereo"]["correlation_low_end_150hz"],
                    3,
                ),
                "spectral_band_pct_deltas": band_deltas,
            },
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def audit_mix_deep(
    audio_path: str,
    target_lufs: float = -11.0,
    target_genre: str = "trap",
    use_neural: bool = True,
    headphone_target: str = "sony_xm6",
) -> str:
    """Audit complet psychoacoustique, spectral 1/3 octave, réverbération/RT60, grésillements, simulation casque AutoEq et scores neuraux (Audiobox Aesthetics + CLAP)."""
    base = _analyze_file(audio_path, target_lufs)
    if "error" in base:
        return json.dumps(base, ensure_ascii=False, indent=2)
    abs_path = os.path.abspath(os.path.expanduser(audio_path))
    data, sr = _load_audio(abs_path)
    report = run_deep_audit(
        audio_path=abs_path,
        base_metrics=base,
        data=data,
        sr=sr,
        target_genre=target_genre,
        use_neural=use_neural,
        headphone_target=headphone_target,
    )
    return json.dumps(report, ensure_ascii=False, indent=2)


@mcp.tool()
def audit_midi_harmony(
    midi_path: str,
    expected_key: str = "",
    min_pitch: int = 24,
    max_pitch: int = 74,
    strict_musicality: bool = True,
    valence: float = 0.0,
    tension: float = 0.3,
    time_signature: str = "",
    phrase_plan: str = "",
) -> str:
    """Audite un fichier MIDI via music21 + pretty_midi + musicality_metrics : justesse, tessiture, clashs, voice leading et musicalité mélodique."""
    return json.dumps(
        audit_midi_file(
            midi_path=midi_path,
            expected_key=expected_key,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            strict_musicality=strict_musicality,
            valence=valence,
            tension=tension,
            time_signature=time_signature or None,
            phrase_plan=phrase_plan,
        ),
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def match_reference_audio(
    mix_path: str,
    reference_path: str,
    out_matched_wav: str = "",
) -> str:
    """Compare la balance fréquentielle du mix à un morceau de référence pro et peut générer un master auto-matché via Matchering."""
    return json.dumps(
        match_with_reference(mix_path=mix_path, reference_path=reference_path, out_matched_wav=out_matched_wav),
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def render_project_stems(
    project_path: str,
    output_dir: str = "",
    mode: str = "tracks",
) -> str:
    """Exporte les pistes/stems séparés d'un fichier .gridline et mesure pour chacun son niveau, centroïde et énergie 200-500Hz / 2-6kHz."""
    return json.dumps(
        render_stems_and_audit(project_path=project_path, output_dir=output_dir, mode=mode),
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def analyze_stem_hierarchy(
    project_path: str = "",
    range_bars: str = "",
    stems_dir: str = "",
    has_drums: bool = True,
) -> str:
    """Audite la hiérarchie inter-stems (Lead vs Harmonie 800Hz-3.5kHz, chevauchement Kick/Bass <150Hz, pente rose 60Hz-12kHz)."""
    return json.dumps(
        _analyze_stem_hierarchy_impl(
            project_path=project_path,
            range_bars=range_bars,
            stems_dir=stems_dir,
            has_drums=has_drums,
        ),
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def audit_track_soundcheck(
    project_dir: str = "",
    wav_path: str = "",
    role: str = "lead_melody",
    track_name: str = "",
    plugin: str = "gl.sampler",
    sample_file: str = "",
    root_note: int = 60,
    fine_tune: int = 0,
    lpf_hz: float | None = None,
    volume: float = 0.80,
    bpm: float = 132.0,
    key: str = "C minor",
    valence: float = 0.0,
) -> str:
    """
    Sound Check unitaire d'une piste isolée (sample ou synthé gl.*) jouée sur 2–4 mesures :
    vérifie audibilité, justesse jouée (<= 8 ct), intégrité chromatique, propreté spectrale
    (2–5 kHz, 0 glitch), enveloppe vs BPM et fidélité sémantique -> verdict PASS / FIX / SWAP.
    """
    from soundcheck_lib import audit_track_soundcheck as _audit_sc

    spec = None if wav_path.strip() else {
        "name": track_name or role,
        "role": role,
        "plugin": plugin,
        "sample_file": sample_file,
        "root_note": root_note,
        "fine_tune": fine_tune,
        "lpf_hz": lpf_hz,
        "volume": volume,
    }
    res = _audit_sc(
        project_dir=project_dir,
        track_spec=spec,
        wav_path=wav_path,
        role=role,
        track_name=track_name or role,
        bpm=bpm,
        key=key,
        valence=valence,
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


@mcp.tool()
def audit_vertical_step(
    stage: str,
    project_path: str = "",
    wav_path: str = "",
    profile: str = "trap_rap",
    blueprint_path: str = "",
    range_bars: str = "",
    stems_dir: str = "",
    render_stems: bool = True,
) -> str:
    """
    Audit incrémental d'un palier vertical du noyau 4/8 mesures (`V1_LOW_END`, `V2_GROOVE`, `V3_HARMONY`, `V4_LEAD`) :
    peut soit muter/rendre automatiquement `core_loop.gridline` (`project_path`), soit auditer directement un WAV (`wav_path`).
    Retourne verdict PASS/FIX/SWAP, score, diagnostic 1–3 phrases et action unique ciblée sur la couche k.
    """
    bp_dict: dict[str, Any] | None = None
    if blueprint_path.strip() and os.path.isfile(os.path.expanduser(blueprint_path)):
        with open(os.path.expanduser(blueprint_path), encoding="utf-8") as fh:
            bp_dict = json.load(fh)

    if project_path.strip() and not wav_path.strip():
        from vertical_tester import run_vertical_stage

        res = run_vertical_stage(
            project_path=project_path,
            stage=stage,
            profile=profile,
            blueprint=bp_dict or (blueprint_path or None),
            range_bars=range_bars,
            render_stems_for_stage=render_stems,
        )
    else:
        res = _analyze_vertical_stage_impl(
            wav_path=wav_path,
            stage=stage,
            profile=profile,
            blueprint=bp_dict,
            stems=stems_dir.strip() or None,
        )
    return json.dumps(res, ensure_ascii=False, indent=2)


@mcp.tool()
def audit_horizontal_seam(
    project_path: str = "",
    wav_path: str = "",
    is_outro: bool = False,
    from_section: str = "Intro",
    to_section: str = "Drop",
    blueprint_path: str = "",
    bpm: float = 132.0,
    beats_per_bar: int = 4,
    bars_before: float = 2.0,
    bars_after: float = 2.0,
    max_seams: int = 6,
) -> str:
    """
    Audit ciblé des coutures horizontales (fenêtres 4 mesures centrées sur chaque transition + Outro) :
    vérifie contraste d'énergie (ΔLUFS), marqueur de transition (beat4_dropout / filter_sweep),
    propreté de jonction (±150 ms, 0 glitch) et extinction propre de l'Outro (< -45 dBFS).
    """
    from horizontal_tester import (
        audit_outro_wav,
        audit_project_horizontal_seams,
        audit_seam_wav,
    )

    if wav_path.strip():
        if is_outro:
            res = audit_outro_wav(wav_path)
        else:
            res = audit_seam_wav(
                wav_path,
                from_section=from_section,
                to_section=to_section,
                bpm=bpm,
                beats_per_bar=beats_per_bar,
                bars_before=bars_before,
                bars_after=bars_after,
            )
    elif project_path.strip():
        res = audit_project_horizontal_seams(
            project_path,
            blueprint=blueprint_path.strip() or None,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            max_seams=max_seams,
        )
    else:
        res = {"error": "Fournir `project_path` ou `wav_path`."}
    return json.dumps(res, ensure_ascii=False, indent=2)


@mcp.tool()
def check_pipewire_health() -> str:
    """Diagnostique PipeWire en temps réel (quantum actif, casque Bluetooth, xruns/craquements sur gridline-app)."""
    return json.dumps(inspect_pipewire(), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
