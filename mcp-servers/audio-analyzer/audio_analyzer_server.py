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
    mix_profile: str = "trap_rap",
) -> str:
    """Audite la hiérarchie inter-stems (matrice bilatérale active_lufs + Lead vs Harmonie 800Hz-3.5kHz, chevauchement Kick/Bass <150Hz, pente rose)."""
    return json.dumps(
        _analyze_stem_hierarchy_impl(
            project_path=project_path,
            range_bars=range_bars,
            stems_dir=stems_dir,
            has_drums=has_drums,
            mix_profile=mix_profile,
        ),
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def audit_track_soundcheck_fast(
    project_dir: str = "",
    role: str = "lead_melody",
    track_name: str = "",
    plugin: str = "gl.sampler",
    sample_file: str = "",
    bpm: float = 132.0,
    key: str = "C minor",
    valence: float = 0.0,
    audition: str = "dsp",
    pattern: int = 1,
    channel: int = 1,
) -> str:
    """Soundcheck RAPIDE d'une piste SANS compiler de .wav : `audition="dsp"` (pur DSP sur le
    sample, ~50 ms, aucun processus Gridline) ou `audition="session"` (sonde `analyze_loudness`
    in-session sur le pattern en solo, toujours pas de .wav). `audition="render"` = legacy (wav)."""
    from soundcheck_lib import audit_track_soundcheck_fast as _fast

    res = _fast(
        track_spec={"name": track_name or role, "role": role, "plugin": plugin, "sample_file": sample_file},
        role=role,
        track_name=track_name,
        bpm=bpm,
        key=key,
        valence=valence,
        audition=audition,
        pattern=pattern,
        channel=channel,
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


@mcp.tool()
def balance_tracks_in_session(
    project_path: str,
    export_dir: str = "",
    mix_profile: str = "trap_rap",
    has_drums: bool = True,
    max_trim_db: float = 6.0,
) -> str:
    """Équilibre les volumes des pistes DANS LA SESSION Gridline (aucun .wav rendu) :
    solo de chaque canal + `analyze_loudness` sur son pattern + trim du fader vers la cible
    du rôle (ancrage -16 LUFS actifs), borné à +/- max_trim_db."""
    from gain_staging_lib import balance_tracks_in_session as _balance
    from studio_lib import GridlineMCP

    abs_proj = os.path.abspath(os.path.expanduser(project_path))
    exp = export_dir.strip() or os.path.dirname(abs_proj)
    mcp = GridlineMCP(abs_proj, exp, client_name="balance_session")
    try:
        channels = mcp.call("list_channels", {}).get("channels", [])
        pats = mcp.call("list_patterns", {}).get("patterns", [])
        tracks = []
        for i, c in enumerate(channels, start=1):
            tracks.append({
                "name": c.get("name") or f"ch{i}",
                "channel": int(c.get("channel", i)),
                "pattern": 1,
                "role": c.get("name") or "lead",
                "volume": float(c.get("volume", 0.8)),
            })
        report = _balance(mcp, tracks, profile=mix_profile, has_drums=has_drums, max_trim_db=max_trim_db)
        mcp.call("save_project", {})
    finally:
        mcp.close()
    return json.dumps(report, ensure_ascii=False, indent=2)


@mcp.tool()
def audit_vertical_stage_session(
    project_path: str,
    export_dir: str = "",
    stage: str = "V4_LEAD",
    profile: str = "trap_rap",
) -> str:
    """Audit vertical V1..V4 EN SESSION (solo par couche, `analyze_loudness` par pattern) :
    aucun rendu .wav, verdict sur la hiérarchie des niveaux (low-end, groove, harmonie, lead)."""
    from vertical_tester import analyze_vertical_stage_session as _vs
    from studio_lib import GridlineMCP

    abs_proj = os.path.abspath(os.path.expanduser(project_path))
    exp = export_dir.strip() or os.path.dirname(abs_proj)
    mcp = GridlineMCP(abs_proj, exp, client_name="vertical_session")
    try:
        channels = mcp.call("list_channels", {}).get("channels", [])
        active = [int(c.get("channel", i)) for i, c in enumerate(channels, start=1) if not c.get("muted", False)]
        res = _vs(mcp, stage, active_channels=active, profile=profile)
    finally:
        mcp.close()
    return json.dumps(res, ensure_ascii=False, indent=2)


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


@mcp.tool(name="extract_target_fingerprint")
def extract_target_fingerprint_tool(
    audio_paths: list[str] | None = None,
    mix_profile: str = "trap_rap",
    emotion_vector: dict | None = None,
    project_dir: str = "",
    blueprint_path: str = "",
    subgenre_tag: str = "",
    has_drums: bool | None = None,
) -> str:
    """
    Phase 6 Top-Down Target Fingerprint & Multi-Scale Contract compiler (0 GPU, <= 400 tokens).
    Saves `multi_scale_contract.json` & `song_target_profile.json` in `project_dir` and returns compact summary.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scripts_dir = os.path.join(root_dir, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from target_fingerprint import extract_target_fingerprint

    res = extract_target_fingerprint(
        audio_paths=audio_paths or [],
        mix_profile=mix_profile,
        emotion_vector=emotion_vector,
        blueprint=blueprint_path or None,
        project_dir=project_dir,
        subgenre_tag=subgenre_tag,
        has_drums=has_drums,
    )
    c = res["multi_scale_contract"]
    s5 = c.get("s5_context") or {}
    compact = {
        "source_mode": res["source_mode"],
        "reference_count": res["reference_count"],
        "mix_profile": res["mix_profile"],
        "has_drums": res["has_drums"],
        "target_lufs": s5.get("target_lufs"),
        "pink_slope_range": s5.get("pink_slope_range"),
        "low_mids_max": s5.get("low_mids_max"),
        "air_min": s5.get("air_min"),
        "sharp_max": s5.get("sharp_max"),
        "spectral_match_min": s5.get("spectral_match_min"),
        "contract_path": res.get("contract_path"),
        "profile_path": res.get("profile_path"),
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


@mcp.tool()
def audit_multiscale_hv(
    project_dir: str,
    stage: str = "ALL",
    contract_path: str = "",
    wav_path: str = "",
    skip_render: bool = False,
    auto_fix: bool = False,
    max_autofix_passes: int = 2,
) -> str:
    """
    Phase 6 Unified Multi-Scale H x V Audit (0 GPU, <= 1,500 chars / <= 400 tokens).
    Writes full diagnostics to `<project_dir>/renders/diagnostics/` and returns Top-3 Deviation Pyramid.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scripts_dir = os.path.join(root_dir, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_autonomous_pipeline import audit_multiscale_hv as _run_audit_hv

    pyramid = _run_audit_hv(
        project_dir=project_dir,
        stage=stage,
        contract_path=contract_path,
        wav_path=wav_path,
        skip_render=skip_render,
        auto_fix=auto_fix,
        max_autofix_passes=max_autofix_passes,
    )
    return json.dumps(pyramid, ensure_ascii=False, separators=(",", ":"))


def _compact_json(obj: Any, limit: int = 1500) -> str:
    """v3.0 token budget: every macro-tool answer is a compact JSON <= 1500 chars (~<= 400 tokens)."""
    txt = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    return txt if len(txt) <= limit else txt[: limit - 3] + "..."


@mcp.tool()
def _analyze_remix_source(remix_source_path: str) -> dict[str, Any]:
    """v3.3 #5 scénario C — diagnostic tonal/BPM d'une source à remixer + grille d'accords
    prête à copier dans `song_blueprint.json` `harmonic_core` (0 GPU)."""
    import numpy as _np
    import soundfile as _sf
    src = os.path.abspath(os.path.expanduser(remix_source_path))
    data, sr = _sf.read(src, always_2d=True, dtype="float32")
    mono = data.mean(axis=1).astype(_np.float64)
    out: dict[str, Any] = {"source": src, "duration_s": round(len(mono) / float(sr), 2)}
    try:
        import librosa as _lb
        tempo, _beats = _lb.beat.beat_track(y=mono, sr=sr)
        out["bpm"] = round(float(tempo), 1)
        chroma = _lb.feature.chroma_cqt(y=mono, sr=sr).mean(axis=1)
        maj = _np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minw = _np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        scores = [float(_np.dot(chroma, _np.roll(maj, -i))) for i in range(12)]
        scores_min = [float(_np.dot(chroma, _np.roll(minw, -i))) for i in range(12)]
        best_maj = int(_np.argmax(scores))
        best_min = int(_np.argmax(scores_min))
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        if scores[best_maj] >= scores_min[best_min]:
            out["key"] = f"{names[best_maj]} major"
            out["suggested_mode"] = "ionian"
            out["suggested_degree_progression"] = ["I", "V", "vi", "IV"]
        else:
            out["key"] = f"{names[best_min]} minor"
            out["suggested_mode"] = "aeolian"
            out["suggested_degree_progression"] = ["i", "bVI", "III", "VII"]
    except Exception as exc:
        out["error"] = str(exc)[:100]
    out["suggested_harmonic_core"] = {
        "key": out.get("key", "C minor"),
        "mode": out.get("suggested_mode", "aeolian"),
        "degree_progression": out.get("suggested_degree_progression", ["i", "bVI", "III", "VII"]),
    }
    return out


def _plan_lyrics(lyrics_spec: dict[str, Any], bpm: float) -> dict[str, Any]:
    """v3.3 #5 scénario D — syllabation + fenêtres respiratoires + tessiture cible
    (`vocal_range_midi`). Repli déterministe si le serveur vocaloid est indisponible."""
    text = str(lyrics_spec.get("text") or lyrics_spec.get("lyrics") or "")
    words = [w for w in text.replace("\n", " ").split() if w.strip()]
    syllables = sum(max(1, len([c for c in w if c.lower() in "aeiouyàâäéèêëîïôöùûüÿ"]) or 1) for w in words)
    beat_s = 60.0 / max(40.0, float(bpm))
    breath_every_syll = 8
    breaths = []
    acc = 0
    for i in range(syllables):
        acc += 1
        if acc % breath_every_syll == 0:
            breaths.append(round((i + 1) * 0.45 * 4 * beat_s, 2))  # ~1 mesure tous les 8 syllabes
    return {
        "n_words": len(words),
        "n_syllables": syllables,
        "breath_windows_s": breaths[:16],
        "vocal_range_midi": lyrics_spec.get("vocal_range_midi") or [48, 72],
        "language": lyrics_spec.get("language", "fr"),
    }


def studio_prepare_session(project_dir: str, style_intent: str = "", reference_paths: list[str] | None = None,
                           remix_source_path: str | None = None, lyrics_spec: dict[str, Any] | None = None) -> str:
    """v3.0 Tour 1/3 — Préparation: indexe Samples/ (scénario B, user_samples_locked), propose 3 palettes
    avec root_note/BPM, calcule le profil cible. v3.3 : `remix_source_path` (scénario C) renvoie
    BPM/clé/grille de départ ; `lyrics_spec` (scénario D) renvoie syllabes/pauses/tessiture.
    Retour JSON compact <= 1500 cars, 0 token LLM côté chaîne."""
    import shutil as _sh
    try:
        proj = os.path.abspath(os.path.expanduser(project_dir))
        bp_path = os.path.join(proj, "song_blueprint.json")
        bp = {}
        if os.path.isfile(bp_path):
            with open(bp_path, encoding="utf-8") as fh:
                bp = json.load(fh)
        cat_path = os.path.join(proj, "samples_catalog.json")
        indexed = False
        has_samples_dir = any(os.path.isdir(os.path.join(proj, d)) for d in ("Samples", "samples"))
        if has_samples_dir and not os.path.isfile(cat_path):
            from index_project_samples import index_project_samples
            index_project_samples(proj, with_clap=False, write_catalog=True)
            indexed = True
        key = str(bp.get("key") or "C minor")
        bpm = float(bp.get("bpm") or 132.0)
        profile = str(bp.get("mix_profile") or "trap_rap")
        palettes_summary = []
        if has_samples_dir:
            try:
                index_dir = os.path.join(_ROOT_DIR, "mcp-servers", "audio-index")
                if index_dir not in sys.path:
                    sys.path.insert(0, index_dir)
                from palette_builder import propose_palettes
                pal = propose_palettes(
                    mix_profile=profile,
                    emotion_vector=bp.get("emotion_vector"),
                    target_key=key,
                    project_samples_dir=proj,
                    bpm=bpm,
                    num_palettes=3,
                )
                for p in (pal.get("palettes") or [])[:3]:
                    palettes_summary.append({
                        "id": p.get("palette_id"),
                        "tracks": [
                            {"role": t.get("role"), "file": os.path.basename(str(t.get("filepath") or t.get("name") or "")),
                             "root_note": t.get("root_note"), "bpm": t.get("bpm")}
                            for t in (p.get("slots") or p.get("blueprint_tracks") or [])[:6]
                        ],
                    })
            except Exception as exc:
                palettes_summary = [{"error": str(exc)[:120]}]
        remix_analysis = None
        if remix_source_path:
            try:
                remix_analysis = _analyze_remix_source(remix_source_path)
            except Exception as exc:
                remix_analysis = {"error": str(exc)[:160]}
        lyrics_plan = None
        if lyrics_spec:
            try:
                lyrics_plan = _plan_lyrics(lyrics_spec, bpm)
            except Exception as exc:
                lyrics_plan = {"error": str(exc)[:160]}
        return _compact_json({
            "ok": True,
            "tour": 1,
            "project_dir": proj,
            "user_samples_locked": has_samples_dir,
            "indexed_now": indexed,
            "key": key,
            "bpm": bpm,
            "mix_profile": profile,
            "valence": (bp.get("emotion_vector") or {}).get("valence"),
            "palettes": palettes_summary,
            "gridline_available": bool(_sh.which("gridline")),
            "scenario": ("C" if remix_source_path else ("D" if lyrics_spec else "A/B")),
            "remix_analysis": remix_analysis,
            "lyrics_plan": lyrics_plan,
        })
    except Exception as exc:
        return _compact_json({"ok": False, "tour": 1, "error": str(exc)[:300]})


@mcp.tool()
def studio_build_and_validate_core(project_dir: str, palette_id: str = "") -> str:
    """v3.0 Tour 2/3 — Noyau validé: gate blueprint_safety (octave/shift/boucles/cloches) puis
    compile_song --core-only (ordre canonique MIDI immuable). Si gridline est dispo: 1 rendu
    stems 8 mesures + stem_matrix_evaluator (V1..V4 + solveur de gains) en NumPy pur.
    Retour: PASS ou Top-3 défauts actionnables, JSON <= 1500 cars."""
    import shutil as _sh
    try:
        from compile_song import compile_song
        rep = compile_song(project_dir, core_only=True)
        if not rep.get("ok"):
            issues = (rep.get("musical_safety") or {}).get("issues") or []
            top3 = [{"code": i.get("code"), "track": i.get("track"), "fix": (i.get("fix") or "")[:140]}
                    for i in issues if i.get("severity") == "error"][:3]
            return _compact_json({"ok": False, "tour": 2, "stage": rep.get("stage"), "top_defects": top3})
        stems_eval = None
        if _sh.which("gridline"):
            try:
                from stem_matrix_evaluator import evaluate_stem_matrix, load_stems_dir
                stems_dir = os.path.join(os.path.abspath(os.path.expanduser(project_dir)), "renders", "stems_core")
                if os.path.isdir(stems_dir):
                    stems, sr = load_stems_dir(stems_dir)
                    bp = {}
                    bp_path = os.path.join(os.path.abspath(os.path.expanduser(project_dir)), "song_blueprint.json")
                    if os.path.isfile(bp_path):
                        with open(bp_path, encoding="utf-8") as fh:
                            bp = json.load(fh)
                    stems_eval = evaluate_stem_matrix(stems, sr, profile=str(bp.get("mix_profile") or "trap_rap"))
            except Exception as exc:
                stems_eval = {"error": str(exc)[:120]}
        # v3.2 #3 — pré-vol symbolique 100 % du morceau (déjà calculé par compile_song) +
        # sonde stems composite 12 mesures si gridline dispo (1 seul rendu, budget préservé).
        full_cov = rep.get("full_coverage") if isinstance(rep, dict) else None
        probe_out = None
        if _sh.which("gridline"):
            try:
                from compile_song import compile_song as _cs
                prep = _cs(project_dir, probe=True)
                probe_out = prep.get("out") if prep.get("ok") else None
            except Exception:
                probe_out = None
        _enr = None
        try:
            _enr_path = os.path.join(os.path.abspath(os.path.expanduser(project_dir)), "renders", "diagnostics", "enrichment_report.json")
            if os.path.isfile(_enr_path):
                with open(_enr_path, encoding="utf-8") as _fh:
                    _enr = json.load(_fh)
        except Exception:
            _enr = None
        return _compact_json({
            "ok": True,
            "tour": 2,
            "stage": rep.get("stage"),
            "enrichment": ({"n_decisions": _enr.get("n_decisions"), "summary": str(_enr.get("summary"))[:200]} if isinstance(_enr, dict) else None),
            "out": rep.get("out"),
            "full_coverage": ({"passed": full_cov.get("passed"), "n_issues": full_cov.get("n_issues"),
                               "issues": (full_cov.get("issues") or [])[:3]} if isinstance(full_cov, dict) and full_cov.get("passed") is not None else full_cov),
            "probe_gridline": probe_out,
            "musical_safety_ok": True,
            "stems_eval": ({"ok": stems_eval.get("ok"), "n_stems": stems_eval.get("n_stems"),
                            "gridline_renders_used": stems_eval.get("gridline_renders_used"),
                            "issues": stems_eval.get("issues")} if stems_eval and "error" not in stems_eval else stems_eval),
            "palette_id": palette_id or None,
        })
    except Exception as exc:
        return _compact_json({"ok": False, "tour": 2, "error": str(exc)[:300]})


def _build_preview_safe(master_wav: str) -> dict[str, Any] | None:
    """v3.3 #5 — génère renders/preview_highlights.wav (24 s) si un master existe."""
    try:
        if not os.path.isfile(master_wav):
            return None
        _sd = os.path.join(_ROOT_DIR, "scripts")
        if _sd not in sys.path:
            sys.path.insert(0, _sd)
        from preview_lib import build_preview_highlights
        r = build_preview_highlights(master_wav)
        if r.get("ok"):
            return {"out": os.path.basename(r["out"]), "duration_s": r["duration_s"]}
    except Exception:
        pass
    return None


def cleanup_temp_dirs(project_dir: str, keep_stems: bool = False) -> dict[str, Any]:
    """v3.3 #5 — hygiène disque : supprime `.tmp_soundcheck/`, `.tmp_probe/` et les stems
    intermédiaires (hors `--keep-stems`) à la fin du Tour 3."""
    import shutil as _sh2
    proj = os.path.abspath(os.path.expanduser(project_dir))
    removed: list[str] = []
    for d in (".tmp_soundcheck", ".tmp_probe"):
        p = os.path.join(proj, d)
        if os.path.isdir(p):
            _sh2.rmtree(p, ignore_errors=True)
            removed.append(d)
    if not keep_stems:
        rd = os.path.join(proj, "renders")
        for d in ("stems_core", "stems_probe", "stems_final"):
            p = os.path.join(rd, d)
            if os.path.isdir(p):
                _sh2.rmtree(p, ignore_errors=True)
                removed.append(f"renders/{d}")
    return {"removed": removed}


@mcp.tool()
def studio_render_final_master(project_dir: str, export_mp3: bool = True, profile: str = "") -> str:
    """v3.0 Tour 3/3 — Master final: compile_song complet puis UN SEUL rendu Gridline master
    + quality gate render_and_audit (1 passe DAG max). Retour: fiche 5 lignes <= 1500 cars."""
    import shutil as _sh
    try:
        proj = os.path.abspath(os.path.expanduser(project_dir))
        if not _sh.which("gridline"):
            return _compact_json({"ok": False, "tour": 3, "error": "binaire gridline indisponible (headless CI)"})
        from compile_song import compile_song
        rep = compile_song(proj, core_only=False)
        if not rep.get("ok"):
            return _compact_json({"ok": False, "tour": 3, "stage": rep.get("stage"),
                                  "musical_safety": rep.get("musical_safety")})
        bp = {}
        bp_path = os.path.join(proj, "song_blueprint.json")
        if os.path.isfile(bp_path):
            with open(bp_path, encoding="utf-8") as fh:
                bp = json.load(fh)
        prof = profile or str(bp.get("mix_profile") or "trap_rap")
        from render_and_audit import render_and_evaluate
        renders_dir = os.path.join(proj, "renders")
        os.makedirs(renders_dir, exist_ok=True)
        out_wav = os.path.join(renders_dir, "master.wav")
        summary = render_and_evaluate(
            os.path.join(proj, "projet.gridline"), out_wav,
            profile=prof, blueprint=bp_path, render_mp3=bool(export_mp3),
        )
        return _compact_json({
            "ok": bool(summary["quality_gate"]["passed"]),
            "tour": 3,
            "wav": out_wav,
            "mp3": os.path.splitext(out_wav)[0] + ".mp3" if export_mp3 else None,
            "lufs": summary["integrated_lufs"],
            "true_peak_dbtp": summary["true_peak_dbtp"],
            "lra_lu": summary.get("lra_lu"),
            "gate_failures": summary["quality_gate"]["failures"][:5],
            "musical_safety_ok": (summary.get("musical_safety") or {}).get("ok", True),
            "render_count": 1,
            "temp_cleaned": cleanup_temp_dirs(proj, keep_stems=False)["removed"],
            "preview_highlights": _build_preview_safe(out_wav),
        })
    except Exception as exc:
        return _compact_json({"ok": False, "tour": 3, "error": str(exc)[:300]})


@mcp.tool()
def studio_refine_track(project_dir: str, delta_intent: Any = "", export_mp3: bool = False) -> str:
    """v3.3 #5 Tour 3b — raffinement chirurgical post-écoute humaine (contrat §6.1) :
    applique UNIQUEMENT le delta déclaré (`mix_adjustments_db`, `section_overrides` avec
    `comping_style`/`energy_delta`/`brightness_shift`, `swap_role`, `feedback_tags`),
    via `autofix_transaction.apply_with_rollback` (aucun invariant Palier 1 ne régresse),
    réutilise le cache SHA-256, enregistre les `feedback_tags` dans la mémoire de goût et
    regénère `renders/preview_highlights.wav` (24 s). Retour <= 1500 cars."""
    try:
        proj = os.path.abspath(os.path.expanduser(project_dir))
        delta = delta_intent
        if isinstance(delta, str):
            delta = json.loads(delta) if delta.strip() else {}
        if not isinstance(delta, dict):
            delta = {}
        sys.path.insert(0, os.path.join(_ROOT_DIR, "scripts")) if os.path.join(_ROOT_DIR, "scripts") not in sys.path else None
        from artifact_cache import sha256_json
        from autofix_transaction import apply_with_rollback
        bp_path = os.path.join(proj, "song_blueprint.json")
        plan_path = os.path.join(proj, "arrangement_plan.json")

        def _read(p: str) -> dict:
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as fh:
                    return json.load(fh)
            return {}

        def _snapshot() -> dict[str, str]:
            snap = {}
            for p in (bp_path, plan_path):
                if os.path.isfile(p):
                    with open(p, encoding="utf-8") as fh:
                        snap[p] = fh.read()
            return snap

        def _restore(snap: dict[str, str]) -> None:
            for p, content in snap.items():
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(content)

        def _apply() -> list[str]:
            applied: list[str] = []
            bp = _read(bp_path)
            if bp and isinstance(delta.get("mix_adjustments_db"), dict):
                for tname, ddb in delta["mix_adjustments_db"].items():
                    for t in bp.get("tracks") or []:
                        if str(t.get("name", "")).lower() == str(tname).lower() or str(tname).lower() in str(t.get("role", "")).lower():
                            t["volume_db"] = round(float(t.get("volume_db", 0.0)) + float(ddb), 2)
                            applied.append(f"mix:{t.get('name')}{float(ddb):+.1f}dB")
            if bp:
                with open(bp_path, "w", encoding="utf-8") as fh:
                    json.dump(bp, fh, ensure_ascii=False, indent=2)
                    fh.write("\n")
            plan = _read(plan_path)
            if plan and isinstance(delta.get("section_overrides"), dict):
                for sec_name, ov in delta["section_overrides"].items():
                    if not isinstance(ov, dict):
                        continue
                    for sec in plan.get("sections") or []:
                        if str(sec.get("name", "")).lower() == str(sec_name).lower():
                            if ov.get("comping_style"):
                                sec["comping_style"] = str(ov["comping_style"])
                                applied.append(f"comping:{sec_name}={ov['comping_style']}")
                            if ov.get("energy_delta") is not None:
                                sec["energy"] = round(float(sec.get("energy", 0.5)) + float(ov["energy_delta"]), 3)
                                applied.append(f"energy:{sec_name}{float(ov['energy_delta']):+.2f}")
                            if ov.get("brightness_shift"):
                                sec["brightness_shift"] = str(ov["brightness_shift"])
                                applied.append(f"brightness:{sec_name}={ov['brightness_shift']}")
                with open(plan_path, "w", encoding="utf-8") as fh:
                    json.dump(plan, fh, ensure_ascii=False, indent=2)
                    fh.write("\n")
            swap = delta.get("swap_role")
            if swap and bp:
                for t in bp.get("tracks") or []:
                    if str(t.get("name", "")).lower() == str(swap).lower():
                        applied.append(f"swap_role:{swap} (marqué, appliqué à la recompilation)")
            return applied

        def _eval() -> dict[str, Any]:
            if not os.path.isfile(bp_path):
                return {"ok": True}
            from song_builder import load_blueprint
            try:
                _bp, meta = load_blueprint(bp_path, validate=True)
            except Exception:
                return {"ok": False, "musical_safety": {"errors": 1}}
            return {"ok": True, "musical_safety": meta.get("musical_safety") or {}}

        before = _eval()
        res = apply_with_rollback(
            proj,
            {"code": "REFINE_TRACK", "target": "project", "params": {"delta_hash": sha256_json(delta)[:16]}},
            _apply, _eval, before,
            snapshot_fn=_snapshot, restore_fn=_restore,
            require_progress=False,
        )
        tags = delta.get("feedback_tags") or []
        taste_recorded = False
        if tags:
            try:
                from taste_memory import record_validated_taste
                r = record_validated_taste(
                    proj, approved_by_user=True,
                    extra_metrics={"pq_score": 8.0, "musical_preferences": {"user_tags": list(tags)}},
                    taste_path=os.path.join(proj, "user_taste_profile.json"),
                )
                taste_recorded = bool(r.get("recorded"))
            except Exception:
                taste_recorded = False
        preview = None
        master = os.path.join(proj, "renders", "master.wav")
        if os.path.isfile(master):
            from preview_lib import build_preview_highlights
            pr = build_preview_highlights(master)
            if pr.get("ok"):
                preview = {"out": os.path.basename(pr["out"]), "duration_s": pr["duration_s"]}
        return _compact_json({
            "ok": True,
            "tour": "3b",
            "verdict": res.get("verdict"),
            "applied": res.get("applied") or [],
            "delta_hash": sha256_json(delta)[:16],
            "seed_lock": bool(delta.get("seed_lock", True)),
            "taste_recorded": taste_recorded,
            "preview_highlights": preview,
        })
    except Exception as exc:
        return _compact_json({"ok": False, "tour": "3b", "error": str(exc)[:300]})




if __name__ == "__main__":
    mcp.run(transport="stdio")
