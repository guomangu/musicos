#!/usr/bin/env python3
"""
Top-Down Target Fingerprint & Multi-Scale Contract Compiler (`target_fingerprint.py`, Phase 6 §2.1 & §3.5).

0 PyTorch / 0 GPU (< 2 s per reference audio):
  1. Extracts DSP fingerprint from 0..3 reference audio files (Bark 24-band, pink slope dB/oct,
     stereo width by band, crest factor, integrated LUFS, onset density, breathing ratio).
  2. Falls back cleanly to `golden_references/spectral_profiles.json` + `emotion_vector` modulation
     when no reference audio is supplied.
  3. Compiles and validates `<project>/song_target_profile.json` and `<project>/multi_scale_contract.json`
     against `schemas/multi_scale_contract.schema.json`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import jsonschema
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
ANALYZER_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from bootstrap_golden_refs import DEFAULT_CORRIDORS_PATH, DEFAULT_PROFILES_PATH, compute_bark24_distribution, load_spectral_corridor_entry
from deep_diagnostics import compute_pink_slope_db_per_oct
from groove_lib import HAAS_POCKET_MS_PROFILES
from taste_memory import query_taste_prior

CONTRACT_SCHEMA_PATH = os.path.join(ROOT_DIR, "schemas", "multi_scale_contract.schema.json")

_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "trap_rap": {
        "target_lufs": -11.0,
        "pink_slope_range": [-5.4, -3.2],
        "low_mids_max": 24.0,
        "air_min": 3.0,
        "dark_air_min": 1.2,
        "sharp_max": 1.90,
        "dark_sharp_max": 1.85,
        "crest_range": [7.5, 13.5],
        "onset_density": 4.8,
        "stereo_width": {"sub": 0.03, "low_mid": 0.16, "high_mid": 0.34, "air": 0.48},
    },
    "pop_rnb": {
        "target_lufs": -10.5,
        "pink_slope_range": [-4.6, -2.9],
        "low_mids_max": 26.0,
        "air_min": 3.0,
        "dark_air_min": 1.5,
        "sharp_max": 1.95,
        "dark_sharp_max": 1.85,
        "crest_range": [7.0, 12.5],
        "onset_density": 4.4,
        "stereo_width": {"sub": 0.03, "low_mid": 0.20, "high_mid": 0.42, "air": 0.56},
    },
    "edm_club": {
        "target_lufs": -9.0,
        "pink_slope_range": [-4.5, -2.6],
        "low_mids_max": 24.0,
        "air_min": 3.5,
        "dark_air_min": 1.75,
        "sharp_max": 2.05,
        "dark_sharp_max": 1.90,
        "crest_range": [6.5, 11.5],
        "onset_density": 5.6,
        "stereo_width": {"sub": 0.02, "low_mid": 0.18, "high_mid": 0.45, "air": 0.62},
    },
    "rock_metal": {
        "target_lufs": -10.0,
        "pink_slope_range": [-4.5, -2.6],
        "low_mids_max": 28.0,
        "air_min": 2.5,
        "dark_air_min": 1.25,
        "sharp_max": 2.10,
        "dark_sharp_max": 1.90,
        "crest_range": [6.5, 12.0],
        "onset_density": 5.2,
        "stereo_width": {"sub": 0.04, "low_mid": 0.24, "high_mid": 0.44, "air": 0.52},
    },
    "lofi_chill": {
        "target_lufs": -14.0,
        "pink_slope_range": [-5.8, -3.5],
        "low_mids_max": 34.0,
        "air_min": 0.5,
        "dark_air_min": 0.25,
        "sharp_max": 1.75,
        "dark_sharp_max": 1.75,
        "crest_range": [9.0, 16.0],
        "onset_density": 3.2,
        "stereo_width": {"sub": 0.04, "low_mid": 0.22, "high_mid": 0.36, "air": 0.42},
    },
    "jazz_acoustic": {
        "target_lufs": -14.5,
        "pink_slope_range": [-5.2, -3.1],
        "low_mids_max": 36.0,
        "air_min": 1.0,
        "dark_air_min": 0.50,
        "sharp_max": 1.85,
        "dark_sharp_max": 1.80,
        "crest_range": [10.0, 18.0],
        "onset_density": 3.6,
        "stereo_width": {"sub": 0.08, "low_mid": 0.26, "high_mid": 0.44, "air": 0.52},
    },
    "classical_ambient": {
        "target_lufs": -16.0,
        "pink_slope_range": [-5.8, -3.2],
        "low_mids_max": 40.0,
        "air_min": 0.3,
        "dark_air_min": 0.15,
        "sharp_max": 1.75,
        "dark_sharp_max": 1.75,
        "crest_range": [11.0, 22.0],
        "onset_density": 1.6,
        "stereo_width": {"sub": 0.12, "low_mid": 0.32, "high_mid": 0.52, "air": 0.65},
    },
}


def _clamp(val: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, val)))


def _band_stereo_width(data: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> float:
    if data.shape[1] < 2 or len(data) < 256:
        return 0.0
    nyq = 0.48 * float(sr)
    hi = min(hi_hz, nyq)
    lo = max(15.0, lo_hz)
    if hi <= lo * 1.1:
        return 0.0
    mid = 0.5 * (data[:, 0] + data[:, 1])
    side = 0.5 * (data[:, 0] - data[:, 1])
    if lo <= 25.0:
        sos = signal.butter(4, hi, btype="lowpass", fs=sr, output="sos")
    else:
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    m_f = signal.sosfiltfilt(sos, mid)
    s_f = signal.sosfiltfilt(sos, side)
    m_rms = float(np.sqrt(np.mean(m_f ** 2))) + 1e-12
    s_rms = float(np.sqrt(np.mean(s_f ** 2)))
    return round(float(_clamp(s_rms / m_rms, 0.0, 2.0)), 4)


def _analyze_single_reference(audio_path: str) -> dict[str, Any]:
    abs_p = os.path.abspath(os.path.expanduser(audio_path))
    if not os.path.isfile(abs_p):
        raise FileNotFoundError(f"Reference audio not found: {abs_p}")

    data, sr = sf.read(abs_p, dtype="float64", always_2d=True)
    if len(data) < max(512, int(0.4 * sr)):
        raise ValueError(f"Reference audio too short (<0.4s): {abs_p}")

    # Cap analysis window at 90s for strict <2s CPU budget on long masters
    max_s = int(90 * sr)
    if len(data) > max_s:
        start = (len(data) - max_s) // 2
        data_win = data[start : start + max_s]
    else:
        data_win = data

    mono = np.mean(data_win, axis=1)
    # Trim dead silence padding for accurate spectral/density metrics
    pk_all = float(np.max(np.abs(mono))) + 1e-12
    active_idx = np.where(np.abs(mono) >= max(1e-5, 0.01 * pk_all))[0]
    if len(active_idx) > int(0.5 * sr):
        data_act = data_win[active_idx[0] : active_idx[-1] + 1]
        mono_act = mono[active_idx[0] : active_idx[-1] + 1]
    else:
        data_act = data_win
        mono_act = mono

    bark24 = compute_bark24_distribution(mono_act, sr)
    pink_slope = compute_pink_slope_db_per_oct(mono_act, sr)

    stereo_width = {
        "sub": _band_stereo_width(data_act, sr, 20.0, 120.0),
        "low_mid": _band_stereo_width(data_act, sr, 120.0, 800.0),
        "high_mid": _band_stereo_width(data_act, sr, 800.0, 4000.0),
        "air": _band_stereo_width(data_act, sr, 4000.0, 18000.0),
    }

    rms = float(np.sqrt(np.mean(mono_act ** 2))) + 1e-12
    pk = float(np.max(np.abs(mono_act))) + 1e-12
    crest_db = round(20.0 * math.log10(pk / rms), 2)

    min_lufs_len = int(0.45 * sr)
    buf_lufs = data_act
    if len(buf_lufs) < min_lufs_len:
        buf_lufs = np.vstack([buf_lufs, np.zeros((min_lufs_len - len(buf_lufs), buf_lufs.shape[1]))])
    meter = pyln.Meter(sr)
    lufs_val = float(meter.integrated_loudness(buf_lufs))
    if math.isinf(lufs_val) or math.isnan(lufs_val):
        lufs_val = round(20.0 * math.log10(rms), 2)
    else:
        lufs_val = round(lufs_val, 2)

    # Onset density (events/sec)
    dur_s = max(0.25, len(mono_act) / float(sr))
    try:
        o_frames = librosa.onset.onset_detect(
            y=mono_act.astype(np.float32),
            sr=sr,
            hop_length=512,
            backtrack=False,
        )
        onset_density = round(len(o_frames) / dur_s, 3)
    except Exception:
        onset_density = 0.0

    # Breathing ratio: percentage of 0.5s windows below -10 dB relative to 90th percentile window RMS
    win_n = max(128, int(0.5 * sr))
    n_w = len(mono_act) // win_n
    if n_w >= 2:
        w_rms = np.sqrt(np.mean(mono_act[: n_w * win_n].reshape(n_w, win_n) ** 2, axis=1))
        p90 = float(np.percentile(w_rms, 90)) + 1e-12
        breathing_ratio = round(float(np.mean(w_rms < p90 * 0.316)), 3)
    else:
        breathing_ratio = 0.15

    # Also measure low_mids & air shares + sharpness-like bright share for threshold calibration
    nperseg = min(16384, max(256, len(mono_act)))
    freqs, psd = signal.welch(mono_act, fs=sr, nperseg=nperseg, scaling="spectrum")
    tot_e = float(np.sum(psd)) + 1e-18
    low_mids_pct = round(100.0 * float(np.sum(psd[(freqs >= 250.0) & (freqs < 500.0)])) / tot_e, 2)
    air_pct = round(100.0 * float(np.sum(psd[(freqs >= 6000.0) & (freqs < min(20000.0, 0.495 * sr))])) / tot_e, 2)

    return {
        "path": abs_p,
        "duration_s": round(len(data) / float(sr), 3),
        "bark24": bark24,
        "pink_slope_db_per_oct": pink_slope,
        "stereo_width_by_band": stereo_width,
        "crest_factor_db": crest_db,
        "integrated_lufs": lufs_val,
        "onset_density_per_sec": onset_density,
        "breathing_ratio": breathing_ratio,
        "low_mids_pct": low_mids_pct,
        "air_pct": air_pct,
    }


def _load_spectral_profile_entry(mix_profile: str, profiles_path: str) -> dict[str, Any] | None:
    """Return the raw `spectral_profiles.json` entry (including `provenance`) if any."""
    if not profiles_path or not os.path.isfile(profiles_path):
        return None
    try:
        with open(profiles_path, encoding="utf-8") as fh:
            data = json.load(fh)
        prof = (data.get("profiles") or {}).get(mix_profile)
        return prof if isinstance(prof, dict) and len(prof.get("bark_24_mean") or []) == 24 else None
    except Exception:
        return None


def _load_fallback_spectral_profile(mix_profile: str, profiles_path: str) -> tuple[list[float], float, float]:
    if os.path.isfile(profiles_path):
        try:
            with open(profiles_path, encoding="utf-8") as fh:
                data = json.load(fh)
            prof = (data.get("profiles") or {}).get(mix_profile)
            if prof and len(prof.get("bark_24_mean") or []) == 24:
                return (
                    [float(x) for x in prof["bark_24_mean"]],
                    float(prof.get("pink_slope_mean", -4.0)),
                    max(0.35, float(prof.get("pink_slope_std", 0.5))),
                )
        except Exception:
            pass
    # Analytic emergency fallback if file missing
    defaults = _PROFILE_DEFAULTS.get(mix_profile, _PROFILE_DEFAULTS["trap_rap"])
    slope = float(sum(defaults["pink_slope_range"]) / 2.0)
    w = np.array([1.0 / ((i + 1) ** 0.85) for i in range(24)], dtype=np.float64)
    w /= np.sum(w)
    return [round(float(x), 6) for x in w], slope, 0.6


def _modulate_bark_with_emotion(bark24: list[float], emotion_vector: dict[str, Any]) -> list[float]:
    val = _clamp(float(emotion_vector.get("valence", 0.0)), -1.0, 1.0)
    energy = _clamp(float(emotion_vector.get("energy", 0.6)), 0.0, 1.0)
    depth = _clamp(float(emotion_vector.get("depth", 0.5)), 0.0, 1.0)

    arr = np.asarray(bark24, dtype=np.float64).copy()
    if len(arr) != 24:
        arr = np.ones(24, dtype=np.float64) / 24.0
    # Negative valence softens upper-mid/air (> 3.1 kHz, indices >= 16)
    if val < -0.15:
        tilt = 1.0 + 0.25 * val  # in [0.75, 0.96]
        arr[16:] *= tilt
    elif val > 0.25:
        arr[16:] *= 1.0 + 0.15 * val
    # High depth slightly reinforces sub/low-end (indices 0..2)
    if depth > 0.6:
        arr[:2] *= 1.0 + 0.15 * (depth - 0.5)
    # High energy reinforces presence/punch slightly (indices 9..16)
    if energy > 0.7:
        arr[9:16] *= 1.0 + 0.12 * (energy - 0.6)

    tot = float(np.sum(arr))
    if tot <= 1e-18:
        return [round(1.0 / 24.0, 6)] * 24
    arr /= tot
    return [round(float(x), 6) for x in arr]


def _section_role_stream_quota(section_name: str, has_drums: bool) -> int:
    low = section_name.lower()
    if any(k in low for k in ("intro", "outro", "coda")):
        return 2
    if any(k in low for k in ("pont", "bridge", "break")):
        return 4 if has_drums else 3
    if any(k in low for k in ("climax", "final", "peak")):
        return 8 if has_drums else 6
    if any(k in low for k in ("refrain", "chorus", "drop", "hook")):
        return 7 if has_drums else 5
    return 5 if has_drums else 4


def _section_default_energy(section_name: str) -> float:
    low = section_name.lower()
    if any(k in low for k in ("intro", "outro", "coda")):
        return 0.30
    if any(k in low for k in ("pont", "bridge", "break")):
        return 0.38
    if any(k in low for k in ("climax", "final", "peak")):
        return 0.98
    if any(k in low for k in ("refrain", "chorus", "drop", "hook")):
        return 0.88
    return 0.62


def _build_sections_energy_arc(
    blueprint: dict[str, Any] | None,
    has_drums: bool,
    base_width: float,
) -> list[dict[str, Any]]:
    structure = (blueprint or {}).get("structure") or []
    if not structure:
        structure = [
            {"section": "Intro", "start_bar": 1, "bars": 4, "energy": 0.30},
            {"section": "Verse", "start_bar": 5, "bars": 8, "energy": 0.60},
            {"section": "Chorus", "start_bar": 13, "bars": 8, "energy": 0.90},
            {"section": "Outro", "start_bar": 21, "bars": 4, "energy": 0.25},
        ]
    arc: list[dict[str, Any]] = []
    cursor = 1
    for item in structure:
        sec_name = str(item.get("section") or item.get("name") or f"Section_{len(arc) + 1}")
        bars = max(1, int(item.get("bars", 4)))
        start_bar = int(item.get("start_bar", cursor))
        cursor = start_bar + bars
        en = float(item.get("energy", _section_default_energy(sec_name)))
        en = _clamp(en, 0.05, 1.0)
        max_streams = _section_role_stream_quota(sec_name, has_drums)
        arc.append({
            "section": sec_name,
            "start_bar": start_bar,
            "bars": bars,
            "lufs_rel": round(en, 3),
            "centroid_norm": round(_clamp(0.25 + 0.65 * en, 0.0, 1.0), 3),
            "onset_density_norm": round(_clamp((0.20 + 0.75 * en) * (1.0 if has_drums else 0.55), 0.0, 1.0), 3),
            "stereo_width": round(_clamp(base_width * (0.75 + 0.45 * en), 0.0, 1.5), 3),
            "max_cognitive_streams": max_streams,
        })
    return arc


def _build_tension_curve_4bar(tension: float) -> list[float]:
    t = _clamp(float(tension), 0.05, 0.95)
    c = [
        round(_clamp(t - 0.14, 0.05, 0.95), 3),
        round(_clamp(t + 0.02, 0.05, 0.95), 3),
        round(_clamp(t + 0.18, 0.08, 0.98), 3),
        round(_clamp(t - 0.06, 0.05, 0.95), 3),
    ]
    return c


def validate_multi_scale_contract(contract: dict[str, Any]) -> None:
    """Validate `multi_scale_contract.json` against `schemas/multi_scale_contract.schema.json`."""
    if os.path.isfile(CONTRACT_SCHEMA_PATH):
        with open(CONTRACT_SCHEMA_PATH, encoding="utf-8") as fh:
            schema = json.load(fh)
        jsonschema.validate(contract, schema)


def load_multi_scale_contract(
    contract_path: str = "",
    project_dir: str = "",
) -> dict[str, Any] | None:
    """Load an explicit or `<project_dir>/multi_scale_contract.json` if it exists."""
    cands: list[str] = []
    if contract_path.strip():
        cands.append(os.path.abspath(os.path.expanduser(contract_path)))
    if project_dir.strip():
        cands.append(os.path.join(os.path.abspath(os.path.expanduser(project_dir)), "multi_scale_contract.json"))
    for p in cands:
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
    return None


def extract_target_fingerprint(
    audio_paths: list[str] | None = None,
    mix_profile: str = "trap_rap",
    emotion_vector: dict[str, Any] | None = None,
    *,
    blueprint: dict[str, Any] | str | None = None,
    project_dir: str = "",
    out_contract_path: str = "",
    out_profile_path: str = "",
    profiles_path: str = DEFAULT_PROFILES_PATH,
    corridors_path: str = DEFAULT_CORRIDORS_PATH,
    subgenre_tag: str = "",
    has_drums: bool | None = None,
    taste_memory_path: str | None = None,
    approve_taste: bool = False,
) -> dict[str, Any]:
    """
    Extract multi-scale target fingerprint from 0..3 reference audio tracks (or fallback
    to `spectral_profiles.json` + `emotion_vector`) and compile `multi_scale_contract.json`.
    """
    bp_dict: dict[str, Any] | None = None
    if isinstance(blueprint, dict):
        bp_dict = blueprint
    elif isinstance(blueprint, str) and blueprint.strip() and os.path.isfile(blueprint):
        with open(blueprint, encoding="utf-8") as fh:
            bp_dict = json.load(fh)
    elif project_dir.strip():
        cand_bp = os.path.join(os.path.abspath(os.path.expanduser(project_dir)), "song_blueprint.json")
        if os.path.isfile(cand_bp):
            with open(cand_bp, encoding="utf-8") as fh:
                bp_dict = json.load(fh)

    eff_profile = (mix_profile or (bp_dict or {}).get("mix_profile") or "trap_rap").strip()
    if eff_profile not in _PROFILE_DEFAULTS:
        eff_profile = "trap_rap"
    p_def = _PROFILE_DEFAULTS[eff_profile]

    ev = dict((bp_dict or {}).get("emotion_vector") or {})
    if emotion_vector:
        ev.update(emotion_vector)
    valence = _clamp(float(ev.get("valence", 0.0)), -1.0, 1.0)
    tension = _clamp(float(ev.get("tension", 0.35)), 0.0, 1.0)
    energy = _clamp(float(ev.get("energy", 0.65)), 0.0, 1.0)
    depth = _clamp(float(ev.get("depth", 0.50)), 0.0, 1.0)
    ev_norm = {"valence": valence, "tension": tension, "energy": energy, "depth": depth}

    if has_drums is None:
        if bp_dict and "has_drums" in bp_dict:
            eff_has_drums = bool(bp_dict["has_drums"])
        else:
            eff_has_drums = eff_profile != "classical_ambient"
    else:
        eff_has_drums = bool(has_drums)

    refs_in = [p for p in (audio_paths or []) if str(p).strip()][:3]
    ref_analyses: list[dict[str, Any]] = []
    for rp in refs_in:
        ref_analyses.append(_analyze_single_reference(rp))

    is_dark = valence < -0.2
    eff_subgenre = subgenre_tag.strip() or str((bp_dict or {}).get("genre") or eff_profile)

    if ref_analyses:
        source_mode = "reference_audio"
        reference_tier = 1  # user_reference_audio
        reference_provenance = "user_reference_audio"
        bark_mat = np.asarray([r["bark24"] for r in ref_analyses], dtype=np.float64)
        med_bark = np.median(bark_mat, axis=0)
        med_bark /= max(1e-18, float(np.sum(med_bark)))
        bark24_target = [round(float(x), 6) for x in med_bark]

        slopes = [float(r["pink_slope_db_per_oct"]) for r in ref_analyses]
        med_slope = float(np.median(slopes))
        slope_tol = max(0.85, float(np.std(slopes)) + 0.65)
        pink_slope_range = [round(med_slope - slope_tol, 2), round(med_slope + slope_tol, 2)]

        stereo_width = {
            band: round(float(np.median([r["stereo_width_by_band"][band] for r in ref_analyses])), 4)
            for band in ("sub", "low_mid", "high_mid", "air")
        }

        crests = [float(r["crest_factor_db"]) for r in ref_analyses]
        med_crest = float(np.median(crests))
        crest_range = [round(max(3.5, med_crest - 2.5), 2), round(med_crest + 2.5, 2)]

        target_lufs = round(float(_clamp(float(np.median([r["integrated_lufs"] for r in ref_analyses])), -26.0, -5.0)), 2)
        target_onset_density = round(float(np.median([r["onset_density_per_sec"] for r in ref_analyses])), 3)
        breathing_ratio = round(float(np.median([r["breathing_ratio"] for r in ref_analyses])), 3)

        med_lm = float(np.median([r["low_mids_pct"] for r in ref_analyses]))
        med_air = float(np.median([r["air_pct"] for r in ref_analyses]))
        low_mids_max = round(max(float(p_def["low_mids_max"]), med_lm + 4.0), 1)
        base_air_min = float(p_def["dark_air_min"] if is_dark else p_def["air_min"])
        air_min = round(max(0.15, min(base_air_min, med_air * 0.70) if med_air < base_air_min else base_air_min), 2)
        sharp_max = float(p_def["dark_sharp_max"] if is_dark else p_def["sharp_max"])
        ref_paths_resolved = [r["path"] for r in ref_analyses]
    else:
        # v3.1 REF-04 — chaîne de repli à 3 niveaux :
        #   Niveau 2 = `golden_references/spectral_corridors.json` versionné (+ taste_memory <= 30 %)
        #   Niveau 3 = baseline psychoacoustique analytique si aucun JSON disponible.
        taste_prior = query_taste_prior(eff_profile, ev_norm, top_k=3, taste_path=taste_memory_path)
        corridor_entry = load_spectral_corridor_entry(eff_profile, corridors_path)
        profile_entry = _load_spectral_profile_entry(eff_profile, profiles_path)
        if corridor_entry is not None or profile_entry is not None:
            reference_tier = 2  # versioned_genre_corridor
            reference_provenance = (corridor_entry or {}).get("provenance") or profile_entry.get("provenance", "internal_bootstrap")
        else:
            reference_tier = 3  # psychoacoustic_baseline
            reference_provenance = "psychoacoustic_baseline"
        base_bark, base_slope, base_slope_std = _load_fallback_spectral_profile(eff_profile, profiles_path)
        if corridor_entry is not None:
            cb = (corridor_entry.get("bark24") or {}).get("p50")
            if isinstance(cb, list) and len(cb) == 24:
                base_bark = [float(x) for x in cb]
                base_slope = float((corridor_entry["metrics"].get("pink_slope_db_per_oct") or {}).get("p50", base_slope))
                base_slope_std = 0.6
        is_external_anchor = bool(profile_entry and profile_entry.get("provenance") == "external_pro_reference")
        if taste_prior is not None:
            source_mode = "user_taste_memory"
            b_prior = taste_prior.get("bark24_mean")
            taste_bark: list[float] | None = None
            if isinstance(b_prior, list) and len(b_prior) == 24:
                b_arr = np.maximum(0.0, np.asarray(b_prior, dtype=np.float64))
                b_sum = float(np.sum(b_arr))
                if b_sum > 1e-12:
                    taste_bark = list(b_arr / b_sum)
            tol = max(0.85, base_slope_std * 1.8)
            if taste_bark is not None and is_external_anchor:
                # Q4.2 bounded merge: external pro anchor dominates automatic memory.
                # v3.1 REF-04 : la fusion taste_memory est plafonnée à 30 % (approve_taste ignoré).
                w_taste = 0.30
                w_ext = 1.0 - w_taste
                ext_bark = np.maximum(0.0, np.asarray(base_bark, dtype=np.float64))
                ext_bark = ext_bark / max(1e-18, float(np.sum(ext_bark)))
                blended = w_ext * ext_bark + w_taste * np.asarray(taste_bark, dtype=np.float64)
                blended = blended / max(1e-18, float(np.sum(blended)))
                bark24_target = [round(float(x), 6) for x in blended]
                slope_taste = float(taste_prior.get("pink_slope_db_per_oct", base_slope))
                eff_slope = w_ext * float(profile_entry.get("pink_slope_mean", base_slope)) + w_taste * slope_taste
                source_mode = "external_pro_reference+taste_memory"
            else:
                bark24_target = (
                    [round(float(x), 6) for x in taste_bark]
                    if taste_bark is not None
                    else _modulate_bark_with_emotion(base_bark, ev_norm)
                )
                eff_slope = float(taste_prior.get("pink_slope_db_per_oct", base_slope))
            pink_slope_range = [round(eff_slope - tol, 2), round(eff_slope + tol, 2)]

            sw_prior = taste_prior.get("stereo_width_by_band") or {}
            stereo_width = {band: round(float(sw_prior.get(band, p_def["stereo_width"][band])), 4) for band in ("sub", "low_mid", "high_mid", "air")}

            if taste_prior.get("crest_factor_db") is not None:
                c_mid = float(taste_prior["crest_factor_db"])
                crest_range = [round(max(3.5, c_mid - 2.0), 2), round(c_mid + 2.0, 2)]
            else:
                crest_lo, crest_hi = p_def["crest_range"]
                if energy > 0.75:
                    crest_lo = max(4.5, crest_lo - 1.0)
                crest_range = [round(crest_lo, 2), round(crest_hi, 2)]

            target_lufs = round(_clamp(float(taste_prior.get("integrated_lufs", p_def["target_lufs"])), -26.0, -5.0), 2)
            fallback_onset = round(float(p_def["onset_density"]) * (0.75 + 0.5 * energy) * (1.0 if eff_has_drums else 0.45), 3)
            target_onset_density = round(float(taste_prior.get("onset_density_per_sec", fallback_onset)), 3)
            breathing_ratio = round(float(taste_prior.get("breathing_ratio", 0.12 + 0.25 * (1.0 - energy))), 3)
            low_mids_max = float(p_def["low_mids_max"])
            air_min = float(p_def["dark_air_min"] if is_dark else p_def["air_min"])
            sharp_max = float(p_def["dark_sharp_max"] if is_dark else p_def["sharp_max"])
            ref_paths_resolved = []
        else:
            source_mode = "external_pro_reference" if is_external_anchor else "profile_fallback"
            bark24_target = _modulate_bark_with_emotion(base_bark, ev_norm)
            slope_shift = -0.25 * abs(valence) if valence < -0.2 else (0.15 * valence if valence > 0.3 else 0.0)
            eff_slope = base_slope + slope_shift
            tol = max(0.85, base_slope_std * 1.8)
            pink_slope_range = [round(eff_slope - tol, 2), round(eff_slope + tol, 2)]
            stereo_width = dict(p_def["stereo_width"])
            crest_lo, crest_hi = p_def["crest_range"]
            if energy > 0.75:
                crest_lo = max(4.5, crest_lo - 1.0)
            crest_range = [round(crest_lo, 2), round(crest_hi, 2)]
            target_lufs = float(p_def["target_lufs"])
            target_onset_density = round(float(p_def["onset_density"]) * (0.75 + 0.5 * energy) * (1.0 if eff_has_drums else 0.45), 3)
            breathing_ratio = round(0.12 + 0.25 * (1.0 - energy), 3)
            low_mids_max = float(p_def["low_mids_max"])
            air_min = float(p_def["dark_air_min"] if is_dark else p_def["air_min"])
            sharp_max = float(p_def["dark_sharp_max"] if is_dark else p_def["sharp_max"])
            ref_paths_resolved = []

    haas_profile = dict(HAAS_POCKET_MS_PROFILES.get(eff_profile, HAAS_POCKET_MS_PROFILES["trap_rap"]))
    if "kick" not in haas_profile:
        haas_profile["kick"] = 0.0

    avg_width = float(np.mean([stereo_width["low_mid"], stereo_width["high_mid"]]))
    sections_arc = _build_sections_energy_arc(bp_dict, eff_has_drums, avg_width)

    profile_payload: dict[str, Any] = {
        "version": 1,
        "source": source_mode,
        "source_mode": source_mode,
        "reference_tier": reference_tier,
        "reference_provenance": reference_provenance,
        "mix_profile": eff_profile,
        "subgenre_tag": eff_subgenre,
        "has_drums": eff_has_drums,
        "emotion_vector": ev_norm,
        "reference_tracks": ref_paths_resolved,
        "bark24_target": bark24_target,
        "pink_slope_range": pink_slope_range,
        "stereo_width_by_band": stereo_width,
        "crest_factor_range_db": crest_range,
        "target_lufs": target_lufs,
        "target_onset_density_per_sec": target_onset_density,
        "breathing_ratio": breathing_ratio,
        "references": ref_analyses,
    }

    contract_payload: dict[str, Any] = {
        "version": "1.0",
        "source": source_mode,
        "reference_tier": reference_tier,
        "reference_provenance": reference_provenance,
        "mix_profile": eff_profile,
        "has_drums": eff_has_drums,
        "emotion_vector": ev_norm,
        "s5_context": {
            "reference_tracks": ref_paths_resolved,
            "subgenre_tag": eff_subgenre,
            "bark24_target": bark24_target,
            "pink_slope_range": pink_slope_range,
            "stereo_width_by_band": stereo_width,
            "crest_factor_range_db": crest_range,
            "target_onset_density_per_sec": target_onset_density,
            "target_lufs": target_lufs,
            "breathing_ratio": breathing_ratio,
            "low_mids_max": low_mids_max,
            "air_min": air_min,
            "sharp_max": sharp_max,
            "spectral_match_min": 0.60 if source_mode == "reference_audio" else 0.45,
        },
        "s4_form": {
            "sections_energy_arc": sections_arc,
            "max_8bar_ssm_similarity": 0.94,
            "boundary_alignment_tolerance_bars": 1.0,
        },
        "s3_syntax": {
            "motif_similarity_band": [0.15, 0.55],
            "tension_curve_4bar": _build_tension_curve_4bar(tension),
            "max_chords_lead_attack_overlap": 0.25,
        },
        "s2_acoustic": {
            "haas_targets_ms": haas_profile,
            "onset_lag_tolerance_ms": 7.0,
            "max_simultaneous_bark_transients": 2,
        },
        "s1_physical": {
            "min_kick_bass_transient_corr": 0.25,
            "min_sustained_spectral_flux": 0.015,
        },
    }

    validate_multi_scale_contract(contract_payload)

    abs_proj = os.path.abspath(os.path.expanduser(project_dir)) if project_dir.strip() else ""
    eff_prof_out = os.path.abspath(os.path.expanduser(out_profile_path)) if out_profile_path.strip() else (
        os.path.join(abs_proj, "song_target_profile.json") if abs_proj else ""
    )
    eff_cont_out = os.path.abspath(os.path.expanduser(out_contract_path)) if out_contract_path.strip() else (
        os.path.join(abs_proj, "multi_scale_contract.json") if abs_proj else ""
    )

    if eff_prof_out:
        os.makedirs(os.path.dirname(eff_prof_out), exist_ok=True)
        with open(eff_prof_out, "w", encoding="utf-8") as fh:
            json.dump(profile_payload, fh, indent=2, ensure_ascii=False)
    if eff_cont_out:
        os.makedirs(os.path.dirname(eff_cont_out), exist_ok=True)
        with open(eff_cont_out, "w", encoding="utf-8") as fh:
            json.dump(contract_payload, fh, indent=2, ensure_ascii=False)

    return {
        "source_mode": source_mode,
        "reference_tier": reference_tier,
        "reference_provenance": reference_provenance,
        "reference_count": len(ref_paths_resolved),
        "mix_profile": eff_profile,
        "has_drums": eff_has_drums,
        "profile_path": eff_prof_out or None,
        "contract_path": eff_cont_out or None,
        "song_target_profile": profile_payload,
        "multi_scale_contract": contract_payload,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Top-Down target fingerprint & compile multi_scale_contract.json (0 GPU).")
    ap.add_argument("--project", default="", help="Project directory to write song_target_profile.json & multi_scale_contract.json")
    ap.add_argument("--refs", nargs="*", default=[], help="0..3 reference audio paths (.wav/.mp3/.flac)")
    ap.add_argument("--profile", default="", help="Mix profile override")
    ap.add_argument("--blueprint", default="", help="Optional song_blueprint.json path")
    ap.add_argument("--subgenre", default="")
    ap.add_argument("--valence", type=float, default=None)
    ap.add_argument("--tension", type=float, default=None)
    ap.add_argument("--energy", type=float, default=None)
    ap.add_argument("--depth", type=float, default=None)
    ap.add_argument("--no-drums", action="store_true")
    ap.add_argument("--taste-memory", default="", help="Optional user_taste_profile.json override")
    ap.add_argument("--approve-taste", action="store_true", help="Allow up to 60% taste-memory weight over external pro anchors")
    ap.add_argument("--json-compact", action="store_true")
    args = ap.parse_args()

    ev: dict[str, Any] = {}
    if args.valence is not None:
        ev["valence"] = args.valence
    if args.tension is not None:
        ev["tension"] = args.tension
    if args.energy is not None:
        ev["energy"] = args.energy
    if args.depth is not None:
        ev["depth"] = args.depth

    res = extract_target_fingerprint(
        audio_paths=args.refs,
        mix_profile=args.profile,
        emotion_vector=ev or None,
        blueprint=args.blueprint or None,
        project_dir=args.project,
        subgenre_tag=args.subgenre,
        has_drums=False if args.no_drums else None,
        taste_memory_path=args.taste_memory or None,
        approve_taste=bool(args.approve_taste),
    )
    if args.json_compact:
        c = res["multi_scale_contract"]
        out = {
            "source_mode": res["source_mode"],
            "reference_count": res["reference_count"],
            "mix_profile": res["mix_profile"],
            "target_lufs": c["s5_context"].get("target_lufs"),
            "pink_slope_range": c["s5_context"].get("pink_slope_range"),
            "contract_path": res["contract_path"],
            "profile_path": res["profile_path"],
        }
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
