#!/usr/bin/env python3
"""
Phase 8 Q1 — Deterministic Active-Loudness Auto-Gain Staging (`gain_staging_lib.py`).

Measures active-frame ITU-R BS.1770 K-weighted loudness (ignoring inter-hit silence
so sparse bells/claps/plucks are never penalized vs sustained 808s/pads), maps tracks
to 9 canonical roles, and computes calibrated fader trims via `gl_dsl.db_to_auto`.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np
from scipy import signal

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from gl_dsl import clamp, db_to_auto  # noqa: E402

ANCHOR_PREMASTER_ACTIVE_LUFS = -16.0

CANONICAL_ROLES: tuple[str, ...] = (
    "kick",
    "bass_sub",
    "snare_clap",
    "hihat_perc",
    "lead",
    "chords_pad",
    "counter_melody",
    "ear_candy_fx",
    "vocals",
)

# Relative target triplets (min_db, target_db, max_db) vs profile anchor (0.0 dB):
# - Anchor is `kick` when `has_drums=True` (except `classical_ambient` which anchors on `lead`)
# - Anchor is `lead` when `has_drums=False`
ROLE_TARGET_OFFSETS_DB: dict[str, dict[str, tuple[float, float, float]]] = {
    "trap_rap": {
        "kick":           (-1.0,  0.0, +1.0),
        "bass_sub":       (-2.5, -1.0, +1.0),
        "snare_clap":     (-3.5, -1.5, +0.5),
        "hihat_perc":     (-13.0, -10.0, -7.0),
        "lead":           (-3.0, -1.0, +1.5),
        "chords_pad":     (-11.0, -8.0, -5.5),
        "counter_melody": (-12.0, -9.0, -6.0),
        "ear_candy_fx":   (-16.0, -12.0, -8.5),
        "vocals":         (-1.0, +1.0, +3.0),
    },
    "pop_rnb": {
        "kick":           (-1.0,  0.0, +1.0),
        "bass_sub":       (-4.0, -2.0,  0.0),
        "snare_clap":     (-3.0, -1.5, +0.5),
        "hihat_perc":     (-13.0, -10.5, -7.5),
        "lead":           (-2.0,  0.0, +2.0),
        "chords_pad":     (-9.5, -7.0, -4.5),
        "counter_melody": (-11.0, -8.0, -5.5),
        "ear_candy_fx":   (-15.0, -11.5, -8.0),
        "vocals":         (-0.5, +1.5, +3.5),
    },
    "edm_club": {
        "kick":           (-1.0,  0.0, +1.0),
        "bass_sub":       (-3.0, -1.5, +0.5),
        "snare_clap":     (-4.0, -2.0,  0.0),
        "hihat_perc":     (-11.0, -8.5, -6.0),
        "lead":           (-2.5, -0.5, +1.5),
        "chords_pad":     (-9.0, -6.5, -4.0),
        "counter_melody": (-11.0, -8.5, -6.0),
        "ear_candy_fx":   (-14.0, -10.5, -7.0),
        "vocals":         (-1.5, +0.5, +2.5),
    },
    "rock_metal": {
        "kick":           (-1.0,  0.0, +1.0),
        "bass_sub":       (-3.5, -2.0,  0.0),
        "snare_clap":     (-2.0, -0.5, +1.5),
        "hihat_perc":     (-10.0, -7.5, -5.0),
        "lead":           (-1.5, +0.5, +2.5),
        "chords_pad":     (-6.5, -4.5, -2.5),
        "counter_melody": (-9.5, -7.0, -4.5),
        "ear_candy_fx":   (-14.0, -11.0, -8.0),
        "vocals":         (-0.5, +1.5, +3.5),
    },
    "lofi_chill": {
        "kick":           (-1.0,  0.0, +1.0),
        "bass_sub":       (-3.5, -1.5, +0.5),
        "snare_clap":     (-4.0, -2.0,  0.0),
        "hihat_perc":     (-14.0, -11.0, -8.0),
        "lead":           (-2.5, -0.5, +1.5),
        "chords_pad":     (-8.0, -5.5, -3.5),
        "counter_melody": (-10.5, -8.0, -5.5),
        "ear_candy_fx":   (-16.0, -12.5, -9.0),
        "vocals":         (-1.0, +1.0, +3.0),
    },
    "jazz_acoustic": {
        "kick":           (-1.0,  0.0, +1.0),
        "bass_sub":       (-3.0, -1.0, +1.0),
        "snare_clap":     (-4.5, -2.5, -0.5),
        "hihat_perc":     (-11.0, -8.5, -6.0),
        "lead":           (-1.0, +1.0, +3.0),
        "chords_pad":     (-7.5, -5.0, -3.0),
        "counter_melody": (-9.5, -7.0, -4.5),
        "ear_candy_fx":   (-15.0, -12.0, -8.5),
        "vocals":         (0.0,  +2.0, +4.0),
    },
    "classical_ambient": {
        "kick":           (-8.0, -5.0, -2.0),
        "bass_sub":       (-7.0, -4.5, -2.0),
        "snare_clap":     (-10.0, -7.0, -4.0),
        "hihat_perc":     (-16.0, -13.0, -10.0),
        "lead":           (-1.0,  0.0, +1.0),
        "chords_pad":     (-8.5, -5.5, -3.0),
        "counter_melody": (-10.0, -7.0, -4.5),
        "ear_candy_fx":   (-15.0, -11.5, -8.0),
        "vocals":         (-0.5, +1.5, +3.5),
    },
}

_PROFILE_ALIASES: dict[str, str] = {
    "trap": "trap_rap",
    "rap": "trap_rap",
    "hiphop": "trap_rap",
    "pop": "pop_rnb",
    "rnb": "pop_rnb",
    "edm": "edm_club",
    "house": "edm_club",
    "techno": "edm_club",
    "rock": "rock_metal",
    "metal": "rock_metal",
    "lofi": "lofi_chill",
    "chill": "lofi_chill",
    "jazz": "jazz_acoustic",
    "acoustic": "jazz_acoustic",
    "classical": "classical_ambient",
    "ambient": "classical_ambient",
}

_KW_SOS_CACHE: dict[int, np.ndarray] = {}


def resolve_mix_profile(mix_profile: str) -> str:
    """Normalize profile alias to one of the 7 canonical `ROLE_TARGET_OFFSETS_DB` profiles."""
    key = str(mix_profile or "trap_rap").strip().lower().replace("-", "_").replace(" ", "_")
    if key in ROLE_TARGET_OFFSETS_DB:
        return key
    for alias, target in _PROFILE_ALIASES.items():
        if alias in key:
            return target
    return "trap_rap"


def normalize_role_for_gain_staging(name_or_role: str) -> str:
    """Map any channel name, `ChannelSpec.role`, or stem label to one of the 9 canonical roles."""
    raw = str(name_or_role or "").strip().lower()
    if raw in CANONICAL_ROLES:
        return raw

    if any(k in raw for k in ("vocal", "vox", "chant", "sing", "voice")):
        return "vocals"
    if any(k in raw for k in ("counter", "contre")):
        return "counter_melody"
    if any(k in raw for k in ("kick", "bd", "kicksculpt")):
        return "kick"
    if any(k in raw for k in ("808", "bass", "sub", "reese", "fingerbass")):
        return "bass_sub"
    if any(k in raw for k in ("snare", "clap", "rim")):
        return "snare_clap"
    if any(k in raw for k in ("hat", "hh", "perc", "shaker", "tom", "cymbal", "crash", "ride", "drum")):
        return "hihat_perc"
    if any(k in raw for k in ("ear_candy", "candy", "riser", "impact", "sweep", "noise", "foley")) or raw == "fx":
        return "ear_candy_fx"
    if any(k in raw for k in ("lead", "melody", "topline", "solo", "pluck", "bell", "arp")):
        return "lead"
    if any(k in raw for k in ("chord", "pad", "keys", "piano", "organ", "rhodes", "string", "harmony", "bed", "guitar")):
        return "chords_pad"
    if "fx" in raw and "room" not in raw and "delay" not in raw and "hall" not in raw:
        return "ear_candy_fx"
    return "lead"


def _get_k_weighting_sos(sr: int) -> np.ndarray:
    """Return cached ITU-R BS.1770 K-weighting SOS matrix for sample rate `sr`."""
    sr_i = int(sr)
    if sr_i in _KW_SOS_CACHE:
        return _KW_SOS_CACHE[sr_i]
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr_i)
        sos = np.vstack([
            np.concatenate([f.b / f.a[0], f.a / f.a[0]])
            for _, f in meter._filters.items()
        ]).astype(np.float64)
    except Exception:
        # Fallback: 2nd-order HPF @ 38 Hz + high-shelf approx
        sos = signal.butter(2, min(38.0, 0.35 * sr_i), btype="highpass", fs=sr_i, output="sos")
    _KW_SOS_CACHE[sr_i] = sos
    return sos


def measure_active_loudness(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 50.0,
    hop_ms: float = 25.0,
    abs_gate_dbfs: float = -45.0,
    rel_gate_db: float = -20.0,
) -> dict[str, Any]:
    """
    Measure ITU-R BS.1770 K-weighted active-frame loudness, ignoring inter-hit silence.

    Returns dict with `active_rms_db`, `active_lufs`, `peak_dbfs`, `crest_factor_db`,
    `active_frame_ratio`, and `is_effectively_silent`.
    """
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    n_samples = int(arr.shape[0])
    n_ch = int(arr.shape[1]) if arr.ndim == 2 else 1
    if n_samples == 0 or n_ch == 0:
        return {
            "active_rms_db": -120.0,
            "active_lufs": -120.0,
            "peak_dbfs": -120.0,
            "crest_factor_db": 0.0,
            "active_frame_ratio": 0.0,
            "is_effectively_silent": True,
        }

    peak_lin = float(np.max(np.abs(arr)))
    peak_dbfs = round(20.0 * math.log10(max(peak_lin, 1e-12)), 2)
    if peak_lin < 1e-9:
        return {
            "active_rms_db": -120.0,
            "active_lufs": -120.0,
            "peak_dbfs": peak_dbfs,
            "crest_factor_db": 0.0,
            "active_frame_ratio": 0.0,
            "is_effectively_silent": True,
        }

    sos_k = _get_k_weighting_sos(sr)
    arr_f = arr.astype(np.float32, copy=False)
    y_k = signal.sosfilt(sos_k.astype(np.float32, copy=False), arr_f, axis=0)
    sq_ch = y_k * y_k  # (N, C), float32 is enough for K-weighted frame energy

    frame_n = max(16, int(round(float(frame_ms) * float(sr) / 1000.0)))
    hop_n = max(8, int(round(float(hop_ms) * float(sr) / 1000.0)))

    if n_samples < frame_n:
        p_mean_frames = np.array([float(np.mean(sq_ch, dtype=np.float64))], dtype=np.float64)
        p_sum_frames = p_mean_frames * float(n_ch)
    else:
        # Prefix-sum O(N) window energy for minimal CPU overhead (< 0.5 ms per stem)
        per_sample_mean = np.mean(sq_ch, axis=1, dtype=np.float64)
        cs_mean = np.concatenate([[0.0], np.cumsum(per_sample_mean, dtype=np.float64)])
        starts = np.arange(0, n_samples - frame_n + 1, hop_n, dtype=np.int64)
        inv_fn = 1.0 / float(frame_n)
        p_mean_frames = np.maximum(0.0, (cs_mean[starts + frame_n] - cs_mean[starts]) * inv_fn)
        p_sum_frames = p_mean_frames * float(n_ch)

    frame_db = 10.0 * np.log10(np.maximum(p_mean_frames, 1e-18))
    e_max_db = float(np.max(frame_db))
    gate_db = max(float(abs_gate_dbfs), e_max_db + float(rel_gate_db))
    active_mask = frame_db > gate_db
    active_frame_ratio = float(np.mean(active_mask))

    if not np.any(active_mask):
        max_sum = float(np.max(p_sum_frames))
        active_rms_db = round(e_max_db, 2)
        active_lufs = round(-0.691 + 10.0 * math.log10(max(max_sum, 1e-18)), 2)
        return {
            "active_rms_db": active_rms_db,
            "active_lufs": active_lufs,
            "peak_dbfs": peak_dbfs,
            "crest_factor_db": round(peak_dbfs - active_rms_db, 2),
            "active_frame_ratio": round(active_frame_ratio, 4),
            "is_effectively_silent": True,
        }

    mean_active_ch_power = float(np.mean(p_mean_frames[active_mask]))
    mean_active_sum_power = float(np.mean(p_sum_frames[active_mask]))
    active_rms_db = round(10.0 * math.log10(max(mean_active_ch_power, 1e-18)), 2)
    active_lufs = round(-0.691 + 10.0 * math.log10(max(mean_active_sum_power, 1e-18)), 2)
    crest_factor_db = round(peak_dbfs - active_rms_db, 2)
    is_silent = (active_frame_ratio < 0.005) or (active_rms_db < -50.0)

    return {
        "active_rms_db": active_rms_db,
        "active_lufs": active_lufs,
        "peak_dbfs": peak_dbfs,
        "crest_factor_db": crest_factor_db,
        "active_frame_ratio": round(active_frame_ratio, 4),
        "is_effectively_silent": bool(is_silent),
    }


def _effective_profile_offsets(
    mix_profile: str,
    has_drums: bool,
    roles_present: set[str] | None = None,
) -> tuple[str, str, dict[str, tuple[float, float, float]]]:
    """Resolve effective anchor role and `(min, target, max)` offsets for `mix_profile`."""
    prof = resolve_mix_profile(mix_profile)
    base_table = ROLE_TARGET_OFFSETS_DB[prof]

    if prof == "classical_ambient" or not has_drums:
        anchor_role = "lead"
        if roles_present and "lead" not in roles_present and "chords_pad" in roles_present:
            anchor_role = "chords_pad"
    else:
        anchor_role = "kick"

    shift = base_table[anchor_role][1]
    if abs(shift) < 1e-9:
        return prof, anchor_role, dict(base_table)
    shifted = {
        r: (round(lo - shift, 2), round(tg - shift, 2), round(hi - shift, 2))
        for r, (lo, tg, hi) in base_table.items()
    }
    return prof, anchor_role, shifted


def compute_gain_staging_plan(
    track_measurements: dict[str, dict[str, Any]],
    mix_profile: str,
    has_drums: bool = True,
    anchor_target_lufs: float = ANCHOR_PREMASTER_ACTIVE_LUFS,
    max_delta_db: float = 12.0,
) -> dict[str, Any]:
    """
    Compute deterministic role-aware gain staging trims using active loudness and peak guard.

    Each input entry in `track_measurements` accepts:
      `{track_key: {role, active_lufs, peak_dbfs, current_volume?, is_effectively_silent?}}`
    """
    roles_present = {
        normalize_role_for_gain_staging(str(m.get("role") or k))
        for k, m in track_measurements.items()
    }
    prof, anchor_role, offsets = _effective_profile_offsets(mix_profile, has_drums, roles_present)

    tracks_plan: dict[str, dict[str, Any]] = {}
    for tkey, meas in track_measurements.items():
        c_role = normalize_role_for_gain_staging(str(meas.get("role") or tkey))
        triplet = offsets.get(c_role, offsets["lead"])
        target_offset_db = float(triplet[1])
        target_active_lufs = round(float(anchor_target_lufs) + target_offset_db, 2)

        measured_lufs = float(meas.get("active_lufs", -120.0))
        peak_dbfs = float(meas.get("peak_dbfs", -120.0))
        current_vol = float(clamp(float(meas.get("current_volume", 0.78) or 0.78), 0.05, 1.25))
        eff_silent = bool(meas.get("is_effectively_silent", False)) or measured_lufs <= -55.0

        raw_delta_db = target_active_lufs - measured_lufs
        clamped_delta_db = clamp(raw_delta_db, -float(max_delta_db), float(max_delta_db))

        clip_guard_active = False
        if not eff_silent and (peak_dbfs + clamped_delta_db > -2.0):
            guard_cap = -2.0 - peak_dbfs
            if guard_cap < clamped_delta_db:
                clamped_delta_db = max(-float(max_delta_db), guard_cap)
                clip_guard_active = True

        # Convert via gl_dsl.db_to_auto (normalized 0..1 = fader / 1.25) -> fader [0.10, 1.15]
        auto_norm = db_to_auto(clamped_delta_db, ref_fader=current_vol)
        raw_fader = auto_norm * 1.25
        rec_vol = round(clamp(raw_fader, 0.10, 1.15), 4)
        actual_fader_delta_db = 40.0 * math.log10(max(rec_vol, 1e-6) / max(current_vol, 1e-6))
        residual_gain_db = round(clamped_delta_db - actual_fader_delta_db, 2)

        if eff_silent or (raw_delta_db > float(max_delta_db) and (measured_lufs + clamped_delta_db) < target_active_lufs - 2.0):
            status = "SOURCE_TOO_QUIET"
        elif clip_guard_active:
            status = "CLIP_GUARD_ACTIVE"
        elif abs(clamped_delta_db) <= 0.75:
            status = "OK"
        elif clamped_delta_db > 0.75:
            status = "BOOSTED"
        else:
            status = "ATTENUATED"

        tracks_plan[tkey] = {
            "role": c_role,
            "measured_active_lufs": round(measured_lufs, 2),
            "peak_dbfs": round(peak_dbfs, 2),
            "current_volume": round(current_vol, 4),
            "target_offset_db": target_offset_db,
            "target_range_db": [float(triplet[0]), float(triplet[1]), float(triplet[2])],
            "target_active_lufs": target_active_lufs,
            "raw_delta_db": round(raw_delta_db, 2),
            "clamped_delta_db": round(clamped_delta_db, 2),
            "predicted_active_lufs": round(measured_lufs + clamped_delta_db, 2),
            "predicted_peak_dbfs": round(peak_dbfs + clamped_delta_db, 2),
            "recommended_volume": rec_vol,
            "recommended_volume_auto": rec_vol,
            "recommended_gain_db": round(clamped_delta_db, 2),
            "residual_gain_db": residual_gain_db,
            "status": status,
        }

    return {
        "mix_profile": prof,
        "has_drums": bool(has_drums),
        "anchor_role": anchor_role,
        "anchor_target_lufs": float(anchor_target_lufs),
        "tracks": tracks_plan,
    }
