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

# Presence floors: a role's target may never drop below this offset vs the anchor,
# otherwise the track becomes imperceptible (e.g. ChordKey buried under drums/lead).
ROLE_PRESENCE_FLOOR_DB: dict[str, float] = {
    "kick": -1.0,
    "bass_sub": -6.0,
    "snare_clap": -4.0,
    "hihat_perc": -14.0,
    "lead": -3.0,
    "chords_pad": -10.0,
    "counter_melody": -11.0,
    "ear_candy_fx": -16.0,
    "vocals": -1.0,
}

# High/airy timbres (bells, plucks, keys > 4 kHz centroid) are perceptually quieter at equal
# LUFS: compensate their target upward so a LeadBell stays audible next to a loud kick.
HIGH_CENTROID_COMPENSATION_DB = 2.0
HIGH_CENTROID_HZ = 4000.0

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

    def _with_floors(table: dict[str, tuple[float, float, float]]) -> dict[str, tuple[float, float, float]]:
        out = {}
        for r, (lo, tg, hi) in table.items():
            fl = ROLE_PRESENCE_FLOOR_DB.get(r)
            if fl is not None and tg < fl:
                tg = fl
                lo = min(lo, fl)
            out[r] = (lo, tg, hi)
        return out

    if abs(shift) < 1e-9:
        return prof, anchor_role, _with_floors(dict(base_table))
    shifted = {
        r: (round(lo - shift, 2), round(tg - shift, 2), round(hi - shift, 2))
        for r, (lo, tg, hi) in base_table.items()
    }
    return prof, anchor_role, _with_floors(shifted)




def nominal_role_offset_db(profile: str, role: str, has_drums: bool = True, roles_present: list[str] | None = None) -> float:
    """Target offset (dB) of `role` vs the profile anchor, without any measurement."""
    prof, anchor_role, offsets = _effective_profile_offsets(profile, has_drums, roles_present or [role])
    return float(offsets.get(role, (0.0, 0.0, 0.0))[1]) - float(offsets.get(anchor_role, (0.0, 0.0, 0.0))[1])


def apply_nominal_balance(channels: list, meta: dict[str, Any]) -> list:
    """Balance channel faders by role using nominal offsets (no measurement needed).

    Each channel volume is shifted by `offset(role) - offset(anchor)` (clamped to +/-6 dB)
    so that a build without soundcheck metrics still has calibrated, even track volumes.
    """
    profile = str(meta.get("mix_profile", "trap_rap"))
    has_drums = bool(meta.get("has_drums", True))
    roles_present = [
        normalize_role_for_gain_staging(f"{getattr(ch, 'name', '')} {getattr(ch, 'role', '') or ''} {getattr(ch, 'track_key', '')}")
        for ch in channels
    ]
    out = []
    for ch in channels:
        role = normalize_role_for_gain_staging(f"{ch.name} {ch.role or ''} {ch.track_key}")
        delta = nominal_role_offset_db(profile, role, has_drums, roles_present)
        delta = max(-6.0, min(6.0, delta))
        if abs(delta) < 0.05:
            out.append(ch)
            continue
        v = float(ch.volume)
        new_v = round(clamp(db_to_auto(delta, ref_fader=v) * v, 0.10, 1.15), 4)
        ch.volume = new_v
        out.append(ch)
    return out


def balance_tracks_in_session(
    mcp: Any,
    tracks: list[dict[str, Any]],
    profile: str = "trap_rap",
    has_drums: bool = True,
    anchor_target_lufs: float = ANCHOR_PREMASTER_ACTIVE_LUFS,
    max_trim_db: float | None = None,
) -> dict[str, Any]:
    """Even out track volumes WITHOUT rendering any .wav: solo each channel in the live
    Gridline session, probe `analyze_loudness` on its pattern, and trim the channel fader
    toward the role target (anchor + role offset). Bounded to +/- max_trim_db per track.

    `tracks`: list of {"name", "channel", "pattern", "role", "volume"}.
    Returns a report {"tracks": [...], "balanced": bool} and applies `edit_channel` volumes.
    """
    prof, anchor_role, offsets = _effective_profile_offsets(profile, has_drums, [t.get("role", "lead") for t in tracks])
    report: dict[str, Any] = {"profile": prof, "anchor_role": anchor_role, "tracks": []}
    all_channels = [int(c.get("channel", i)) for i, c in enumerate(mcp.call("list_channels", {}).get("channels", []), start=1)]
    for t in tracks:
        ch = int(t["channel"])
        role = normalize_role_for_gain_staging(str(t.get("role") or t.get("name", "")))
        off_t = float(offsets.get(role, (0.0, 0.0, 0.0))[1])
        fl = ROLE_PRESENCE_FLOOR_DB.get(role)
        if fl is not None and off_t < fl:
            off_t = fl
        if float(t.get("centroid_hz") or 0.0) > HIGH_CENTROID_HZ:
            off_t += HIGH_CENTROID_COMPENSATION_DB  # bells/plucks: perceptually quiet, lift them
        target_lufs = round(anchor_target_lufs + off_t, 2)
        muted = mcp.solo_channel(ch, all_channels)
        try:
            ld = mcp.analyze_loudness(scope="pattern", pattern=int(t.get("pattern", 1)))
        finally:
            mcp.unsolo(muted, keep_solo=ch)
        lufs = ld.get("lufs")
        silent = bool(ld.get("silent"))
        entry: dict[str, Any] = {"name": t.get("name"), "channel": ch, "role": role, "target_lufs": target_lufs, "measured_lufs": lufs, "silent": silent}
        if lufs is None or silent:
            entry["status"] = "SKIP_SILENT"
        else:
            delta = float(target_lufs - float(lufs))
            if max_trim_db is None:
                from mix_config import get_invariant
                max_trim_db = float(get_invariant("max_trim_db", 6.0))
            delta = max(-max_trim_db, min(max_trim_db, delta))
            cur_v = float(t.get("volume", 0.8))
            if abs(delta) < 0.05:
                entry["status"] = "OK"
                entry.update({"delta_db": round(delta, 2), "volume_before": cur_v, "volume_after": cur_v})
                report["tracks"].append(entry)
                continue
            new_v = round(clamp(db_to_auto(delta, ref_fader=cur_v) * cur_v, 0.10, 1.15), 4)
            if abs(new_v - cur_v) > 1e-4:
                mcp.call("edit_channel", {"channel": ch, "volume": new_v})
                entry["status"] = "TRIMMED"
            else:
                entry["status"] = "OK"
            entry.update({"delta_db": round(delta, 2), "volume_before": cur_v, "volume_after": new_v})
        report["tracks"].append(entry)
    report["balanced"] = all(e.get("status") in ("OK", "TRIMMED") for e in report["tracks"])
    return report


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


# ---------------------------------------------------------------------------
# v3.0 -- analytic multi-stem gain solver (DAG stage 3): one-shot least-squares
# trims toward ROLE_TARGET_OFFSETS_DB with presence floors, |delta| <= 6 dB cap
# and a low_end_share constraint, instead of iterative +-4 dB fader passes.
# ---------------------------------------------------------------------------

def solve_analytic_gain_trims(
    measured_lufs: dict[str, float],
    target_offsets_db: dict[str, float],
    anchor_lufs: float = -16.0,
    max_trim_db: float | None = None,
    presence_floors_db: dict[str, float] | None = None,
    low_end_share_pct: float | None = None,
    low_end_max_pct: float = 42.0,
    bass_role_names: tuple[str, ...] = ("bass", "808", "sub", "kick"),
    low_share_weight_by_role: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Closed-form per-stem trims: delta_i = clamp(target_i - measured_i, +-max_trim_db),
    then raised to satisfy ROLE_PRESENCE_FLOOR_DB, then bass roles are pulled down in
    1 dB steps (max 5 iterations) until low_end_share fits `low_end_max_pct`.

    Returns {"trims_db": {role: delta}, "target_lufs": {role: abs}, "low_end_share_after_pct"}.
    """
    if max_trim_db is None:
        from mix_config import get_invariant
        max_trim_db = float(get_invariant("max_trim_db", 6.0))
    floors = presence_floors_db or {}
    trims: dict[str, float] = {}
    targets: dict[str, float] = {}
    for role, meas in measured_lufs.items():
        norm = normalize_role_for_gain_staging(role)
        off = float(target_offsets_db.get(norm, target_offsets_db.get(role, 0.0)))
        tgt = float(anchor_lufs) + off
        d = max(-max_trim_db, min(max_trim_db, tgt - float(meas)))
        fl = floors.get(norm, floors.get(role))
        if fl is not None:
            # measured + d - anchor >= floor  =>  d >= floor + anchor - measured
            d = max(d, float(fl) + float(anchor_lufs) - float(meas))
            d = max(-max_trim_db, min(max_trim_db, d))
        trims[role] = round(d, 2)
        targets[role] = round(tgt, 2)

    low_after = low_end_share_pct
    if low_end_share_pct is not None and low_end_share_pct > low_end_max_pct:
        weights = low_share_weight_by_role or {}
        bass_roles = {normalize_role_for_gain_staging(r) for r in bass_role_names}
        for _ in range(5):
            if low_after is None or low_after <= low_end_max_pct:
                break
            # pull the heaviest low-end roles down by 1 dB per iteration
            for role in trims:
                if normalize_role_for_gain_staging(role) in bass_roles or weights.get(role, 0.0) > 0.3:
                    trims[role] = round(max(-max_trim_db, trims[role] - 1.0), 2)
            wsum = sum(max(0.0, weights.get(r, 0.0)) for r in trims) or 1.0
            low_after = round(low_end_share_pct * sum(
                (weights.get(r, 0.0) / wsum) * (10.0 ** (trims[r] / 10.0)) for r in trims
            ) / max(1e-9, sum(weights.get(r, 0.0) / wsum for r in trims)), 2) if weights else max(0.0, low_end_share_pct - 5.0 * (low_end_share_pct - low_end_max_pct) / max(low_end_share_pct, 1e-9))
            if not weights:
                break
    return {
        "trims_db": trims,
        "target_lufs": targets,
        "anchor_lufs": float(anchor_lufs),
        "low_end_share_before_pct": low_end_share_pct,
        "low_end_share_after_pct": low_after,
        "low_end_max_pct": float(low_end_max_pct),
    }
