#!/usr/bin/env python3
"""
Macro-Structure & Holistic Musical Form Audit (`macro_structure_audit.py`, Phase 6 §3.4).

Scale S4 (16 bars -> complete song), 0 PyTorch / 0 GPU (< 2.0 s CPU for 3-min track):
  - Axis H: Bar-synchronized Self-Similarity Matrix (SSM) combining L2-normalized
    12-dim CQT Chroma (0.35), MFCC 1..13 (0.40), and 16-step onset rhythm profile (0.25).
    - Strict 8-bar cloned-loop detector: S_bar8(b) > 0.94 -> MUTATE_BLOCK_DENSITY_OR_LPF.
    - Gaussian Checkerboard Novelty Curve N(m) (L=4 bars) aligned with blueprint section
      boundaries (±1 bar tolerance) -> REINFORCE_SECTION_TRANSITION if missing.
  - Axis H x V: 5D Dramatic Arc A(b) = (LUFS_rel, SpecCentroid, OnsetDensity, MS_Width, CogStreams)
    computed per 4-bar block / section and compared against `multi_scale_contract.json`
    (`s4_form.sections_energy_arc`), supporting `has_drums=False` (`classical_ambient`).
  - Axis V: Per-bar Cognitive Ceiling audit on separated stems (tracks > -28 dBFS RMS vs
    Arrangement Matrix quotas) -> PRUNE_TRACK_IN_SECTION if >2 consecutive bars exceed quota.

Full SSM, novelty curve, 5D arc, and stem activity matrices are saved to
`<project_dir>/renders/diagnostics/s4_macro_structure.json` while returning compact verdicts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from gain_staging_lib import measure_active_loudness, normalize_role_for_gain_staging
from target_fingerprint import load_multi_scale_contract

DEFAULT_SECTION_QUOTAS: dict[str, int] = {
    "intro": 2,
    "outro": 2,
    "pont": 4,
    "bridge": 4,
    "pre_chorus": 4,
    "break": 4,
    "breakdown": 4,
    "couplet": 5,
    "verse": 5,
    "buildup": 5,
    "refrain": 7,
    "chorus": 7,
    "drop": 7,
    "climax": 8,
    "final": 8,
}

_NON_INSTRUMENT_STEM_KW = ("master", "reverb", "fx room", "bus_master", "bus ")


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _l2_normalize_rows(mat: np.ndarray, *, fallback_unit: bool = False) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    valid = (norms[:, 0] > 1e-8)
    out = np.zeros_like(mat, dtype=np.float64)
    out[valid] = mat[valid] / norms[valid]
    if fallback_unit and np.any(~valid) and mat.shape[1] > 0:
        out[~valid, 0] = 0.0
    return out, valid


def _section_quota_from_name(name: str, contract_arc: list[dict[str, Any]] | None = None) -> int:
    low = (name or "").strip().lower()
    if contract_arc:
        for item in contract_arc:
            if isinstance(item, dict) and str(item.get("section") or "").strip().lower() == low:
                if item.get("max_cognitive_streams"):
                    return int(item["max_cognitive_streams"])
    for kw, quota in DEFAULT_SECTION_QUOTAS.items():
        if kw in low:
            return quota
    return 6


def _parse_sections_spec(
    sections: str | list[dict[str, Any]] | None,
    blueprint: str | dict[str, Any] | None,
    contract: dict[str, Any] | None,
    total_bars: int,
) -> list[dict[str, Any]]:
    """Resolve section timeline `[{section, start_bar, bars, quota, target_arc}]` (1-based start_bar)."""
    contract_arc_raw = ((contract or {}).get("s4_form") or {}).get("sections_energy_arc")
    contract_arc: list[dict[str, Any]] = []
    if isinstance(contract_arc_raw, list):
        contract_arc = [x for x in contract_arc_raw if isinstance(x, dict)]
    elif isinstance(contract_arc_raw, dict):
        cursor = 1
        for k, v in contract_arc_raw.items():
            if isinstance(v, dict):
                item = dict(v)
                item.setdefault("section", k)
                item.setdefault("start_bar", cursor)
                item.setdefault("bars", 4)
                cursor = int(item["start_bar"]) + int(item["bars"])
                contract_arc.append(item)

    raw_secs: list[dict[str, Any]] = []
    if isinstance(sections, list) and sections:
        cursor = 1
        for item in sections:
            if isinstance(item, dict):
                nm = str(item.get("section") or item.get("name") or f"Section_{len(raw_secs)+1}")
                bars = max(1, int(item.get("bars", 4)))
                sb = int(item.get("start_bar", cursor))
                cursor = sb + bars
                raw_secs.append({"section": nm, "start_bar": sb, "bars": bars, **item})
    elif isinstance(sections, str) and sections.strip():
        # Format: "Intro:1-4,Couplet:5-12,Refrain:13-20,Outro:21-24"
        for tok in sections.split(","):
            tok = tok.strip()
            if not tok or ":" not in tok:
                continue
            nm, rng = tok.split(":", 1)
            if "-" in rng:
                a, b = rng.split("-", 1)
                sb, eb = int(a), int(b)
                raw_secs.append({"section": nm.strip(), "start_bar": sb, "bars": max(1, eb - sb + 1)})

    if not raw_secs and blueprint:
        bp_obj: dict[str, Any] | None = None
        if isinstance(blueprint, dict):
            bp_obj = blueprint
        elif isinstance(blueprint, str) and os.path.isfile(os.path.expanduser(blueprint)):
            with open(os.path.expanduser(blueprint), encoding="utf-8") as fh:
                bp_obj = json.load(fh)
        if bp_obj and isinstance(bp_obj.get("structure"), list):
            cursor = 1
            for item in bp_obj["structure"]:
                if isinstance(item, dict):
                    nm = str(item.get("section") or item.get("name") or f"Section_{len(raw_secs)+1}")
                    bars = max(1, int(item.get("bars", 4)))
                    sb = int(item.get("start_bar", cursor))
                    cursor = sb + bars
                    raw_secs.append({"section": nm, "start_bar": sb, "bars": bars, **item})

    if not raw_secs and contract_arc:
        raw_secs = [dict(x) for x in contract_arc]

    if not raw_secs:
        # Infer coarse 4-bar blocks as pseudo-sections so 5D arc still runs cleanly
        b = 1
        while b <= total_bars:
            blk_bars = min(4, total_bars - b + 1)
            raw_secs.append({"section": f"Block_{(b-1)//4 + 1}", "start_bar": b, "bars": blk_bars})
            b += blk_bars

    # Enrich with quotas and matched target 5D arc
    enriched: list[dict[str, Any]] = []
    for idx, sec in enumerate(raw_secs):
        sb = max(1, int(sec.get("start_bar", 1)))
        bars = max(1, int(sec.get("bars", 4)))
        if sb > total_bars:
            continue
        eb = min(total_bars, sb + bars - 1)
        nm = str(sec.get("section") or sec.get("name") or f"Section_{idx + 1}")
        target_arc = None
        for ca in contract_arc:
            if str(ca.get("section") or "").strip().lower() == nm.lower():
                target_arc = ca
                break
        if target_arc is None and idx < len(contract_arc):
            target_arc = contract_arc[idx]
        quota = _section_quota_from_name(nm, contract_arc)
        if sec.get("max_cognitive_streams"):
            quota = int(sec["max_cognitive_streams"])
        enriched.append({
            "section": nm,
            "start_bar": sb,
            "end_bar": eb,
            "bars": eb - sb + 1,
            "quota": int(quota),
            "target_arc": target_arc,
        })
    return enriched


def _load_downsample_for_mir(
    wav_path: str,
    preloaded_data: tuple[np.ndarray, int] | None = None,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, int]:
    """
    Load audio once (or reuse preloaded `(data, sr)`) and produce:
      - data_stereo (float64, N x C), orig_sr
      - mono_orig (float64, N)
      - mono_22k (float32), sr_22k (~22050 Hz for fast CQT/MFCC/onset)
    """
    if preloaded_data is not None:
        raw_d = np.asarray(preloaded_data[0], dtype=np.float32)
        if raw_d.ndim == 1:
            data = np.column_stack([raw_d, raw_d])
        else:
            data = raw_d
        sr = int(preloaded_data[1])
    else:
        data, sr = sf.read(os.path.abspath(os.path.expanduser(wav_path)), dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.hstack([data, data])
    mono = 0.5 * (data[:, 0] + data[:, 1])

    target_sr = 11025
    if sr > target_sr:
        if sr % target_sr == 0:
            step = sr // target_sr
            mono_22k = mono[::step].astype(np.float32)
            eff_sr = target_sr
        else:
            g = math.gcd(int(sr), target_sr)
            mono_22k = signal.resample_poly(mono, target_sr // g, int(sr) // g).astype(np.float32)
            eff_sr = target_sr
    else:
        mono_22k = mono.astype(np.float32)
        eff_sr = int(sr)
    return data, int(sr), mono, mono_22k, eff_sr


def compute_bar_synchronized_ssm(
    mono_22k: np.ndarray,
    sr_22k: int,
    mono_orig: np.ndarray,
    sr_orig: int,
    *,
    bpm: float = 132.0,
    beats_per_bar: int = 4,
) -> dict[str, Any]:
    """
    Compute bar-synchronized SSM S in R^(M x M):
      S_ij = 0.35 * cos(chroma) + 0.40 * cos(mfcc[1..13]) + 0.25 * cos(rhythm16)
    """
    bar_s = (60.0 / max(40.0, float(bpm))) * max(1, int(beats_per_bar))
    total_s = len(mono_orig) / float(sr_orig)
    n_bars = int(math.floor(total_s / bar_s + 1e-6))
    if n_bars < 1 and total_s >= 0.45 * bar_s:
        n_bars = 1
    if n_bars < 2:
        return {
            "n_bars": n_bars,
            "bar_s": round(bar_s, 5),
            "ssm": np.ones((max(1, n_bars), max(1, n_bars)), dtype=np.float64),
            "active_bars": np.array([True] * max(1, n_bars), dtype=bool),
            "bar_rms_db": [-120.0] * max(1, n_bars),
            "chroma": np.zeros((max(1, n_bars), 12), dtype=np.float64),
            "mfcc": np.zeros((max(1, n_bars), 13), dtype=np.float64),
            "rhythm": np.zeros((max(1, n_bars), 16), dtype=np.float64),
        }

    hop = 512
    chroma_hop = 1024 if n_bars >= 16 else 512
    # 12-dim Chroma CQT (bins_per_octave=12, 5 octaves C2..B6; hop=1024 on long tracks cuts CQT cost in half)
    chroma_frames = librosa.feature.chroma_cqt(
        y=mono_22k,
        sr=sr_22k,
        hop_length=chroma_hop,
        fmin=float(librosa.note_to_hz("C2")),
        n_octaves=5,
        bins_per_octave=12,
    )
    # Single STFT + Mel spectrogram shared by MFCC, onset_strength, and spectral_centroid
    n_fft_mfcc = min(2048, max(512, 1 << int(math.log2(max(512, sr_22k // 12)))))
    stft_mag = np.abs(librosa.stft(mono_22k, n_fft=n_fft_mfcc, hop_length=hop))
    frame_rms = np.sqrt(np.mean(stft_mag * stft_mag, axis=0))
    mel_pow = librosa.feature.melspectrogram(S=stft_mag * stft_mag, sr=sr_22k, n_fft=n_fft_mfcc, hop_length=hop)
    mel_db = librosa.power_to_db(mel_pow, ref=1.0)
    mfcc_all = librosa.feature.mfcc(S=mel_db, sr=sr_22k, n_mfcc=14)
    mfcc_frames = mfcc_all[1:14, :]
    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr_22k, hop_length=hop)

    n_frames = min(mfcc_frames.shape[1], len(onset_env))
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr_22k, hop_length=hop)
    if chroma_hop == hop:
        chroma_times = frame_times
        n_cframes = min(n_frames, chroma_frames.shape[1])
    else:
        n_cframes = chroma_frames.shape[1]
        chroma_times = librosa.frames_to_time(np.arange(n_cframes), sr=sr_22k, hop_length=chroma_hop)

    bar_chroma = np.zeros((n_bars, 12), dtype=np.float64)
    bar_mfcc = np.zeros((n_bars, 13), dtype=np.float64)
    bar_rhythm = np.zeros((n_bars, 16), dtype=np.float64)
    bar_rms_db = np.zeros(n_bars, dtype=np.float64)

    bar_samples_orig = bar_s * float(sr_orig)
    for m in range(n_bars):
        t0 = m * bar_s
        t1 = (m + 1) * bar_s
        s0 = int(round(m * bar_samples_orig))
        s1 = min(len(mono_orig), int(round((m + 1) * bar_samples_orig)))
        seg_orig = mono_orig[s0:s1]
        rms = math.sqrt(float(np.dot(seg_orig, seg_orig)) / len(seg_orig)) if len(seg_orig) else 0.0
        bar_rms_db[m] = 20.0 * math.log10(rms + 1e-12)

        c0 = int(np.searchsorted(chroma_times, t0, side="left"))
        c1 = int(np.searchsorted(chroma_times, t1, side="left"))
        if c1 <= c0:
            c1 = min(n_cframes, c0 + 1)
        if c0 < n_cframes:
            bar_chroma[m] = np.mean(chroma_frames[:, c0:c1], axis=1)

        f0 = int(np.searchsorted(frame_times, t0, side="left"))
        f1 = int(np.searchsorted(frame_times, t1, side="left"))
        if f1 <= f0:
            f1 = min(n_frames, f0 + 1)
        if f0 < n_frames:
            bar_mfcc[m] = np.mean(mfcc_frames[:, f0:f1], axis=1)
            # Bar-relative 16-step RMS: smooth, phase-insensitive, so identical bars stay identical
            # (narrow onset peaks sampled at fixed midpoints jitter with non-integer frame counts).
            rms_times = frame_times[:n_frames]
            rms_vals = frame_rms[:n_frames]
            for st_idx in range(16):
                a = t0 + st_idx * (bar_s / 16.0)
                b = a + (bar_s / 16.0)
                i0s = int(np.searchsorted(rms_times, a, side="left"))
                i1s = int(np.searchsorted(rms_times, b, side="left"))
                if i1s > i0s:
                    bar_rhythm[m, st_idx] = float(np.mean(rms_vals[i0s:i1s]))
                else:
                    bar_rhythm[m, st_idx] = float(np.interp((a + b) * 0.5, rms_times, rms_vals))

    active_bars = bar_rms_db > -45.0

    # Chroma cosine (0.35)
    c_norm, c_ok = _l2_normalize_rows(bar_chroma)
    s_chroma = c_norm @ c_norm.T

    # MFCC 1..13 centered normalized similarity (0.40):
    # Raw MFCCs are dominated by MFCC_1 spectral slope (+50..+180), which hides LPF / layer changes.
    # When active bars show timbral variation (ptp >= 2.5), center relative to track median;
    # if completely flat/cloned across active bars, similarity stays ~1.0 so clones are reliably flagged.
    if np.sum(active_bars) >= 2:
        act_mfcc = bar_mfcc[active_bars]
        max_ptp = float(np.max(np.ptp(act_mfcc, axis=0)))
        if max_ptp >= 2.5:
            med = np.median(act_mfcc, axis=0)
            scale = np.maximum(np.std(act_mfcc, axis=0), 2.5)
            mfcc_centered = (bar_mfcc - med) / scale
            m_norm, m_ok = _l2_normalize_rows(mfcc_centered)
            s_mfcc = m_norm @ m_norm.T
            # Handle rare bars right at the exact median vector (near zero centered norm)
            near_zero = active_bars & (~m_ok)
            if np.any(near_zero):
                raw_m, _ = _l2_normalize_rows(bar_mfcc)
                s_raw_m = raw_m @ raw_m.T
                for idx in np.where(near_zero)[0]:
                    s_mfcc[idx, :] = s_raw_m[idx, :]
                    s_mfcc[:, idx] = s_raw_m[:, idx]
        else:
            m_norm, _ = _l2_normalize_rows(bar_mfcc)
            s_mfcc = m_norm @ m_norm.T
    else:
        m_norm, _ = _l2_normalize_rows(bar_mfcc)
        s_mfcc = m_norm @ m_norm.T

    # Rhythm 16-step cosine * energy-balance guard (0.25) so added hi-hats/density alter similarity
    r_norm, _ = _l2_normalize_rows(bar_rhythm)
    r_energy = np.linalg.norm(bar_rhythm, axis=1) + 1e-8
    e_ratio = (2.0 * np.outer(r_energy, r_energy)) / (
        np.outer(r_energy ** 2, np.ones(n_bars)) + np.outer(np.ones(n_bars), r_energy ** 2) + 1e-12
    )
    s_rhythm = (r_norm @ r_norm.T) * e_ratio

    ssm = np.clip(0.35 * s_chroma + 0.40 * s_mfcc + 0.25 * s_rhythm, -1.0, 1.0)
    np.fill_diagonal(ssm, 1.0)

    return {
        "n_bars": n_bars,
        "bar_s": round(bar_s, 5),
        "ssm": ssm,
        "active_bars": active_bars,
        "bar_rms_db": [round(float(x), 2) for x in bar_rms_db],
        "onset_env": onset_env,
        "onset_frame_times": frame_times,
        "stft_mag": stft_mag,
        "stft_n_fft": n_fft_mfcc,
        "chroma": bar_chroma,
        "mfcc": bar_mfcc,
        "rhythm": bar_rhythm,
    }


def _detect_8bar_clones(
    ssm: np.ndarray,
    active_bars: np.ndarray,
    *,
    max_sim_threshold: float = 0.94,
) -> dict[str, Any]:
    """
    Evaluate S_bar8(b) = (1/8) * sum_{k=0..7} S[b+k, b+8+k].
    If > max_sim_threshold on active 8-bar pairs -> MUTATE_BLOCK_DENSITY_OR_LPF on (b+8)..(b+15).
    """
    n_bars = ssm.shape[0]
    if n_bars < 16:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "max_similarity": None,
            "cloned_blocks": [],
            "next_action": {"code": "KEEP", "params": {}},
        }

    pair_scores: list[dict[str, Any]] = []
    failing: list[dict[str, Any]] = []
    max_sim = -1.0

    for b in range(0, n_bars - 15):
        if np.sum(active_bars[b : b + 8]) < 5 or np.sum(active_bars[b + 8 : b + 16]) < 5:
            continue
        diag_vals = [float(ssm[b + k, b + 8 + k]) for k in range(8)]
        s8 = float(np.mean(diag_vals))
        max_sim = max(max_sim, s8)
        entry = {
            "source_bars": f"{b + 1}-{b + 8}",
            "target_bars": f"{b + 9}-{b + 16}",
            "start_bar_target": b + 9,
            "end_bar_target": b + 16,
            "similarity": round(s8, 4),
        }
        pair_scores.append(entry)
        if s8 > float(max_sim_threshold):
            failing.append(entry)

    if not pair_scores:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "max_similarity": None,
            "cloned_blocks": [],
            "next_action": {"code": "KEEP", "params": {}},
        }

    # Deduplicate overlapping windows by keeping disjoint peaks/worst starts >= 4 bars apart
    dedup_fail: list[dict[str, Any]] = []
    for item in sorted(failing, key=lambda x: (-x["similarity"], x["start_bar_target"])):
        sb = item["start_bar_target"]
        if not any(abs(sb - kept["start_bar_target"]) < 4 for kept in dedup_fail):
            dedup_fail.append(item)
    dedup_fail.sort(key=lambda x: x["start_bar_target"])

    if dedup_fail:
        worst = max(dedup_fail, key=lambda x: x["similarity"])
        # Score decays from ~0.65 at 0.941 down to 0.15 at 1.0
        excess = max(0.0, worst["similarity"] - float(max_sim_threshold))
        score = round(max(0.15, 0.72 - excess * 9.0), 3)
        return {
            "status": "FAIL",
            "verdict": "FAIL",
            "score": score,
            "max_similarity": round(worst["similarity"], 4),
            "threshold": float(max_sim_threshold),
            "cloned_blocks": dedup_fail,
            "next_action": {
                "code": "MUTATE_BLOCK_DENSITY_OR_LPF",
                "params": {
                    "bars": worst["target_bars"],
                    "start_bar": worst["start_bar_target"],
                    "end_bar": worst["end_bar_target"],
                    "similarity": worst["similarity"],
                },
            },
        }

    margin = float(max_sim_threshold) - max_sim
    score = round(min(1.0, 0.85 + max(0.0, margin) * 1.2), 3)
    return {
        "status": "PASS",
        "verdict": "PASS",
        "score": score,
        "max_similarity": round(max_sim, 4),
        "threshold": float(max_sim_threshold),
        "cloned_blocks": [],
        "next_action": {"code": "KEEP", "params": {}},
    }


def _checkerboard_novelty_curve(ssm: np.ndarray, kernel_half: int = 4) -> np.ndarray:
    """Compute Gaussian Checkerboard Novelty Curve N(m) across bar boundaries 0..M."""
    M = ssm.shape[0]
    L = max(2, min(int(kernel_half), M // 2))
    coords = np.arange(-L, L, dtype=np.float64) + 0.5
    sigma = max(1.0, 0.5 * L)
    gx, gy = np.meshgrid(coords, coords)
    gauss = np.exp(-(gx ** 2 + gy ** 2) / (2.0 * sigma ** 2))
    sign_mat = np.ones((2 * L, 2 * L), dtype=np.float64)
    sign_mat[:L, L:] = -1.0
    sign_mat[L:, :L] = -1.0
    kernel = gauss * sign_mat
    pos_sum = float(np.sum(kernel[kernel > 0]))
    if pos_sum > 0:
        kernel /= pos_sum

    # Symmetric pad SSM by reflecting border blocks cleanly so early/late boundaries stay valid
    padded = np.pad(ssm, pad_width=L, mode="edge")
    # Diagonal index m represents boundary before bar m (0-based bar start m in 0..M)
    nov = np.zeros(M, dtype=np.float64)
    for m in range(1, M):
        # Boundary between bar m-1 and bar m -> center at m+L in padded coordinates
        c = m + L
        patch = padded[c - L : c + L, c - L : c + L]
        nov[m] = max(0.0, float(np.sum(patch * kernel)))
    return nov


def _audit_boundary_novelty(
    ssm: np.ndarray,
    sections: list[dict[str, Any]],
    *,
    tolerance_bars: float = 1.0,
) -> dict[str, Any]:
    """Check whether declared section transitions exhibit local novelty peaks within ±tol bars."""
    M = ssm.shape[0]
    if M < 8 or len(sections) < 2:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "novelty_curve": [],
            "boundaries": [],
            "missing_boundaries": [],
            "next_action": {"code": "KEEP", "params": {}},
        }

    nov = _checkerboard_novelty_curve(ssm, kernel_half=min(4, max(2, M // 4)))
    tol = max(0, int(round(tolerance_bars)))
    max_nov = float(np.max(nov)) if len(nov) else 0.0
    # Contrast floor: novelty must meaningfully rise above negligible jitter (>= 0.06 and >= 18% of track peak)
    peak_floor = max(0.055, 0.18 * max_nov) if max_nov > 0 else 0.055

    # Local peaks in novelty curve (0-based boundary index m corresponds to start_bar = m + 1)
    peaks: list[int] = []
    for m in range(1, M):
        left = nov[m - 1]
        right = nov[m + 1] if m + 1 < M else 0.0
        if nov[m] >= left and nov[m] >= right and nov[m] >= peak_floor:
            peaks.append(m)

    boundaries: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for idx in range(1, len(sections)):
        prev_sec = sections[idx - 1]
        cur_sec = sections[idx]
        target_sb = int(cur_sec["start_bar"])  # 1-based
        b_idx = target_sb - 1  # 0-based index in nov
        if b_idx < 1 or b_idx >= M:
            continue

        lo = max(1, b_idx - tol)
        hi = min(M - 1, b_idx + tol)
        win_vals = nov[lo : hi + 1]
        win_peak_val = float(np.max(win_vals)) if len(win_vals) else 0.0
        win_peak_idx = lo + int(np.argmax(win_vals)) if len(win_vals) else b_idx

        matched_peaks = [p for p in peaks if abs(p - b_idx) <= tol]
        # Accept if matched peak OR window contains genuine sharp boundary value >= peak_floor & > neighborhood baseline
        nb_lo = max(1, b_idx - max(2, tol + 1))
        nb_hi = min(M - 1, b_idx + max(2, tol + 1))
        outer_vals = [float(nov[k]) for k in range(nb_lo, nb_hi + 1) if abs(k - win_peak_idx) > tol]
        contrast_ok = True
        if outer_vals and win_peak_val < 0.09:
            contrast_ok = win_peak_val >= 1.15 * float(np.median(outer_vals))

        aligned = (bool(matched_peaks) or (win_peak_val >= peak_floor and contrast_ok)) and (max_nov >= 0.055)
        b_entry = {
            "transition": f"{prev_sec['section']}->{cur_sec['section']}",
            "expected_bar": target_sb,
            "detected_peak_bar": win_peak_idx + 1 if aligned else None,
            "novelty_value": round(win_peak_val, 4),
            "aligned": bool(aligned),
        }
        boundaries.append(b_entry)
        if not aligned:
            missing.append(b_entry)

    if not boundaries:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "novelty_curve": [round(float(x), 4) for x in nov],
            "boundaries": [],
            "missing_boundaries": [],
            "next_action": {"code": "KEEP", "params": {}},
        }

    hit_ratio = (len(boundaries) - len(missing)) / float(len(boundaries))
    if missing:
        first_m = missing[0]
        return {
            "status": "WARN" if hit_ratio >= 0.5 else "FAIL",
            "verdict": "FIX",
            "score": round(max(0.25, 0.35 + 0.55 * hit_ratio), 3),
            "novelty_curve": [round(float(x), 4) for x in nov],
            "boundaries": boundaries,
            "missing_boundaries": missing,
            "next_action": {
                "code": "REINFORCE_SECTION_TRANSITION",
                "params": {
                    "transition": first_m["transition"],
                    "bar": first_m["expected_bar"],
                },
            },
        }

    return {
        "status": "PASS",
        "verdict": "PASS",
        "score": round(min(1.0, 0.85 + 0.15 * min(1.0, max_nov * 3.0)), 3),
        "novelty_curve": [round(float(x), 4) for x in nov],
        "boundaries": boundaries,
        "missing_boundaries": [],
        "next_action": {"code": "KEEP", "params": {}},
    }


def _extract_stems_per_bar_activity(
    stems_dir_or_map: str | dict[str, Any] | None,
    *,
    n_bars: int,
    bar_s: float,
    bpm: float,
    active_threshold_db: float = -28.0,
) -> dict[str, Any] | None:
    """Measure active track count (> -28 dBFS RMS per bar) across non-bus stems."""
    if not stems_dir_or_map:
        return None

    stem_entries: list[tuple[str, Any]] = []
    if isinstance(stems_dir_or_map, str):
        sdir = os.path.abspath(os.path.expanduser(stems_dir_or_map))
        if not os.path.isdir(sdir):
            return None
        for fn in sorted(os.listdir(sdir)):
            if not fn.lower().endswith((".wav", ".flac")):
                continue
            low = fn.lower()
            if any(kw in low for kw in _NON_INSTRUMENT_STEM_KW):
                continue
            stem_entries.append((os.path.splitext(fn)[0], os.path.join(sdir, fn)))
    elif isinstance(stems_dir_or_map, dict):
        for k, v in stems_dir_or_map.items():
            if any(kw in k.lower() for kw in _NON_INSTRUMENT_STEM_KW):
                continue
            stem_entries.append((str(k), v))

    if not stem_entries:
        return None

    per_bar_active_tracks: list[list[str]] = [[] for _ in range(n_bars)]
    per_bar_counts = [0] * n_bars

    for stem_name, src in stem_entries:
        if isinstance(src, str):
            if not os.path.isfile(src):
                continue
            data, s_sr = sf.read(src, dtype="float64", always_2d=True)
            mono = np.mean(data, axis=1)
        elif isinstance(src, (tuple, list)) and len(src) == 2 and isinstance(src[0], np.ndarray):
            arr, s_sr = src[0], int(src[1])
            mono = np.mean(arr, axis=1) if arr.ndim == 2 else arr
        else:
            continue

        bar_n = bar_s * float(s_sr)
        for m in range(n_bars):
            i0 = int(round(m * bar_n))
            i1 = min(len(mono), int(round((m + 1) * bar_n)))
            if i1 - i0 < 64:
                continue
            rms = float(np.sqrt(np.mean(mono[i0:i1] ** 2, dtype=np.float64)))
            rms_db = 20.0 * math.log10(rms + 1e-12)
            if rms_db > float(active_threshold_db):
                per_bar_active_tracks[m].append(stem_name)
                per_bar_counts[m] += 1

    return {
        "stem_names": [nm for nm, _ in stem_entries],
        "per_bar_counts": per_bar_counts,
        "per_bar_active_tracks": per_bar_active_tracks,
    }


def _audit_cognitive_ceiling(
    stems_activity: dict[str, Any] | None,
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    if not stems_activity or not stems_activity.get("stem_names"):
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "sections": [],
            "violations": [],
            "next_action": {"code": "KEEP", "params": {}},
        }

    counts = stems_activity["per_bar_counts"]
    sec_reports: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []

    for sec in sections:
        sb = int(sec["start_bar"])
        eb = min(len(counts), int(sec["end_bar"]))
        if sb > len(counts):
            continue
        quota = int(sec["quota"])
        consec_over = 0
        max_consec_over = 0
        over_bars: list[int] = []
        sec_counts: list[int] = []
        for b1 in range(sb, eb + 1):
            c = int(counts[b1 - 1])
            sec_counts.append(c)
            if c > quota:
                consec_over += 1
                max_consec_over = max(max_consec_over, consec_over)
                over_bars.append(b1)
            else:
                consec_over = 0

        obs_max = max(sec_counts) if sec_counts else 0
        exceeded = max_consec_over > 2
        entry = {
            "section": sec["section"],
            "bars": f"{sb}-{eb}",
            "quota": quota,
            "observed_max": obs_max,
            "consecutive_over": max_consec_over,
            "over_bars": over_bars,
            "passed": not exceeded,
        }
        sec_reports.append(entry)
        if exceeded:
            violations.append(entry)

    if violations:
        worst = max(violations, key=lambda x: (x["observed_max"] - x["quota"], x["consecutive_over"]))
        score = round(max(0.20, 0.72 - 0.12 * (worst["observed_max"] - worst["quota"])), 3)
        return {
            "status": "FAIL",
            "verdict": "FAIL",
            "score": score,
            "sections": sec_reports,
            "violations": violations,
            "next_action": {
                "code": "PRUNE_TRACK_IN_SECTION",
                "params": {
                    "section": worst["section"],
                    "bars": worst["bars"],
                    "quota": worst["quota"],
                    "observed_max": worst["observed_max"],
                },
            },
        }

    return {
        "status": "PASS",
        "verdict": "PASS",
        "score": 1.0,
        "sections": sec_reports,
        "violations": [],
        "next_action": {"code": "KEEP", "params": {}},
    }


_PRIMARY_AUDIBLE_ROLES = {"kick", "bass_sub", "snare_clap", "lead", "vocals"}
_SECONDARY_AUDIBLE_ROLES = {"chords_pad", "hihat_perc", "counter_melody"}


def _clean_stem_label(stem_name: str) -> str:
    raw = stem_name.split(" - ", 1)[1] if " - " in stem_name else stem_name
    cleaned = re.sub(r"^\d+\s+", "", raw.strip())
    return cleaned or raw.strip() or stem_name


def audit_section_track_audibility(
    stems_dir_or_map: str | dict[str, Any] | None,
    sections: list[dict[str, Any]] | None = None,
    *,
    bar_s: float | None = None,
    bpm: float = 132.0,
    beats_per_bar: int = 4,
    arrangement_plan: dict[str, Any] | str | None = None,
    blueprint: dict[str, Any] | str | None = None,
    project_dir: str = "",
    mix_data: tuple[np.ndarray, int] | None = None,
) -> dict[str, Any]:
    """
    Phase 8 Q2 — Per-section minimum audibility floor audit (`TRACK_UNDER_AUDIBLE_FLOOR`):
    Checks every role listed in `active_roles` for a section against the section mix LUFS:
      - Primary (`kick`, `bass_sub`, `snare_clap`, `lead`, `vocals`): >= section_mix_lufs - 13.0 dB
      - Secondary (`chords_pad`, `hihat_perc`, `counter_melody`): >= section_mix_lufs - 18.0 dB
      - FX / Ear candy (`ear_candy_fx`): >= section_mix_lufs - 22.0 dB
    Emits `ADJUST_TRACK_FADER` bounded to `[+1.5, +4.0] dB` on violations.
    """
    if not stems_dir_or_map:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "violations": [],
            "recommendations": [],
            "next_action": {"code": "KEEP", "params": {}},
        }

    # Resolve arrangement_plan or blueprint for active_roles declarations
    plan_obj: dict[str, Any] | None = None
    if isinstance(arrangement_plan, dict):
        plan_obj = arrangement_plan
    elif isinstance(arrangement_plan, str) and arrangement_plan.strip() and os.path.isfile(os.path.expanduser(arrangement_plan)):
        with open(os.path.expanduser(arrangement_plan), encoding="utf-8") as fh:
            plan_obj = json.load(fh)
    elif project_dir:
        cand_plan = os.path.join(os.path.abspath(os.path.expanduser(project_dir)), "arrangement_plan.json")
        if os.path.isfile(cand_plan):
            try:
                with open(cand_plan, encoding="utf-8") as fh:
                    plan_obj = json.load(fh)
            except Exception:
                plan_obj = None

    bp_obj: dict[str, Any] | None = None
    if isinstance(blueprint, dict):
        bp_obj = blueprint
    elif isinstance(blueprint, str) and blueprint.strip() and os.path.isfile(os.path.expanduser(blueprint)):
        with open(os.path.expanduser(blueprint), encoding="utf-8") as fh:
            bp_obj = json.load(fh)
    elif project_dir:
        cand_bp = os.path.join(os.path.abspath(os.path.expanduser(project_dir)), "song_blueprint.json")
        if os.path.isfile(cand_bp):
            try:
                with open(cand_bp, encoding="utf-8") as fh:
                    bp_obj = json.load(fh)
            except Exception:
                bp_obj = None

    eff_bpm = float(((plan_obj or {}).get("meta") or {}).get("bpm") or (bp_obj or {}).get("bpm") or bpm)
    eff_bpb = int((((plan_obj or {}).get("meta") or {}).get("time_signature") or {}).get("numerator") or ((bp_obj or {}).get("time_signature") or {}).get("numerator") or beats_per_bar)
    eff_bar_s = float(bar_s) if bar_s is not None else (float(eff_bpb) * 60.0 / max(20.0, eff_bpm))

    # Build timeline entries with active_roles (1-indexed start_bar/end_bar)
    plan_sections: list[dict[str, Any]] = []
    if plan_obj and isinstance(plan_obj.get("sections"), list):
        for s in plan_obj["sections"]:
            if isinstance(s, dict) and s.get("active_roles"):
                sb_1 = int(s.get("start_bar", 0)) + 1
                nb = max(1, int(s.get("bars", 4)))
                plan_sections.append({
                    "section": str(s.get("name") or "Section"),
                    "start_bar": sb_1,
                    "end_bar": sb_1 + nb - 1,
                    "active_roles": list(s["active_roles"]),
                })
    elif bp_obj and isinstance(bp_obj.get("structure"), list):
        cursor = 1
        for s in bp_obj["structure"]:
            if isinstance(s, dict):
                nb = max(1, int(s.get("bars", 4)))
                sb_1 = int(s.get("start_bar", cursor))
                cursor = sb_1 + nb
                ar = s.get("active_roles")
                if ar:
                    plan_sections.append({
                        "section": str(s.get("section") or s.get("name") or "Section"),
                        "start_bar": sb_1,
                        "end_bar": sb_1 + nb - 1,
                        "active_roles": list(ar),
                    })
    if not plan_sections and sections:
        for s in sections:
            if isinstance(s, dict) and s.get("active_roles"):
                sb = int(s.get("start_bar", 1))
                eb = int(s.get("end_bar", sb + int(s.get("bars", 4)) - 1))
                plan_sections.append({
                    "section": str(s.get("section") or s.get("name") or "Section"),
                    "start_bar": sb,
                    "end_bar": eb,
                    "active_roles": list(s["active_roles"]),
                })

    if not plan_sections:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "violations": [],
            "recommendations": [],
            "next_action": {"code": "KEEP", "params": {}},
        }

    # Load non-bus stems
    loaded_stems: list[dict[str, Any]] = []
    master_loaded: tuple[np.ndarray, int] | None = mix_data
    if isinstance(stems_dir_or_map, str):
        sdir = os.path.abspath(os.path.expanduser(stems_dir_or_map))
        if not os.path.isdir(sdir):
            return {"status": "SKIP", "verdict": "PASS", "score": 1.0, "violations": [], "recommendations": [], "next_action": {"code": "KEEP", "params": {}}}
        for fn in sorted(os.listdir(sdir)):
            if not fn.lower().endswith((".wav", ".flac")):
                continue
            fpath = os.path.join(sdir, fn)
            low = fn.lower()
            if " - " not in fn or "master" in low:
                if master_loaded is None and ("master" in low or " - " not in fn):
                    try:
                        d_m, sr_m = sf.read(fpath, dtype="float64", always_2d=True)
                        master_loaded = (d_m, sr_m)
                    except Exception:
                        pass
                continue
            if any(kw in low for kw in _NON_INSTRUMENT_STEM_KW) or any(fxk in low for fxk in ("fx delay", "fx hall", "fx_room")):
                continue
            try:
                d_s, sr_s = sf.read(fpath, dtype="float64", always_2d=True)
                cname = _clean_stem_label(os.path.splitext(fn)[0])
                loaded_stems.append({
                    "name": cname,
                    "raw_key": os.path.splitext(fn)[0],
                    "role": normalize_role_for_gain_staging(cname),
                    "data": d_s,
                    "sr": sr_s,
                })
            except Exception:
                continue
    elif isinstance(stems_dir_or_map, dict):
        for k, v in stems_dir_or_map.items():
            low = str(k).lower()
            if any(kw in low for kw in _NON_INSTRUMENT_STEM_KW):
                continue
            arr: np.ndarray | None = None
            s_sr = 44100
            if isinstance(v, str) and os.path.isfile(v):
                arr, s_sr = sf.read(v, dtype="float64", always_2d=True)
            elif isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], np.ndarray):
                arr = np.asarray(v[0], dtype=np.float64)
                if arr.ndim == 1:
                    arr = np.column_stack([arr, arr])
                s_sr = int(v[1])
            if arr is not None:
                cname = _clean_stem_label(str(k))
                loaded_stems.append({
                    "name": cname,
                    "raw_key": str(k),
                    "role": normalize_role_for_gain_staging(cname),
                    "data": arr,
                    "sr": s_sr,
                })

    if not loaded_stems:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "violations": [],
            "recommendations": [],
            "next_action": {"code": "KEEP", "params": {}},
        }

    violations: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    section_reports: list[dict[str, Any]] = []

    for sec in plan_sections:
        sb = int(sec["start_bar"])
        eb = int(sec["end_bar"])
        t0 = max(0.0, float(sb - 1) * eff_bar_s)
        t1 = float(eb) * eff_bar_s

        # Determine section mix active LUFS
        if master_loaded is not None:
            m_arr, m_sr = master_loaded
            i0 = int(round(t0 * m_sr))
            i1 = min(len(m_arr), int(round(t1 * m_sr)))
            if i1 - i0 < 64:
                continue
            sec_mix_lufs = float(measure_active_loudness(m_arr[i0:i1], m_sr)["active_lufs"])
        else:
            ref_sr = int(loaded_stems[0]["sr"])
            i0 = int(round(t0 * ref_sr))
            max_i1 = max(min(len(st["data"]), int(round(t1 * st["sr"]))) for st in loaded_stems)
            if max_i1 - i0 < 64:
                continue
            mix_sum = np.zeros((max_i1 - i0, 2), dtype=np.float64)
            for st in loaded_stems:
                arr_s = st["data"]
                s_i0 = int(round(t0 * st["sr"]))
                s_i1 = min(len(arr_s), int(round(t1 * st["sr"])))
                if s_i1 > s_i0:
                    n_c = min(len(mix_sum), s_i1 - s_i0)
                    seg = arr_s[s_i0 : s_i0 + n_c]
                    if seg.shape[1] == 1:
                        seg = np.repeat(seg, 2, axis=1)
                    mix_sum[:n_c] += seg[:, :2]
            sec_mix_lufs = float(measure_active_loudness(mix_sum, ref_sr)["active_lufs"])

        sec_role_results: list[dict[str, Any]] = []
        for raw_role in sec["active_roles"]:
            rl_str = str(raw_role).strip().lower()
            canon_role = normalize_role_for_gain_staging(rl_str)
            if canon_role in _PRIMARY_AUDIBLE_ROLES:
                margin_db = 13.0
            elif canon_role in _SECONDARY_AUDIBLE_ROLES:
                margin_db = 18.0
            else:
                margin_db = 22.0
            min_allowed = sec_mix_lufs - margin_db

            matches = [
                st for st in loaded_stems
                if st["role"] == canon_role or rl_str in st["name"].lower() or rl_str in st["raw_key"].lower()
            ]
            if not matches:
                continue

            for st in matches:
                s_arr, s_sr = st["data"], int(st["sr"])
                s_i0 = int(round(t0 * s_sr))
                s_i1 = min(len(s_arr), int(round(t1 * s_sr)))
                if s_i1 - s_i0 < 64:
                    continue
                act = measure_active_loudness(s_arr[s_i0:s_i1], s_sr)
                tr_lufs = float(act["active_lufs"])
                ok = tr_lufs >= min_allowed
                sec_role_results.append({
                    "track": st["name"],
                    "role": canon_role,
                    "active_lufs": tr_lufs,
                    "min_allowed_lufs": round(min_allowed, 2),
                    "passed": ok,
                })
                if not ok:
                    deficit = min_allowed - tr_lufs
                    delta_db = round(min(4.0, max(1.5, deficit)), 2)
                    viol = {
                        "code": "TRACK_UNDER_AUDIBLE_FLOOR",
                        "action": "ADJUST_TRACK_FADER",
                        "section": sec["section"],
                        "bars": f"{sb}-{eb}",
                        "target_role": canon_role,
                        "target_track": st["name"],
                        "track_active_lufs": round(tr_lufs, 2),
                        "section_mix_lufs": round(sec_mix_lufs, 2),
                        "min_allowed_lufs": round(min_allowed, 2),
                        "deficit_db": round(deficit, 2),
                        "delta_db": delta_db,
                        "reason": (
                            f"TRACK_UNDER_AUDIBLE_FLOOR: '{st['name']}' ({canon_role}) à {tr_lufs:.1f} LUFS "
                            f"dans '{sec['section']}' (< {min_allowed:.1f} LUFS) -> ADJUST_TRACK_FADER +{delta_db:.1f} dB."
                        ),
                    }
                    violations.append(viol)
                    recommendations.append({
                        "code": "ADJUST_TRACK_FADER",
                        "params": {"track": st["name"], "role": canon_role, "delta_db": delta_db, "section": sec["section"]},
                    })

        section_reports.append({
            "section": sec["section"],
            "bars": f"{sb}-{eb}",
            "section_mix_lufs": round(sec_mix_lufs, 2),
            "tracks": sec_role_results,
        })

    if violations:
        worst = max(violations, key=lambda v: v["deficit_db"])
        score = round(max(0.25, 0.85 - 0.04 * worst["deficit_db"]), 3)
        return {
            "status": "FIX",
            "verdict": "FIX",
            "score": score,
            "sections": section_reports,
            "violations": violations,
            "recommendations": recommendations,
            "next_action": {
                "code": "ADJUST_TRACK_FADER",
                "params": {
                    "track": worst["target_track"],
                    "role": worst["target_role"],
                    "delta_db": worst["delta_db"],
                    "section": worst["section"],
                },
            },
        }

    return {
        "status": "PASS",
        "verdict": "PASS",
        "score": 1.0,
        "sections": section_reports,
        "violations": [],
        "recommendations": [],
        "next_action": {"code": "KEEP", "params": {}},
    }


def _audit_5d_dramatic_arc(
    data_stereo: np.ndarray,
    sr_orig: int,
    mono_22k: np.ndarray,
    sr_22k: int,
    sections: list[dict[str, Any]],
    *,
    bar_s: float,
    n_bars: int,
    has_drums: bool,
    stems_activity: dict[str, Any] | None,
    precomputed_onset_env: np.ndarray | None = None,
    precomputed_frame_times: np.ndarray | None = None,
    precomputed_stft_mag: np.ndarray | None = None,
    precomputed_n_fft: int = 2048,
) -> dict[str, Any]:
    """
    Compute normalized 5D Dramatic Arc A(b) = (LUFS_rel, C_spec, D_onset, W_MS, K_cog):
    validates multi-dimensional contrast & alignment with `sections_energy_arc`.
    """
    if n_bars < 1 or not sections:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "blocks": [],
            "arc_contrast_span": 0.0,
            "next_action": {"code": "KEEP", "params": {}},
        }

    hop = 512
    # Precompute fast frame-wise spectral centroid & onset peaks on downsampled mono once (<5 ms peak pick)
    if precomputed_stft_mag is not None:
        cent_frames = librosa.feature.spectral_centroid(S=precomputed_stft_mag, sr=sr_22k, n_fft=precomputed_n_fft, hop_length=hop)[0]
    else:
        cent_frames = librosa.feature.spectral_centroid(y=mono_22k, sr=sr_22k, hop_length=hop)[0]
    o_env = precomputed_onset_env if precomputed_onset_env is not None else librosa.onset.onset_strength(y=mono_22k, sr=sr_22k, hop_length=hop)
    frame_times = precomputed_frame_times if precomputed_frame_times is not None else librosa.frames_to_time(np.arange(len(cent_frames)), sr=sr_22k, hop_length=hop)
    o_max = float(np.max(o_env)) if len(o_env) else 0.0
    if o_max > 1e-6:
        o_pks, _ = signal.find_peaks(o_env, height=max(0.5, 0.25 * o_max), distance=max(2, int(round(0.08 * sr_22k / hop))))
        onset_frames = frame_times[np.minimum(o_pks, len(frame_times) - 1)]
    else:
        onset_frames = np.zeros(0, dtype=np.float64)
    # Fast vectorized K-weighting proxy on 22k mono (<15 ms even on 3-min tracks)
    sos_kw = signal.butter(2, min(110.0, 0.35 * sr_22k), btype="highpass", fs=sr_22k, output="sos")
    mono_kw = signal.sosfilt(sos_kw, mono_22k)

    # Group into evaluation windows: contract sections
    eval_windows: list[dict[str, Any]] = []
    for sec in sections:
        sb = int(sec["start_bar"])
        eb = min(n_bars, int(sec["end_bar"]))
        if sb > eb:
            continue
        eval_windows.append(sec)

    raw_rows: list[dict[str, Any]] = []
    for win in eval_windows:
        sb, eb = int(win["start_bar"]), int(win["end_bar"])
        t0 = (sb - 1) * bar_s
        t1 = eb * bar_s
        i0 = int(round(t0 * sr_orig))
        i1 = min(len(data_stereo), int(round(t1 * sr_orig)))
        seg = data_stereo[i0:i1]
        if len(seg) < 64:
            continue

        k0 = int(round(t0 * sr_22k))
        k1 = min(len(mono_kw), int(round(t1 * sr_22k)))
        kw_seg = mono_kw[k0:k1]
        kw_rms = float(np.sqrt(np.mean(kw_seg ** 2))) + 1e-12 if len(kw_seg) else 1e-12
        lufs_val = 20.0 * math.log10(kw_rms) - 0.691

        # Mid/Side stereo width ratio W_MS
        mid = 0.5 * (seg[:, 0] + seg[:, 1])
        side = 0.5 * (seg[:, 0] - seg[:, 1])
        mid_rms = float(np.sqrt(np.mean(mid ** 2))) + 1e-12
        side_rms = float(np.sqrt(np.mean(side ** 2)))
        w_ms = _clamp(side_rms / mid_rms, 0.0, 2.0)

        # Spectral centroid in Hz (weighted by frame mask inside [t0, t1])
        f0 = int(np.searchsorted(frame_times, t0, side="left"))
        f1 = int(np.searchsorted(frame_times, t1, side="left"))
        if f1 > f0:
            c_hz = float(np.median(cent_frames[f0:f1]))
        else:
            c_hz = 1000.0
        # Map 180 Hz -> 5500 Hz in log2 space cleanly into [0, 1]
        c_norm = _clamp((math.log2(max(90.0, c_hz)) - math.log2(180.0)) / (math.log2(5500.0) - math.log2(180.0)), 0.0, 1.0)

        # Onset density (events / s)
        n_onsets = int(np.sum((onset_frames >= t0) & (onset_frames < t1)))
        dur_w = max(0.25, t1 - t0)
        onset_per_s = n_onsets / dur_w

        # Cognitive stream proxy K_cog
        if stems_activity and stems_activity.get("per_bar_counts"):
            counts = stems_activity["per_bar_counts"][sb - 1 : eb]
            k_cog = float(np.mean(counts)) if counts else 1.0
        else:
            # Fast multi-band spectral occupancy on 22k mono slice
            sl_22 = mono_22k[k0:k1]
            nperseg = min(2048, max(256, len(sl_22)))
            freqs, psd = signal.welch(sl_22, fs=sr_22k, nperseg=nperseg, scaling="spectrum")
            tot = float(np.sum(psd)) + 1e-18
            bands = [(20, 120), (120, 500), (500, 3500), (3500, min(10500, 0.48 * sr_22k))]
            active_b = 0
            for lo_f, hi_f in bands:
                share = float(np.sum(psd[(freqs >= lo_f) & (freqs < hi_f)])) / tot
                if share > 0.04:
                    active_b += 1
            k_cog = float(max(1, min(8, active_b + (1 if onset_per_s > 2.0 else 0) + (1 if w_ms > 0.25 else 0))))

        raw_rows.append({
            "section": win["section"],
            "bars": f"{sb}-{eb}",
            "lufs": lufs_val,
            "centroid_hz": round(c_hz, 1),
            "centroid_norm": round(c_norm, 3),
            "onset_per_s": round(onset_per_s, 3),
            "stereo_width": round(w_ms, 3),
            "cognitive_streams": round(k_cog, 2),
            "target_arc": win.get("target_arc"),
        })

    if not raw_rows:
        return {
            "status": "SKIP",
            "verdict": "PASS",
            "score": 1.0,
            "blocks": [],
            "arc_contrast_span": 0.0,
            "next_action": {"code": "KEEP", "params": {}},
        }

    # Relative LUFS and onset normalization across sections
    lufs_vals = [r["lufs"] for r in raw_rows if r["lufs"] > -55.0]
    min_lufs = min(lufs_vals) if lufs_vals else -24.0
    max_lufs = max(lufs_vals) if lufs_vals else -12.0
    lufs_span = max(0.0, max_lufs - min_lufs)

    onset_max_ref = max(max(r["onset_per_s"] for r in raw_rows), 3.0 if not has_drums else 5.0)
    cent_vals = [r["centroid_norm"] for r in raw_rows]
    width_vals = [r["stereo_width"] for r in raw_rows]
    onset_vals = [_clamp(r["onset_per_s"] / onset_max_ref, 0.0, 1.0) for r in raw_rows]

    cent_span = max(cent_vals) - min(cent_vals) if len(cent_vals) > 1 else 0.0
    width_span = max(width_vals) - min(width_vals) if len(width_vals) > 1 else 0.0
    onset_span = max(onset_vals) - min(onset_vals) if len(onset_vals) > 1 else 0.0

    # Weighted normalized vectors vs contract target_arc (if provided)
    dist_errors: list[float] = []
    out_blocks: list[dict[str, Any]] = []
    for idx, r in enumerate(raw_rows):
        if lufs_span >= 1.5:
            lufs_rel = _clamp(0.25 + 0.70 * ((r["lufs"] - min_lufs) / max(1e-6, lufs_span)), 0.05, 1.0)
        else:
            lufs_rel = 0.50 if r["lufs"] > -45.0 else 0.10
        d_norm = onset_vals[idx]
        meas_vec = [
            round(lufs_rel, 3),
            r["centroid_norm"],
            round(d_norm, 3),
            r["stereo_width"],
            round(_clamp(r["cognitive_streams"] / 8.0, 0.0, 1.0), 3),
        ]
        dist_to_target: float | None = None
        targ = r.get("target_arc")
        if isinstance(targ, dict):
            t_vec = [
                float(targ.get("lufs_rel", lufs_rel)),
                float(targ.get("centroid_norm", r["centroid_norm"])),
                float(targ.get("onset_density_norm", d_norm)),
                min(1.0, float(targ.get("stereo_width", r["stereo_width"]))),
                _clamp(float(targ.get("max_cognitive_streams", r["cognitive_streams"])) / 8.0, 0.0, 1.0),
            ]
            # For drumless profiles, down-weight raw LUFS pumping in distance and emphasize spectral/spatial/density arc
            w = np.array([0.12, 0.30, 0.25, 0.20, 0.13] if not has_drums else [0.28, 0.24, 0.22, 0.14, 0.12])
            diff = np.array(meas_vec) - np.array([t_vec[0], t_vec[1], t_vec[2], min(1.0, t_vec[3]), t_vec[4]])
            dist_to_target = round(float(np.sqrt(np.sum(w * (diff ** 2)))), 4)
            dist_errors.append(dist_to_target)

        out_blocks.append({
            "section": r["section"],
            "bars": r["bars"],
            "lufs": round(r["lufs"], 2),
            "lufs_rel": round(lufs_rel, 3),
            "centroid_hz": r["centroid_hz"],
            "centroid_norm": r["centroid_norm"],
            "onset_density_norm": round(d_norm, 3),
            "stereo_width": r["stereo_width"],
            "cognitive_streams": r["cognitive_streams"],
            "vector_5d": meas_vec,
            "target_distance": dist_to_target,
        })

    # Composite multi-dimensional dramatic contrast check:
    # A song has healthy dramatic arc if EITHER loudness evolves (>= 2.5 dB span) OR spectral+width+density evolve
    dim_contrast = max(
        _clamp(lufs_span / 6.0, 0.0, 1.0),
        0.45 * _clamp(cent_span / 0.20, 0.0, 1.0)
        + 0.30 * _clamp(width_span / 0.15, 0.0, 1.0)
        + 0.25 * _clamp(onset_span / 0.25, 0.0, 1.0),
    )

    mean_target_dist = round(float(np.mean(dist_errors)), 4) if dist_errors else None
    if len(out_blocks) == 1:
        score = 0.88
        status = "PASS"
        verdict = "PASS"
        act = {"code": "KEEP", "params": {}}
    else:
        contrast_score = _clamp(0.45 + 0.55 * min(1.0, dim_contrast / 0.45), 0.30, 1.0)
        align_score = 1.0 if mean_target_dist is None else _clamp(1.0 - max(0.0, mean_target_dist - 0.18) * 1.6, 0.35, 1.0)
        score = round(0.55 * contrast_score + 0.45 * align_score if mean_target_dist is not None else contrast_score, 3)

        flat_monotone = (len(out_blocks) >= 3) and (dim_contrast < 0.18) and (lufs_span < 1.2)
        if flat_monotone or (mean_target_dist is not None and mean_target_dist > 0.48):
            status = "WARN" if not flat_monotone else "FAIL"
            verdict = "FIX"
            act = {
                "code": "MUTATE_BLOCK_DENSITY_OR_LPF" if flat_monotone else "REINFORCE_SECTION_TRANSITION",
                "params": {"dim_contrast": round(dim_contrast, 3), "mean_target_dist": mean_target_dist},
            }
        else:
            status = "PASS"
            verdict = "PASS"
            act = {"code": "KEEP", "params": {}}

    return {
        "status": status,
        "verdict": verdict,
        "score": score,
        "has_drums": has_drums,
        "lufs_span_db": round(lufs_span, 2),
        "centroid_span_norm": round(cent_span, 3),
        "width_span": round(width_span, 3),
        "onset_span_norm": round(onset_span, 3),
        "arc_contrast_span": round(dim_contrast, 3),
        "mean_target_distance": mean_target_dist,
        "blocks": out_blocks,
        "next_action": act,
    }


def analyze_macro_structure(
    wav_path: str,
    *,
    bpm: float = 132.0,
    beats_per_bar: int = 4,
    sections: str | list[dict[str, Any]] | None = None,
    blueprint: str | dict[str, Any] | None = None,
    contract: str | dict[str, Any] | None = None,
    stems_dir_or_map: str | dict[str, Any] | None = None,
    project_dir: str = "",
    has_drums: bool | None = None,
    save_diagnostics: bool = True,
    preloaded_data: tuple[np.ndarray, int] | None = None,
) -> dict[str, Any]:
    """
    Run full S4 macro-structure audit (SSM + 8-bar clone check, Checkerboard Novelty,
    5D Dramatic Arc, and per-bar Cognitive Ceiling on stems) and save full report to disk.
    """
    t0 = time.perf_counter()
    abs_wav = os.path.abspath(os.path.expanduser(wav_path))
    if preloaded_data is None and not os.path.isfile(abs_wav):
        raise FileNotFoundError(f"Audio file not found for macro structure audit: {abs_wav}")

    eff_proj = os.path.abspath(os.path.expanduser(project_dir)) if project_dir.strip() else os.path.dirname(os.path.dirname(abs_wav))
    contract_obj: dict[str, Any] | None = None
    if isinstance(contract, dict):
        contract_obj = contract
    elif isinstance(contract, str) and contract.strip():
        contract_obj = load_multi_scale_contract(contract_path=contract, project_dir=eff_proj)
    else:
        contract_obj = load_multi_scale_contract(project_dir=eff_proj)

    eff_has_drums = bool(has_drums) if has_drums is not None else bool((contract_obj or {}).get("has_drums", True))
    s4_cfg = (contract_obj or {}).get("s4_form") or {}
    max_8bar_sim = float(s4_cfg.get("max_8bar_ssm_similarity", 0.94))
    tol_bars = float(s4_cfg.get("boundary_alignment_tolerance_bars", 1.0))

    data_st, sr_orig, mono_orig, mono_22k, sr_22k = _load_downsample_for_mir(abs_wav, preloaded_data=preloaded_data)
    ssm_data = compute_bar_synchronized_ssm(
        mono_22k,
        sr_22k,
        mono_orig,
        sr_orig,
        bpm=float(bpm),
        beats_per_bar=int(beats_per_bar),
    )
    n_bars = int(ssm_data["n_bars"])
    bar_s = float(ssm_data["bar_s"])
    ssm = ssm_data["ssm"]
    active_bars = ssm_data["active_bars"]

    parsed_sections = _parse_sections_spec(sections, blueprint, contract_obj, n_bars)
    clones_res = _detect_8bar_clones(ssm, active_bars, max_sim_threshold=max_8bar_sim)
    novelty_res = _audit_boundary_novelty(ssm, parsed_sections, tolerance_bars=tol_bars)
    stems_act = _extract_stems_per_bar_activity(
        stems_dir_or_map, n_bars=n_bars, bar_s=bar_s, bpm=float(bpm), active_threshold_db=-28.0
    )
    cog_res = _audit_cognitive_ceiling(stems_act, parsed_sections)
    aud_res = audit_section_track_audibility(
        stems_dir_or_map,
        parsed_sections,
        bar_s=bar_s,
        bpm=float(bpm),
        beats_per_bar=int(beats_per_bar),
        blueprint=blueprint,
        project_dir=eff_proj,
        mix_data=(data_st, sr_orig),
    )
    arc_res = _audit_5d_dramatic_arc(
        data_st,
        sr_orig,
        mono_22k,
        sr_22k,
        parsed_sections,
        bar_s=bar_s,
        n_bars=n_bars,
        has_drums=eff_has_drums,
        stems_activity=stems_act,
        precomputed_onset_env=ssm_data.get("onset_env"),
        precomputed_frame_times=ssm_data.get("onset_frame_times"),
        precomputed_stft_mag=ssm_data.get("stft_mag"),
        precomputed_n_fft=int(ssm_data.get("stft_n_fft", 2048)),
    )

    deviations: list[dict[str, Any]] = []
    if clones_res["verdict"] != "PASS":
        act = clones_res["next_action"]
        deviations.append({
            "scale": "S4_form",
            "axis": "H",
            "location": f"bars {act['params'].get('bars', '')}".strip(),
            "metric": f"S_bar8={clones_res['max_similarity']} (> {max_8bar_sim})",
            "diagnosis": "Cloned 8-bar block detected in SSM",
            "next_action": f"{act['code']}({act['params'].get('bars', '')})",
            "action_payload": act,
            "blocking": True,
            "severity": 0.90,
        })
    if novelty_res["verdict"] != "PASS":
        act = novelty_res["next_action"]
        miss = novelty_res.get("missing_boundaries") or []
        loc = miss[0]["transition"] if miss else "section_boundary"
        deviations.append({
            "scale": "S4_form",
            "axis": "H",
            "location": loc,
            "metric": f"novelty_missing={len(miss)} boundary",
            "diagnosis": f"Section transition '{loc}' lacks checkerboard novelty peak (±{int(tol_bars)} bar)",
            "next_action": act["code"],
            "action_payload": act,
            "blocking": len(miss) > 1,
            "severity": 0.72,
        })
    if cog_res["verdict"] != "PASS":
        act = cog_res["next_action"]
        p = act["params"]
        deviations.append({
            "scale": "S4_form",
            "axis": "V",
            "location": f"{p.get('section', 'section')} ({p.get('bars', '')})",
            "metric": f"streams={p.get('observed_max')}/{p.get('quota')} (>2 bars)",
            "diagnosis": "Cognitive ceiling exceeded for >2 consecutive bars",
            "next_action": f"{act['code']}({p.get('section', '')})",
            "action_payload": act,
            "blocking": True,
            "severity": 0.82,
        })
    if aud_res["verdict"] != "PASS":
        act = aud_res["next_action"]
        p = act["params"]
        v0 = aud_res["violations"][0]
        deviations.append({
            "scale": "S4_form",
            "axis": "V",
            "location": f"{v0.get('section', 'section')} ({v0.get('bars', '')})",
            "metric": f"TRACK_UNDER_AUDIBLE_FLOOR={v0.get('track_active_lufs')}LUFS (<{v0.get('min_allowed_lufs')})",
            "diagnosis": v0.get("reason", "TRACK_UNDER_AUDIBLE_FLOOR"),
            "next_action": f"ADJUST_TRACK_FADER(track={p.get('track', '')},delta_db={p.get('delta_db', 2.0)})",
            "action_payload": act,
            "blocking": False,
            "severity": 0.78,
        })
    if arc_res["verdict"] != "PASS":
        act = arc_res["next_action"]
        deviations.append({
            "scale": "S4_form",
            "axis": "HV",
            "location": "sections_energy_arc",
            "metric": f"arc_span={arc_res['arc_contrast_span']}, dist={arc_res['mean_target_distance']}",
            "diagnosis": "Flat or unaligned 5D dramatic arc across sections",
            "next_action": act["code"],
            "action_payload": act,
            "blocking": arc_res["status"] == "FAIL",
            "severity": 0.65,
        })

    # Weighted S4 score (only active/non-SKIP components dominate, blocking failure caps score)
    sub_scores = [
        x["score"]
        for x in (clones_res, novelty_res, arc_res, cog_res, aud_res)
        if x.get("status") != "SKIP"
    ]
    overall_score = round(float(np.mean(sub_scores)), 3) if sub_scores else 1.0
    if any(d["blocking"] for d in deviations):
        overall_score = round(min(overall_score, min(x["score"] for x in (clones_res, cog_res, novelty_res, arc_res, aud_res) if x["verdict"] != "PASS")), 3)

    verdict = "FAIL" if any(d["blocking"] for d in deviations) else ("FIX" if deviations else "PASS")
    elapsed_s = round(time.perf_counter() - t0, 3)

    diag_path = ""
    if save_diagnostics and eff_proj:
        diag_dir = os.path.join(eff_proj, "renders", "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        diag_path = os.path.join(diag_dir, "s4_macro_structure.json")
        full_dump = {
            "wav_path": abs_wav,
            "bpm": bpm,
            "beats_per_bar": beats_per_bar,
            "n_bars": n_bars,
            "elapsed_s": elapsed_s,
            "verdict": verdict,
            "score": overall_score,
            "clones_8bar": clones_res,
            "novelty": novelty_res,
            "arc_5d": arc_res,
            "cognitive_ceiling": cog_res,
            "section_track_audibility": aud_res,
            "bar_rms_db": ssm_data["bar_rms_db"],
            "ssm_matrix": np.round(ssm, 4).tolist(),
        }
        with open(diag_path, "w", encoding="utf-8") as fh:
            json.dump(full_dump, fh, indent=2, ensure_ascii=False)

    return {
        "verdict": verdict,
        "score": overall_score,
        "n_bars": n_bars,
        "elapsed_s": elapsed_s,
        "clones_8bar": {k: v for k, v in clones_res.items()},
        "novelty": {
            "status": novelty_res["status"],
            "verdict": novelty_res["verdict"],
            "score": novelty_res["score"],
            "boundaries": novelty_res["boundaries"],
            "missing_boundaries": novelty_res["missing_boundaries"],
            "next_action": novelty_res["next_action"],
        },
        "arc_5d": {
            "status": arc_res["status"],
            "verdict": arc_res["verdict"],
            "score": arc_res["score"],
            "lufs_span_db": arc_res.get("lufs_span_db"),
            "centroid_span_norm": arc_res.get("centroid_span_norm"),
            "width_span": arc_res.get("width_span"),
            "onset_span_norm": arc_res.get("onset_span_norm"),
            "arc_contrast_span": arc_res.get("arc_contrast_span"),
            "mean_target_distance": arc_res.get("mean_target_distance"),
            "next_action": arc_res["next_action"],
        },
        "cognitive_ceiling": cog_res,
        "section_track_audibility": aud_res,
        "deviations": deviations,
        "full_report_path": diag_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="S4 Macro-Structure Audit (SSM, Checkerboard Novelty, 5D Arc, Cognitive Ceiling).")
    ap.add_argument("wav_path", help="Audio render path (.wav/.flac/.mp3)")
    ap.add_argument("--bpm", type=float, default=132.0)
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--sections", default="", help="e.g. Intro:1-8,Drop:9-24,Outro:25-32")
    ap.add_argument("--blueprint", default="", help="Path to song_blueprint.json")
    ap.add_argument("--contract", default="", help="Path to multi_scale_contract.json")
    ap.add_argument("--stems-dir", default="", help="Directory of separated track stems")
    ap.add_argument("--project-dir", default="")
    ap.add_argument("--no-drums", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    res = analyze_macro_structure(
        args.wav_path,
        bpm=args.bpm,
        beats_per_bar=args.beats_per_bar,
        sections=args.sections or None,
        blueprint=args.blueprint or None,
        contract=args.contract or None,
        stems_dir_or_map=args.stems_dir or None,
        project_dir=args.project_dir,
        has_drums=False if args.no_drums else None,
        save_diagnostics=not args.no_save,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
