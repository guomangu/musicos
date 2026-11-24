#!/usr/bin/env python3
"""
Unified Multi-Scale H x V Audit & One-Shot Autonomous Pipeline
(`run_autonomous_pipeline.py`, Phase 6 §4.1 & §4.4).

Exposes:
  - `audit_multiscale_hv(project_dir, stage="ALL", contract_path="", ...)`:
    Unified H x V analyzer shared by both MCP (`audio-analyzer:audit_multiscale_hv`)
    and CLI (`--stage E1|E2|E3|E4|ALL --auto-fix --max-autofix-passes 2 --json-compact`).
  - Strictly enforces CPU budget (< 4 s on 2-min pre-rendered audio + stems + MIDI)
    and LLM token budget (<= 1,500 chars / <= 400 tokens on stdout / MCP return).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import math
import os
import sys
import time
from typing import Any

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

from bootstrap_golden_refs import DEFAULT_PROFILES_PATH, compute_bark24_distribution, compute_spectral_match
from deep_diagnostics import (
    LRA_PROFILE_THRESHOLDS,
    _third_octave_profile,
    analyze_audio_pocket_alignment,
    analyze_bass_harmonic_translation,
    analyze_dynamic_bark_masking,
    analyze_kick_bass_transient_phase,
    analyze_sustained_spectral_flux,
    analyze_transient_stacking,
    audit_midi_file,
    compute_ebu_r128_stats,
    compute_lra_ebu,
    compute_pink_slope_db_per_oct,
)
from horizontal_tester import audit_project_horizontal_seams
from loop_first_runner import LoopFirstStateMachine
from macro_structure_audit import analyze_macro_structure
from multiscale_autofix import dispatch_multiscale_autofix
from multiscale_pyramid import build_multiscale_pyramid
from render_and_audit import GENRE_GATES, render_and_evaluate
from soundcheck_lib import audit_soundcheck_wav
from studio_lib import GridlineMCP
from target_fingerprint import load_multi_scale_contract
from vertical_tester import _extract_expected_onsets_from_gridline, run_all_vertical_stages


def _find_project_files(project_dir: str) -> dict[str, Any]:
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    bp_path = os.path.join(p_dir, "song_blueprint.json")
    bp: dict[str, Any] = {}
    if os.path.isfile(bp_path):
        with open(bp_path, encoding="utf-8") as fh:
            bp = json.load(fh)

    core_gl = os.path.join(p_dir, "core_loop.gridline")
    main_gl = ""
    for cand in ("projet.gridline", "project.gridline"):
        p = os.path.join(p_dir, cand)
        if os.path.isfile(p):
            main_gl = p
            break
    if not main_gl and os.path.isdir(p_dir):
        for fn in sorted(os.listdir(p_dir)):
            if fn.endswith(".gridline") and fn != "core_loop.gridline":
                main_gl = os.path.join(p_dir, fn)
                break

    master_wav = ""
    for cand in (
        os.path.join(p_dir, "renders", "final_mix.wav"),
        os.path.join(p_dir, "renders", "master.wav"),
        os.path.join(p_dir, "master.wav"),
    ):
        if os.path.isfile(cand):
            master_wav = cand
            break
    renders_dir = os.path.join(p_dir, "renders")
    if not master_wav and os.path.isdir(renders_dir):
        wavs = [
            os.path.join(renders_dir, f)
            for f in sorted(os.listdir(renders_dir))
            if f.lower().endswith((".wav", ".flac")) and not f.startswith("preview")
        ]
        if wavs:
            master_wav = wavs[0]

    stems_dir = os.path.join(renders_dir, "stems")
    if not os.path.isdir(stems_dir):
        stems_dir = ""

    midi_path = ""
    for cand in ("master_sketch.mid", "master_sketch.midi"):
        p = os.path.join(p_dir, cand)
        if os.path.isfile(p):
            midi_path = p
            break

    samples_dir = os.path.join(p_dir, "samples")
    if not os.path.isdir(samples_dir):
        samples_dir = ""

    return {
        "project_dir": p_dir,
        "blueprint_path": bp_path if os.path.isfile(bp_path) else "",
        "blueprint": bp,
        "core_gridline": core_gl if os.path.isfile(core_gl) else "",
        "main_gridline": main_gl,
        "master_wav": master_wav,
        "stems_dir": stems_dir,
        "midi_path": midi_path,
        "samples_dir": samples_dir,
    }


def _load_stems_into_memory(stems_dir: str) -> dict[str, tuple[np.ndarray, int]]:
    out: dict[str, tuple[np.ndarray, int]] = {}
    if not stems_dir or not os.path.isdir(stems_dir):
        return out
    files_to_read = [
        (os.path.splitext(fn)[0], os.path.join(stems_dir, fn))
        for fn in sorted(os.listdir(stems_dir))
        if fn.lower().endswith((".wav", ".flac"))
    ]
    if not files_to_read:
        return out

    def _read_one(item: tuple[str, str]) -> tuple[str, tuple[np.ndarray, int]]:
        stem, p = item
        data, sr = sf.read(p, dtype="float32", always_2d=True)
        mono = 0.5 * (data[:, 0] + data[:, 1]) if data.shape[1] >= 2 else data[:, 0]
        return stem, (mono, int(sr))

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(files_to_read))) as ex:
        for stem, val in ex.map(_read_one, files_to_read):
            out[stem] = val
    return out


def _fast_master_metrics(
    data: np.ndarray,
    sr: int,
    *,
    profile: str,
) -> dict[str, Any]:
    """
    Vectorized CPU-light (< 0.45 s for 2-min stereo) master physical & spectral metrics
    operating on preloaded `data (N, 2), sr` without spawning Essentia / duplicate I/O.
    """
    ch0 = data[:, 0]
    ch1 = data[:, 1]
    mono = 0.5 * (ch0 + ch1)

    pk0_hi, pk0_lo = float(np.max(ch0)), float(np.min(ch0))
    pk1_hi, pk1_lo = float(np.max(ch1)), float(np.min(ch1))
    max_abs = max(abs(pk0_hi), abs(pk0_lo), abs(pk1_hi), abs(pk1_lo))
    clipped = int(np.sum(np.abs(data) >= 0.999)) if max_abs >= 0.999 else 0

    # Local 4x True Peak around peak sample (< 1 ms)
    pk_idx = int(np.argmax(np.abs(ch0))) if max(abs(pk0_hi), abs(pk0_lo)) >= max(abs(pk1_hi), abs(pk1_lo)) else int(np.argmax(np.abs(ch1)))
    lo = max(0, pk_idx - 2048)
    hi = min(len(data), pk_idx + 2048)
    seg_pk = data[lo:hi]
    if len(seg_pk) >= 16:
        os_seg = signal.resample_poly(seg_pk, 4, 1, axis=0)
        tp_lin = float(np.max(np.abs(os_seg)))
    else:
        tp_lin = max_abs
    true_peak_dbtp = round(20.0 * math.log10(max(tp_lin, 1e-12)), 2)

    # Single-pass BS.1770 loudness + EBU R128 LRA via fast C SOS filter (~0.38s vs 3.5s dual pyloudnorm)
    ebu_stats = compute_ebu_r128_stats(data, sr)
    lufs = float(ebu_stats["integrated_lufs"])
    lra_lu = float(ebu_stats["lra_lu"])
    if math.isinf(lufs) or math.isnan(lufs):
        rms = float(np.sqrt(np.mean(mono ** 2))) + 1e-12
        lufs = 20.0 * math.log10(rms)

    # Low-end (< 150 Hz) and low-mid (200-800 Hz) phase correlation on integer-decimated ~4.4 kHz stereo (< 20 ms)
    dec = max(1, int(sr // 4410))
    st_low = data[::dec]
    sr_low = sr / float(dec)
    l_ch, r_ch = st_low[:, 0], st_low[:, 1]

    def _band_corr(lo_hz: float | None, hi_hz: float) -> float:
        if lo_hz is None:
            sos = signal.butter(4, min(hi_hz, 0.45 * sr_low), btype="lowpass", fs=sr_low, output="sos")
        else:
            sos = signal.butter(4, [lo_hz, min(hi_hz, 0.45 * sr_low)], btype="bandpass", fs=sr_low, output="sos")
        lf = signal.sosfilt(sos, l_ch)
        rf = signal.sosfilt(sos, r_ch)
        den = math.sqrt(float(np.sum(lf ** 2) * np.sum(rf ** 2)))
        return round(float(np.sum(lf * rf) / den), 3) if den > 1e-12 else 1.0

    low_end_corr = _band_corr(None, 150.0)
    low_mid_corr = _band_corr(200.0, 800.0)

    # Single Welch PSD reused for low_mids/air shares + 1/3-oct resonances (sample evenly <= 16s on > 24s tracks)
    if len(mono) > int(24.0 * sr):
        n_chunks = 8
        chunk_len = int(2.0 * sr)
        starts = np.linspace(0, len(mono) - chunk_len, n_chunks, dtype=int)
        rep_mono = np.concatenate([mono[s : s + chunk_len] for s in starts])
    else:
        rep_mono = mono

    nperseg = min(16384, max(512, len(rep_mono)))
    freqs, psd = signal.welch(rep_mono, fs=sr, nperseg=nperseg, scaling="spectrum")
    tot_e = float(np.sum(psd)) + 1e-18
    low_mids_pct = round(100.0 * float(np.sum(psd[(freqs >= 250.0) & (freqs < 500.0)])) / tot_e, 2)
    air_pct = round(100.0 * float(np.sum(psd[(freqs >= 6000.0) & (freqs < min(20000.0, 0.495 * sr))])) / tot_e, 2)

    resonances: list[dict[str, float]] = []
    try:
        prof_13 = _third_octave_profile(freqs, psd)
        vals = [x["level_db"] for x in prof_13]
        for idx in range(2, len(prof_13) - 2):
            fc = prof_13[idx]["freq_hz"]
            if 150.0 <= fc <= 8000.0:
                nb = 0.25 * (vals[idx - 2] + vals[idx - 1] + vals[idx + 1] + vals[idx + 2])
                exc = vals[idx] - nb
                if exc >= 10.0:
                    resonances.append({"freq_hz": float(fc), "local_excess_db": round(float(exc), 2)})
    except Exception:
        pass

    bark24 = compute_bark24_distribution(rep_mono, sr)
    pink_slope = compute_pink_slope_db_per_oct(rep_mono, sr)
    sm_res = compute_spectral_match(mono=rep_mono, sr=sr, profile=profile, bark_24=bark24, pink_slope=pink_slope)

    return {
        "clipped_samples": clipped,
        "true_peak_dbtp": true_peak_dbtp,
        "integrated_lufs": round(lufs, 2),
        "lra_lu": lra_lu,
        "low_end_corr": low_end_corr,
        "low_mid_corr": low_mid_corr,
        "low_mids_pct": low_mids_pct,
        "air_pct": air_pct,
        "resonances": resonances,
        "bark24": bark24,
        "pink_slope_db_per_oct": pink_slope,
        "spectral_match": sm_res,
    }


def _evaluate_pre_rendered_audio(
    *,
    files: dict[str, Any],
    contract: dict[str, Any] | None,
    stage_norm: str,
    bpm: float,
    beats_per_bar: int,
    profile: str,
    has_drums: bool,
    master_wav: str,
) -> dict[str, Any]:
    """
    Fast CPU-only (< 4s on 2-min tracks) H x V evaluation over already-rendered
    `master_wav`, `stems_dir`, and `master_sketch.mid` without invoking Gridline renders.
    """
    bp = files["blueprint"]
    stems_map = _load_stems_into_memory(files["stems_dir"])
    scale_matrix: dict[str, dict[str, float | None]] = {
        "S1_physical": {"V": None, "H": None},
        "S2_acoustic": {"V": None, "H": None},
        "S3_syntax": {"V": None, "H": None},
        "S4_form": {"V": None, "H": None},
        "S5_context": {"V": None, "H": None},
    }
    deviations: list[dict[str, Any]] = []
    raw_details: dict[str, Any] = {}

    s1_v_scores: list[float] = []
    s1_h_scores: list[float] = []
    s2_v_scores: list[float] = []
    s2_h_scores: list[float] = []

    if stems_map:
        kick_stem = next((v for k, v in stems_map.items() if any(w in k.lower() for w in ("kick", "bd"))), None)
        bass_stem = next((v for k, v in stems_map.items() if any(w in k.lower() for w in ("808", "sub", "bass"))), None)
        if has_drums and kick_stem is not None and bass_stem is not None:
            kb_res = analyze_kick_bass_transient_phase(kick_stem, bass_stem)
            raw_details["s1_v_kick_bass"] = kb_res
            if kb_res.get("status") != "SKIP":
                s1_v_scores.append(float(kb_res.get("score", 1.0)))
                if kb_res.get("verdict") != "PASS":
                    act = kb_res.get("next_action") or {}
                    deviations.append({
                        "scale": "S1_physical",
                        "axis": "V",
                        "location": "Kick/808",
                        "metric": f"r_0={kb_res.get('r_0')}, r_max={kb_res.get('r_max')}",
                        "diagnosis": "Kick/808 transient phase cancellation or micro-shift",
                        "next_action": act.get("code", "INVERT_808_POLARITY"),
                        "action_payload": act,
                        "blocking": True,
                        "severity": 0.95,
                    })

        if bass_stem is not None:
            bt_res = analyze_bass_harmonic_translation(bass_stem)
            raw_details["s1_v_bass_translation"] = bt_res
            if bt_res.get("status") != "SKIP":
                s1_v_scores.append(1.0 if bt_res.get("verdict") == "PASS" else 0.58)
                if bt_res.get("verdict") != "PASS":
                    act = bt_res.get("next_action") or {}
                    f0_val = bt_res.get("f0_hz")
                    deviations.append({
                        "scale": "S1_physical",
                        "axis": "V",
                        "location": "Bass/Sub",
                        "metric": f"f0={f0_val}Hz, R_trans={bt_res.get('r_trans_db')}dB",
                        "diagnosis": "Sub-bass fundamental lacks 2nd/3rd harmonics for small speakers",
                        "next_action": f"BOOST_BASS_HARMONICS({f0_val})" if f0_val else act.get("code", "BOOST_BASS_HARMONICS"),
                        "action_payload": act,
                        "blocking": False,
                        "severity": 0.72,
                    })

        lead_stem = next(
            (v for k, v in stems_map.items() if any(w in k.lower() for w in ("lead", "melody", "topline", "vox", "vocal", "solo"))),
            None,
        )
        harmony_stem = next(
            (
                v
                for k, v in stems_map.items()
                if any(w in k.lower() for w in ("chord", "pad", "harmony", "string", "rhodes", "arp", "counter", "keys", "piano", "guitar", "organ"))
            ),
            None,
        )
        if lead_stem is not None and harmony_stem is not None:
            bm_res = analyze_dynamic_bark_masking(lead_stem, harmony_stem)
            raw_details["s2_v_bark_masking"] = bm_res
            if bm_res.get("status") != "SKIP":
                s2_v_scores.append(1.0 if bm_res.get("verdict") == "PASS" else 0.60)
                if bm_res.get("verdict") != "PASS":
                    act = bm_res.get("next_action") or {}
                    f_m = bm_res.get("f_mask_hz")
                    deviations.append({
                        "scale": "S2_acoustic",
                        "axis": "V",
                        "location": "Lead vs Harmony",
                        "metric": f"masked={bm_res.get('masked_ratio')}, SMR={bm_res.get('mean_smr_db')}dB@{f_m}Hz",
                        "diagnosis": f"Dynamic Bark masking of Lead by Harmony near {f_m} Hz",
                        "next_action": f"APPLY_DYNAMIC_MIRROR_EQ({f_m})" if f_m else act.get("code", "APPLY_DYNAMIC_MIRROR_EQ"),
                        "action_payload": act,
                        "blocking": False,
                        "severity": 0.76,
                    })

        # Sustained spectral flux (S1/H) on sustained stems
        for s_nm, s_data in stems_map.items():
            if any(w in s_nm.lower() for w in ("bass", "808", "pad", "chord", "lead")):
                sf_res = analyze_sustained_spectral_flux(s_data, role=s_nm)
                raw_details.setdefault("s1_h_spectral_flux", {})[s_nm] = sf_res
                if sf_res.get("status") != "SKIP":
                    s1_h_scores.append(float(sf_res.get("score", 1.0)))
                    flux_val = sf_res.get("spectral_flux_norm")
                    if sf_res.get("verdict") in ("WARN", "FIX", "FAIL") and flux_val is not None and float(flux_val) < 0.008:
                        deviations.append({
                            "scale": "S1_physical",
                            "axis": "H",
                            "location": s_nm,
                            "metric": f"flux={flux_val}",
                            "diagnosis": f"Static timbre on sustained stem '{s_nm}'",
                            "next_action": "ADD_LFO_OR_FILTER_MOD",
                            "blocking": False,
                            "severity": 0.45,
                        })

        # Transient stacking (S2/V) on non-percussive stems
        non_perc = {
            k: v for k, v in stems_map.items()
            if not any(kw in k.lower() for kw in ("kick", "snare", "clap", "hat", "perc", "drum", "master", "bus"))
        }
        if len(non_perc) >= 3:
            max_sim = int(((contract or {}).get("s2_acoustic") or {}).get("max_simultaneous_bark_transients", 2))
            ts_res = analyze_transient_stacking(non_perc, bpm=bpm, max_simultaneous=max_sim)
            raw_details["s2_v_transient_stacking"] = ts_res
            if ts_res.get("status") != "SKIP":
                s2_v_scores.append(float(ts_res.get("score", 1.0)))
                if ts_res.get("verdict") != "PASS":
                    act = ts_res.get("next_action") or {}
                    deviations.append({
                        "scale": "S2_acoustic",
                        "axis": "V",
                        "location": ",".join(ts_res.get("colliding_tracks", [])[:2]) or "non_perc_stems",
                        "metric": f"collision_ratio={ts_res.get('collision_ratio')}",
                        "diagnosis": "Simultaneous >2 non-percussive attacks in 1-4 kHz / LowMid",
                        "next_action": act.get("code", "STAGGER_ATTACK_OR_STRUM"),
                        "action_payload": act,
                        "blocking": float(ts_res.get("collision_ratio", 0.0)) > 0.25,
                        "severity": 0.80,
                    })

        # Audio pocket alignment (S2/H) if project gridline exists to provide expected onsets
        gl_ref = files["main_gridline"] or files["core_gridline"]
        if gl_ref:
            gl_bpm, exp_onsets = _extract_expected_onsets_from_gridline(gl_ref)
            if exp_onsets:
                s2_cfg = (contract or {}).get("s2_acoustic") or {}
                tol_ms = float(s2_cfg.get("onset_lag_tolerance_ms", 7.0))
                haas_targets = s2_cfg.get("haas_targets_ms")
                pk_res = analyze_audio_pocket_alignment(
                    stems_map, exp_onsets, bpm=gl_bpm or bpm, haas_targets_ms=haas_targets, tolerance_ms=tol_ms
                )
                raw_details["s2_h_pocket"] = pk_res
                if pk_res.get("status") != "SKIP":
                    s2_h_scores.append(float(pk_res.get("score", 1.0)))
                    if pk_res.get("verdict") != "PASS":
                        act = pk_res.get("next_action") or {}
                        tr = (act.get("params") or {}).get("track", "stem")
                        deviations.append({
                            "scale": "S2_acoustic",
                            "axis": "H",
                            "location": tr,
                            "metric": f"lag={((pk_res.get('roles') or {}).get(tr) or {}).get('median_lag_ms')}ms",
                            "diagnosis": f"Audio onset pocket lag on '{tr}'",
                            "next_action": act.get("code", "TRIM_SAMPLE_START_MS"),
                            "action_payload": act,
                            "blocking": float(pk_res.get("score", 1.0)) < 0.70,
                            "severity": 0.85,
                        })

    # Master audio analysis (S1/S4/S5 fast vectorized DSP; 0 redundant file reads)
    s5_ctx = (contract or {}).get("s5_context") or {}
    gate_cfg = GENRE_GATES.get(profile, GENRE_GATES["trap_rap"])
    target_lufs = float(s5_ctx.get("target_lufs", gate_cfg["default_lufs"]))
    low_mids_max = float(s5_ctx.get("low_mids_max", gate_cfg["low_mids_max"]))
    air_min = float(s5_ctx.get("air_min", gate_cfg["air_min"]))

    if master_wav and os.path.isfile(master_wav):
        data, sr = sf.read(master_wav, dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.hstack([data, data])
        m_dsp = _fast_master_metrics(data, sr, profile=profile)
        raw_details["master_dsp"] = {k: v for k, v in m_dsp.items() if k != "bark24"}

        clipped = m_dsp["clipped_samples"]
        tp = m_dsp["true_peak_dbtp"]
        le_corr = m_dsp["low_end_corr"]
        lm_corr = m_dsp["low_mid_corr"]
        res_list = m_dsp["resonances"]

        s1_phys = 1.0
        if clipped > 0 or tp > -0.5:
            s1_phys = min(s1_phys, 0.35)
            deviations.append({
                "scale": "S1_physical",
                "axis": "V",
                "location": "Master",
                "metric": f"clip={clipped}, tp={tp}dBTP",
                "diagnosis": "Master clipping or True Peak > -0.5 dBTP",
                "next_action": "ADJUST_MASTER_GAIN",
                "blocking": True,
                "severity": 0.98,
            })
        if has_drums and le_corr < 0.45:
            s1_phys = min(s1_phys, 0.45)
            deviations.append({
                "scale": "S1_physical",
                "axis": "V",
                "location": "LowEnd(<150Hz)",
                "metric": f"corr={le_corr}",
                "diagnosis": "Low-end phase cancellation below 150 Hz",
                "next_action": "INVERT_808_POLARITY",
                "blocking": True,
                "severity": 0.90,
            })
        if res_list:
            s1_phys = min(s1_phys, 0.68)
            r0 = res_list[0]
            deviations.append({
                "scale": "S1_physical",
                "axis": "V",
                "location": f"Master@{r0['freq_hz']}Hz",
                "metric": f"+{r0['local_excess_db']}dB",
                "diagnosis": f"Severe 1/3-oct resonance at {r0['freq_hz']} Hz",
                "next_action": "EQ_RESONANCE_NOTCH",
                "blocking": False,
                "severity": 0.65,
            })
        s1_v_scores.append(s1_phys)
        if not s2_v_scores:
            s2_v_scores.append(1.0 if lm_corr >= 0.55 else 0.55)

        duration_s = float(len(data)) / float(max(1, sr))
        lra_lu = float(m_dsp["lra_lu"])
        lra_min, lra_max = LRA_PROFILE_THRESHOLDS.get(profile, (3.5, 9.0))
        if stage_norm in ("E4", "ALL") and duration_s >= 12.0 and lra_lu < lra_min:
            s2_v_scores.append(0.65)
            deviations.append({
                "scale": "S2_acoustic",
                "axis": "V",
                "location": "BUS_Master/Dyn",
                "metric": f"LRA={lra_lu}LU < {lra_min}LU (max {lra_max})",
                "diagnosis": "Over-compressed dynamic range (EBU R128 LRA below genre floor)",
                "next_action": "RELAX_BUS_COMPRESSION",
                "blocking": False,
                "severity": 0.74,
            })

        # S4 Macro Structure (SSM + Novelty + 5D Arc + Cognitive Ceiling) reusing preloaded `data, sr` + stems
        if stage_norm in ("E3", "E4", "ALL"):
            macro_res = analyze_macro_structure(
                master_wav,
                bpm=bpm,
                beats_per_bar=beats_per_bar,
                blueprint=bp or None,
                contract=contract,
                stems_dir_or_map=stems_map or None,
                project_dir=files["project_dir"],
                has_drums=has_drums,
                preloaded_data=(data, sr),
            )
            raw_details["s4_macro"] = macro_res
            s4_h_sub = [
                float(macro_res["clones_8bar"].get("score", 1.0)),
                float(macro_res["novelty"].get("score", 1.0)),
            ]
            s4_v_sub = [
                float(macro_res["arc_5d"].get("score", 1.0)),
                float(macro_res["cognitive_ceiling"].get("score", 1.0)),
            ]
            scale_matrix["S4_form"] = {
                "V": round(min(s4_v_sub), 3),
                "H": round(min(s4_h_sub), 3),
            }
            deviations.extend(macro_res.get("deviations") or [])

        # S5 Context (Spectral Match vs Contract / Golden Refs + LUFS/Balance)
        if stage_norm in ("E4", "ALL"):
            bark24 = m_dsp["bark24"]
            sm_res = m_dsp["spectral_match"]
            target_b24 = s5_ctx.get("bark24_target")
            if target_b24 and len(target_b24) == 24:
                bm = np.asarray(bark24, dtype=np.float64)
                bt = np.asarray(target_b24, dtype=np.float64)
                den = float(np.linalg.norm(bm) * np.linalg.norm(bt))
                cos_val = float(np.dot(bm, bt) / den) if den > 1e-12 else 0.0
                spec_score = cos_val
            else:
                cos_val = float(sm_res.get("cosine_similarity", 0.85))
                spec_score = float(sm_res.get("spectral_match_score", 0.85))
            raw_details["s5_spectral_match"] = {**sm_res, "contract_cosine": round(cos_val, 4)}

            lufs_err = abs(float(m_dsp["integrated_lufs"]) - target_lufs)
            lm_pct = float(m_dsp["low_mids_pct"])
            air_pct = float(m_dsp["air_pct"])

            s5_v = min(1.0, max(0.25, 0.55 * spec_score + 0.45 * max(0.0, 1.0 - lufs_err / 6.0)))
            if lm_pct > low_mids_max or air_pct < air_min or lufs_err > 1.8:
                s5_v = min(s5_v, 0.70)
                deviations.append({
                    "scale": "S5_context",
                    "axis": "V",
                    "location": f"Master({profile})",
                    "metric": f"LUFS={m_dsp['integrated_lufs']}/{target_lufs}, cos={round(cos_val,3)}",
                    "diagnosis": "Mix balance / LUFS deviates from S5 target fingerprint",
                    "next_action": "ADJUST_MASTER_GAIN",
                    "blocking": False,
                    "severity": 0.60,
                })
            scale_matrix["S5_context"]["V"] = round(s5_v, 3)

    # S3 MIDI Syntax & Harmony
    if files["midi_path"] and stage_norm in ("E2", "E3", "E4", "ALL"):
        hc = (bp or {}).get("harmonic_core") or {}
        ts_obj = (bp or {}).get("time_signature") or {}
        ts_tup = (int(ts_obj["numerator"]), int(ts_obj["denominator"])) if ts_obj.get("numerator") else (beats_per_bar, 4)
        s3_res = audit_midi_file(
            midi_path=files["midi_path"],
            expected_key=str(hc.get("key") or ""),
            time_signature=ts_tup,
            strict_s3=True,
            s3_contract=(contract or {}).get("s3_syntax"),
        )
        raw_details["s3_midi"] = s3_res
        s3_diag = s3_res.get("s3_diagnostics") or {}
        s3_v = 1.0 if s3_res.get("gate_passed") else 0.62
        s3_h = 1.0 if s3_res.get("gate_passed") else 0.70
        if s3_diag:
            co = s3_diag.get("motif_cohesion") or {}
            inter = s3_diag.get("rhythmic_interlocking") or {}
            if co.get("verdict") in ("WARN", "FAIL"):
                s3_h = min(s3_h, 0.52)
                act = co.get("next_action") or {}
                deviations.append({
                    "scale": "S3_syntax",
                    "axis": "H",
                    "location": "Topline/Lead",
                    "metric": f"motif_sim={co.get('adjacent_bar_similarity')}",
                    "diagnosis": co.get("reason") or "Motif cohesion violation",
                    "next_action": act.get("code", "FIX_MOTIF_CLONE"),
                    "action_payload": act,
                    "blocking": True,
                    "severity": 0.85,
                })
            if inter.get("verdict") in ("WARN", "FAIL"):
                s3_v = min(s3_v, 0.60)
                act = inter.get("next_action") or {}
                deviations.append({
                    "scale": "S3_syntax",
                    "axis": "V",
                    "location": "Chords vs Lead",
                    "metric": f"attack_overlap={inter.get('chords_lead_attack_overlap')}",
                    "diagnosis": "Excessive simultaneous Chords/Lead strong attacks",
                    "next_action": act.get("code", "FIX_CHORDS_LEAD_OVERLAP"),
                    "action_payload": act,
                    "blocking": False,
                    "severity": 0.70,
                })
        scale_matrix["S3_syntax"] = {"V": round(s3_v, 3), "H": round(s3_h, 3)}

    if s1_v_scores:
        scale_matrix["S1_physical"]["V"] = round(min(s1_v_scores), 3)
    if s1_h_scores:
        scale_matrix["S1_physical"]["H"] = round(min(s1_h_scores), 3)
    if s2_v_scores:
        scale_matrix["S2_acoustic"]["V"] = round(min(s2_v_scores), 3)
    if s2_h_scores:
        scale_matrix["S2_acoustic"]["H"] = round(min(s2_h_scores), 3)

    return {
        "scale_matrix": scale_matrix,
        "deviations": deviations,
        "raw_details": raw_details,
    }


def _evaluate_gridline_stages(
    *,
    files: dict[str, Any],
    contract: dict[str, Any] | None,
    stage_norm: str,
    bpm: float,
    beats_per_bar: int,
    profile: str,
    has_drums: bool,
    contract_path: str,
    skip_e1_e2_if_cached: bool = False,
) -> dict[str, Any]:
    """Run incremental Gridline-backed E1..E4 checks (using SHA-256 artifact cache)."""
    p_dir = files["project_dir"]
    bp = files["blueprint"]
    scale_matrix: dict[str, dict[str, float | None]] = {
        "S1_physical": {"V": None, "H": None},
        "S2_acoustic": {"V": None, "H": None},
        "S3_syntax": {"V": None, "H": None},
        "S4_form": {"V": None, "H": None},
        "S5_context": {"V": None, "H": None},
    }
    deviations: list[dict[str, Any]] = []
    raw_details: dict[str, Any] = {}

    e1_e2_reused = False
    if skip_e1_e2_if_cached and os.path.isdir(p_dir):
        try:
            sm_cached = LoopFirstStateMachine(p_dir, has_drums=has_drums)
            st_c = sm_cached.state
            if st_c.get("core_loop_locked") and (st_c.get("artifact_hashes") or (st_c.get("multiscale_scores") or {}).get("E2")):
                prev_e2 = (st_c.get("multiscale_scores") or {}).get("E2") or {}
                prev_sm = prev_e2.get("scale_matrix") or {}
                for k_sc in ("S1_physical", "S2_acoustic", "S3_syntax"):
                    if isinstance(prev_sm.get(k_sc), dict):
                        scale_matrix[k_sc] = dict(prev_sm[k_sc])
                    else:
                        scale_matrix[k_sc] = {"V": 1.0, "H": 1.0}
                e1_e2_reused = True
                raw_details["e1_e2_cache_reused"] = True
        except Exception:
            e1_e2_reused = False

    if not e1_e2_reused and stage_norm in ("E1", "ALL") and files["samples_dir"]:
        s1_fluxes: list[float] = []
        sdir = files["samples_dir"]
        for fn in sorted(os.listdir(sdir)):
            if not fn.lower().endswith((".wav", ".flac")):
                continue
            sp = os.path.join(sdir, fn)
            sc = audit_soundcheck_wav(
                sp,
                track_name=os.path.splitext(fn)[0],
                role=os.path.splitext(fn)[0],
                bpm=bpm,
                use_neural=False,
            )
            raw_details.setdefault("e1_samples", {})[fn] = sc
            fl = (sc.get("spectral_flux") or {}).get("score", 1.0)
            s1_fluxes.append(float(fl))
            if sc.get("verdict") != "PASS":
                deviations.append({
                    "scale": "S1_physical",
                    "axis": "H",
                    "location": fn,
                    "metric": f"soundcheck={sc.get('verdict')}",
                    "diagnosis": "; ".join(sc.get("human_diagnosis") or ["Sample soundcheck issue"]),
                    "next_action": (sc.get("next_action") or {}).get("code", "SWAP_SAMPLE"),
                    "action_payload": sc.get("next_action") or {},
                    "blocking": sc.get("verdict") == "SWAP",
                    "severity": 0.75,
                })
        if s1_fluxes:
            scale_matrix["S1_physical"]["H"] = round(min(s1_fluxes), 3)

    if not e1_e2_reused and stage_norm in ("E2", "ALL"):
        core_or_main = files["core_gridline"] or files["main_gridline"]
        if core_or_main:
            v_rep = run_all_vertical_stages(
                core_or_main,
                profile=profile,
                blueprint=bp or None,
                contract=contract,
                stop_on_fail=False,
                use_cache=True,
            )
            raw_details["e2_vertical"] = v_rep
            s1_v_scores: list[float] = []
            s2_v_scores: list[float] = []
            s2_h_scores: list[float] = []
            for st_name, st_res in (v_rep.get("stages") or {}).items():
                kb = st_res.get("kick_bass_phase")
                if isinstance(kb, dict) and kb.get("status") != "SKIP":
                    s1_v_scores.append(float(kb.get("score", 1.0)))
                    if kb.get("verdict") != "PASS":
                        act = kb.get("next_action") or {}
                        deviations.append({
                            "scale": "S1_physical",
                            "axis": "V",
                            "location": f"{st_name}:Kick/808",
                            "metric": f"r_0={kb.get('r_0')}",
                            "diagnosis": "Kick/808 phase or micro-timing clash",
                            "next_action": act.get("code", "INVERT_808_POLARITY"),
                            "action_payload": act,
                            "blocking": True,
                            "severity": 0.95,
                        })
                pk = st_res.get("pocket_alignment")
                if isinstance(pk, dict) and pk.get("status") != "SKIP":
                    s2_h_scores.append(float(pk.get("score", 1.0)))
                    if pk.get("verdict") != "PASS":
                        act = pk.get("next_action") or {}
                        deviations.append({
                            "scale": "S2_acoustic",
                            "axis": "H",
                            "location": f"{st_name}:Pocket",
                            "metric": f"score={pk.get('score')}",
                            "diagnosis": "Audio onset lag / Haas pocket drift",
                            "next_action": act.get("code", "TRIM_SAMPLE_START_MS"),
                            "action_payload": act,
                            "blocking": float(pk.get("score", 1.0)) < 0.70,
                            "severity": 0.85,
                        })
                ts = st_res.get("transient_stacking")
                if isinstance(ts, dict) and ts.get("status") != "SKIP":
                    s2_v_scores.append(float(ts.get("score", 1.0)))
                    if ts.get("verdict") != "PASS":
                        act = ts.get("next_action") or {}
                        deviations.append({
                            "scale": "S2_acoustic",
                            "axis": "V",
                            "location": f"{st_name}:Stacking",
                            "metric": f"coll_ratio={ts.get('collision_ratio')}",
                            "diagnosis": "Simultaneous non-percussive transient stacking",
                            "next_action": act.get("code", "STAGGER_ATTACK_OR_STRUM"),
                            "action_payload": act,
                            "blocking": False,
                            "severity": 0.78,
                        })
                bt = st_res.get("bass_harmonic_translation")
                if isinstance(bt, dict) and bt.get("status") != "SKIP":
                    s1_v_scores.append(1.0 if bt.get("verdict") == "PASS" else 0.58)
                    if bt.get("verdict") != "PASS":
                        act = bt.get("next_action") or {}
                        f0_val = bt.get("f0_hz")
                        deviations.append({
                            "scale": "S1_physical",
                            "axis": "V",
                            "location": f"{st_name}:BassTranslation",
                            "metric": f"f0={f0_val}Hz, R_trans={bt.get('r_trans_db')}dB",
                            "diagnosis": "Sub-bass fundamental lacks 2nd/3rd harmonics for small speakers",
                            "next_action": f"BOOST_BASS_HARMONICS({f0_val})" if f0_val else act.get("code", "BOOST_BASS_HARMONICS"),
                            "action_payload": act,
                            "blocking": False,
                            "severity": 0.72,
                        })
                bm = st_res.get("dynamic_bark_masking")
                if isinstance(bm, dict) and bm.get("status") != "SKIP":
                    s2_v_scores.append(1.0 if bm.get("verdict") == "PASS" else 0.60)
                    if bm.get("verdict") != "PASS":
                        act = bm.get("next_action") or {}
                        f_m = bm.get("f_mask_hz")
                        deviations.append({
                            "scale": "S2_acoustic",
                            "axis": "V",
                            "location": f"{st_name}:BarkMasking",
                            "metric": f"masked={bm.get('masked_ratio')}, SMR={bm.get('mean_smr_db')}dB",
                            "diagnosis": f"Dynamic Bark masking of Lead by Harmony near {f_m} Hz",
                            "next_action": f"APPLY_DYNAMIC_MIRROR_EQ({f_m})" if f_m else act.get("code", "APPLY_DYNAMIC_MIRROR_EQ"),
                            "action_payload": act,
                            "blocking": False,
                            "severity": 0.76,
                        })
                s2_v_scores.append(1.0 if st_res.get("verdict") == "PASS" else 0.55)
                if st_res.get("verdict") != "PASS" and not deviations:
                    act = st_res.get("next_action") or {}
                    deviations.append({
                        "scale": "S2_acoustic",
                        "axis": "V",
                        "location": st_name,
                        "metric": f"verdict={st_res.get('verdict')}",
                        "diagnosis": "; ".join(st_res.get("human_diagnosis") or ["Vertical stage failed"]),
                        "next_action": act.get("code", "REVIEW"),
                        "action_payload": act,
                        "blocking": True,
                        "severity": 0.80,
                    })
            if s1_v_scores:
                scale_matrix["S1_physical"]["V"] = round(min(s1_v_scores), 3)
            if s2_v_scores:
                scale_matrix["S2_acoustic"]["V"] = round(min(s2_v_scores), 3)
            if s2_h_scores:
                scale_matrix["S2_acoustic"]["H"] = round(min(s2_h_scores), 3)

        pre_s3 = _evaluate_pre_rendered_audio(
            files=files,
            contract=contract,
            stage_norm="E2",
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            profile=profile,
            has_drums=has_drums,
            master_wav="",
        )
        if pre_s3["scale_matrix"]["S3_syntax"]["V"] is not None:
            scale_matrix["S3_syntax"] = pre_s3["scale_matrix"]["S3_syntax"]
            deviations.extend(pre_s3["deviations"])
            raw_details["s3_midi"] = pre_s3["raw_details"].get("s3_midi")

    if stage_norm in ("E3", "ALL") and files["main_gridline"] and bp:
        h_rep = audit_project_horizontal_seams(
            files["main_gridline"],
            blueprint=bp,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            use_cache=True,
            include_macro_ssm=bool(files["master_wav"]),
            full_mix_wav=files["master_wav"],
            contract_path=contract_path,
            stems_dir=files["stems_dir"],
        )
        raw_details["e3_horizontal"] = h_rep
        s4_h = float(h_rep.get("score", 1.0))
        s4_v: float | None = None
        if isinstance(h_rep.get("macro_structure"), dict):
            ms = h_rep["macro_structure"]
            s4_h = min(s4_h, float(ms.get("score", 1.0)))
            s4_v = float((ms.get("arc_5d") or {}).get("score", 1.0))
            deviations.extend(ms.get("deviations") or [])
        if h_rep.get("verdict") != "PASS":
            act = h_rep.get("next_action") or {}
            deviations.append({
                "scale": "S4_form",
                "axis": "H",
                "location": "HorizontalSeams",
                "metric": f"score={h_rep.get('score')}",
                "diagnosis": "; ".join(h_rep.get("human_diagnosis") or ["Seam/outro transition failure"]),
                "next_action": act.get("code", "REINFORCE_SECTION_TRANSITION"),
                "action_payload": act,
                "blocking": False,
                "severity": 0.72,
            })
        scale_matrix["S4_form"] = {"V": round(s4_v, 3) if s4_v is not None else None, "H": round(s4_h, 3)}

    if stage_norm in ("E4", "ALL") and files["main_gridline"]:
        out_wav = files["master_wav"] or os.path.join(p_dir, "renders", "final_mix.wav")
        e4_rep = render_and_evaluate(
            project=files["main_gridline"],
            out_wav=out_wav,
            profile=profile,
            bpm=bpm,
            beats_per_bar=float(beats_per_bar),
            blueprint=files["blueprint_path"],
            has_drums_arg=has_drums,
            use_neural=False,
            contract_path=contract_path,
            run_macro_audit=True,
            stems_dir=files["stems_dir"],
            export_fingerprint=True,
        )
        raw_details["e4_final"] = {k: v for k, v in e4_rep.items() if k != "neural"}
        if isinstance(e4_rep.get("macro_structure"), dict):
            ms = e4_rep["macro_structure"]
            scale_matrix["S4_form"] = {
                "V": round(float((ms.get("arc_5d") or {}).get("score", 1.0)), 3),
                "H": round(float(ms.get("score", 1.0)), 3),
            }
            deviations.extend(ms.get("deviations") or [])

        sm_score = float((e4_rep.get("spectral_match") or {}).get("spectral_match_score", 0.85))
        qg = e4_rep.get("quality_gate") or {}
        s5_v = sm_score if qg.get("passed") else min(0.68, sm_score)
        scale_matrix["S5_context"]["V"] = round(s5_v, 3)
        p3_diag = e4_rep.get("p3_diagnostics") or {}
        for fail_msg in (qg.get("failures") or []):
            act_payload = None
            if fail_msg.startswith("lra="):
                act_code = "RELAX_BUS_COMPRESSION"
            elif fail_msg.startswith("bass_translation_r_trans="):
                act_payload = (p3_diag.get("bass_harmonic_translation") or {}).get("next_action")
                f0_val = (act_payload or {}).get("params", {}).get("f0_hz")
                act_code = f"BOOST_BASS_HARMONICS({f0_val})" if f0_val else "BOOST_BASS_HARMONICS"
            elif fail_msg.startswith("bark_masking_lead_harmony="):
                act_payload = (p3_diag.get("dynamic_bark_masking") or {}).get("next_action")
                f_m = (act_payload or {}).get("params", {}).get("freq_hz")
                act_code = f"APPLY_DYNAMIC_MIRROR_EQ({f_m})" if f_m else "APPLY_DYNAMIC_MIRROR_EQ"
            elif any(k in fail_msg for k in ("resonance", "low_mids", "air=")):
                act_code = "EQ_RESONANCE_NOTCH"
            else:
                act_code = "ADJUST_MASTER_GAIN"
            dev_item: dict[str, Any] = {
                "scale": "S5_context",
                "axis": "V",
                "location": "MasterGate",
                "metric": fail_msg[:54],
                "diagnosis": f"E4 gate failure: {fail_msg}",
                "next_action": act_code,
                "blocking": False,
                "severity": 0.78,
            }
            if isinstance(act_payload, dict) and act_payload:
                dev_item["action_payload"] = act_payload
            deviations.append(dev_item)

    return {
        "scale_matrix": scale_matrix,
        "deviations": deviations,
        "raw_details": raw_details,
    }


def _export_e2_checkpoint_15s(
    project_dir: str,
    files: dict[str, Any],
    *,
    profile: str,
    bpm: float,
    beats_per_bar: int,
    eval_out: dict[str, Any],
    wav_path: str = "",
    skip_render: bool = False,
) -> dict[str, Any]:
    """
    Phase 7 P5: Export a 12-15s core-loop preview normalized to -14 LUFS at
    `<project>/renders/e2_checkpoint_15s.wav` and return a <= 300-token compact
    checkpoint JSON with 5-line musical summary.
    """
    p_dir = os.path.abspath(os.path.expanduser(project_dir))
    renders_dir = os.path.join(p_dir, "renders")
    os.makedirs(renders_dir, exist_ok=True)
    out_wav = os.path.join(renders_dir, "e2_checkpoint_15s.wav")

    bar_sec = float(beats_per_bar) * 60.0 / max(40.0, float(bpm))
    target_s = min(15.0, max(12.0, round(8.0 * bar_sec, 2)))

    src_wav = ""
    if wav_path.strip() and os.path.isfile(os.path.expanduser(wav_path)):
        src_wav = os.path.abspath(os.path.expanduser(wav_path))
    elif files.get("master_wav") and os.path.isfile(files["master_wav"]):
        src_wav = files["master_wav"]
    else:
        for cand in (os.path.join(renders_dir, "core_loop.wav"), os.path.join(renders_dir, "e2_core.wav")):
            if os.path.isfile(cand):
                src_wav = cand
                break

    if not src_wav and not skip_render:
        gl_core = files.get("core_gridline") or files.get("main_gridline") or ""
        if gl_core and os.path.isfile(gl_core):
            mcp = GridlineMCP(gl_core, os.path.dirname(gl_core))
            try:
                rel_out = os.path.relpath(out_wav, os.path.dirname(gl_core))
                mcp.call("render_audio", {"path": rel_out, "maxSeconds": 15.0})
                src_wav = out_wav
            finally:
                mcp.close()

    sr = 44100
    if src_wav and os.path.isfile(src_wav):
        audio, sr = sf.read(src_wav, dtype="float32", always_2d=True)
    else:
        audio = np.zeros((int(round(target_s * sr)), 2), dtype=np.float32)

    if audio.shape[1] == 1:
        audio = np.hstack([audio, audio])

    target_n = int(round(target_s * sr))
    if len(audio) >= int(round(12.0 * sr)):
        clip = audio[: min(len(audio), int(round(15.0 * sr)))]
        if len(clip) > target_n and target_n >= int(round(12.0 * sr)):
            clip = clip[:target_n]
    elif len(audio) > max(64, sr // 4):
        reps = int(math.ceil(float(target_n) / float(len(audio))))
        clip = np.tile(audio, (reps, 1))[:target_n]
    else:
        clip = np.zeros((target_n, 2), dtype=np.float32)

    # Normalize cleanly to -14.0 LUFS with -1.0 dBTP peak guard
    pk = float(np.max(np.abs(clip)))
    lufs_meas = -14.0
    if pk > 1e-5:
        cur_lufs = float(compute_ebu_r128_stats(clip, sr)["integrated_lufs"])
        if math.isfinite(cur_lufs) and cur_lufs > -65.0:
            gain = 10.0 ** ((-14.0 - cur_lufs) / 20.0)
            scaled = clip * np.float32(gain)
            new_pk = float(np.max(np.abs(scaled)))
            if new_pk > 0.891:
                # Smooth soft-knee ceiling at -1.0 dBTP without breaking -14 LUFS loudness
                clip = np.tanh(scaled / np.float32(0.891)) * np.float32(0.891)
            else:
                clip = scaled
        else:
            clip = clip * np.float32(0.25 / pk)
        lufs_meas = float(compute_ebu_r128_stats(clip, sr)["integrated_lufs"])

    sf.write(out_wav, clip.astype(np.float32, copy=False), sr, subtype="PCM_24")
    dur_s = round(float(len(clip)) / float(max(1, sr)), 2)

    bp = files.get("blueprint") or {}
    hc = bp.get("harmonic_core") or {}
    key_str = str(hc.get("key") or "unknown")
    prog = hc.get("degree_progression") or bp.get("degree_progression") or []
    prog_str = "-".join(str(x) for x in prog[:8]) if isinstance(prog, list) and prog else "default"

    roles: list[str] = []
    if files.get("stems_dir") and os.path.isdir(files["stems_dir"]):
        for fn in sorted(os.listdir(files["stems_dir"])):
            if fn.lower().endswith(".wav") and "master" not in fn.lower() and "bus " not in fn.lower():
                lbl = os.path.splitext(fn)[0].split(" - ", 1)[-1]
                if lbl not in roles:
                    roles.append(lbl)
    if not roles and isinstance(bp.get("tracks"), list):
        roles = [str(t.get("name") or t.get("role") or "") for t in bp["tracks"] if isinstance(t, dict) and (t.get("name") or t.get("role"))]

    sm = eval_out.get("scale_matrix") or {}
    def _sc(k: str) -> str:
        v = sm.get(k) or {}
        vals = [x for x in (v.get("V"), v.get("H")) if x is not None]
        return f"{min(vals):.2f}" if vals else "-"

    rel_wav = os.path.relpath(out_wav, p_dir).replace(os.sep, "/")
    summary_lines = [
        f"{int(round(bpm))} BPM | {key_str} | {profile}",
        f"Prog: {prog_str}"[:72],
        f"Roles: {', '.join(roles[:5]) or 'core_loop'}"[:72],
        f"Scores S1={_sc('S1_physical')} S2={_sc('S2_acoustic')} S3={_sc('S3_syntax')}",
        f"WAV: {rel_wav} ({dur_s}s, {round(lufs_meas, 1)} LUFS)",
    ]
    return {
        "status": "AWAITING_E2_CHECKPOINT",
        "checkpoint_wav": rel_wav,
        "duration_s": dur_s,
        "lufs": round(lufs_meas, 2),
        "summary": summary_lines,
        "resume_hint": "--resume-from-checkpoint",
    }


def audit_multiscale_hv(
    project_dir: str,
    stage: str = "ALL",
    contract_path: str = "",
    *,
    wav_path: str = "",
    skip_render: bool = False,
    auto_fix: bool = False,
    max_autofix_passes: int = 2,
    checkpoint_e2: bool = False,
    resume_from_checkpoint: bool = False,
) -> dict[str, Any]:
    """
    Unified Multi-Scale H x V Audit (used by both MCP `audit_multiscale_hv` and CLI).

    Strictly returns compact Bottom-Up Deviation Pyramid JSON (<= 1,500 chars / <= 400 tokens)
    while writing full matrices & diagnostics to `<project_dir>/renders/diagnostics/`.
    """
    t0 = time.perf_counter()
    files = _find_project_files(project_dir)
    p_dir = files["project_dir"]
    stage_norm = (stage or "ALL").strip().upper()
    if stage_norm not in ("E1", "E2", "E3", "E4", "ALL"):
        stage_norm = "ALL"

    contract = load_multi_scale_contract(contract_path=contract_path, project_dir=p_dir)
    bp = files["blueprint"]
    profile = str((contract or {}).get("mix_profile") or bp.get("mix_profile") or "trap_rap")
    if profile not in GENRE_GATES:
        profile = "trap_rap"
    bpm = float(bp.get("bpm") or 132.0)
    ts = bp.get("time_signature") or {}
    beats_per_bar = int(ts.get("numerator") or 4)
    has_drums = bool((contract or {}).get("has_drums", bp.get("has_drums", True)))

    sm_ctrl = LoopFirstStateMachine(p_dir, has_drums=has_drums, e2_checkpoint=True if checkpoint_e2 else None) if os.path.isdir(p_dir) else None
    if resume_from_checkpoint and sm_ctrl is not None:
        sm_ctrl.resume_from_checkpoint()

    chk_active = bool(
        (checkpoint_e2 or (sm_ctrl is not None and sm_ctrl.state.get("e2_checkpoint_enabled") and not sm_ctrl.state.get("e2_checkpoint_approved")))
        and not resume_from_checkpoint
        and stage_norm in ("E2", "ALL")
    )
    eff_stage_norm = "E2" if chk_active else stage_norm

    eff_wav = os.path.abspath(os.path.expanduser(wav_path)) if wav_path.strip() else files["master_wav"]
    use_prerendered = bool(skip_render or wav_path.strip() or (eff_wav and not files["main_gridline"] and not files["core_gridline"]))

    def _run_once() -> dict[str, Any]:
        if use_prerendered:
            return _evaluate_pre_rendered_audio(
                files=files,
                contract=contract,
                stage_norm=eff_stage_norm,
                bpm=bpm,
                beats_per_bar=beats_per_bar,
                profile=profile,
                has_drums=has_drums,
                master_wav=eff_wav,
            )
        eval_res = _evaluate_gridline_stages(
            files=files,
            contract=contract,
            stage_norm=eff_stage_norm,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            profile=profile,
            has_drums=has_drums,
            contract_path=contract_path,
            skip_e1_e2_if_cached=bool(resume_from_checkpoint),
        )
        sm = eval_res["scale_matrix"]
        has_any = any(
            isinstance(v, dict) and (v.get("V") is not None or v.get("H") is not None)
            for v in sm.values()
        )
        if eff_wav and os.path.isfile(eff_wav) and not has_any:
            return _evaluate_pre_rendered_audio(
                files=files,
                contract=contract,
                stage_norm=eff_stage_norm,
                bpm=bpm,
                beats_per_bar=beats_per_bar,
                profile=profile,
                has_drums=has_drums,
                master_wav=eff_wav,
            )
        return eval_res

    eval_out = _run_once()
    autofix_tags: list[str] = []

    if auto_fix:
        for _pass_idx in range(max(1, int(max_autofix_passes))):
            devs = eval_out.get("deviations") or []
            if not devs:
                break
            e4_summary = (eval_out.get("raw_details") or {}).get("e4_final")
            fix_res = dispatch_multiscale_autofix(
                p_dir,
                devs,
                gridline_path=files["main_gridline"] or files["core_gridline"],
                render_summary=e4_summary,
            )
            applied = fix_res.get("applied") or []
            if not applied:
                break
            for a in applied:
                c_tag = str(a.get("code") or "FIX")
                if c_tag not in autofix_tags:
                    autofix_tags.append(c_tag)
            eval_out = _run_once()

    blocking_e2 = [d for d in (eval_out.get("deviations") or []) if d.get("blocking")]
    if chk_active and not blocking_e2:
        chk_payload = _export_e2_checkpoint_15s(
            p_dir,
            files,
            profile=profile,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            eval_out=eval_out,
            wav_path=wav_path,
            skip_render=skip_render,
        )
        if sm_ctrl is not None:
            sm_ctrl.record_multiscale_scores("E2", {
                "stage": "E2",
                "verdict": "PASS",
                "scale_matrix": eval_out["scale_matrix"],
            })
            sm_ctrl.trigger_e2_checkpoint(
                wav_path=chk_payload["checkpoint_wav"],
                summary_lines=chk_payload["summary"],
            )
        return chk_payload

    elapsed_s = round(time.perf_counter() - t0, 3)
    full_details = {
        **(eval_out.get("raw_details") or {}),
        "elapsed_s": elapsed_s,
        "autofix_tags": autofix_tags,
        "mode": "pre_rendered" if use_prerendered else "gridline_cached",
    }

    pyramid = build_multiscale_pyramid(
        stage=eff_stage_norm,
        scale_matrix=eval_out["scale_matrix"],
        deviations=eval_out["deviations"],
        autofix_applied=autofix_tags,
        project_dir=p_dir,
        full_details=full_details,
    )

    if os.path.isdir(p_dir):
        try:
            sm_state = LoopFirstStateMachine(p_dir, has_drums=has_drums)
            sm_state.record_multiscale_scores(eff_stage_norm, pyramid)
        except Exception:
            pass

    return pyramid


def main() -> None:
    ap = argparse.ArgumentParser(description="One-Shot Multi-Scale H x V Autonomous Pipeline (`run_autonomous_pipeline.py`).")
    ap.add_argument("--project", required=True, help="Project directory path")
    ap.add_argument("--stage", default="ALL", choices=["E1", "E2", "E3", "E4", "ALL"], help="Pipeline stage or ALL")
    ap.add_argument("--contract", default="", help="Optional multi_scale_contract.json path")
    ap.add_argument("--wav", default="", help="Optional pre-rendered master WAV path")
    ap.add_argument("--skip-render", action="store_true", help="Audit existing renders/stems/MIDI without invoking Gridline")
    ap.add_argument("--auto-fix", action="store_true", help="Run closed deterministic Python autofix before escalating to LLM")
    ap.add_argument("--max-autofix-passes", type=int, default=2, help="Max deterministic autofix passes (default 2)")
    ap.add_argument("--checkpoint-e2", action="store_true", help="Export 15s -14 LUFS core loop at end of E2 and pause for human approval")
    ap.add_argument("--resume-from-checkpoint", action="store_true", help="Approve E2 checkpoint and advance to E3/E4 with cached E1/E2")
    ap.add_argument("--json-compact", action="store_true", help="Emit only compact <=1500-char Pyramid JSON on stdout")
    args = ap.parse_args()

    if args.json_compact:
        with contextlib.redirect_stdout(sys.stderr):
            res = audit_multiscale_hv(
                project_dir=args.project,
                stage=args.stage,
                contract_path=args.contract,
                wav_path=args.wav,
                skip_render=args.skip_render,
                auto_fix=args.auto_fix,
                max_autofix_passes=args.max_autofix_passes,
                checkpoint_e2=args.checkpoint_e2,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
        print(json.dumps(res, ensure_ascii=False, separators=(",", ":")))
    else:
        res = audit_multiscale_hv(
            project_dir=args.project,
            stage=args.stage,
            contract_path=args.contract,
            wav_path=args.wav,
            skip_render=args.skip_render,
            auto_fix=args.auto_fix,
            max_autofix_passes=args.max_autofix_passes,
            checkpoint_e2=args.checkpoint_e2,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
