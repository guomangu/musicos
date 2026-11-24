#!/usr/bin/env python3
"""
Fast quiet wrapper around `gridline render` + `audio-analyzer` (`analyze_mix` + `audit_mix_deep`)
+ lightweight spectral reference match (`spectral_profiles.json`, 0 PyTorch)
+ closed-loop deterministic `--auto-fix --max-passes 2` (`GridlineMCP` + `gl_dsl`).
Supports `--range BAR1-BAR2` for rapid 4x faster mix/EQ verification loops.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _p in (SCRIPTS_DIR, ANALYZER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audio_analyzer_server import _analyze_file
from bootstrap_golden_refs import DEFAULT_PROFILES_PATH, compute_bark24_distribution, compute_spectral_match
from catalog_coherence import build_master_fingerprint
from deep_diagnostics import (
    LRA_PROFILE_THRESHOLDS,
    _find_role_stems_in_dir,
    _load_and_sum_stems,
    analyze_bass_harmonic_translation,
    analyze_dynamic_bark_masking,
    analyze_stem_hierarchy,
    compute_lra_ebu,
    run_deep_audit,
)
from gain_staging_lib import normalize_role_for_gain_staging
from gl_dsl import clamp, db_to_auto, eq_band, eq_params, limiter_master, to_mixer_params
from macro_structure_audit import analyze_macro_structure
from studio_lib import GridlineMCP
from target_fingerprint import load_multi_scale_contract
from taste_memory import record_validated_taste
from blueprint_safety import summarize_musical_safety
from mix_config import get_invariant, limiter_invariants, notch_caps, profile_settings, tuning_invariants


def _should_record_taste(proj_dir: str, taste_memory_path: str = "", approve_taste: bool = False) -> bool:
    if taste_memory_path or approve_taste:
        return True
    tmp_root = os.path.abspath(tempfile.gettempdir())
    abs_p = os.path.abspath(proj_dir)
    return not (abs_p == tmp_root or abs_p.startswith(tmp_root + os.sep))


def parse_sections(spec: str) -> list[tuple[str, int, int]]:
    out = []
    if not spec.strip():
        return out
    for item in spec.split(","):
        name, rng = item.split(":")
        b0, b1 = rng.split("-")
        out.append((name.strip(), int(b0), int(b1)))
    return out


GENRE_GATES: dict[str, dict[str, float]] = {
    "trap_rap":          {"low_mids_max": 24.0, "air_min": 3.0, "dark_air_min": 1.2,  "sharp_max": 1.90, "dark_sharp_max": 1.85, "arc_min": 2.5, "default_lufs": -11.0, "low_end_max": 42.0},
    "pop_rnb":           {"low_mids_max": 26.0, "air_min": 3.0, "dark_air_min": 1.5,  "sharp_max": 1.95, "dark_sharp_max": 1.85, "arc_min": 2.5, "default_lufs": -10.5, "low_end_max": 44.0},
    "edm_club":          {"low_mids_max": 24.0, "air_min": 3.5, "dark_air_min": 1.75, "sharp_max": 2.05, "dark_sharp_max": 1.90, "arc_min": 2.5, "default_lufs": -9.0, "low_end_max": 46.0},
    "rock_metal":        {"low_mids_max": 28.0, "air_min": 2.5, "dark_air_min": 1.25, "sharp_max": 2.10, "dark_sharp_max": 1.90, "arc_min": 2.0, "default_lufs": -10.0, "low_end_max": 44.0},
    "lofi_chill":        {"low_mids_max": 34.0, "air_min": 0.5, "dark_air_min": 0.25, "sharp_max": 1.75, "dark_sharp_max": 1.75, "arc_min": 1.8, "default_lufs": -14.0, "low_end_max": 50.0},
    "jazz_acoustic":     {"low_mids_max": 36.0, "air_min": 1.0, "dark_air_min": 0.50, "sharp_max": 1.85, "dark_sharp_max": 1.80, "arc_min": 2.5, "default_lufs": -14.5, "low_end_max": 52.0},
    "classical_ambient": {"low_mids_max": 40.0, "air_min": 0.3, "dark_air_min": 0.15, "sharp_max": 1.75, "dark_sharp_max": 1.75, "arc_min": 3.0, "default_lufs": -16.0, "low_end_max": 58.0},
}


def _resolve_valence_and_dark_mode(
    project_path: str,
    blueprint_path: str,
    dark_flag: bool,
    valence_arg: float | None,
) -> tuple[float | None, bool, str | None]:
    """Read emotion_vector.valence from song_blueprint.json (or CLI overrides) to adapt dark-mix thresholds."""
    valence, is_dark, _, used_bp = _resolve_blueprint_metadata(
        project_path=project_path,
        blueprint_path=blueprint_path,
        dark_flag=dark_flag,
        valence_arg=valence_arg,
        has_drums_arg=None,
        profile="trap_rap",
    )
    return valence, is_dark, used_bp


def _resolve_blueprint_metadata(
    project_path: str,
    blueprint_path: str,
    dark_flag: bool,
    valence_arg: float | None,
    has_drums_arg: bool | None,
    profile: str,
) -> tuple[float | None, bool, bool, str | None]:
    """Read `emotion_vector.valence` and `has_drums` from `song_blueprint.json` with safe defaults."""
    valence: float | None = valence_arg
    has_drums: bool | None = has_drums_arg
    used_bp: str | None = None
    candidates = [blueprint_path] if blueprint_path else [
        os.path.join(os.path.dirname(os.path.abspath(project_path)), "song_blueprint.json")
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            try:
                with open(cand, encoding="utf-8") as fh:
                    bp = json.load(fh)
                ev = bp.get("emotion_vector") or {}
                if valence is None and "valence" in ev and ev["valence"] is not None:
                    valence = float(ev["valence"])
                if has_drums is None and "has_drums" in bp and bp["has_drums"] is not None:
                    has_drums = bool(bp["has_drums"])
                used_bp = os.path.abspath(cand)
                break
            except Exception:
                pass
    is_dark = bool(dark_flag) or (valence is not None and valence < -0.2)
    if has_drums is None:
        has_drums = profile != "classical_ambient"
    return valence, is_dark, bool(has_drums), used_bp


def _extract_song_metadata_for_fingerprint(project_path: str, blueprint_used: str | None, fallback_bpm: float) -> dict[str, Any]:
    bp_obj: dict[str, Any] = {}
    cand_paths = []
    if blueprint_used and os.path.isfile(blueprint_used):
        cand_paths.append(blueprint_used)
    proj_dir = os.path.dirname(os.path.abspath(os.path.expanduser(project_path)))
    default_bp = os.path.join(proj_dir, "song_blueprint.json")
    if os.path.isfile(default_bp) and default_bp not in cand_paths:
        cand_paths.append(default_bp)
    for p in cand_paths:
        try:
            with open(p, encoding="utf-8") as fh:
                bp_obj = json.load(fh)
            if isinstance(bp_obj, dict):
                break
        except Exception:
            pass

    eff_bpm = float(bp_obj.get("bpm") or fallback_bpm or 132.0)
    hc = bp_obj.get("harmonic_core") or {}
    key = str(hc.get("key") or bp_obj.get("key") or "")
    deg = hc.get("degree_progression") or bp_obj.get("degree_progression") or ""

    if os.path.isfile(os.path.expanduser(project_path)):
        try:
            with open(os.path.expanduser(project_path), encoding="utf-8") as fh:
                gdata = json.load(fh)
            if isinstance(gdata, dict):
                if not bp_obj.get("bpm") and gdata.get("tempo"):
                    eff_bpm = float(gdata["tempo"])
                if not key and (gdata.get("key") or gdata.get("scale")):
                    key = str(gdata.get("key") or gdata.get("scale"))
        except Exception:
            pass

    return {
        "project_name": os.path.basename(proj_dir) or os.path.splitext(os.path.basename(project_path))[0],
        "project_dir": proj_dir,
        "bpm": eff_bpm,
        "key": key,
        "degree_progression": deg,
    }


def render_and_evaluate(
    project: str,
    out_wav: str,
    *,
    profile: str = "trap_rap",
    range_bars: str = "",
    target_lufs: float | None = None,
    bpm: float = 132.0,
    beats_per_bar: float = 4.0,
    sections: str = "",
    blueprint: str = "",
    valence_arg: float | None = None,
    dark_flag: bool = False,
    has_drums_arg: bool | None = None,
    render_mp3: bool = False,
    use_neural: bool = True,
    profiles_path: str = DEFAULT_PROFILES_PATH,
    contract_path: str = "",
    run_macro_audit: bool = True,
    stems_dir: str = "",
    export_fingerprint: bool = True,
    approve_taste: bool = False,
    taste_memory_path: str = "",
    record_taste: bool = True,
) -> dict[str, Any]:
    """Renders `.gridline` (optionally scoped by bar range) and computes full Quality Gate report."""
    gate_cfg = dict(GENRE_GATES[profile])
    # v3.0: config/mix_profiles.json is the single source of truth — overlay genre values
    _ps = profile_settings(profile)
    _gate_overlay = {
        "low_end_max": "low_end_max_pct", "low_mids_max": "low_mids_max_pct",
        "air_min": "air_min_pct", "dark_air_min": "air_min_dark_pct",
        "sharp_max": "sharpness_max", "dark_sharp_max": "sharpness_max_dark",
        "arc_min": "dynamic_arc_min_db", "default_lufs": "target_lufs",
    }
    for _gk, _ck in _gate_overlay.items():
        if _ck in _ps:
            gate_cfg[_gk] = float(_ps[_ck])
    proj_dir = os.path.dirname(os.path.abspath(os.path.expanduser(project)))
    contract_obj = load_multi_scale_contract(contract_path=contract_path, project_dir=proj_dir)
    s5_ctx = (contract_obj or {}).get("s5_context") or {}

    valence, is_dark, has_drums, blueprint_used = _resolve_blueprint_metadata(
        project_path=project,
        blueprint_path=blueprint,
        dark_flag=dark_flag,
        valence_arg=valence_arg,
        has_drums_arg=has_drums_arg,
        profile=profile,
    )
    if contract_obj is not None and has_drums_arg is None and "has_drums" in contract_obj:
        has_drums = bool(contract_obj["has_drums"])

    air_min = float(s5_ctx["air_min"]) if s5_ctx.get("air_min") is not None else (
        gate_cfg["dark_air_min"] if is_dark else gate_cfg["air_min"]
    )
    sharp_max = float(s5_ctx["sharp_max"]) if s5_ctx.get("sharp_max") is not None else (
        gate_cfg["dark_sharp_max"] if is_dark else gate_cfg["sharp_max"]
    )
    low_mids_max = float(s5_ctx["low_mids_max"]) if s5_ctx.get("low_mids_max") is not None else gate_cfg["low_mids_max"]
    pink_slope_range = s5_ctx.get("pink_slope_range")
    eff_target_lufs = (
        target_lufs
        if target_lufs is not None
        else (float(s5_ctx["target_lufs"]) if s5_ctx.get("target_lufs") is not None else gate_cfg["default_lufs"])
    )

    abs_out = os.path.abspath(out_wav)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    cmd = ["gridline", "render", project, "-o", abs_out, "--bits", "24"]
    if range_bars:
        cmd.extend(["--scope", f"range:{range_bars}"])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"gridline render failed ({proc.returncode}): {proc.stderr or proc.stdout}")

    if render_mp3:
        mp3_out = os.path.splitext(abs_out)[0] + ".mp3"
        cmd_mp3 = ["gridline", "render", project, "-o", mp3_out, "--format", "mp3", "--bitrate", "320"]
        if range_bars:
            cmd_mp3.extend(["--scope", f"range:{range_bars}"])
        subprocess.run(cmd_mp3, capture_output=True, text=True, timeout=180, check=True)

    base = _analyze_file(abs_out, target_lufs=eff_target_lufs)
    no_drums_ambient = (not has_drums) or (profile == "classical_ambient")
    if is_dark and base["spectral_bands"]["air"]["energy_pct"] >= air_min:
        base["recommendations"] = [
            r for r in base["recommendations"] if "Manque d'air dans les aigus" not in r
        ] or ["Mix équilibré (mode sombre) : loudness, True Peak, spectre et phase dans les cibles."]
    if no_drums_ambient:
        base["recommendations"] = [
            r
            for r in base["recommendations"]
            if not any(kw in r.lower() for kw in ("sous-grave", "sub", "kick", "sidechain"))
        ] or ["Mix acoustique/ambient équilibré : dynamique, clarté et image stéréo dans les cibles."]

    data, sr = sf.read(abs_out, dtype="float64", always_2d=True)
    deep = run_deep_audit(
        audio_path=abs_out,
        base_metrics=base,
        data=data,
        sr=sr,
        target_genre=profile,
        use_neural=use_neural,
        headphone_target="sony_xm6",
    )

    mono = data.mean(axis=1)
    bark24_measured = compute_bark24_distribution(mono, sr)
    spectral_match = compute_spectral_match(
        mono=mono,
        sr=sr,
        profile=profile,
        profiles_path=profiles_path,
        bark_24=bark24_measured,
    )
    if s5_ctx.get("bark24_target") and len(s5_ctx["bark24_target"]) == 24:
        b_meas = np.asarray(bark24_measured, dtype=np.float64)
        b_targ = np.asarray(s5_ctx["bark24_target"], dtype=np.float64)
        denom = float(np.linalg.norm(b_meas) * np.linalg.norm(b_targ))
        c_cos = round(float(np.dot(b_meas, b_targ) / denom), 4) if denom > 1e-12 else 0.0
        spectral_match["contract_bark24_cosine"] = c_cos
        spectral_match["reference_source"] = "multi_scale_contract"
    # v3.1 REF-04 : toujours renseigner le tier de référence utilisé (1=user audio, 2=corridor
    # versionné, 3=baseline psychoacoustique) dans le rapport final.
    if contract_obj:
        for _k in ("reference_tier", "reference_provenance"):
            if contract_obj.get(_k) is not None:
                spectral_match[_k] = contract_obj.get(_k)

    sec_lufs = {}
    if sections and not range_bars:
        meter = pyln.Meter(sr)
        bar_s = float(beats_per_bar) * 60.0 / float(bpm)
        for name, b0, b1 in parse_sections(sections):
            s0 = int((b0 - 1) * bar_s * sr)
            s1 = min(len(data), int(b1 * bar_s * sr))
            if s1 - s0 > int(0.45 * sr):
                sec_lufs[name] = round(float(meter.integrated_loudness(data[s0:s1])), 2)

    max_jump = round(float(np.max(np.abs(np.diff(mono)))) if len(mono) > 1 else 0.0, 4)

    sub_bass_pct = (
        float(base["spectral_bands"]["sub"]["energy_pct"])
        + float(base["spectral_bands"]["bass"]["energy_pct"])
    )
    # Low-end dominance cap: too much energy < 120 Hz reads as boomy/muddy, not "natural".
    low_end_max = float(gate_cfg.get("low_end_max", 42.0)) if not no_drums_ambient else 55.0
    low_end_share = sub_bass_pct
    gate_failures: list[str] = []
    # v3.1 SSOT-01: gate thresholds come from config/mix_profiles.json (single source)
    _lim_inv = limiter_invariants()
    _tp_limit = _lim_inv["ceiling_dbtp"] + float(get_invariant("true_peak_margin_db", 0.2))
    _lufs_tol = float(get_invariant("lufs_tolerance_db", 1.2))
    _le_corr_min = float(get_invariant("low_end_corr_min", 0.50))
    _lm_corr_min = float(get_invariant("low_mid_corr_min", 0.55))
    _bal_max = float(get_invariant("stereo_balance_max_db", 1.2))
    # Project-wide musical safety audit of the blueprint (root_note presence/correctness,
    # non-playable samples in melodic roles, key-fit): error-severity issues block the gate
    # so a dissonant/"etrange" track can never ship in a render silently.
    musical_safety: dict[str, Any] = {"checked": False, "issues": [], "errors": 0, "warnings": 0, "ok": True}
    bp_safety_path = blueprint_used or os.path.join(proj_dir, "song_blueprint.json")
    if bp_safety_path and os.path.isfile(bp_safety_path):
        try:
            with open(bp_safety_path, encoding="utf-8") as fh:
                bp_safety_obj = json.load(fh)
            cat_path = os.path.join(proj_dir, "samples_catalog.json")
            musical_safety = summarize_musical_safety(
                bp_safety_obj if isinstance(bp_safety_obj, dict) else {},
                catalog_path=cat_path if os.path.isfile(cat_path) else None,
            )
            musical_safety["blueprint"] = bp_safety_path
        except Exception as exc:
            musical_safety = {"checked": False, "error": str(exc), "issues": [], "errors": 0, "warnings": 0, "ok": True}
    for _iss in musical_safety.get("issues") or []:
        if _iss.get("severity") == "error":
            gate_failures.append(f"musical_safety[{_iss.get('code')}] track={_iss.get('track')} role={_iss.get('role')}")
    if base["clipped_samples"] > 0:
        gate_failures.append(f"clipped_samples={base['clipped_samples']} > 0")
    if base["true_peak_dbtp"] > _tp_limit:
        gate_failures.append(f"true_peak_dbtp={base['true_peak_dbtp']} > {_tp_limit} (ceiling+margin from config)")
    if abs(base["integrated_lufs"] - eff_target_lufs) > _lufs_tol:
        gate_failures.append(f"lufs={base['integrated_lufs']} vs target={eff_target_lufs} (tol {_lufs_tol})")

    low_end_corr = float(base["stereo"]["correlation_low_end_150hz"])
    if no_drums_ambient:
        if sub_bass_pct >= 5.0 and low_end_corr < _le_corr_min * 0.4:
            gate_failures.append(f"low_end_corr={low_end_corr} < {_le_corr_min * 0.4:.2f}")
    else:
        if low_end_corr < _le_corr_min:
            gate_failures.append(f"low_end_corr={low_end_corr} < {_le_corr_min}")

    lm_corr = deep["room_and_transients"]["low_mid_200_800hz_correlation"]
    if lm_corr < _lm_corr_min:
        gate_failures.append(f"low_mid_corr={lm_corr} < {_lm_corr_min}")
    stereo_img = deep.get("stereo_image") or {}
    if stereo_img:
        bal_db = float(stereo_img.get("stereo_balance_db", 0.0))
        if abs(bal_db) > _bal_max:
            gate_failures.append(f"stereo_balance={bal_db:+.2f}dB > {_bal_max}dB")
        w_db = float(stereo_img.get("mid_high_side_to_mid_db", 0.0))
        w_lo, w_hi = (float(v) for v in (stereo_img.get("mid_high_width_range_db") or [-14.0, -1.5]))
        if w_db < w_lo:
            gate_failures.append(f"stereo_width={w_db}dB < {w_lo}dB")
        elif w_db > w_hi:
            gate_failures.append(f"stereo_width={w_db}dB > {w_hi}dB")
    sharp = deep["psychoacoustics"].get("sharpness_din_acum", 0.0)
    if sharp >= sharp_max:
        gate_failures.append(f"sharpness={sharp} >= {sharp_max}")
    if deep["glitches_and_artifacts"]["micro_glitch_or_crackle_events"] > 0:
        gate_failures.append(f"micro_glitches={deep['glitches_and_artifacts']['micro_glitch_or_crackle_events']} > 0")
    if no_drums_ambient:
        severe_res = [
            r
            for r in deep["third_octave_resonances"]
            if 150.0 <= float(r.get("freq_hz", 0.0)) <= 6000.0
            and float(r.get("local_excess_db", 0.0)) >= 13.0
        ]
        if severe_res:
            gate_failures.append(f"resonances={[r['freq_hz'] for r in severe_res]}")
    elif deep["third_octave_resonances"]:
        gate_failures.append(f"resonances={[r['freq_hz'] for r in deep['third_octave_resonances']]}")
    if base["spectral_bands"]["low_mids"]["energy_pct"] > low_mids_max:
        gate_failures.append(f"low_mids={base['spectral_bands']['low_mids']['energy_pct']}% > {low_mids_max}%")
    if low_end_share > low_end_max:
        gate_failures.append(f"low_end_share={low_end_share:.1f}% > {low_end_max}% (boomy/muddy)")
    if base["spectral_bands"]["air"]["energy_pct"] < air_min:
        gate_failures.append(f"air={base['spectral_bands']['air']['energy_pct']}% < {air_min}%")
    tune_ct = abs(float(deep["psychoacoustics"].get("tuning_cents_offset", 0.0)))
    _tune_inv = tuning_invariants()
    # v3.0: on the full master, vibrato/expressive tracks may oscillate +-12 ct around the
    # juste note -> only the expressive PEAK cap applies at master level. The strict
    # foundation threshold (<= 6.0 ct) is enforced on stable Bass/Chords stems below.
    if tune_ct > _tune_inv["expressive_peak_ct"]:
        gate_failures.append(
            f"tuning_offset={tune_ct}ct > {_tune_inv['expressive_peak_ct']}ct (master peak cap; "
            f"foundation stems must stay <= {_tune_inv['foundation_max_ct']}ct)"
        )
    if len(sec_lufs) >= 2:
        arc_db = round(max(sec_lufs.values()) - min(sec_lufs.values()), 2)
        if arc_db < gate_cfg["arc_min"]:
            gate_failures.append(f"dynamic_arc={arc_db}dB < {gate_cfg['arc_min']}dB")

    # Phase 7 P3 — LRA (EBU R128), Bass Harmonic Translation & Dynamic Bark Masking
    lra_lu = compute_lra_ebu(data, sr)
    lra_min, lra_max = LRA_PROFILE_THRESHOLDS.get(profile, (3.5, 9.0))
    deep["psychoacoustics"]["lra_lu"] = lra_lu
    deep["psychoacoustics"]["lra_range"] = [lra_min, lra_max]
    if not range_bars and float(base["duration_seconds"]) >= 12.0 and lra_lu < lra_min:
        gate_failures.append(f"lra={lra_lu}LU < {lra_min}LU")

    eff_stems_dir = stems_dir.strip() or (
        os.path.join(proj_dir, "renders", "stems") if os.path.isdir(os.path.join(proj_dir, "renders", "stems")) else ""
    )
    bass_trans_res: dict[str, Any] | None = None
    bark_mask_res: dict[str, Any] | None = None
    stem_hier_res: dict[str, Any] | None = None
    if eff_stems_dir and os.path.isdir(eff_stems_dir):
        st_roles = _find_role_stems_in_dir(eff_stems_dir)
        # v3.0: strict tuning on stable foundation stems (Bass/808 + Chords), not on the
        # vibrato-bearing master — expressive vibrato +-12 ct is NOT an out-of-tune render.
        try:
            from sample_prep_lib import measure_fft_tuning
            _found_stems = list(st_roles.get("bass") or []) + list(st_roles.get("chords") or []) + list(st_roles.get("harmony_list") or [])
            for _stem in _found_stems[:2]:
                _ti = measure_fft_tuning(_stem)
                _ct = abs(float(_ti.get("harmonic_tuning_ct") if _ti.get("harmonic_tuning_ct") is not None else (_ti.get("fft_cents_offset") or 0.0)))
                if _ct > _tune_inv["foundation_max_ct"]:
                    gate_failures.append(f"tuning_offset_stem={_ct}ct > {_tune_inv['foundation_max_ct']}ct on {os.path.basename(_stem)}")
        except Exception:
            pass
        if st_roles.get("bass"):
            bass_trans_res = analyze_bass_harmonic_translation(st_roles["bass"], sr=sr)
            if bass_trans_res.get("verdict") == "FIX":
                gate_failures.append(
                    f"bass_translation_r_trans={bass_trans_res['r_trans_db']}dB < -14.0dB (f0={bass_trans_res['f0_hz']}Hz)"
                )
        if st_roles.get("lead_list") and st_roles.get("harmony_list"):
            from pathlib import Path as _Path
            l_arr, l_sr = _load_and_sum_stems([_Path(p) for p in st_roles["lead_list"]], target_sr=sr)
            h_arr, h_sr = _load_and_sum_stems([_Path(p) for p in st_roles["harmony_list"]], target_sr=l_sr)
            bark_mask_res = analyze_dynamic_bark_masking((l_arr, l_sr), (h_arr, h_sr), sr=l_sr)
            if bark_mask_res.get("verdict") == "FIX":
                gate_failures.append(
                    f"bark_masking_lead_harmony={bark_mask_res['masked_ratio']} > 0.25 @ {bark_mask_res['f_mask_hz']}Hz"
                )
        if any(fn.lower().endswith(".wav") for fn in os.listdir(eff_stems_dir)):
            stem_hier_res = analyze_stem_hierarchy(
                stems_dir=eff_stems_dir,
                has_drums=has_drums,
                mix_profile=profile,
            )
            if isinstance(stem_hier_res, dict) and not stem_hier_res.get("error"):
                if not (stem_hier_res.get("checks") or {}).get("bounded_hierarchy_ok", True):
                    for rec in stem_hier_res.get("recommendations") or []:
                        gate_failures.append(
                            f"stem_balance_{rec.get('metric', 'pair')}={rec.get('measured_diff_db')}dB"
                        )

    macro_res: dict[str, Any] | None = None
    if run_macro_audit and not range_bars:
        try:
            macro_res = analyze_macro_structure(
                abs_out,
                bpm=bpm,
                beats_per_bar=max(1, int(round(beats_per_bar))),
                sections=sections or None,
                blueprint=blueprint_used or None,
                contract=contract_obj,
                stems_dir_or_map=eff_stems_dir or None,
                project_dir=proj_dir,
                has_drums=has_drums,
            )
            if contract_obj is not None:
                if macro_res["clones_8bar"]["verdict"] == "FAIL":
                    gate_failures.append(
                        f"macro_8bar_clone={macro_res['clones_8bar']['max_similarity']}"
                    )
                if macro_res["cognitive_ceiling"]["verdict"] == "FAIL":
                    gate_failures.append("macro_cognitive_ceiling_exceeded")
        except Exception as exc:
            macro_res = {"verdict": "SKIP", "error": str(exc)}

    gate_passed = len(gate_failures) == 0
    fp_path = ""
    taste_res: dict[str, Any] | None = None
    if export_fingerprint and gate_passed and not range_bars:
        rms_all = float(np.sqrt(np.mean(mono ** 2))) + 1e-12
        pk_all = float(np.max(np.abs(mono))) + 1e-12
        crest_db = round(20.0 * math.log10(pk_all / rms_all), 2)
        song_meta = _extract_song_metadata_for_fingerprint(project, blueprint_used, bpm)
        fp_obj = build_master_fingerprint(
            project_name=song_meta["project_name"],
            project_dir=song_meta["project_dir"],
            mix_profile=profile,
            bpm=song_meta["bpm"],
            key=song_meta["key"],
            degree_progression=song_meta["degree_progression"],
            integrated_lufs=base["integrated_lufs"],
            true_peak_dbtp=base["true_peak_dbtp"],
            crest_factor_db=crest_db,
            pink_slope_db_per_oct=float(spectral_match.get("pink_slope_db_per_oct", -4.0)),
            bark24=bark24_measured,
            duration_s=base["duration_seconds"],
        )
        renders_dir = os.path.join(proj_dir, "renders")
        os.makedirs(renders_dir, exist_ok=True)
        fp_path = os.path.join(renders_dir, "master_fingerprint.json")
        with open(fp_path, "w", encoding="utf-8") as fh:
            json.dump(fp_obj, fh, indent=2, ensure_ascii=False)
        if record_taste and _should_record_taste(proj_dir, taste_memory_path, approve_taste):
            pq_score = (((deep.get("neural_perception") or {}).get("audiobox_aesthetics") or {}).get("PQ"))
            taste_res = record_validated_taste(
                project_dir=proj_dir,
                blueprint_path=blueprint_used or "",
                fingerprint_path=fp_path,
                approved_by_user=bool(approve_taste),
                taste_path=taste_memory_path or None,
                pq_score=None if pq_score is None else float(pq_score),
                extra_metrics={"lra_lu": lra_lu},
            )

    return {
        "profile": profile,
        "has_drums": has_drums,
        "dark_mode": is_dark,
        "valence": valence,
        "blueprint_used": blueprint_used,
        "musical_safety": musical_safety,
        "contract_used": bool(contract_obj is not None),
        "effective_thresholds": {
            "air_min": air_min,
            "sharp_max": sharp_max,
            "low_mids_max": low_mids_max,
            "target_lufs": eff_target_lufs,
            "pink_slope_range": pink_slope_range,
            "arc_min": gate_cfg["arc_min"],
            "lra_range": [lra_min, lra_max],
        },
        "target_lufs": eff_target_lufs,
        "out_wav": abs_out,
        "duration_s": base["duration_seconds"],
        "integrated_lufs": base["integrated_lufs"],
        "lra_lu": lra_lu,
        "true_peak_dbtp": base["true_peak_dbtp"],
        "clipped_samples": base["clipped_samples"],
        "max_sample_jump": max_jump,
        "bands_pct": {k: v["energy_pct"] for k, v in base["spectral_bands"].items()},
        "stereo": base["stereo"],
        "stereo_image": stereo_img,
        "spectral_match": spectral_match,
        "recommendations": base["recommendations"],
        "psychoacoustics": deep["psychoacoustics"],
        "p3_diagnostics": {
            "lra_lu": lra_lu,
            "lra_range": [lra_min, lra_max],
            "bass_harmonic_translation": bass_trans_res,
            "dynamic_bark_masking": bark_mask_res,
            "stem_hierarchy": stem_hier_res,
        },
        "stem_hierarchy": stem_hier_res,
        "micro_glitches": deep["glitches_and_artifacts"]["micro_glitch_or_crackle_events"],
        "resonances_third_oct": deep["third_octave_resonances"],
        "low_mid_corr": lm_corr,
        "neural": deep.get("neural_perception"),
        "section_lufs": sec_lufs,
        "macro_structure": macro_res,
        "master_fingerprint_path": fp_path or None,
        "taste_memory": taste_res,
        "quality_gate": {
            "passed": gate_passed,
            "failures": gate_failures,
        },
    }


def _find_mixer_targets(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    by_idx = {int(t["mixer"]): t for t in tracks if "mixer" in t}
    bus_music = next((t for t in tracks if "bus music" in (t.get("name") or "").lower()), by_idx.get(19))
    bus_bass = next(
        (t for t in tracks if any(k in (t.get("name") or "").lower() for k in ("bus bass", "bus 808"))),
        by_idx.get(18),
    )
    fx_room = next(
        (t for t in tracks if any(k in (t.get("name") or "").lower() for k in ("fx room", "reverb"))),
        by_idx.get(20),
    )
    if bus_music is None:
        bus_music = by_idx.get(0)
    return {
        "by_idx": by_idx,
        "master": by_idx.get(0),
        "bus_music": bus_music,
        "bus_bass": bus_bass,
        "fx_room": fx_room,
    }


def _param_lookup(mcp: GridlineMCP, mixer_idx: int, slot_idx: int) -> dict[str, float]:
    res = mcp.call("get_parameters", {"mixer": int(mixer_idx), "slot": int(slot_idx)})
    out: dict[str, float] = {}
    for p in res.get("params", []):
        name = p.get("name")
        if name and "value" in p:
            out[str(name)] = float(p["value"])
    return out


def _slots_for_plugin(track: dict[str, Any] | None, plugin_id: str) -> list[int]:
    if not track:
        return []
    return [
        int(e["slot"])
        for e in track.get("effects", [])
        if e.get("plugin") == plugin_id and "slot" in e
    ]


def _insert_tracks(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return instrument insert tracks (exclude Master, buses 17..22 and FX buses)."""
    out: list[dict[str, Any]] = []
    for t in tracks:
        try:
            m_i = int(t.get("mixer", -1))
        except Exception:
            continue
        if m_i <= 0 or m_i in (17, 18, 19, 20, 21, 22):
            continue
        nm = str(t.get("name") or "")
        if any(k in nm.lower() for k in ("bus ", "fx room", "fx delay", "fx hall")):
            continue
        out.append(t)
    return out


def _fix_stereo_balance_mcp(
    mcp: GridlineMCP,
    tracks: list[dict[str, Any]],
    balance_db: float,
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Phase 8 Q3.2 `FIX_STEREO_BALANCE`: nudge 1-2 panoramic inserts toward the quieter
    channel by `+/-0.10..0.16 pan` (max 2 passes, bounded to `[-1, 1]`).
    """
    passes = int(state.get("stereo_balance_passes", 0))
    if passes >= 2:
        return {"applied": False, "code": "FIX_STEREO_BALANCE", "reason": "max 2 passes reached"}
    direction = 1.0 if balance_db > 0.0 else -1.0  # +1: left louder -> move content right
    step = 0.10 if abs(balance_db) < 3.0 else 0.16
    candidates: list[tuple[float, dict[str, Any]]] = []
    for t in _insert_tracks(tracks):
        role = normalize_role_for_gain_staging(str(t.get("name") or ""))
        if role not in ("chords_pad", "hihat_perc", "counter_melody", "ear_candy_fx", "lead"):
            continue
        pan = float(t.get("pan", 0.0) or 0.0)
        # Prefer tracks already offset opposite to the correction, then centred tracks
        score = (1.0 if pan * direction < 0.0 else 0.5) - abs(pan)
        candidates.append((score, t))
    candidates.sort(key=lambda it: it[0], reverse=True)
    edits: list[dict[str, Any]] = []
    changed: list[str] = []
    for _s, t in candidates[:2]:
        pan = float(t.get("pan", 0.0) or 0.0)
        new_pan = round(clamp(pan + direction * step, -1.0, 1.0), 3)
        if abs(new_pan - pan) < 1e-4:
            continue
        edits.append({"tool": "edit_mixer_track", "arguments": {"mixer": int(t["mixer"]), "pan": new_pan}})
        changed.append(str(t.get("name") or t.get("mixer")))
    if not edits:
        return {"applied": False, "code": "FIX_STEREO_BALANCE", "reason": "no pannable insert"}
    mcp.batch(edits)
    state["stereo_balance_passes"] = passes + 1
    return {
        "rule": "stereo_balance",
        "applied": True,
        "code": "FIX_STEREO_BALANCE",
        "balance_db": round(balance_db, 2),
        "tracks": changed,
        "pan_delta": round(direction * step, 3),
        "passes": passes + 1,
    }


def _widen_mix_mcp(
    mcp: GridlineMCP,
    tracks: list[dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Phase 8 Q3.2 `WIDEN_HARMONY_OR_PERC`: complementary `+/-0.22` pans on Chords/Pad and
    Counter/Hats/Perc, plus `+0.06` `FX Hall (22)` send on harmonic layers (once).
    """
    if state.get("widen_applied"):
        return {"applied": False, "code": "WIDEN_HARMONY_OR_PERC", "reason": "already applied"}
    inserts = _insert_tracks(tracks)
    edits: list[dict[str, Any]] = []
    param_changes: list[tuple[str, Any]] = []
    changed: list[str] = []
    wants_hall = any(int(t.get("mixer", -1)) == 22 for t in tracks)
    for t in inserts:
        role = normalize_role_for_gain_staging(str(t.get("name") or ""))
        if wants_hall and role in ("chords_pad", "counter_melody"):
            edits.append({"tool": "set_send", "arguments": {"mixer": int(t["mixer"]), "to": 22, "level": 0.22}})
            changed.append(str(t.get("name") or t.get("mixer")))
        elif role == "hihat_perc":
            changed.append(str(t.get("name") or t.get("mixer")))
    # Deterministic mid/high-only widener on BUS Music (legacy engine, keeps < 250 Hz mono)
    bus_music = next((t for t in tracks if int(t.get("mixer", -1)) == 19), None)
    if bus_music is not None:
        st_slots = _slots_for_plugin(bus_music, "gl.stereo")
        if not st_slots:
            mcp.call("add_plugin", {"mixer": 19, "plugin": "gl.stereo"})
            tracks = mcp.call("get_mixer", {}).get("tracks", [])
            bus_music = next((t for t in tracks if int(t.get("mixer", -1)) == 19), None)
            st_slots = _slots_for_plugin(bus_music, "gl.stereo") if bus_music else []
        if st_slots:
            sl = int(st_slots[-1])
            param_changes.extend([
                (f"mixer/19/slot/{sl}/Reference engine", 0),
                (f"mixer/19/slot/{sl}/Width", 2.0),
                (f"mixer/19/slot/{sl}/Side level", 6.0),
                (f"mixer/19/slot/{sl}/Dephase", 0.0),
                (f"mixer/19/slot/{sl}/Mono below", 250.0),
            ])
            changed.append("BUS Music gl.stereo")
    if not edits and not param_changes:
        return {"applied": False, "code": "WIDEN_HARMONY_OR_PERC", "reason": "no harmonic/perc insert"}
    if edits:
        mcp.batch(edits)
    if param_changes:
        mcp.set_params(param_changes)
    state["widen_applied"] = True
    return {"rule": "widen_stereo", "applied": True, "code": "WIDEN_HARMONY_OR_PERC", "tracks": changed}


def _narrow_mix_mcp(
    mcp: GridlineMCP,
    tracks: list[dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Phase 8 Q3.2 `NARROW_EXCESSIVE_SIDE`: cap gl.stereo widths and pull extreme pans inward."""
    if state.get("narrow_applied"):
        return {"applied": False, "code": "NARROW_EXCESSIVE_SIDE", "reason": "already applied"}
    param_changes: list[tuple[str, Any]] = []
    edits: list[dict[str, Any]] = []
    for t in tracks:
        m_i = int(t.get("mixer", -1))
        for slot in _slots_for_plugin(t, "gl.stereo"):
            if m_i in (17, 19, 20, 21, 22):
                param_changes.extend([
                    (f"mixer/{m_i}/slot/{slot}/Reference engine", 1),
                    (f"mixer/{m_i}/slot/{slot}/Width", 0.80),
                    (f"mixer/{m_i}/slot/{slot}/Dephase", 0.0),
                    (f"mixer/{m_i}/slot/{slot}/Side level", 0.0),
                ])
        if m_i in (17, 18, 19, 20, 21, 22, 0):
            continue
        pan = float(t.get("pan", 0.0) or 0.0)
        if abs(pan) > 0.35:
            new_pan = round(math.copysign(max(0.20, abs(pan) - 0.12), pan), 3)
            edits.append({"tool": "edit_mixer_track", "arguments": {"mixer": m_i, "pan": new_pan}})
    if param_changes:
        mcp.set_params(param_changes)
    if edits:
        mcp.batch(edits)
    if not param_changes and not edits:
        return {"applied": False, "code": "NARROW_EXCESSIVE_SIDE", "reason": "no stereo width target"}
    state["narrow_applied"] = True
    return {"rule": "narrow_stereo", "applied": True, "code": "NARROW_EXCESSIVE_SIDE"}


def _apply_autofix_mutations(
    project_path: str,
    summary: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Applies deterministic mixer mutations per Axe 3 §3.2 using `GridlineMCP` & `gl_dsl`:
      1) Clip / True Peak > -1.0 dBTP -> Master `gl.limiter` look-ahead + safety + Ceiling -= 0.4 dB (max 2 drops)
      2) |LUFS - target| > 1.2 -> Master `gl.limiter Gain += clamp(target - lufs, -4, +4)` (blocked if true_peak > -1.3 on boost)
      3) resonances -> surgical `gl.eq` on BUS Music (`Peak -4.0 dB @ F_i`, `Q=2.8 -> width=0.22`, max 3 bands in [200, 8000])
      4) low_mids > max -> BUS Music `gl.eq` HPF = max(HPF, 135) + Peak -3 dB @ 280 Hz (`Q=1.2`), once only
      5) sharpness >= max -> BUS Music `gl.eq` Peak -3 dB @ 3200 Hz (`Q=1.4`) + LPF *= 0.85 (floor 7.5 kHz)
      6) low_end_corr < 0.50 -> BUS Bass `gl.stereo Width = 0`, pan = 0 on Kick/Bass, FX Room send = 0
    """
    abs_proj = os.path.abspath(project_path)
    mcp = GridlineMCP(abs_proj, os.path.dirname(abs_proj))
    mutations: list[dict[str, Any]] = []
    param_changes: list[tuple[str, Any]] = []

    try:
        mx = mcp.call("get_mixer", {})
        tracks = mx.get("tracks", [])
        targets = _find_mixer_targets(tracks)
        master_track = targets["master"]
        bus_music_track = targets["bus_music"]
        bus_bass_track = targets["bus_bass"]
        fx_room_track = targets["fx_room"]

        failures = summary["quality_gate"]["failures"]
        clipped = int(summary["clipped_samples"])
        true_peak = float(summary["true_peak_dbtp"])
        lufs = float(summary["integrated_lufs"])
        target_lufs = float(summary["target_lufs"])
        low_mids_max = float(summary["effective_thresholds"]["low_mids_max"])
        sharp_max = float(summary["effective_thresholds"]["sharp_max"])
        low_mids_val = float(summary["bands_pct"]["low_mids"])
        sharp_val = float(summary["psychoacoustics"].get("sharpness_din_acum", 0.0))
        low_end_corr = float(summary["stereo"]["correlation_low_end_150hz"])
        resonances = [
            r for r in (summary.get("resonances_third_oct") or [])
            if 200.0 <= float(r.get("freq_hz", 0.0)) <= 8000.0
        ]

        # Rules 1 & 2: Master gl.limiter
        need_tp_fix = clipped > 0 or true_peak > _tp_limit
        need_lufs_fix = abs(lufs - target_lufs) > 1.2
        if (need_tp_fix or need_lufs_fix) and master_track is not None:
            lim_slots = _slots_for_plugin(master_track, "gl.limiter")
            if not lim_slots:
                mcp.call("add_plugin", {"mixer": 0, "plugin": "gl.limiter"})
                tracks = mcp.call("get_mixer", {}).get("tracks", [])
                targets = _find_mixer_targets(tracks)
                master_track = targets["master"]
                lim_slots = _slots_for_plugin(master_track, "gl.limiter")
            if lim_slots:
                lim_slot = lim_slots[-1]
                cur_lim = _param_lookup(mcp, 0, lim_slot)
                cur_gain = cur_lim.get("Gain", 0.0)
                cur_ceil = cur_lim.get("Ceiling", -2.2)

                eff_ceil = cur_ceil
                if need_tp_fix:
                    if state.get("ceiling_drops", 0) < 2:
                        eff_ceil = max(-3.0, round(cur_ceil - 0.4, 2))
                        state["ceiling_drops"] = state.get("ceiling_drops", 0) + 1
                    lim_dict = limiter_master(gain_db=cur_gain, ceiling_db=eff_ceil)
                    param_changes.extend(to_mixer_params(0, lim_slot, lim_dict))
                    mutations.append(
                        {
                            "rule": "true_peak_or_clip",
                            "target": f"mixer/0/slot/{lim_slot}",
                            "old_ceiling_db": cur_ceil,
                            "new_ceiling_db": eff_ceil,
                            "ceiling_drops": state["ceiling_drops"],
                        }
                    )

                if need_lufs_fix:
                    raw_delta = clamp(target_lufs - lufs, -4.0, 4.0)
                    if true_peak > -1.3 and raw_delta > 0.0:
                        mutations.append(
                            {
                                "rule": "lufs_adjust",
                                "skipped": True,
                                "reason": f"true_peak={true_peak:.2f} > -1.3 dBTP safety guard",
                            }
                        )
                    else:
                        new_gain = round(clamp(cur_gain + raw_delta, -36.0, 24.0), 2)
                        param_changes.append((f"mixer/0/slot/{lim_slot}/Gain", new_gain))
                        mutations.append(
                            {
                                "rule": "lufs_adjust",
                                "target": f"mixer/0/slot/{lim_slot}/Gain",
                                "old_gain_db": cur_gain,
                                "new_gain_db": new_gain,
                                "delta_db": round(new_gain - cur_gain, 2),
                            }
                        )

        # BUS Music EQ rules (3, 4, 5)
        if bus_music_track is not None:
            bm_idx = int(bus_music_track["mixer"])
            bm_eq_slots = _slots_for_plugin(bus_music_track, "gl.eq")

            # Rule 3: surgical resonances (-4 dB @ F_i, Q=2.8 -> width=0.22) with the v3.0
            # notch budget from config: beyond max_eq_notches_total the fix is SWAP_SAMPLE,
            # not more comb-like EQ (a sample needing 4 surgical cuts is the wrong timbre).
            if resonances:
                _per_track_cap, _total_cap = notch_caps()
                _already = int(state.get("resonance_notches_total", 0))
                if _already >= _total_cap:
                    mutations.append({
                        "rule": "resonances_surgical_eq",
                        "skipped": True,
                        "action": "SWAP_SAMPLE",
                        "reason": f"notch budget exhausted ({_already}/{_total_cap} total) — stop EQ, swap the sample",
                    })
                    top_res = []
                else:
                    top_res = sorted(
                        resonances,
                        key=lambda r: float(r.get("local_excess_db", r.get("excess_db", 0.0))),
                        reverse=True,
                    )[: max(0, _total_cap - _already)]
                if len(bm_eq_slots) >= 2:
                    surg_slot = bm_eq_slots[-1]
                else:
                    mcp.call("add_plugin", {"mixer": bm_idx, "plugin": "gl.eq"})
                    tracks = mcp.call("get_mixer", {}).get("tracks", [])
                    targets = _find_mixer_targets(tracks)
                    bus_music_track = targets["bus_music"]
                    bm_eq_slots = _slots_for_plugin(bus_music_track, "gl.eq")
                    surg_slot = bm_eq_slots[-1]

                surg_bands = [
                    eq_band(
                        idx,
                        "peak",
                        float(r["freq_hz"]),
                        -min(12.0, max(6.0, float(r.get("local_excess_db", 5.0)) + 1.0)),
                        q=2.8,
                    )
                    for idx, r in enumerate(top_res, start=1)
                ]
                if top_res:
                    param_changes.extend(eq_params(bm_idx, surg_bands, slot=surg_slot, quality=0))
                    for unused_idx in range(len(top_res) + 1, 8):
                        param_changes.append((f"mixer/{bm_idx}/slot/{surg_slot}/Band {unused_idx} enabled", 0))
                    state["resonance_notches_total"] = _already + len(top_res)
                    mutations.append(
                        {
                            "rule": "resonances_surgical_eq",
                            "target": f"mixer/{bm_idx}/slot/{surg_slot}",
                            "freqs_hz": [float(r["freq_hz"]) for r in top_res],
                            "gain_db": [-min(12.0, max(6.0, float(r.get("local_excess_db", 5.0)) + 1.0)) for r in top_res],
                            "q": 2.8,
                            "notches_total": state["resonance_notches_total"],
                            "notch_cap_total": _total_cap,
                        }
                    )

            # Rules 4 & 5 on primary BUS Music EQ
            need_low_mids = (low_mids_val > low_mids_max) and not state.get("low_mids_applied", False)
            need_sharp = sharp_val >= sharp_max
            if need_low_mids or need_sharp:
                if not bm_eq_slots:
                    mcp.call("add_plugin", {"mixer": bm_idx, "plugin": "gl.eq"})
                    tracks = mcp.call("get_mixer", {}).get("tracks", [])
                    targets = _find_mixer_targets(tracks)
                    bus_music_track = targets["bus_music"]
                    bm_eq_slots = _slots_for_plugin(bus_music_track, "gl.eq")
                prim_slot = bm_eq_slots[0]
                cur_eq = _param_lookup(mcp, bm_idx, prim_slot)
                bm_bands: list[dict[str, Any]] = []

                if need_low_mids:
                    cur_hpf = cur_eq.get("Band 1 freq", 120.0)
                    new_hpf = round(max(cur_hpf, 135.0), 1)
                    bm_bands.append(eq_band(1, "hpf", new_hpf))
                    bm_bands.append(eq_band(2, "peak", 280.0, -3.0, q=1.2))
                    state["low_mids_applied"] = True
                    mutations.append(
                        {
                            "rule": "low_mids_clean",
                            "target": f"mixer/{bm_idx}/slot/{prim_slot}",
                            "hpf_hz": new_hpf,
                            "dip_280hz_db": -3.0,
                        }
                    )

                if need_sharp:
                    cur_lpf = cur_eq.get("Band 7 freq", 12500.0)
                    if cur_lpf <= 1000.0:
                        cur_lpf = 12500.0
                    new_lpf = round(max(7500.0, cur_lpf * 0.85), 1)
                    bm_bands.append(eq_band(6, "peak", 3200.0, -3.0, q=1.4))
                    bm_bands.append(eq_band(7, "lpf", new_lpf))
                    mutations.append(
                        {
                            "rule": "sharpness_tame",
                            "target": f"mixer/{bm_idx}/slot/{prim_slot}",
                            "dip_3200hz_db": -3.0,
                            "old_lpf_hz": cur_lpf,
                            "new_lpf_hz": new_lpf,
                        }
                    )

                if bm_bands:
                    param_changes.extend(eq_params(bm_idx, bm_bands, slot=prim_slot))

        # Rule 6b: low_end_share > max -> trim BUS Bass insert (max 2 passes, -1.5 dB each) + raise BUS Music HPF once
        low_end_share_val = float(summary.get("spectral_bands", {}).get("sub", {}).get("energy_pct", 0.0)) + float(
            summary.get("spectral_bands", {}).get("bass", {}).get("energy_pct", 0.0)
        )
        low_end_max_eff = float((summary.get("effective_thresholds") or {}).get("low_end_max", 42.0))
        if any(f.startswith("low_end_share=") for f in failures) and low_end_share_val > low_end_max_eff:
            if bus_bass_track is not None and state.get("bass_trims", 0) < 2:
                bb_idx = int(bus_bass_track["mixer"])
                cur_bb = float(bus_bass_track.get("volume", 1.0))
                new_bb = round(max(0.5, cur_bb * (10.0 ** (-1.5 / 20.0))), 4)
                param_changes.append((f"mixer/{bb_idx}/volume", new_bb))
                state["bass_trims"] = state.get("bass_trims", 0) + 1
                mutations.append({
                    "rule": "low_end_dominance_bass_trim",
                    "target": f"mixer/{bb_idx}/volume",
                    "old": cur_bb, "new": new_bb, "low_end_share": round(low_end_share_val, 1),
                })
            if bus_music_track is not None and not state.get("music_hpf_raised"):
                bm_idx = int(bus_music_track["mixer"])
                param_changes.extend(eq_params(bm_idx, [eq_band(1, "hpf", 160.0)], slot=2, quality=0))
                state["music_hpf_raised"] = True
                mutations.append({"rule": "low_end_dominance_music_hpf", "target": f"mixer/{bm_idx}/slot/2 Band 1", "hpf_hz": 160.0})

        # Rule 6: low_end_corr < 0.50 (idempotent mono low-end enforcement)
        if any(f.startswith("low_end_corr=") for f in failures):
            if bus_bass_track is not None:
                bb_idx = int(bus_bass_track["mixer"])
                st_slots = _slots_for_plugin(bus_bass_track, "gl.stereo")
                if not st_slots:
                    mcp.call("add_plugin", {"mixer": bb_idx, "plugin": "gl.stereo"})
                    tracks = mcp.call("get_mixer", {}).get("tracks", [])
                    targets = _find_mixer_targets(tracks)
                    bus_bass_track = targets["bus_bass"]
                    st_slots = _slots_for_plugin(bus_bass_track, "gl.stereo")
                if st_slots:
                    param_changes.append((f"mixer/{bb_idx}/slot/{st_slots[0]}/Width", 0.0))

            batch_mono: list[dict[str, Any]] = []
            fx_room_idx = int(fx_room_track["mixer"]) if fx_room_track is not None else None
            fixed_tracks: list[int] = []
            for t in tracks:
                m_i = int(t["mixer"])
                nm = (t.get("name") or "").lower()
                if m_i == (int(bus_bass_track["mixer"]) if bus_bass_track else -1) or any(
                    k in nm for k in ("kick", "808", "bass", "sub")
                ):
                    if m_i == 0:
                        continue
                    fixed_tracks.append(m_i)
                    batch_mono.append({"tool": "edit_mixer_track", "arguments": {"mixer": m_i, "pan": 0.0}})
                    if fx_room_idx is not None:
                        batch_mono.append({"tool": "set_send", "arguments": {"mixer": m_i, "to": fx_room_idx, "level": 0.0}})
            if batch_mono:
                mcp.batch(batch_mono)
            mutations.append(
                {
                    "rule": "low_end_mono_enforce",
                    "bus_bass": int(bus_bass_track["mixer"]) if bus_bass_track else None,
                    "centered_tracks": fixed_tracks,
                }
            )

        # Phase 7 P3 Rules: Dynamic Mirror EQ, LRA Bus Compression Relax, Bass Harmonic Translation
        p3 = summary.get("p3_diagnostics") or {}
        bark_m = p3.get("dynamic_bark_masking") or {}
        if (bark_m.get("verdict") == "FIX" and bark_m.get("f_mask_hz")) or any(
            f.startswith("bark_masking_lead_harmony=") for f in failures
        ):
            f_hz = float(bark_m.get("f_mask_hz") or 1850.0)
            m_res = _apply_mirror_eq_mcp(mcp, tracks, f_hz, gain_db=-3.5, q=1.8)
            if m_res.get("applied"):
                mutations.append(m_res)
                tracks = mcp.call("get_mixer", {}).get("tracks", [])

        if any(f.startswith("lra=") for f in failures):
            r_res = _relax_bus_compression_mcp(mcp, tracks)
            if r_res.get("applied"):
                mutations.append(r_res)
                tracks = mcp.call("get_mixer", {}).get("tracks", [])

        b_tr = p3.get("bass_harmonic_translation") or {}
        if (b_tr.get("verdict") == "FIX" and b_tr.get("f0_hz")) or any(
            f.startswith("bass_translation_r_trans=") for f in failures
        ):
            f0_v = float(b_tr.get("f0_hz") or 50.0)
            b_res = _boost_bass_harmonics_mcp(mcp, tracks, f0_v)
            if b_res.get("applied"):
                mutations.append(b_res)

        # Phase 8 Q2 Rules: ADJUST_TRACK_FADER from bounded stem hierarchy & section audibility
        fader_recs: list[dict[str, Any]] = []
        sh_res = p3.get("stem_hierarchy") or summary.get("stem_hierarchy") or {}
        for rec in sh_res.get("recommendations") or []:
            if isinstance(rec, dict) and rec.get("action") == "ADJUST_TRACK_FADER":
                fader_recs.append(rec)
        sec_aud = (summary.get("macro_structure") or {}).get("section_track_audibility") or {}
        for rec in sec_aud.get("recommendations") or []:
            if isinstance(rec, dict) and rec.get("code") == "ADJUST_TRACK_FADER":
                p = rec.get("params") or {}
                fader_recs.append({
                    "target_track": p.get("track", ""),
                    "target_role": p.get("role", ""),
                    "delta_db": p.get("delta_db", 0.0),
                })
        seen_fader_targets: set[str] = set()
        for frec in fader_recs:
            t_nm = str(frec.get("target_track") or "")
            t_rl = str(frec.get("target_role") or "")
            key_dedup = (t_nm or t_rl).lower()
            if not key_dedup or key_dedup in seen_fader_targets:
                continue
            seen_fader_targets.add(key_dedup)
            f_res = _adjust_track_fader_mcp(
                mcp,
                tracks,
                track_name=t_nm,
                role_hint=t_rl,
                delta_db=float(frec.get("delta_db") or 0.0),
                state=state,
            )
            if f_res.get("applied"):
                mutations.append(f_res)
                state["refresh_stems"] = True
                tracks = mcp.call("get_mixer", {}).get("tracks", [])

        # Phase 8 Q3.2 — Stereo image deterministic corrections
        s_img = summary.get("stereo_image") or {}
        bal_db = float(s_img.get("stereo_balance_db", 0.0))
        w_db = float(s_img.get("mid_high_side_to_mid_db", 0.0))
        w_lo, w_hi = (float(v) for v in (s_img.get("mid_high_width_range_db") or [-14.0, -1.5]))
        if abs(bal_db) > 1.2:
            b_res = _fix_stereo_balance_mcp(mcp, tracks, bal_db, state)
            if b_res.get("applied"):
                mutations.append(b_res)
                tracks = mcp.call("get_mixer", {}).get("tracks", [])
        if w_db < w_lo:
            w_res = _widen_mix_mcp(mcp, tracks, state)
            if w_res.get("applied"):
                mutations.append(w_res)
                tracks = mcp.call("get_mixer", {}).get("tracks", [])
        elif w_db > w_hi:
            n_res = _narrow_mix_mcp(mcp, tracks, state)
            if n_res.get("applied"):
                mutations.append(n_res)
                tracks = mcp.call("get_mixer", {}).get("tracks", [])

        if param_changes:
            mcp.set_params(param_changes)
        if param_changes or mutations:
            # Persist ALWAYS when any parameter changed: otherwise the next render pass is
            # bit-identical to the previous one (the "versions don't differ" bug).
            mcp.call("save_project", {})
    finally:
        mcp.close()

    return mutations


def _apply_mirror_eq_mcp(
    mcp: GridlineMCP,
    tracks: list[dict[str, Any]],
    freq_hz: float,
    *,
    gain_db: float = -3.5,
    q: float = 1.8,
    prefer_bus_music: bool = False,
) -> dict[str, Any]:
    targets = _find_mixer_targets(tracks)
    bm_track = targets["bus_music"]
    bm_idx = int(bm_track["mixer"]) if bm_track else 19

    # Target Chords/Pad/Harmony inserts so Lead stays untouched; fall back to BUS Music if none exist
    chords_tracks = [
        t for t in tracks
        if int(t["mixer"]) not in (0, 17, 18, 19, 20)
        and any(k in (t.get("name") or "").lower() for k in ("chord", "pad", "harmony"))
    ]
    target_mixers = (
        [int(t["mixer"]) for t in chords_tracks]
        if (chords_tracks and not prefer_bus_music)
        else [bm_idx]
    )

    by_idx = {int(t["mixer"]): t for t in tracks if "mixer" in t}
    p_changes: list[tuple[str, Any]] = []
    used_slots: list[dict[str, int]] = []
    for mx_i in target_mixers:
        tr = by_idx.get(mx_i)
        eq_slots = _slots_for_plugin(tr, "gl.eq")
        if not eq_slots:
            mcp.call("add_plugin", {"mixer": mx_i, "plugin": "gl.eq"})
            refreshed = mcp.call("get_mixer", {}).get("tracks", [])
            by_idx = {int(t["mixer"]): t for t in refreshed if "mixer" in t}
            eq_slots = _slots_for_plugin(by_idx.get(mx_i), "gl.eq")
        if not eq_slots:
            continue
        # On BUS Music, slot 2 is the primary EQ (slot 1 is gl.dist in setup_standard_buses); on inserts use last gl.eq
        slot_i = eq_slots[1] if (mx_i == bm_idx and len(eq_slots) >= 2) else eq_slots[-1]
        cur_eq = _param_lookup(mcp, mx_i, slot_i)
        band_idx = 4
        for cand_b in (4, 5, 3, 2, 6):
            # Band is free if not already active peak/hpf/lpf (or reuse band 4 if already at similar freq)
            bt = int(round(cur_eq.get(f"Band {cand_b} type", 0)))
            if bt == 0 or (cand_b == 4 and abs(cur_eq.get("Band 4 freq", 0.0) - freq_hz) < 120.0):
                band_idx = cand_b
                break
        p_changes.extend(eq_params(mx_i, [eq_band(band_idx, "peak", float(freq_hz), float(gain_db), q=float(q))], slot=slot_i, quality=0))
        used_slots.append({"mixer": mx_i, "slot": slot_i, "band": band_idx})

    if p_changes:
        mcp.set_params(p_changes)
    return {
        "applied": bool(used_slots),
        "code": "APPLY_DYNAMIC_MIRROR_EQ",
        "rule": "dynamic_bark_mirror_eq",
        "freq_hz": round(float(freq_hz), 1),
        "gain_db": float(gain_db),
        "q": float(q),
        "targets": used_slots,
    }


def _relax_bus_compression_mcp(
    mcp: GridlineMCP,
    tracks: list[dict[str, Any]],
) -> dict[str, Any]:
    targets = _find_mixer_targets(tracks)
    bm_track = targets["bus_music"]
    master_track = targets["master"]
    if bm_track is None:
        return {"applied": False, "code": "RELAX_BUS_COMPRESSION", "reason": "BUS Music not found"}

    bm_idx = int(bm_track["mixer"])
    comp_slots = _slots_for_plugin(bm_track, "gl.comp")
    if not comp_slots:
        mcp.call("add_plugin", {"mixer": bm_idx, "plugin": "gl.comp"})
        tracks = mcp.call("get_mixer", {}).get("tracks", [])
        targets = _find_mixer_targets(tracks)
        bm_track = targets["bus_music"]
        comp_slots = _slots_for_plugin(bm_track, "gl.comp")

    if not comp_slots:
        return {"applied": False, "code": "RELAX_BUS_COMPRESSION", "reason": "gl.comp missing"}

    # Prefer glue compressor (Detector input == 0) over sidechain ducking slot
    glue_slot = comp_slots[0]
    glue_pars: dict[str, float] = {}
    for sl in comp_slots:
        pars = _param_lookup(mcp, bm_idx, sl)
        if int(round(pars.get("Detector input", 0))) == 0:
            glue_slot = sl
            glue_pars = pars
            break
    if not glue_pars:
        glue_pars = _param_lookup(mcp, bm_idx, glue_slot)

    cur_thresh = float(glue_pars.get("Threshold", -13.5))
    new_thresh = round(clamp(cur_thresh + 2.5, -40.0, 0.0), 2)
    p_changes: list[tuple[str, Any]] = [(f"mixer/{bm_idx}/slot/{glue_slot}/Threshold", new_thresh)]

    old_gain: float | None = None
    new_gain: float | None = None
    if master_track is not None:
        lim_slots = _slots_for_plugin(master_track, "gl.limiter")
        if lim_slots:
            lim_slot = lim_slots[-1]
            cur_lim = _param_lookup(mcp, 0, lim_slot)
            old_gain = float(cur_lim.get("Gain", 0.0))
            new_gain = round(clamp(old_gain - 1.5, -36.0, 24.0), 2)
            p_changes.append((f"mixer/0/slot/{lim_slot}/Gain", new_gain))

    mcp.set_params(p_changes)
    return {
        "applied": True,
        "code": "RELAX_BUS_COMPRESSION",
        "rule": "relax_bus_compression",
        "bus_music": bm_idx,
        "comp_slot": glue_slot,
        "old_threshold_db": cur_thresh,
        "new_threshold_db": new_thresh,
        "old_master_gain_db": old_gain,
        "new_master_gain_db": new_gain,
    }


def _boost_bass_harmonics_mcp(
    mcp: GridlineMCP,
    tracks: list[dict[str, Any]],
    f0_hz: float,
) -> dict[str, Any]:
    targets = _find_mixer_targets(tracks)
    bb_track = targets["bus_bass"]
    if bb_track is None:
        bb_track = next(
            (
                t for t in tracks
                if int(t["mixer"]) != 0 and any(k in (t.get("name") or "").lower() for k in ("808", "bass", "sub"))
            ),
            None,
        )
    if bb_track is None:
        return {"applied": False, "code": "BOOST_BASS_HARMONICS", "reason": "BUS Bass / bass insert not found"}

    bb_idx = int(bb_track["mixer"])
    dist_slots = _slots_for_plugin(bb_track, "gl.dist")
    if not dist_slots:
        mcp.call("add_plugin", {"mixer": bb_idx, "plugin": "gl.dist"})
        tracks = mcp.call("get_mixer", {}).get("tracks", [])
        bb_track = next((t for t in tracks if int(t["mixer"]) == bb_idx), bb_track)
        dist_slots = _slots_for_plugin(bb_track, "gl.dist")

    dist_slot = dist_slots[-1]
    cur_dist = _param_lookup(mcp, bb_idx, dist_slot)
    cur_mix = float(cur_dist.get("Mix", 0.0))
    eff_mix = 0.24 if (dist_slots and 0.10 <= cur_mix <= 0.20) else 0.14

    # Ensure an EQ slot sits AFTER gl.dist so boost applies cleanly to the saturated signal
    eq_slots = [s for s in _slots_for_plugin(bb_track, "gl.eq") if s > dist_slot]
    if not eq_slots:
        mcp.call("add_plugin", {"mixer": bb_idx, "plugin": "gl.eq"})
        tracks = mcp.call("get_mixer", {}).get("tracks", [])
        bb_track = next((t for t in tracks if int(t["mixer"]) == bb_idx), bb_track)
        eq_slots = [s for s in _slots_for_plugin(bb_track, "gl.eq") if s > dist_slot] or _slots_for_plugin(bb_track, "gl.eq")
    eq_slot = eq_slots[-1]

    boost_freq = float(round(clamp(2.5 * float(f0_hz), 70.0, 220.0)))
    p_changes: list[tuple[str, Any]] = []
    p_changes.extend(
        to_mixer_params(
            bb_idx,
            dist_slot,
            {
                "Mode": 0,
                "Mix": eff_mix,
                "Tone": 1200.0,
                "Remove DC": 1,
                "Drive": 36.0,
                "Color": 0.7,
                "Output": 6.0,
                "Oversampling": 2,
            },
        )
    )
    p_changes.extend(eq_params(bb_idx, [eq_band(2, "peak", boost_freq, 2.5, q=1.5)], slot=eq_slot, quality=0))
    mcp.set_params(p_changes)
    return {
        "applied": True,
        "code": "BOOST_BASS_HARMONICS",
        "rule": "boost_bass_harmonics",
        "mixer": bb_idx,
        "dist_slot": dist_slot,
        "eq_slot": eq_slot,
        "mix": eff_mix,
        "f0_hz": round(float(f0_hz), 2),
        "boost_freq_hz": boost_freq,
        "gain_db": 2.5,
        "q": 1.5,
    }


def _adjust_track_fader_mcp(
    mcp: GridlineMCP,
    tracks: list[dict[str, Any]],
    *,
    track_name: str = "",
    insert_idx: int | None = None,
    role_hint: str = "",
    delta_db: float = 0.0,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Phase 8 Q2: Deterministic `ADJUST_TRACK_FADER` mutation via `GridlineMCP` + `gl_dsl.db_to_auto`.
    Enforces per-pass cap `|delta_db| <= 4.0` and cumulative cap `|sum delta_db| <= 8.0`.
    """
    step_req = clamp(float(delta_db), -4.0, 4.0)
    if abs(step_req) < 0.05:
        return {"applied": False, "code": "ADJUST_TRACK_FADER", "reason": "delta_db too small"}

    target: dict[str, Any] | None = None
    if insert_idx is not None:
        idx_i = int(insert_idx)
        target = next((t for t in tracks if int(t.get("mixer", -1)) == idx_i), None)

    if target is None and track_name.strip():
        tn_low = track_name.strip().lower()
        # Exact match first, then substring match (excluding Master and buses unless explicitly named)
        for t in tracks:
            m_i = int(t.get("mixer", 0))
            if m_i == 0:
                continue
            nm = str(t.get("name") or "").strip().lower()
            if nm == tn_low:
                target = t
                break
        if target is None:
            for t in tracks:
                m_i = int(t.get("mixer", 0))
                if m_i == 0:
                    continue
                nm = str(t.get("name") or "").strip().lower()
                if nm and (tn_low in nm or nm in tn_low):
                    target = t
                    break

    if target is None and role_hint.strip():
        canon = normalize_role_for_gain_staging(role_hint)
        for t in tracks:
            m_i = int(t.get("mixer", 0))
            nm = str(t.get("name") or "").strip()
            if m_i == 0 or not nm or nm.lower().startswith(("bus ", "fx ")):
                continue
            if normalize_role_for_gain_staging(nm) == canon:
                target = t
                break

    if target is None or int(target.get("mixer", 0)) == 0:
        return {
            "applied": False,
            "code": "ADJUST_TRACK_FADER",
            "reason": f"Mixer insert not found for track={track_name!r}, role={role_hint!r}, insert={insert_idx}",
        }

    m_idx = int(target["mixer"])
    t_label = str(target.get("name") or f"Insert {m_idx}")
    st = state if state is not None else {}
    cum_map = st.setdefault("fader_cumulative_db", {})
    cum_key = f"mixer:{m_idx}"
    prev_cum = float(cum_map.get(cum_key, 0.0))

    new_cum = clamp(prev_cum + step_req, -8.0, 8.0)
    eff_delta = round(new_cum - prev_cum, 3)
    if abs(eff_delta) < 0.05:
        return {
            "applied": False,
            "code": "ADJUST_TRACK_FADER",
            "track": t_label,
            "mixer": m_idx,
            "cumulative_delta_db": prev_cum,
            "escalate": "SWAP_SAMPLE",
            "reason": f"Cumulative +-8.0 dB fader cap reached on '{t_label}'; escalate to SWAP_SAMPLE / CHECK_SOURCE_GAIN.",
        }

    cur_vol = float(target.get("volume", 0.80))
    # gl_dsl.db_to_auto returns normalized [0, 1] automation scale (fader / 1.25)
    auto_norm = db_to_auto(eff_delta, ref_fader=cur_vol)
    new_vol = round(clamp(auto_norm * 1.25, 0.10, 1.25), 4)
    mcp.call("edit_mixer_track", {"mixer": m_idx, "volume": new_vol})
    cum_map[cum_key] = round(new_cum, 3)

    return {
        "applied": True,
        "code": "ADJUST_TRACK_FADER",
        "rule": "adjust_track_fader",
        "track": t_label,
        "mixer": m_idx,
        "old_volume": round(cur_vol, 4),
        "new_volume": new_vol,
        "delta_db": eff_delta,
        "cumulative_delta_db": cum_map[cum_key],
    }


def adjust_track_fader(
    project_path: str,
    *,
    track_name: str = "",
    insert_idx: int | None = None,
    role_hint: str = "",
    delta_db: float = 0.0,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Patch mixer insert volume in a `.gridline` file via `GridlineMCP` + `gl_dsl.db_to_auto`."""
    abs_proj = os.path.abspath(project_path)
    mcp = GridlineMCP(abs_proj, os.path.dirname(abs_proj))
    try:
        tracks = mcp.call("get_mixer", {}).get("tracks", [])
        res = _adjust_track_fader_mcp(
            mcp,
            tracks,
            track_name=track_name,
            insert_idx=insert_idx,
            role_hint=role_hint,
            delta_db=delta_db,
            state=state,
        )
        if res.get("applied"):
            mcp.call("save_project", {})
        return res
    finally:
        mcp.close()


def apply_dynamic_mirror_eq(project_path: str, freq_hz: float, *, gain_db: float = -3.5, q: float = 1.8) -> dict[str, Any]:
    abs_proj = os.path.abspath(project_path)
    mcp = GridlineMCP(abs_proj, os.path.dirname(abs_proj))
    try:
        tracks = mcp.call("get_mixer", {}).get("tracks", [])
        res = _apply_mirror_eq_mcp(mcp, tracks, freq_hz, gain_db=gain_db, q=q)
        if res.get("applied"):
            mcp.call("save_project", {})
        return res
    finally:
        mcp.close()


def relax_bus_compression(project_path: str) -> dict[str, Any]:
    abs_proj = os.path.abspath(project_path)
    mcp = GridlineMCP(abs_proj, os.path.dirname(abs_proj))
    try:
        tracks = mcp.call("get_mixer", {}).get("tracks", [])
        res = _relax_bus_compression_mcp(mcp, tracks)
        if res.get("applied"):
            mcp.call("save_project", {})
        return res
    finally:
        mcp.close()


def boost_bass_harmonics(project_path: str, f0_hz: float) -> dict[str, Any]:
    abs_proj = os.path.abspath(project_path)
    mcp = GridlineMCP(abs_proj, os.path.dirname(abs_proj))
    try:
        tracks = mcp.call("get_mixer", {}).get("tracks", [])
        res = _boost_bass_harmonics_mcp(mcp, tracks, f0_hz)
        if res.get("applied"):
            mcp.call("save_project", {})
        return res
    finally:
        mcp.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Quiet render + combined audio-analyzer + optional deterministic auto-fix.")
    ap.add_argument("project", help="Path to .gridline project")
    ap.add_argument("-o", "--out", required=True, help="Output WAV path")
    ap.add_argument("--range", default="", help="Optional bar range e.g. 5-12 for fast iteration")
    ap.add_argument("--profile", default="trap_rap", choices=list(GENRE_GATES.keys()), help="Genre mix gate profile")
    ap.add_argument("--target-lufs", type=float, default=None, help="Target LUFS (defaults to profile target)")
    ap.add_argument("--bpm", type=float, default=132.0)
    ap.add_argument("--beats-per-bar", type=float, default=4.0, help="Beats per bar (4.0 for 4/4, 3.0 for 3/4, 6.0 for 6/8)")
    ap.add_argument("--sections", default="", help="Optional e.g. 'Intro:1-4,Refrain1:5-12,Outro:29-33'")
    ap.add_argument("--blueprint", default="", help="Optional path to song_blueprint.json (auto-detected next to project)")
    ap.add_argument("--valence", type=float, default=None, help="Override emotion_vector.valence [-1..+1]")
    ap.add_argument("--dark", action="store_true", help="Force dark mix thresholds (reduced air_min, stricter sharpness)")
    ap.add_argument("--no-drums", action="store_true", help="Declare drumless/ambient texture so sub/sidechain gates relax")
    ap.add_argument("--mp3", action="store_true", help="Also render 320 kbps MP3 alongside WAV")
    ap.add_argument("--no-neural", action="store_true", help="Skip Audiobox/CLAP for ultra-fast check")
    ap.add_argument("--auto-fix", action="store_true", help="Enable closed-loop deterministic mix auto-fix via GridlineMCP + gl_dsl")
    ap.add_argument("--max-passes", type=int, default=2, help="Maximum deterministic auto-fix passes (default 2)")
    ap.add_argument("--contract", default="", help="Optional path to multi_scale_contract.json (auto-detected next to project)")
    ap.add_argument("--approve-taste", action="store_true", help="Mark validated mix as user-approved in episodic taste memory (weight 2.0)")
    ap.add_argument("--taste-memory", default="", help="Optional path override for user_taste_profile.json")
    args = ap.parse_args()

    has_drums_override: bool | None = False if args.no_drums else None
    summary = render_and_evaluate(
        project=args.project,
        out_wav=args.out,
        profile=args.profile,
        range_bars=args.range,
        target_lufs=args.target_lufs,
        bpm=args.bpm,
        beats_per_bar=args.beats_per_bar,
        sections=args.sections,
        blueprint=args.blueprint,
        valence_arg=args.valence,
        dark_flag=args.dark,
        has_drums_arg=has_drums_override,
        render_mp3=args.mp3 and not args.auto_fix,
        use_neural=not args.no_neural,
        contract_path=args.contract,
        approve_taste=args.approve_taste,
        taste_memory_path=args.taste_memory,
        record_taste=not args.auto_fix,
    )

    if args.auto_fix:
        history_passes: list[dict[str, Any]] = [
            {
                "pass": 0,
                "stage": "initial",
                "out_wav": summary["out_wav"],
                "integrated_lufs": summary["integrated_lufs"],
                "true_peak_dbtp": summary["true_peak_dbtp"],
                "clipped_samples": summary["clipped_samples"],
                "bands_pct": summary["bands_pct"],
                "sharpness_din_acum": summary["psychoacoustics"].get("sharpness_din_acum"),
                "resonances_third_oct": summary["resonances_third_oct"],
                "low_end_corr": summary["stereo"]["correlation_low_end_150hz"],
                "quality_gate": summary["quality_gate"],
                "mutations_applied": [],
            }
        ]
        state: dict[str, Any] = {"ceiling_drops": 0, "low_mids_applied": False}
        max_p = max(1, int(args.max_passes))

        for pass_idx in range(1, max_p + 1):
            if summary["quality_gate"]["passed"]:
                break
            mutations = _apply_autofix_mutations(args.project, summary, state)
            history_passes[-1]["mutations_applied"] = mutations
            if not mutations:
                break

            if state.pop("refresh_stems", False):
                stems_auto_dir = os.path.join(os.path.dirname(os.path.abspath(args.project)), "renders", "stems")
                if os.path.isdir(stems_auto_dir):
                    cmd_st = ["gridline", "render", args.project, "-o", stems_auto_dir.rstrip("/") + "/", "--stems"]
                    if args.range:
                        cmd_st.extend(["--scope", f"range:{args.range}"])
                    subprocess.run(cmd_st, capture_output=True, text=True, timeout=180)

            summary = render_and_evaluate(
                project=args.project,
                out_wav=args.out,
                profile=args.profile,
                range_bars=args.range,
                target_lufs=args.target_lufs,
                bpm=args.bpm,
                beats_per_bar=args.beats_per_bar,
                sections=args.sections,
                blueprint=args.blueprint,
                valence_arg=args.valence,
                dark_flag=args.dark,
                has_drums_arg=has_drums_override,
                render_mp3=args.mp3 and (pass_idx == max_p),
                use_neural=not args.no_neural,
                contract_path=args.contract,
                approve_taste=args.approve_taste,
                taste_memory_path=args.taste_memory,
                record_taste=False,
            )
            history_passes.append(
                {
                    "pass": pass_idx,
                    "stage": f"after_fix_{pass_idx}",
                    "out_wav": summary["out_wav"],
                    "integrated_lufs": summary["integrated_lufs"],
                    "true_peak_dbtp": summary["true_peak_dbtp"],
                    "clipped_samples": summary["clipped_samples"],
                    "bands_pct": summary["bands_pct"],
                    "sharpness_din_acum": summary["psychoacoustics"].get("sharpness_din_acum"),
                    "resonances_third_oct": summary["resonances_third_oct"],
                    "low_end_corr": summary["stereo"]["correlation_low_end_150hz"],
                    "quality_gate": summary["quality_gate"],
                    "mutations_applied": [],
                }
            )

        if args.mp3 and summary["quality_gate"]["passed"]:
            mp3_out = os.path.splitext(os.path.abspath(args.out))[0] + ".mp3"
            cmd_mp3 = ["gridline", "render", args.project, "-o", mp3_out, "--format", "mp3", "--bitrate", "320"]
            if args.range:
                cmd_mp3.extend(["--scope", f"range:{args.range}"])
            subprocess.run(cmd_mp3, capture_output=True, text=True, timeout=180, check=True)

        out_dir = os.path.dirname(os.path.abspath(args.out))
        proj_renders = os.path.join(os.path.dirname(os.path.abspath(args.project)), "renders")
        history_dir = out_dir if os.path.basename(out_dir) == "renders" else proj_renders
        os.makedirs(history_dir, exist_ok=True)
        history_path = os.path.join(history_dir, "autofix_history.json")
        history_payload = {
            "project_path": os.path.abspath(args.project),
            "profile": args.profile,
            "range": args.range or "full",
            "target_lufs": summary["target_lufs"],
            "max_passes": max_p,
            "final_passed": summary["quality_gate"]["passed"],
            "passes": history_passes,
        }
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(history_payload, fh, ensure_ascii=False, indent=2)

        summary["autofix"] = {
            "passes_run": len(history_passes) - 1,
            "history_path": history_path,
            "mutations": [m for p in history_passes for m in p.get("mutations_applied", [])],
        }
        proj_dir = os.path.dirname(os.path.abspath(os.path.expanduser(args.project)))
        if (
            summary["quality_gate"]["passed"]
            and not args.range
            and summary.get("master_fingerprint_path")
            and _should_record_taste(proj_dir, args.taste_memory, args.approve_taste)
        ):
            summary["taste_memory"] = record_validated_taste(
                project_dir=proj_dir,
                blueprint_path=args.blueprint or "",
                fingerprint_path=str(summary["master_fingerprint_path"]),
                approved_by_user=bool(args.approve_taste),
                taste_path=args.taste_memory or None,
                extra_metrics={"lra_lu": summary.get("lra_lu")},
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
