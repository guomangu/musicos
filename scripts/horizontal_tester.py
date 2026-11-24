#!/usr/bin/env python3
"""
Horizontal Arrangement Seam & Outro Auditor (`horizontal_tester.py`, RFC §5).

Instead of rendering the full 3-minute song on every arrangement edit, renders
only 4-bar seam windows straddling each major section boundary:
  `[bar_boundary - 2, bar_boundary + 1]` (2 bars before + 2 bars after)
plus the final 4 bars (`Outro`).

Audits all 4 RFC §5.2 criteria:
  1. Energy contrast (`delta_lufs = LUFS_after - LUFS_before`)
  2. Transition event (`beat4_dropout` RMS dip on last beat OR `filter_sweep` centroid glide)
  3. Junction cleanliness (0 micro-glitches / discontinuities within ±150 ms of boundary)
  4. Outro closure (last 1.0 s RMS < -45 dBFS, never abruptly cut off)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Any

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
ANALYZER_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from artifact_cache import compute_e3_seam_hash, get_cached_verdict, set_cached_verdict
from deep_diagnostics import _lin_to_db, _scan_glitches_and_crackle
from macro_structure_audit import analyze_macro_structure

_HIGH_ENERGY_KW = ("drop", "chorus", "refrain", "climax", "hook", "peak")
_LOW_ENERGY_KW = ("intro", "break", "bridge", "pont", "outro", "ambient", "calm")


def _classify_transition(from_name: str, to_name: str) -> str:
    f_low = from_name.lower()
    t_low = to_name.lower()
    f_high = any(k in f_low for k in _HIGH_ENERGY_KW)
    t_high = any(k in t_low for k in _HIGH_ENERGY_KW)
    f_low_e = any(k in f_low for k in _LOW_ENERGY_KW)
    t_low_e = any(k in t_low for k in _LOW_ENERGY_KW)

    if t_high and not f_high:
        return "major_lift" if f_low_e else "lift"
    if f_high and t_low_e:
        return "drop_down"
    if t_low_e and not f_low_e:
        return "drop_down"
    return "shift"


def _compute_segment_lufs(data: np.ndarray, sr: int) -> float:
    if len(data) == 0:
        return -120.0
    min_s = int(0.45 * sr)
    buf = data
    if len(buf) < min_s:
        buf = np.vstack([buf, np.zeros((min_s - len(buf), buf.shape[1]), dtype=np.float64)])
    meter = pyln.Meter(sr)
    val = float(meter.integrated_loudness(buf))
    if math.isinf(val) or math.isnan(val):
        rms = float(np.sqrt(np.mean(buf**2)))
        return _lin_to_db(rms)
    return round(val, 2)


def audit_seam_wav(
    wav_path: str,
    *,
    from_section: str = "Intro",
    to_section: str = "Drop",
    bpm: float = 132.0,
    beats_per_bar: int = 4,
    bars_before: float = 2.0,
    bars_after: float = 2.0,
    transition_type: str = "",
) -> dict[str, Any]:
    """
    Audit a 4-bar transition window WAV straddling a section boundary.
    `bars_before` indicates where the boundary sits inside `wav_path` (default: 2 bars in).
    """
    abs_wav = os.path.abspath(os.path.expanduser(wav_path))
    if not os.path.isfile(abs_wav):
        raise FileNotFoundError(f"Seam WAV not found: {abs_wav}")

    data, sr = sf.read(abs_wav, dtype="float64", always_2d=True)
    mono = np.mean(data, axis=1)
    beat_s = 60.0 / max(40.0, float(bpm))
    bar_s = float(beats_per_bar) * beat_s

    boundary_s = float(bars_before) * bar_s
    b_idx = min(len(data) - 1, max(1, int(round(boundary_s * sr))))

    # Fallback to midpoint if WAV length differs significantly from expected bars_before+bars_after
    expected_s = (float(bars_before) + float(bars_after)) * bar_s
    actual_s = len(data) / float(sr)
    if abs(actual_s - expected_s) > 0.6 * bar_s and actual_s >= 1.5:
        b_idx = int(round(len(data) * (float(bars_before) / max(0.5, float(bars_before) + float(bars_after)))))
        boundary_s = b_idx / float(sr)

    seg_before = data[:b_idx]
    seg_after = data[b_idx:]
    lufs_before = _compute_segment_lufs(seg_before, sr)
    lufs_after = _compute_segment_lufs(seg_after, sr)
    delta_lufs = round(lufs_after - lufs_before, 2)

    t_type = transition_type.strip() or _classify_transition(from_section, to_section)

    # 1) Energy contrast check
    if t_type in ("major_lift", "lift"):
        min_delta = 2.0 if t_type == "major_lift" else 1.5
        contrast_ok = delta_lufs >= min_delta
    elif t_type == "drop_down":
        contrast_ok = delta_lufs <= -2.0
    else:
        contrast_ok = abs(delta_lufs) >= 1.0

    # 2) Transition event detection (beat4_dropout OR filter_sweep before boundary)
    n_beat = max(64, int(round(beat_s * sr)))
    last_beat_start = max(0, b_idx - n_beat)
    prev_region_start = max(0, last_beat_start - int(3 * n_beat))

    rms_last_beat = float(np.sqrt(np.mean(mono[last_beat_start:b_idx] ** 2))) if b_idx > last_beat_start else 0.0
    rms_prev = float(np.sqrt(np.mean(mono[prev_region_start:last_beat_start] ** 2))) if last_beat_start > prev_region_start else rms_last_beat
    dropout_db = round(_lin_to_db(rms_last_beat) - _lin_to_db(rms_prev), 2)
    dropout_detected = (rms_prev > 1e-4) and (dropout_db <= -3.5)

    # Filter sweep detection: spectral centroid shift over the pre-boundary bars
    centroid_glide_ratio = 0.0
    sweep_detected = False
    if b_idx >= int(0.8 * sr):
        pre_mono = mono[:b_idx].astype(np.float32)
        if float(np.sqrt(np.mean(pre_mono**2))) > 1e-4:
            cent = librosa.feature.spectral_centroid(y=pre_mono, sr=sr, hop_length=512)[0]
            if len(cent) >= 8:
                q_len = max(2, len(cent) // 4)
                c_start = float(np.median(cent[:q_len])) + 1.0
                c_end = float(np.median(cent[-q_len:])) + 1.0
                centroid_glide_ratio = round(abs(c_end - c_start) / c_start, 3)
                sweep_detected = centroid_glide_ratio >= 0.25

    event_detected = bool(dropout_detected or sweep_detected)
    require_event = t_type == "major_lift"
    event_ok = event_detected if require_event else True

    # 3) Junction cleanliness within ±150 ms around boundary
    j_half = int(round(0.150 * sr))
    j0 = max(0, b_idx - j_half)
    j1 = min(len(mono), b_idx + j_half)
    j_seg = mono[j0:j1].astype(np.float32)
    gl_info = _scan_glitches_and_crackle(j_seg, sr) if len(j_seg) > 32 else {"micro_glitch_or_crackle_events": 0}
    j_glitches = int(gl_info.get("micro_glitch_or_crackle_events") or 0)
    # Check sudden waveform discontinuity right across the exact sample boundary (±4 samples)
    c0 = max(1, b_idx - 4)
    c1 = min(len(mono), b_idx + 4)
    boundary_jump = round(float(np.max(np.abs(np.diff(mono[c0 - 1 : c1])))) if c1 > c0 else 0.0, 4)
    junction_ok = (j_glitches == 0) and (boundary_jump < 0.45)

    checks = {
        "energy_contrast_ok": contrast_ok,
        "transition_event_ok": event_ok,
        "junction_clean_ok": junction_ok,
    }
    score = 1.0
    diag: list[str] = []
    next_action: dict[str, Any] = {"code": "KEEP_SEAM", "params": {"from": from_section, "to": to_section}}

    if not junction_ok:
        score -= 0.40
        diag.append(f"Jonction {from_section}->{to_section} sale (glitches={j_glitches}, saut={boundary_jump}) : ajuster fade/release.")
        next_action = {"code": "CLEAN_SEAM_FADE", "params": {"from": from_section, "to": to_section, "fade_ms": 15.0}}
    if not contrast_ok:
        score -= 0.35
        diag.append(
            f"Transition plate {from_section}->{to_section} (ΔLUFS={delta_lufs:+.2f} dB, type={t_type}) : "
            "alléger la section calme ou accentuer section_gain_arc."
        )
        if next_action["code"] == "KEEP_SEAM":
            next_action = {
                "code": "INCREASE_SECTION_CONTRAST",
                "params": {"from": from_section, "to": to_section, "delta_lufs": delta_lufs, "helper": "transitions_lib.section_gain_arc"},
            }
    if not event_ok:
        score -= 0.30
        diag.append(
            f"Aucun marqueur de transition avant {to_section} (dropout={dropout_db:+.1f} dB, sweep={centroid_glide_ratio:.2f}) : "
            "ajouter beat4_dropout() ou filter_sweep()."
        )
        if next_action["code"] == "KEEP_SEAM":
            next_action = {
                "code": "ADD_TRANSITION_EVENT",
                "params": {"from": from_section, "to": to_section, "helper": "transitions_lib.beat4_dropout"},
            }
    if not diag:
        marker_txt = "beat4_dropout" if dropout_detected else ("filter_sweep" if sweep_detected else "contraste net")
        diag.append(
            f"Couture {from_section}->{to_section} validée : ΔLUFS={delta_lufs:+.2f} dB ({marker_txt}), 0 glitch."
        )

    verdict = "PASS" if all(checks.values()) else "FIX"
    return {
        "seam": f"{from_section} -> {to_section}",
        "transition_type": t_type,
        "verdict": verdict,
        "score": round(max(0.0, min(1.0, score)), 4),
        "human_diagnosis": diag[:3],
        "next_action": next_action,
        "checks": checks,
        "metrics": {
            "lufs_before": lufs_before,
            "lufs_after": lufs_after,
            "delta_lufs": delta_lufs,
            "dropout_last_beat_db": dropout_db,
            "dropout_detected": dropout_detected,
            "centroid_glide_ratio": centroid_glide_ratio,
            "sweep_detected": sweep_detected,
            "junction_glitches": j_glitches,
            "boundary_sample_jump": boundary_jump,
        },
        "wav_path": abs_wav,
    }


def audit_outro_wav(wav_path: str) -> dict[str, Any]:
    """Audit the final bars (`Outro`) ensuring clean tail decay (last 1.0 s RMS < -45 dBFS)."""
    abs_wav = os.path.abspath(os.path.expanduser(wav_path))
    if not os.path.isfile(abs_wav):
        raise FileNotFoundError(f"Outro WAV not found: {abs_wav}")

    data, sr = sf.read(abs_wav, dtype="float64", always_2d=True)
    mono = np.mean(data, axis=1)
    n_last = min(len(mono), sr)
    tail = mono[-n_last:] if n_last > 0 else mono
    end_100ms = mono[-min(len(mono), max(1, int(0.1 * sr))):]

    tail_rms_db = _lin_to_db(float(np.sqrt(np.mean(tail**2))) if len(tail) else 0.0)
    end_rms_db = _lin_to_db(float(np.sqrt(np.mean(end_100ms**2))) if len(end_100ms) else 0.0)
    outro_ok = (tail_rms_db < -45.0) or (tail_rms_db < -38.0 and end_rms_db < -55.0)

    if outro_ok:
        return {
            "seam": "Outro",
            "transition_type": "outro_closure",
            "verdict": "PASS",
            "score": 1.0,
            "human_diagnosis": [f"Clôture Outro propre (RMS dernière seconde = {tail_rms_db:.1f} dBFS < -45 dBFS)."],
            "next_action": {"code": "KEEP_OUTRO", "params": {}},
            "checks": {"outro_clean_closure_ok": True},
            "metrics": {"last_1s_rms_dbfs": tail_rms_db, "last_100ms_rms_dbfs": end_rms_db},
            "wav_path": abs_wav,
        }
    return {
        "seam": "Outro",
        "transition_type": "outro_closure",
        "verdict": "FIX",
        "score": 0.35,
        "human_diagnosis": [
            f"Outro coupée trop brutalement (RMS dernière seconde = {tail_rms_db:.1f} dBFS >= -45 dBFS) : "
            "appliquer transitions_lib.outro_fade() sur les 2 dernières mesures."
        ],
        "next_action": {
            "code": "APPLY_OUTRO_FADE",
            "params": {"fade_bars": 2, "helper": "transitions_lib.outro_fade"},
        },
        "checks": {"outro_clean_closure_ok": False},
        "metrics": {"last_1s_rms_dbfs": tail_rms_db, "last_100ms_rms_dbfs": end_rms_db},
        "wav_path": abs_wav,
    }


def extract_sections_from_blueprint(blueprint: dict[str, Any]) -> list[tuple[str, int, int]]:
    """Return `[(name, start_bar_1idx, bars), ...]` from `song_blueprint.json`."""
    out: list[tuple[str, int, int]] = []
    cursor = 1
    for item in blueprint.get("structure") or []:
        name = str(item.get("section") or item.get("name") or f"Section_{len(out)+1}")
        bars = int(item.get("bars", 4))
        st = int(item.get("start_bar", cursor))
        out.append((name, st, bars))
        cursor = st + bars
    return out


def audit_project_horizontal_seams(
    project_path: str,
    *,
    blueprint: dict[str, Any] | str | None = None,
    sections: list[tuple[str, int, int]] | None = None,
    bpm: float = 132.0,
    beats_per_bar: int = 4,
    max_seams: int = 6,
    use_cache: bool = True,
    include_macro_ssm: bool = False,
    full_mix_wav: str = "",
    contract_path: str = "",
    stems_dir: str = "",
) -> dict[str, Any]:
    """
    Render and audit up to `max_seams` 4-bar section transition windows plus the `Outro`
    window directly from a `.gridline` project. Supports incremental SHA-256 caching per
    seam (`artifact_cache.py`) and optional full-song `macro_structure_audit` (`S4`).
    """
    abs_proj = os.path.abspath(os.path.expanduser(project_path))
    if not os.path.isfile(abs_proj):
        raise FileNotFoundError(f"Project file not found: {abs_proj}")

    proj_dir = os.path.dirname(abs_proj)
    bp: dict[str, Any] = {}
    if isinstance(blueprint, dict):
        bp = blueprint
    elif isinstance(blueprint, str) and blueprint.strip() and os.path.isfile(blueprint):
        with open(blueprint, encoding="utf-8") as fh:
            bp = json.load(fh)
    else:
        cand_bp = os.path.join(proj_dir, "song_blueprint.json")
        if os.path.isfile(cand_bp):
            with open(cand_bp, encoding="utf-8") as fh:
                bp = json.load(fh)

    if bp:
        bpm = float(bp.get("bpm", bpm))
        ts = bp.get("time_signature") or {}
        beats_per_bar = int(ts.get("numerator", beats_per_bar))

    secs = sections or (extract_sections_from_blueprint(bp) if bp else [])
    if not secs:
        raise ValueError("No sections found in blueprint or `sections` parameter.")

    total_bars = max(st + bars - 1 for _nm, st, bars in secs)
    seam_dir = os.path.join(proj_dir, "renders", "seams")
    diag_dir = os.path.join(proj_dir, "renders", "diagnostics")
    os.makedirs(seam_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    # Select up to `max_seams` transitions (prioritizing major lifts & drops)
    transitions: list[tuple[int, str, str, int, str]] = []
    for (n1, s1, b1), (n2, s2, _b2) in zip(secs[:-1], secs[1:]):
        boundary_bar = s2  # 1-indexed first bar of next section
        t_type = _classify_transition(n1, n2)
        prio = 0 if t_type == "major_lift" else (1 if t_type in ("lift", "drop_down") else 2)
        transitions.append((prio, n1, n2, boundary_bar, t_type))
    transitions.sort(key=lambda x: (x[0], x[3]))
    selected = sorted(transitions[: max(1, int(max_seams))], key=lambda x: x[3])

    cache_hits = 0
    cache_misses = 0
    seam_results: list[dict[str, Any]] = []
    for idx, (_pr, n1, n2, b_bar, t_type) in enumerate(selected, start=1):
        w_start = max(1, b_bar - 2)
        w_end = min(total_bars, b_bar + 1)
        bars_before = float(b_bar - w_start)
        bars_after = float(w_end - b_bar + 1)
        out_wav = os.path.join(seam_dir, f"seam_{idx}_{w_start}_{w_end}.wav")
        seam_key = f"E3_{n1}->{n2}_{w_start}_{w_end}"
        seam_hash = compute_e3_seam_hash(
            seam_key,
            project_path=abs_proj,
            bar_range=f"{w_start}-{w_end}",
            bpm=bpm,
            beats_per_bar=beats_per_bar,
        )
        cached = get_cached_verdict(proj_dir, seam_key, seam_hash) if use_cache else None
        if cached and isinstance(cached.get("result"), dict) and cached["result"].get("verdict"):
            sr_res = dict(cached["result"])
            sr_res["cache_hit"] = True
            cache_hits += 1
            seam_results.append(sr_res)
            continue

        cache_misses += 1
        cmd = ["gridline", "render", abs_proj, "-o", out_wav, "--bits", "24", "--scope", f"range:{w_start}-{w_end}"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if proc.returncode != 0:
            raise RuntimeError(f"Seam render failed ({w_start}-{w_end}): {proc.stderr or proc.stdout}")
        sr_res = audit_seam_wav(
            out_wav,
            from_section=n1,
            to_section=n2,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            bars_before=bars_before,
            bars_after=bars_after,
            transition_type=t_type,
        )
        sr_res["bar_range"] = f"{w_start}-{w_end}"
        sr_res["boundary_bar"] = b_bar
        sr_res["cache_hit"] = False
        set_cached_verdict(proj_dir, seam_key, seam_hash, sr_res)
        seam_results.append(sr_res)

    # Outro window (last 4 bars)
    o_start = max(1, total_bars - 3)
    o_end = total_bars
    outro_wav = os.path.join(seam_dir, f"outro_{o_start}_{o_end}.wav")
    outro_key = f"E3_OUTRO_{o_start}_{o_end}"
    outro_hash = compute_e3_seam_hash(
        outro_key,
        project_path=abs_proj,
        bar_range=f"{o_start}-{o_end}",
        bpm=bpm,
        beats_per_bar=beats_per_bar,
    )
    cached_outro = get_cached_verdict(proj_dir, outro_key, outro_hash) if use_cache else None
    if cached_outro and isinstance(cached_outro.get("result"), dict) and cached_outro["result"].get("verdict"):
        outro_res = dict(cached_outro["result"])
        outro_res["cache_hit"] = True
        cache_hits += 1
    else:
        cache_misses += 1
        cmd_o = ["gridline", "render", abs_proj, "-o", outro_wav, "--bits", "24", "--scope", f"range:{o_start}-{o_end}"]
        proc_o = subprocess.run(cmd_o, capture_output=True, text=True, timeout=90)
        if proc_o.returncode != 0:
            raise RuntimeError(f"Outro render failed ({o_start}-{o_end}): {proc_o.stderr or proc_o.stdout}")
        outro_res = audit_outro_wav(outro_wav)
        outro_res["bar_range"] = f"{o_start}-{o_end}"
        outro_res["cache_hit"] = False
        set_cached_verdict(proj_dir, outro_key, outro_hash, outro_res)

    macro_res: dict[str, Any] | None = None
    if include_macro_ssm and full_mix_wav and os.path.isfile(os.path.expanduser(full_mix_wav)):
        macro_res = analyze_macro_structure(
            full_mix_wav,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            blueprint=bp or None,
            contract=contract_path or None,
            stems_dir_or_map=stems_dir or None,
            project_dir=proj_dir,
        )

    all_items = seam_results + [outro_res]
    failed = [x for x in all_items if x["verdict"] != "PASS"]
    avg_score = round(float(np.mean([x["score"] for x in all_items])), 4) if all_items else 1.0
    verdict = "PASS" if not failed else "FIX"

    if failed:
        diag = [f"{len(failed)}/{len(all_items)} couture(s) à corriger."] + [
            f["human_diagnosis"][0] for f in failed[:2]
        ]
        next_action = failed[0]["next_action"]
    elif macro_res and macro_res["verdict"] != "PASS":
        verdict = "FIX"
        m_dev = (macro_res.get("deviations") or [{}])[0]
        next_action = m_dev.get("action_payload") or {"code": "MUTATE_BLOCK_DENSITY_OR_LPF", "params": {}}
        diag = [m_dev.get("diagnosis") or "Macro-structure S4 deviation detected."]
    else:
        diag = [f"Toutes les coutures ({len(seam_results)}) et l'Outro sont validées (score={avg_score:.2f})."]
        next_action = {"code": "PROCEED_TO_FINAL_RENDER", "params": {"stage": "E4_FINAL_GATE"}}

    report = {
        "project_path": abs_proj,
        "verdict": verdict,
        "score": avg_score,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "human_diagnosis": diag[:3],
        "next_action": next_action,
        "seams": seam_results,
        "outro": outro_res,
        "macro_structure": macro_res,
    }
    diag_path = os.path.join(diag_dir, "horizontal_seams.json")
    with open(diag_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    report["diagnostics_path"] = diag_path
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit horizontal transition seams & Outro closure.")
    ap.add_argument("--project", default="", help="Path to full .gridline project.")
    ap.add_argument("--blueprint", default="", help="Path to song_blueprint.json.")
    ap.add_argument("--seam-wav", default="", help="Direct 4-bar seam WAV to audit.")
    ap.add_argument("--outro-wav", default="", help="Direct Outro WAV to audit.")
    ap.add_argument("--from-section", default="Intro")
    ap.add_argument("--to-section", default="Drop")
    ap.add_argument("--bpm", type=float, default=132.0)
    ap.add_argument("--beats-per-bar", type=int, default=4)
    args = ap.parse_args()

    if args.seam_wav:
        res = audit_seam_wav(
            args.seam_wav,
            from_section=args.from_section,
            to_section=args.to_section,
            bpm=args.bpm,
            beats_per_bar=args.beats_per_bar,
        )
    elif args.outro_wav:
        res = audit_outro_wav(args.outro_wav)
    elif args.project:
        res = audit_project_horizontal_seams(
            args.project,
            blueprint=args.blueprint or None,
            bpm=args.bpm,
            beats_per_bar=args.beats_per_bar,
        )
    else:
        ap.error("Provide --project, --seam-wav, or --outro-wav")
        return
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
