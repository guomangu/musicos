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
import os
import subprocess
import sys
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT = "/home/gamo/Bureau/musicos"
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _p in (SCRIPTS_DIR, ANALYZER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audio_analyzer_server import _analyze_file
from bootstrap_golden_refs import DEFAULT_PROFILES_PATH, compute_spectral_match
from deep_diagnostics import run_deep_audit
from gl_dsl import clamp, eq_band, eq_params, limiter_master, to_mixer_params
from studio_lib import GridlineMCP


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
    "trap_rap":          {"low_mids_max": 24.0, "air_min": 3.0, "dark_air_min": 1.2,  "sharp_max": 1.90, "dark_sharp_max": 1.85, "arc_min": 2.5, "default_lufs": -11.0},
    "pop_rnb":           {"low_mids_max": 26.0, "air_min": 3.0, "dark_air_min": 1.5,  "sharp_max": 1.95, "dark_sharp_max": 1.85, "arc_min": 2.5, "default_lufs": -10.5},
    "edm_club":          {"low_mids_max": 24.0, "air_min": 3.5, "dark_air_min": 1.75, "sharp_max": 2.05, "dark_sharp_max": 1.90, "arc_min": 2.5, "default_lufs": -9.0},
    "rock_metal":        {"low_mids_max": 28.0, "air_min": 2.5, "dark_air_min": 1.25, "sharp_max": 2.10, "dark_sharp_max": 1.90, "arc_min": 2.0, "default_lufs": -10.0},
    "lofi_chill":        {"low_mids_max": 34.0, "air_min": 0.5, "dark_air_min": 0.25, "sharp_max": 1.75, "dark_sharp_max": 1.75, "arc_min": 1.8, "default_lufs": -14.0},
    "jazz_acoustic":     {"low_mids_max": 36.0, "air_min": 1.0, "dark_air_min": 0.50, "sharp_max": 1.85, "dark_sharp_max": 1.80, "arc_min": 2.5, "default_lufs": -14.5},
    "classical_ambient": {"low_mids_max": 40.0, "air_min": 0.3, "dark_air_min": 0.15, "sharp_max": 1.75, "dark_sharp_max": 1.75, "arc_min": 3.0, "default_lufs": -16.0},
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
) -> dict[str, Any]:
    """Renders `.gridline` (optionally scoped by bar range) and computes full Quality Gate report."""
    gate_cfg = GENRE_GATES[profile]
    valence, is_dark, has_drums, blueprint_used = _resolve_blueprint_metadata(
        project_path=project,
        blueprint_path=blueprint,
        dark_flag=dark_flag,
        valence_arg=valence_arg,
        has_drums_arg=has_drums_arg,
        profile=profile,
    )
    air_min = gate_cfg["dark_air_min"] if is_dark else gate_cfg["air_min"]
    sharp_max = gate_cfg["dark_sharp_max"] if is_dark else gate_cfg["sharp_max"]
    eff_target_lufs = target_lufs if target_lufs is not None else gate_cfg["default_lufs"]

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
    spectral_match = compute_spectral_match(
        mono=mono,
        sr=sr,
        profile=profile,
        profiles_path=profiles_path,
    )

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
    gate_failures: list[str] = []
    if base["clipped_samples"] > 0:
        gate_failures.append(f"clipped_samples={base['clipped_samples']} > 0")
    if base["true_peak_dbtp"] > -1.0:
        gate_failures.append(f"true_peak_dbtp={base['true_peak_dbtp']} > -1.0")
    if abs(base["integrated_lufs"] - eff_target_lufs) > 1.2:
        gate_failures.append(f"lufs={base['integrated_lufs']} vs target={eff_target_lufs}")

    low_end_corr = float(base["stereo"]["correlation_low_end_150hz"])
    if no_drums_ambient:
        if sub_bass_pct >= 5.0 and low_end_corr < 0.20:
            gate_failures.append(f"low_end_corr={low_end_corr} < 0.20")
    else:
        if low_end_corr < 0.50:
            gate_failures.append(f"low_end_corr={low_end_corr} < 0.50")

    lm_corr = deep["room_and_transients"]["low_mid_200_800hz_correlation"]
    if lm_corr < 0.55:
        gate_failures.append(f"low_mid_corr={lm_corr} < 0.55")
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
    if base["spectral_bands"]["low_mids"]["energy_pct"] > gate_cfg["low_mids_max"]:
        gate_failures.append(f"low_mids={base['spectral_bands']['low_mids']['energy_pct']}% > {gate_cfg['low_mids_max']}%")
    if base["spectral_bands"]["air"]["energy_pct"] < air_min:
        gate_failures.append(f"air={base['spectral_bands']['air']['energy_pct']}% < {air_min}%")
    tune_ct = abs(float(deep["psychoacoustics"].get("tuning_cents_offset", 0.0)))
    if tune_ct > 6.5:
        gate_failures.append(f"tuning_offset={tune_ct}ct > 6.5ct")
    if len(sec_lufs) >= 2:
        arc_db = round(max(sec_lufs.values()) - min(sec_lufs.values()), 2)
        if arc_db < gate_cfg["arc_min"]:
            gate_failures.append(f"dynamic_arc={arc_db}dB < {gate_cfg['arc_min']}dB")

    return {
        "profile": profile,
        "has_drums": has_drums,
        "dark_mode": is_dark,
        "valence": valence,
        "blueprint_used": blueprint_used,
        "effective_thresholds": {
            "air_min": air_min,
            "sharp_max": sharp_max,
            "low_mids_max": gate_cfg["low_mids_max"],
            "arc_min": gate_cfg["arc_min"],
        },
        "target_lufs": eff_target_lufs,
        "out_wav": abs_out,
        "duration_s": base["duration_seconds"],
        "integrated_lufs": base["integrated_lufs"],
        "true_peak_dbtp": base["true_peak_dbtp"],
        "clipped_samples": base["clipped_samples"],
        "max_sample_jump": max_jump,
        "bands_pct": {k: v["energy_pct"] for k, v in base["spectral_bands"].items()},
        "stereo": base["stereo"],
        "spectral_match": spectral_match,
        "recommendations": base["recommendations"],
        "psychoacoustics": deep["psychoacoustics"],
        "micro_glitches": deep["glitches_and_artifacts"]["micro_glitch_or_crackle_events"],
        "resonances_third_oct": deep["third_octave_resonances"],
        "low_mid_corr": lm_corr,
        "neural": deep.get("neural_perception"),
        "section_lufs": sec_lufs,
        "quality_gate": {
            "passed": len(gate_failures) == 0,
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
        need_tp_fix = clipped > 0 or true_peak > -1.0
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

            # Rule 3: surgical resonances (-4 dB @ F_i, Q=2.8 -> width=0.22, max 3 bands in [200, 8000] Hz)
            if resonances:
                top_res = sorted(
                    resonances,
                    key=lambda r: float(r.get("excess_db", 0.0)),
                    reverse=True,
                )[:3]
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
                    eq_band(idx, "peak", float(r["freq_hz"]), -4.0, q=2.8)
                    for idx, r in enumerate(top_res, start=1)
                ]
                param_changes.extend(eq_params(bm_idx, surg_bands, slot=surg_slot, quality=0))
                for unused_idx in range(len(top_res) + 1, 8):
                    param_changes.append((f"mixer/{bm_idx}/slot/{surg_slot}/Band {unused_idx} enabled", 0))
                mutations.append(
                    {
                        "rule": "resonances_surgical_eq",
                        "target": f"mixer/{bm_idx}/slot/{surg_slot}",
                        "freqs_hz": [float(r["freq_hz"]) for r in top_res],
                        "gain_db": -4.0,
                        "q": 2.8,
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

        if param_changes:
            mcp.set_params(param_changes)
        if mutations:
            mcp.call("save_project", {})
    finally:
        mcp.close()

    return mutations


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

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
