#!/usr/bin/env python3
"""
Fast quiet wrapper around `gridline render` + `audio-analyzer` (`analyze_mix` + `audit_mix_deep`)
+ per-section LUFS dynamics + Nyquist sample-jump check in ONE token-efficient call.
Supports `--range BAR1-BAR2` for rapid 4x faster mix/EQ verification loops.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT = "/home/gamo/Bureau/musicos"
sys.path.insert(0, f"{ROOT}/mcp-servers/audio-analyzer")
from audio_analyzer_server import _analyze_file
from deep_diagnostics import run_deep_audit


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
    "trap_rap":          {"low_mids_max": 24.0, "air_min": 3.0, "sharp_max": 2.2, "arc_min": 2.5, "default_lufs": -11.0},
    "pop_rnb":           {"low_mids_max": 26.0, "air_min": 3.0, "sharp_max": 2.2, "arc_min": 2.5, "default_lufs": -10.5},
    "edm_club":          {"low_mids_max": 24.0, "air_min": 3.5, "sharp_max": 2.3, "arc_min": 2.5, "default_lufs": -9.0},
    "rock_metal":        {"low_mids_max": 28.0, "air_min": 2.5, "sharp_max": 2.4, "arc_min": 2.0, "default_lufs": -10.0},
    "lofi_chill":        {"low_mids_max": 34.0, "air_min": 0.5, "sharp_max": 2.0, "arc_min": 1.8, "default_lufs": -14.0},
    "jazz_acoustic":     {"low_mids_max": 36.0, "air_min": 1.0, "sharp_max": 2.1, "arc_min": 2.5, "default_lufs": -14.5},
    "classical_ambient": {"low_mids_max": 40.0, "air_min": 0.3, "sharp_max": 2.0, "arc_min": 3.0, "default_lufs": -16.0},
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Quiet render + combined audio-analyzer + section LUFS audit.")
    ap.add_argument("project", help="Path to .gridline project")
    ap.add_argument("-o", "--out", required=True, help="Output WAV path")
    ap.add_argument("--range", default="", help="Optional bar range e.g. 5-12 for fast iteration")
    ap.add_argument("--profile", default="trap_rap", choices=list(GENRE_GATES.keys()), help="Genre mix gate profile")
    ap.add_argument("--target-lufs", type=float, default=None, help="Target LUFS (defaults to profile target)")
    ap.add_argument("--bpm", type=float, default=132.0)
    ap.add_argument("--beats-per-bar", type=float, default=4.0, help="Beats per bar (4.0 for 4/4, 3.0 for 3/4, 6.0 for 6/8)")
    ap.add_argument("--sections", default="", help="Optional e.g. 'Intro:1-4,Refrain1:5-12,Outro:29-33'")
    ap.add_argument("--mp3", action="store_true", help="Also render 320 kbps MP3 alongside WAV")
    ap.add_argument("--no-neural", action="store_true", help="Skip Audiobox/CLAP for ultra-fast check")
    args = ap.parse_args()

    gate_cfg = GENRE_GATES[args.profile]
    target_lufs = args.target_lufs if args.target_lufs is not None else gate_cfg["default_lufs"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cmd = ["gridline", "render", args.project, "-o", args.out, "--bits", "24"]
    if args.range:
        cmd.extend(["--scope", f"range:{args.range}"])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"gridline render failed ({proc.returncode}): {proc.stderr or proc.stdout}")

    if args.mp3:
        mp3_out = os.path.splitext(args.out)[0] + ".mp3"
        cmd_mp3 = ["gridline", "render", args.project, "-o", mp3_out, "--format", "mp3", "--bitrate", "320"]
        if args.range:
            cmd_mp3.extend(["--scope", f"range:{args.range}"])
        subprocess.run(cmd_mp3, capture_output=True, text=True, timeout=180, check=True)

    base = _analyze_file(args.out, target_lufs=target_lufs)
    data, sr = sf.read(args.out, dtype="float64", always_2d=True)
    deep = run_deep_audit(
        audio_path=os.path.abspath(args.out),
        base_metrics=base,
        data=data,
        sr=sr,
        target_genre=args.profile,
        use_neural=not args.no_neural,
        headphone_target="sony_xm6",
    )

    # Compact per-section LUFS
    sec_lufs = {}
    if args.sections and not args.range:
        meter = pyln.Meter(sr)
        bar_s = float(args.beats_per_bar) * 60.0 / float(args.bpm)
        for name, b0, b1 in parse_sections(args.sections):
            s0 = int((b0 - 1) * bar_s * sr)
            s1 = min(len(data), int(b1 * bar_s * sr))
            if s1 - s0 > int(0.45 * sr):
                sec_lufs[name] = round(float(meter.integrated_loudness(data[s0:s1])), 2)

    mono = data.mean(axis=1)
    max_jump = round(float(np.max(np.abs(np.diff(mono)))) if len(mono) > 1 else 0.0, 4)

    # Studio Quality Gate evaluation (genre-aware)
    gate_failures: list[str] = []
    if base["clipped_samples"] > 0:
        gate_failures.append(f"clipped_samples={base['clipped_samples']} > 0")
    if base["true_peak_dbtp"] > -1.0:
        gate_failures.append(f"true_peak_dbtp={base['true_peak_dbtp']} > -1.0")
    if abs(base["integrated_lufs"] - target_lufs) > 1.2:
        gate_failures.append(f"lufs={base['integrated_lufs']} vs target={target_lufs}")
    if base["stereo"]["correlation_low_end_150hz"] < 0.5:
        gate_failures.append(f"low_end_corr={base['stereo']['correlation_low_end_150hz']} < 0.50")
    lm_corr = deep["room_and_transients"]["low_mid_200_800hz_correlation"]
    if lm_corr < 0.55:
        gate_failures.append(f"low_mid_corr={lm_corr} < 0.55")
    sharp = deep["psychoacoustics"].get("sharpness_din_acum", 0.0)
    if sharp >= gate_cfg["sharp_max"]:
        gate_failures.append(f"sharpness={sharp} >= {gate_cfg['sharp_max']}")
    if deep["glitches_and_artifacts"]["micro_glitch_or_crackle_events"] > 0:
        gate_failures.append(f"micro_glitches={deep['glitches_and_artifacts']['micro_glitch_or_crackle_events']} > 0")
    if deep["third_octave_resonances"]:
        gate_failures.append(f"resonances={[r['freq_hz'] for r in deep['third_octave_resonances']]}")
    if base["spectral_bands"]["low_mids"]["energy_pct"] > gate_cfg["low_mids_max"]:
        gate_failures.append(f"low_mids={base['spectral_bands']['low_mids']['energy_pct']}% > {gate_cfg['low_mids_max']}%")
    if base["spectral_bands"]["air"]["energy_pct"] < gate_cfg["air_min"]:
        gate_failures.append(f"air={base['spectral_bands']['air']['energy_pct']}% < {gate_cfg['air_min']}%")
    tune_ct = abs(float(deep["psychoacoustics"].get("tuning_cents_offset", 0.0)))
    if tune_ct > 6.5:
        gate_failures.append(f"tuning_offset={tune_ct}ct > 6.5ct")
    if len(sec_lufs) >= 2:
        arc_db = round(max(sec_lufs.values()) - min(sec_lufs.values()), 2)
        if arc_db < gate_cfg["arc_min"]:
            gate_failures.append(f"dynamic_arc={arc_db}dB < {gate_cfg['arc_min']}dB")

    summary = {
        "profile": args.profile,
        "target_lufs": target_lufs,
        "out_wav": os.path.abspath(args.out),
        "duration_s": base["duration_seconds"],
        "integrated_lufs": base["integrated_lufs"],
        "true_peak_dbtp": base["true_peak_dbtp"],
        "clipped_samples": base["clipped_samples"],
        "max_sample_jump": max_jump,
        "bands_pct": {k: v["energy_pct"] for k, v in base["spectral_bands"].items()},
        "stereo": base["stereo"],
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
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
