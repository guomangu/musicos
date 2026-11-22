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


def main() -> None:
    ap = argparse.ArgumentParser(description="Quiet render + combined audio-analyzer + section LUFS audit.")
    ap.add_argument("project", help="Path to .gridline project")
    ap.add_argument("-o", "--out", required=True, help="Output WAV path")
    ap.add_argument("--range", default="", help="Optional bar range e.g. 5-12 for fast iteration")
    ap.add_argument("--target-lufs", type=float, default=-11.0)
    ap.add_argument("--bpm", type=float, default=132.0)
    ap.add_argument("--sections", default="", help="Optional e.g. 'Intro:1-4,Refrain1:5-12,Outro:29-33'")
    ap.add_argument("--mp3", action="store_true", help="Also render 320 kbps MP3 alongside WAV")
    ap.add_argument("--no-neural", action="store_true", help="Skip Audiobox/CLAP for ultra-fast check")
    args = ap.parse_args()

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

    base = _analyze_file(args.out, target_lufs=args.target_lufs)
    data, sr = sf.read(args.out, dtype="float64", always_2d=True)
    deep = run_deep_audit(
        audio_path=os.path.abspath(args.out),
        base_metrics=base,
        data=data,
        sr=sr,
        target_genre="trap",
        use_neural=not args.no_neural,
        headphone_target="sony_xm6",
    )

    # Compact per-section LUFS
    sec_lufs = {}
    if args.sections and not args.range:
        meter = pyln.Meter(sr)
        bar_s = 4.0 * 60.0 / float(args.bpm)
        for name, b0, b1 in parse_sections(args.sections):
            s0 = int((b0 - 1) * bar_s * sr)
            s1 = min(len(data), int(b1 * bar_s * sr))
            if s1 - s0 > int(0.45 * sr):
                sec_lufs[name] = round(float(meter.integrated_loudness(data[s0:s1])), 2)

    mono = data.mean(axis=1)
    max_jump = round(float(np.max(np.abs(np.diff(mono)))) if len(mono) > 1 else 0.0, 4)

    summary = {
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
        "low_mid_corr": deep["room_and_transients"]["low_mid_200_800hz_correlation"],
        "neural": deep.get("neural_perception"),
        "section_lufs": sec_lufs,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
