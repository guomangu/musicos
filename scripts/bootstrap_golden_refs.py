#!/usr/bin/env python3
"""
Bootstrap Lightweight Spectral Reference Profiles (`golden_references/spectral_profiles.json`).

Extracts 24-Bark relative energy profiles + pink spectral slope (dB/octave in 60 Hz - 12 kHz)
grouped by `mix_profile`, from reference mixes (`golden_references/mixes/<profile>/`) and
validated studio renders (`projects/*/`, `morceaux/*/`), with zero mandatory PyTorch/CLAP usage.

Also provides `compute_spectral_match(...)` for fast zero-PyTorch `spectral_match_score`
inside `scripts/render_and_audit.py`.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = "/home/gamo/Bureau/musicos"
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _p in (SCRIPTS_DIR, ANALYZER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audio_metrics import BARK_EDGES_HZ
from deep_diagnostics import compute_pink_slope_db_per_oct

DEFAULT_PROFILES_PATH = os.path.join(ROOT, "golden_references", "spectral_profiles.json")
PROFILES = (
    "trap_rap",
    "pop_rnb",
    "edm_club",
    "rock_metal",
    "lofi_chill",
    "jazz_acoustic",
    "classical_ambient",
)

# Analytic spectral slope targets (pink dB/oct & low-end/air tilt weights across 24 Bark bands)
# Used ONLY when a genre has zero rendered references on disk yet.
_ANALYTIC_GENRE_HINTS: dict[str, dict[str, float]] = {
    "trap_rap": {"target_slope": -4.2, "sub_boost": 1.35, "air_scale": 0.90},
    "pop_rnb": {"target_slope": -3.7, "sub_boost": 1.10, "air_scale": 1.05},
    "edm_club": {"target_slope": -3.5, "sub_boost": 1.25, "air_scale": 1.05},
    "rock_metal": {"target_slope": -3.5, "sub_boost": 0.90, "air_scale": 1.00},
    "lofi_chill": {"target_slope": -4.6, "sub_boost": 1.05, "air_scale": 0.65},
    "jazz_acoustic": {"target_slope": -4.0, "sub_boost": 0.85, "air_scale": 0.95},
    "classical_ambient": {"target_slope": -4.4, "sub_boost": 0.75, "air_scale": 0.85},
}


def compute_bark24_distribution(mono: np.ndarray, sr: int) -> list[float]:
    """Computes normalized 24-Bark spectral energy distribution (~1-2 ms, pure numpy)."""
    if len(mono) < 512 or sr <= 0:
        return [round(1.0 / len(BARK_EDGES_HZ), 6)] * len(BARK_EDGES_HZ)
    mag_sq = np.abs(np.fft.rfft(mono.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1.0 / float(sr))
    nyq = 0.5 * float(sr)
    bark = np.zeros(len(BARK_EDGES_HZ), dtype=np.float64)
    for i, (flo, fhi) in enumerate(BARK_EDGES_HZ):
        hi = min(fhi, nyq)
        if hi <= flo:
            continue
        mask = (freqs >= flo) & (freqs < hi)
        if np.any(mask):
            bark[i] = float(np.sum(mag_sq[mask]))
    tot = float(np.sum(bark))
    if tot <= 1e-18:
        return [round(1.0 / len(BARK_EDGES_HZ), 6)] * len(BARK_EDGES_HZ)
    bark /= tot
    return [round(float(x), 6) for x in bark]


def _analytic_bark24_profile(profile: str) -> dict[str, Any]:
    hints = _ANALYTIC_GENRE_HINTS.get(profile, {"target_slope": -3.9, "sub_boost": 1.0, "air_scale": 1.0})
    slope = float(hints["target_slope"])
    # Approximate energy distribution of a 1/f^k mix integrated per Bark bandwidth:
    # PSD(f) ~ (f / 1000)^(slope / 3.01), bandwidth dw_i = (fhi - flo)
    weights = np.zeros(len(BARK_EDGES_HZ), dtype=np.float64)
    exp = slope / 3.0103  # since 10*log10(f^exp) = (10*exp*log10(2)) * log2(f)
    for i, (flo, fhi) in enumerate(BARK_EDGES_HZ):
        fc = math.sqrt(max(20.0, flo) * max(20.0, fhi))
        bw = max(10.0, fhi - flo)
        val = ((fc / 1000.0) ** exp) * bw
        if fc < 110.0:
            val *= float(hints["sub_boost"])
        elif fc > 6000.0:
            val *= float(hints["air_scale"])
        weights[i] = val
    weights /= max(1e-18, float(np.sum(weights)))
    return {
        "mix_profile": profile,
        "source_type": "analytic_default",
        "source_count": 0,
        "sources": [],
        "pink_slope_mean": round(slope, 3),
        "pink_slope_std": 0.5,
        "bark_24_mean": [round(float(x), 6) for x in weights],
        "bark_24_std": [round(float(x) * 0.35, 6) for x in weights],
    }


def _find_project_master(proj_dir: str) -> str | None:
    renders_dir = os.path.join(proj_dir, "renders")
    candidates: list[str] = []
    if os.path.isdir(renders_dir):
        for pat in ("master_final.wav", "master*.wav", "*.wav", "*.mp3"):
            for p in sorted(glob.glob(os.path.join(renders_dir, pat))):
                bn = os.path.basename(p).lower()
                if any(skip in bn for skip in ("_tmp", "_temp", "stem", "autofix")):
                    continue
                if os.path.isfile(p) and os.path.getsize(p) > 200_000:
                    candidates.append(p)
        if candidates:
            return candidates[0]
    exports_dir = os.path.join(proj_dir, "exports")
    if os.path.isdir(exports_dir):
        for pat in ("master_final.wav", "master*.wav"):
            hits = sorted(glob.glob(os.path.join(exports_dir, pat)))
            if hits:
                return hits[0]
    return None


def _discover_project_renders() -> list[tuple[str, str, str]]:
    """Returns `(mix_profile, audio_path, label)` pairs from `golden_references/mixes/`, `projects/`, `morceaux/`."""
    found: list[tuple[str, str, str]] = []
    seen_paths: set[str] = set()

    # 1) External reference mixes dropped in golden_references/mixes/<profile>/
    mixes_root = os.path.join(ROOT, "golden_references", "mixes")
    if os.path.isdir(mixes_root):
        for prof in PROFILES:
            pdir = os.path.join(mixes_root, prof)
            if not os.path.isdir(pdir):
                continue
            for p in sorted(Path(pdir).rglob("*")):
                if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg") and str(p) not in seen_paths:
                    seen_paths.add(str(p))
                    found.append((prof, str(p), f"mixes/{prof}/{p.name}"))

    # 2) Existing studio renders across projects/*/ and morceaux/*/
    for base_name in ("projects", "morceaux"):
        for root_dir in (os.path.join(ROOT, base_name), os.path.join(os.path.dirname(ROOT), base_name)):
            if not os.path.isdir(root_dir):
                continue
            for entry in sorted(os.listdir(root_dir)):
                low = entry.lower()
                if "(copie)" in low or "copy" in low or low.startswith("."):
                    continue
                proj_dir = os.path.join(root_dir, entry)
                if not os.path.isdir(proj_dir):
                    continue
                bp_path = os.path.join(proj_dir, "song_blueprint.json")
                if not os.path.isfile(bp_path):
                    continue
                try:
                    with open(bp_path, encoding="utf-8") as fh:
                        bp = json.load(fh)
                    prof = str(bp.get("mix_profile") or "trap_rap")
                except Exception:
                    continue
                master = _find_project_master(proj_dir)
                if master and os.path.abspath(master) not in seen_paths:
                    seen_paths.add(os.path.abspath(master))
                    found.append((prof, os.path.abspath(master), f"{base_name}/{entry}/{os.path.basename(master)}"))
    return found


def _cached_pq_for_render(audio_path: str) -> float | None:
    """Reads cached PQ score from sibling *audit*.json if present."""
    parent = os.path.dirname(audio_path)
    for jp in sorted(Path(parent).glob("*audit*.json")):
        try:
            with open(jp, encoding="utf-8") as fh:
                data = json.load(fh)
            pq = ((data.get("neural") or {}).get("audiobox_aesthetics") or {}).get("PQ")
            if pq is not None:
                return float(pq)
        except Exception:
            continue
    return None


def _analyze_reference_audio(audio_path: str, min_pq: float | None = None, use_neural: bool = False) -> dict[str, Any] | None:
    try:
        data, sr = sf.read(audio_path, dtype="float64", always_2d=True)
    except Exception:
        return None
    if len(data) < sr * 4:
        return None
    mono = np.mean(data, axis=1)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    if rms < 1e-4 or peak > 1.05:
        return None

    # Exclude silent head/tail padding by trimming active window (> -45 dBFS)
    thresh = max(1e-5, rms * 0.05)
    idx = np.where(np.abs(mono) >= thresh)[0]
    if len(idx) > sr * 2:
        mono = mono[idx[0] : idx[-1] + 1]

    if min_pq is not None:
        pq = _cached_pq_for_render(audio_path)
        if pq is None and use_neural:
            try:
                from deep_diagnostics import _run_neural_perceptual_audit

                n_res = _run_neural_perceptual_audit(audio_path, data, mono, sr)
                pq = ((n_res or {}).get("audiobox_aesthetics") or {}).get("PQ")
                if pq is not None:
                    pq = float(pq)
            except Exception:
                pq = None
        if pq is not None and pq < float(min_pq):
            return None

    bark_24 = compute_bark24_distribution(mono, sr)
    pink_slope = compute_pink_slope_db_per_oct(mono, sr)
    return {"bark_24": bark_24, "pink_slope_db_per_oct": pink_slope}


def build_spectral_profiles(
    output_path: str = DEFAULT_PROFILES_PATH,
    min_pq: float | None = 7.8,
    use_neural: bool = False,
    extra_sources: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Builds `golden_references/spectral_profiles.json` across all 7 genres."""
    discovered = list(extra_sources) if extra_sources is not None else _discover_project_renders()
    by_profile: dict[str, list[dict[str, Any]]] = {p: [] for p in PROFILES}

    for prof, apath, label in discovered:
        sig = _analyze_reference_audio(apath, min_pq=min_pq, use_neural=use_neural)
        if sig is None:
            continue
        sig["label"] = label
        by_profile.setdefault(prof, []).append(sig)

    profiles_out: dict[str, Any] = {}
    for prof in PROFILES:
        items = by_profile.get(prof, [])
        if items:
            mat = np.asarray([it["bark_24"] for it in items], dtype=np.float64)
            mean_b = np.mean(mat, axis=0)
            mean_b /= max(1e-18, float(np.sum(mean_b)))
            std_b = np.std(mat, axis=0)
            slopes = np.asarray([it["pink_slope_db_per_oct"] for it in items], dtype=np.float64)
            profiles_out[prof] = {
                "mix_profile": prof,
                "source_type": "measured",
                "source_count": len(items),
                "sources": [it["label"] for it in items],
                "pink_slope_mean": round(float(np.mean(slopes)), 3),
                "pink_slope_std": round(float(np.std(slopes)), 3),
                "bark_24_mean": [round(float(x), 6) for x in mean_b],
                "bark_24_std": [round(float(x), 6) for x in std_b],
            }
        else:
            profiles_out[prof] = _analytic_bark24_profile(prof)

    payload = {
        "version": 1,
        "bark_band_count": len(BARK_EDGES_HZ),
        "bark_edges_hz": [[float(lo), float(hi)] for lo, hi in BARK_EDGES_HZ],
        "profiles": profiles_out,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return {
        "output_path": output_path,
        "measured_profiles": {p: v["source_count"] for p, v in profiles_out.items() if v["source_count"] > 0},
        "analytic_fallback_profiles": [p for p, v in profiles_out.items() if v["source_count"] == 0],
    }


def compute_spectral_match(
    mono: np.ndarray,
    sr: int,
    profile: str,
    profiles_path: str = DEFAULT_PROFILES_PATH,
    bark_24: list[float] | None = None,
    pink_slope: float | None = None,
) -> dict[str, Any]:
    """
    Fast zero-PyTorch spectral distance comparison between a mix and `spectral_profiles.json`.
    Returns `spectral_match_score` in `[0.0, 1.0]` plus `pink_slope_db_per_oct`.
    """
    cur_bark = np.asarray(
        bark_24 if bark_24 is not None else compute_bark24_distribution(mono, sr),
        dtype=np.float64,
    )
    cur_bark /= max(1e-18, float(np.sum(cur_bark)))
    cur_slope = float(pink_slope if pink_slope is not None else compute_pink_slope_db_per_oct(mono, sr))

    ref_entry: dict[str, Any] | None = None
    if profiles_path and os.path.isfile(profiles_path):
        try:
            with open(profiles_path, encoding="utf-8") as fh:
                data = json.load(fh)
            ref_entry = (data.get("profiles") or {}).get(profile)
        except Exception:
            ref_entry = None
    if ref_entry is None:
        ref_entry = _analytic_bark24_profile(profile)

    ref_bark = np.asarray(ref_entry["bark_24_mean"], dtype=np.float64)
    ref_bark /= max(1e-18, float(np.sum(ref_bark)))
    ref_slope = float(ref_entry.get("pink_slope_mean", -4.0))

    denom = float(np.linalg.norm(cur_bark) * np.linalg.norm(ref_bark)) + 1e-18
    cosine_sim = float(np.dot(cur_bark, ref_bark) / denom)
    l1_sim = max(0.0, 1.0 - 0.5 * float(np.sum(np.abs(cur_bark - ref_bark))))
    slope_sim = max(0.0, 1.0 - abs(cur_slope - ref_slope) / 4.0)
    match_score = round(max(0.0, min(1.0, 0.45 * cosine_sim + 0.35 * l1_sim + 0.20 * slope_sim)), 4)

    return {
        "spectral_match_score": match_score,
        "cosine_similarity": round(cosine_sim, 4),
        "l1_similarity": round(l1_sim, 4),
        "pink_slope_db_per_oct": round(cur_slope, 3),
        "reference_pink_slope_db_per_oct": round(ref_slope, 3),
        "reference_source": ref_entry.get("source_type", "analytic_default"),
        "reference_count": int(ref_entry.get("source_count", 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap lightweight 24-Bark + pink-slope spectral profiles")
    ap.add_argument("--out", default=DEFAULT_PROFILES_PATH, help="Output spectral_profiles.json path")
    ap.add_argument("--min-pq", type=float, default=7.8, help="Optional min PQ threshold (applied when PQ is known or --use-neural is set)")
    ap.add_argument("--use-neural", action="store_true", help="Optionally evaluate Audiobox PQ on renders without cached audit JSON")
    args = ap.parse_args()

    res = build_spectral_profiles(output_path=args.out, min_pq=args.min_pq, use_neural=args.use_neural)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
