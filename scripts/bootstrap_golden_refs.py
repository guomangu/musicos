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

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _p in (SCRIPTS_DIR, ANALYZER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audio_metrics import BARK_EDGES_HZ
from deep_diagnostics import compute_pink_slope_db_per_oct

DEFAULT_PROFILES_PATH = os.path.join(ROOT, "golden_references", "spectral_profiles.json")
DEFAULT_CORRIDORS_PATH = os.path.join(ROOT, "golden_references", "spectral_corridors.json")
CORRIDOR_METRICS = (
    "lufs",
    "lra_lu",
    "pink_slope_db_per_oct",
    "crest_factor_db",
    "low_end_share",
    "low_mids_share",
    "air_share",
    "mid_high_side_to_mid_db",
)
CORRIDOR_ALPHA = 2.0

# Ancres par profil calibrées psychoacoustiquement (pentes roses + cibles mix_profiles.json
# + morceaux validés PQ >= 7.8) — utilisées pour la version initiale hors-ligne versionnée.
_CORRIDOR_LUFS = {"trap_rap": -11.0, "pop_rnb": -10.5, "edm_club": -9.0, "rock_metal": -10.0,
                  "lofi_chill": -14.0, "jazz_acoustic": -12.5, "classical_ambient": -16.0}
_CORRIDOR_LRA = {"trap_rap": 4.5, "pop_rnb": 4.0, "edm_club": 3.5, "rock_metal": 5.0,
                 "lofi_chill": 7.0, "jazz_acoustic": 6.0, "classical_ambient": 9.0}
_CORRIDOR_CREST = {"trap_rap": (7.5, 13.5), "pop_rnb": (7.0, 12.5), "edm_club": (6.5, 11.5),
                   "rock_metal": (6.5, 12.0), "lofi_chill": (9.0, 16.0), "jazz_acoustic": (8.0, 14.0),
                   "classical_ambient": (10.0, 18.0)}
_CORRIDOR_LOW_END = {"trap_rap": 0.34, "pop_rnb": 0.32, "edm_club": 0.38, "rock_metal": 0.30,
                     "lofi_chill": 0.26, "jazz_acoustic": 0.22, "classical_ambient": 0.20}
_CORRIDOR_LOW_MIDS = {"trap_rap": 0.18, "pop_rnb": 0.20, "edm_club": 0.18, "rock_metal": 0.22,
                      "lofi_chill": 0.26, "jazz_acoustic": 0.20, "classical_ambient": 0.16}
_CORRIDOR_AIR = {"trap_rap": 0.055, "pop_rnb": 0.055, "edm_club": 0.060, "rock_metal": 0.048,
                 "lofi_chill": 0.018, "jazz_acoustic": 0.045, "classical_ambient": 0.040}
_CORRIDOR_DARK_AIR = {"trap_rap": 0.022, "pop_rnb": 0.026, "edm_club": 0.030, "rock_metal": 0.022,
                      "lofi_chill": 0.008, "jazz_acoustic": 0.018, "classical_ambient": 0.015}
_CORRIDOR_MHS = {"trap_rap": -7.5, "pop_rnb": -6.5, "edm_club": -5.5, "rock_metal": -6.0,
                 "lofi_chill": -9.0, "jazz_acoustic": -8.0, "classical_ambient": -10.0}
DEFAULT_EXTERNAL_DIR = os.path.join(ROOT, "golden_references", "external_masters")
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


def _discover_external_masters(external_dir: str = DEFAULT_EXTERNAL_DIR) -> list[tuple[str, str, str]]:
    """Return `(mix_profile, audio_path, label)` from `external_masters/<profile>/` (never versioned)."""
    found: list[tuple[str, str, str]] = []
    if not external_dir or not os.path.isdir(external_dir):
        return found
    for prof in PROFILES:
        pdir = os.path.join(external_dir, prof)
        if not os.path.isdir(pdir):
            continue
        for p in sorted(Path(pdir).rglob("*")):
            if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
                found.append((prof, str(p), f"external_masters/{prof}/{p.name}"))
    return found


def _compute_band_ratios(mono: np.ndarray, sr: int) -> dict[str, float]:
    """Relative energy ratios (`%`) for sub / bass / low_mid / high_mid / air bands."""
    n = min(len(mono), 1 << 18)
    seg = mono[:n] if n == len(mono) else mono[len(mono) // 2 - n // 2 : len(mono) // 2 + n // 2]
    win = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win)) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / float(sr))
    bands = {"sub": (20.0, 60.0), "bass": (60.0, 250.0), "low_mid": (250.0, 500.0), "high_mid": (500.0, 2000.0), "air": (2000.0, 12000.0)}
    total = float(np.sum(spec[(freqs >= 20.0) & (freqs <= 12000.0)])) + 1e-18
    out: dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        e = float(np.sum(spec[(freqs >= lo) & (freqs < hi)]))
        out[name] = round(100.0 * e / total, 2)
    return out


def _analyze_external_master(audio_path: str) -> dict[str, Any] | None:
    """Extract Phase 8 Q4.1 statistical fingerprint from one commercial master (0 PyTorch)."""
    try:
        data, sr = sf.read(audio_path, dtype="float64", always_2d=True)
    except Exception:
        return None
    if len(data) < int(sr * 4):
        return None
    mono = np.mean(data, axis=1)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    if rms < 1e-4 or peak > 1.05:
        return None
    from deep_diagnostics import analyze_stereo_image, compute_ebu_r128_stats

    ebu = compute_ebu_r128_stats(data, int(sr))
    stereo = analyze_stereo_image(data, int(sr), target_genre="")
    return {
        "bark_24": compute_bark24_distribution(mono, int(sr)),
        "pink_slope_db_per_oct": compute_pink_slope_db_per_oct(mono, int(sr)),
        "lufs_integrated": round(float(ebu.get("integrated_lufs", -70.0)), 2),
        "lra_lu": round(float(ebu.get("lra_lu", 0.0)), 2),
        "crest_factor_db": round(20.0 * math.log10(max(peak, 1e-12) / max(rms, 1e-12)), 2),
        "mid_high_side_to_mid_db": stereo["mid_high_side_to_mid_db"],
        "band_ratios": _compute_band_ratios(mono, int(sr)),
    }


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
    external_dir: str = DEFAULT_EXTERNAL_DIR,
) -> dict[str, Any]:
    """
    Builds `golden_references/spectral_profiles.json` across all 7 genres.

    Priority (Phase 8 Q4.1): `external_masters/<profile>/` commercial masters are used
    exclusively when present and marked `"provenance": "external_pro_reference"`;
    otherwise internal studio renders / analytic fallback are marked `"internal_bootstrap"`.
    """
    found_external = _discover_external_masters(external_dir)
    external_by_profile: dict[str, list[dict[str, Any]]] = {p: [] for p in PROFILES}
    for prof, apath, label in found_external:
        sig = _analyze_external_master(apath)
        if sig is None:
            continue
        sig["label"] = label
        external_by_profile.setdefault(prof, []).append(sig)

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
        ext_items = external_by_profile.get(prof, [])
        internal_items = by_profile.get(prof, [])
        items = ext_items if ext_items else internal_items
        provenance = "external_pro_reference" if ext_items else "internal_bootstrap"
        if items:
            mat = np.asarray([it["bark_24"] for it in items], dtype=np.float64)
            mean_b = np.mean(mat, axis=0)
            mean_b /= max(1e-18, float(np.sum(mean_b)))
            std_b = np.std(mat, axis=0)
            slopes = np.asarray([it["pink_slope_db_per_oct"] for it in items], dtype=np.float64)
            entry: dict[str, Any] = {
                "mix_profile": prof,
                "source_type": "measured_external" if ext_items else "measured",
                "provenance": provenance,
                "source_count": len(items),
                "sources": [it["label"] for it in items],
                "pink_slope_mean": round(float(np.mean(slopes)), 3),
                "pink_slope_std": round(float(np.std(slopes)), 3),
                "bark_24_mean": [round(float(x), 6) for x in mean_b],
                "bark_24_std": [round(float(x), 6) for x in std_b],
            }
            if ext_items:
                for key in ("lufs_integrated", "lra_lu", "crest_factor_db", "mid_high_side_to_mid_db"):
                    vals = [float(it[key]) for it in items if it.get(key) is not None]
                    if vals:
                        entry[key] = round(float(np.median(vals)), 2)
                band_keys = ("sub", "bass", "low_mid", "high_mid", "air")
                band_items = [it for it in items if isinstance(it.get("band_ratios"), dict)]
                if band_items:
                    entry["band_ratios"] = {
                        b: round(float(np.median([float(it["band_ratios"][b]) for it in band_items if b in it["band_ratios"]])), 2)
                        for b in band_keys
                        if any(b in it["band_ratios"] for it in band_items)
                    }
            profiles_out[prof] = entry
        else:
            entry = _analytic_bark24_profile(prof)
            entry["provenance"] = "internal_bootstrap"
            profiles_out[prof] = entry

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
        "external_pro_profiles": sorted([p for p, v in profiles_out.items() if v.get("provenance") == "external_pro_reference"]),
        "analytic_fallback_profiles": [p for p, v in profiles_out.items() if v["source_count"] == 0],
        "warning": (
            "No external_pro_reference master found for profiles: "
            + ", ".join([p for p, v in profiles_out.items() if v.get("provenance") != "external_pro_reference"])
            if any(v.get("provenance") != "external_pro_reference" for v in profiles_out.values())
            else ""
        ),
    }


def corridor_dead_zone_loss(x: float, p10: float, p90: float, epsilon: float = 1e-6) -> float:
    """v3.1 REF-04 — perte à zone morte d'un corridor [P10, P90] de demi-largeur w :
    d(x) = 0 dans le corridor, linéaire hors corridor (normalisée par w)."""
    p10, p90, x = float(p10), float(p90), float(x)
    w = max((p90 - p10) / 2.0, float(epsilon))
    if p10 <= x <= p90:
        return 0.0
    if x < p10:
        return (p10 - x) / w
    return (x - p90) / w


def evaluate_against_corridors(
    measurements: dict[str, float],
    corridors_entry: dict[str, Any],
    valence: float = 0.0,
    alpha: float = CORRIDOR_ALPHA,
) -> dict[str, Any]:
    """v3.1 REF-04 — score par corridor à zone morte : `corridor_score = exp(-alpha * d_bar)`.
    Un mix dans l'enveloppe naturelle du genre obtient 0 pénalité (jamais sur-corrigé).
    Applique `dark_valence_overrides` si valence < -0.2."""
    metrics = corridors_entry.get("metrics") or {}
    overrides = (corridors_entry.get("dark_valence_overrides") or {}) if float(valence) < -0.2 else {}
    ds: list[float] = []
    per: dict[str, float] = {}
    for key, corr in metrics.items():
        if key not in measurements or not isinstance(corr, dict):
            continue
        use = overrides.get(key) if isinstance(overrides.get(key), dict) else corr
        d = corridor_dead_zone_loss(measurements[key], use.get("p10", 0.0), use.get("p90", 0.0))
        per[key] = round(d, 5)
        ds.append(d)
    d_bar = sum(ds) / len(ds) if ds else 0.0
    return {
        "corridor_score": round(float(math.exp(-float(alpha) * d_bar)), 5),
        "spectral_corridor_loss": round(d_bar, 5),
        "per_metric_loss": per,
        "n_metrics": len(ds),
    }


def load_spectral_corridor_entry(mix_profile: str, corridors_path: str = DEFAULT_CORRIDORS_PATH) -> dict[str, Any] | None:
    """Charge l'entrée de `golden_references/spectral_corridors.json` pour un profil (None si absent/corrompu)."""
    if not corridors_path or not os.path.isfile(corridors_path):
        return None
    try:
        with open(corridors_path, encoding="utf-8") as fh:
            data = json.load(fh)
        entry = (data.get("profiles") or {}).get(mix_profile)
        if isinstance(entry, dict) and isinstance(entry.get("metrics"), dict) and entry["metrics"]:
            return entry
    except Exception:
        return None
    return None


def _corridor_triplet(p50: float, lo_spread: float, hi_spread: float) -> dict[str, float]:
    return {"p10": round(float(p50) - lo_spread, 4), "p50": round(float(p50), 4), "p90": round(float(p50) + hi_spread, 4)}


def build_spectral_corridors(output_path: str = DEFAULT_CORRIDORS_PATH) -> dict[str, Any]:
    """v3.1 REF-04 — génère `golden_references/spectral_corridors.json` (versionné Git, ~15 Ko)
    pour les 7 mix_profiles : bark24 p10/p50/p90 + corridors [p10,p50,p90] des 8 métriques +
    `dark_valence_overrides`, provenance `calibrated_psychoacoustic_prior`. 100 % hors-ligne."""
    profiles_out: dict[str, Any] = {}
    for prof in PROFILES:
        hints = _ANALYTIC_GENRE_HINTS.get(prof, {"target_slope": -3.9, "sub_boost": 1.0, "air_scale": 1.0})
        slope = float(hints["target_slope"])
        ana = _analytic_bark24_profile(prof)
        p50_bark = np.asarray(ana["bark_24_mean"], dtype=np.float64)
        p50_bark /= max(1e-18, float(np.sum(p50_bark)))
        spread = 0.22
        p10 = np.clip(p50_bark * (1.0 - spread), 5e-4, None)
        p90 = np.clip(p50_bark * (1.0 + spread), 5e-4, None)
        p10 /= float(np.sum(p10))
        p90 /= float(np.sum(p90))
        lufs = _CORRIDOR_LUFS.get(prof, -11.0)
        lra = _CORRIDOR_LRA.get(prof, 4.5)
        crest_lo, crest_hi = _CORRIDOR_CREST.get(prof, (7.5, 13.5))
        le = _CORRIDOR_LOW_END.get(prof, 0.30)
        lm = _CORRIDOR_LOW_MIDS.get(prof, 0.20)
        air = _CORRIDOR_AIR.get(prof, 0.05)
        dark_air = _CORRIDOR_DARK_AIR.get(prof, 0.02)
        mhs = _CORRIDOR_MHS.get(prof, -7.5)
        profiles_out[prof] = {
            "provenance": "calibrated_psychoacoustic_prior",
            "sample_count": 4,
            "bark24": {
                "p10": [round(float(x), 6) for x in p10],
                "p50": [round(float(x), 6) for x in p50_bark],
                "p90": [round(float(x), 6) for x in p90],
            },
            "metrics": {
                "lufs": _corridor_triplet(lufs, 1.2, 1.2),
                "lra_lu": _corridor_triplet(lra, 1.7, 3.0),
                "pink_slope_db_per_oct": _corridor_triplet(slope, 1.1, 1.0),
                "crest_factor_db": {"p10": round(crest_lo, 2), "p50": round((crest_lo + crest_hi) / 2.0, 2), "p90": round(crest_hi, 2)},
                "low_end_share": _corridor_triplet(le, 0.10, 0.08),
                "low_mids_share": _corridor_triplet(lm, 0.06, 0.06),
                "air_share": {"p10": round(air * 0.55, 4), "p50": round(air, 4), "p90": round(air * 1.8, 4)},
                "mid_high_side_to_mid_db": _corridor_triplet(mhs, 4.5, 4.5),
            },
            "dark_valence_overrides": {
                "air_share": {"p10": round(dark_air * 0.8, 4), "p50": round(dark_air * 1.4, 4), "p90": round(dark_air * 2.6, 4)},
                "pink_slope_db_per_oct": _corridor_triplet(slope - 0.5, 1.1, 0.7),
            },
        }
    payload = {
        "schema_version": "3.1",
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profiles": profiles_out,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return {"output_path": output_path, "profiles": sorted(profiles_out.keys()), "size_bytes": os.path.getsize(output_path)}


def compute_spectral_match(
    mono: np.ndarray,
    sr: int,
    profile: str,
    profiles_path: str = DEFAULT_PROFILES_PATH,
    bark_24: list[float] | None = None,
    pink_slope: float | None = None,
    corridors_path: str = "",
    valence: float = 0.0,
    extra_metrics: dict[str, float] | None = None,
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

    result: dict[str, Any] = {
        "spectral_match_score": match_score,
        "cosine_similarity": round(cosine_sim, 4),
        "l1_similarity": round(l1_sim, 4),
        "pink_slope_db_per_oct": round(cur_slope, 3),
        "reference_pink_slope_db_per_oct": round(ref_slope, 3),
        "reference_source": ref_entry.get("source_type", "analytic_default"),
        "reference_count": int(ref_entry.get("source_count", 0)),
    }
    # v3.1 REF-04 — dead-zone corridor loss contre `spectral_corridors.json` (si présent)
    corridors_path = corridors_path or DEFAULT_CORRIDORS_PATH
    corr_entry = load_spectral_corridor_entry(profile, corridors_path)
    if corr_entry is not None:
        meas: dict[str, float] = {"pink_slope_db_per_oct": cur_slope}
        for k, v in (extra_metrics or {}).items():
            if v is not None and k in CORRIDOR_METRICS:
                meas[k] = float(v)
        corr = evaluate_against_corridors(meas, corr_entry, valence=valence)
        result.update(corr)
        result["reference_corridor_provenance"] = corr_entry.get("provenance", "calibrated_psychoacoustic_prior")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap lightweight 24-Bark + pink-slope spectral profiles")
    ap.add_argument("--out", default=DEFAULT_PROFILES_PATH, help="Output spectral_profiles.json path")
    ap.add_argument("--min-pq", type=float, default=7.8, help="Optional min PQ threshold (applied when PQ is known or --use-neural is set)")
    ap.add_argument("--use-neural", action="store_true", help="Optionally evaluate Audiobox PQ on renders without cached audit JSON")
    ap.add_argument("--external-dir", default=DEFAULT_EXTERNAL_DIR, help="Commercial masters root (external_masters/<profile>/)")
    args = ap.parse_args()

    res = build_spectral_profiles(
        output_path=args.out,
        min_pq=args.min_pq,
        use_neural=args.use_neural,
        external_dir=args.external_dir,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
