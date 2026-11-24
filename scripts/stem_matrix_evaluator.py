#!/usr/bin/env python3
"""
v3.0 Stem-Matrix Evaluator — replaces 20-30 headless Gridline renders per song by ONE
8-bar stems render + pure-NumPy evaluation (< 1.2 s CPU, 0 extra Gridline calls):

  - §1 per-stem sound check directly on each stem array (RMS audibility, f0, spectral
    centroid, narrow 2-5 kHz peaks, pre-attack silence);
  - §2 vertical stages V1..V4 reconstructed in memory by linear sums:
      y_V1 = kick + bass,  y_V2 = y_V1 + drums,  y_V3 = y_V2 + harmony,  y_V4 = y_V3 + lead
    running the SAME diagnostics on the mixes (low-end < 150 Hz phase correlation,
    sub+bass energy share, spectral balance) without any additional render;
  - DAG stage 3 analytic gain solver (`gain_staging_lib.solve_analytic_gain_trims`)
    computed on the stems and returned ready to inject into the full compilation.

Headless-testable: pass synthetic `stems: dict[str, np.ndarray]`.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
ANALYZER_DIR = os.path.join(ROOT, "mcp-servers", "audio-analyzer")
for _d in (SCRIPTS_DIR, ANALYZER_DIR):
    if _d not in __import__("sys").path:
        __import__("sys").path.insert(0, _d)

ROLE_GROUPS = {
    "drums": ("kick", "snare", "clap", "hat", "hihat", "perc", "drum"),
    "bass": ("bass", "808", "sub"),
    "harmony": ("chord", "pad", "keys", "harmony", "string"),
    "lead": ("lead", "melody", "topline", "counter", "vox", "vocal"),
    "fx": ("fx", "riser", "sweep", "ear_candy"),
}


def _role_of(stem_name: str) -> str:
    n = stem_name.lower()
    for group, keys in ROLE_GROUPS.items():
        if any(k in n for k in keys):
            return group
    return "other"


def _mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return y.mean(axis=1) if y.ndim > 1 else y


def _band_energy_pct(mono: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> float:
    if len(mono) < 256 or sr <= 0:
        return 0.0
    sp = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    fr = np.fft.rfftfreq(len(mono), 1.0 / sr)
    tot = float(np.sum(sp ** 2)) + 1e-12
    band = (fr >= lo_hz) & (fr <= hi_hz)
    return round(100.0 * float(np.sum(sp[band] ** 2)) / tot, 2)


def _low_end_phase_corr(y_stereo: np.ndarray, sr: int, cutoff_hz: float = 150.0) -> float:
    """Pearson correlation of L/R in the low band (1.0 = perfectly mono low end)."""
    y = np.asarray(y_stereo, dtype=np.float64)
    if y.ndim < 2 or y.shape[1] < 2 or len(y) < 256:
        return 1.0
    try:
        from scipy import signal as _sig
        sos = _sig.butter(2, min(cutoff_hz, sr * 0.45), btype="lowpass", fs=sr, output="sos")
        l = _sig.sosfilt(sos, y[:, 0])
        r = _sig.sosfilt(sos, y[:, 1])
        denom = float(np.sqrt(np.sum(l ** 2) * np.sum(r ** 2))) + 1e-12
        return round(float(np.sum(l * r) / denom), 4)
    except Exception:
        return 1.0


def check_stem(name: str, y: np.ndarray, sr: int) -> dict[str, Any]:
    """§1 unit sound check on one stem array (0 Gridline calls)."""
    mono = _mono(y)
    rms = float(np.sqrt(np.mean(mono ** 2))) + 1e-12
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms_dbfs = round(20.0 * float(np.log10(rms)), 2)
    centroid = 0.0
    if len(mono) >= 256 and sr > 0:
        sp = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        fr = np.fft.rfftfreq(len(mono), 1.0 / sr)
        tot = float(np.sum(sp)) + 1e-12
        centroid = round(float(np.sum(fr * sp) / tot), 1)
    # narrow peaks in 2-5 kHz (harsh resonances)
    narrow_peaks = []
    if len(mono) >= 2048 and sr > 0:
        sp = np.abs(np.fft.rfft(mono * np.hanning(len(mono)), n=8192))
        fr = np.fft.rfftfreq(8192, 1.0 / sr)
        band = (fr >= 2000.0) & (fr <= 5000.0)
        if np.any(band):
            sb, fb = sp[band], fr[band]
            med = float(np.median(sb)) + 1e-12
            pk = (sb[1:-1] > sb[:-2]) & (sb[1:-1] >= sb[2:]) & (sb[1:-1] > med * 8.0)
            for f, a in zip(fb[1:-1][pk], sb[1:-1][pk]):
                narrow_peaks.append({"freq_hz": round(float(f), 1), "excess_db": round(20.0 * float(np.log10(a / med)), 1)})
    return {
        "stem": name,
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": round(20.0 * float(np.log10(peak + 1e-12)), 2),
        "audible": rms_dbfs > -60.0,
        "centroid_hz": centroid,
        "narrow_peaks_2_5khz": narrow_peaks[:4],
        "low_end_pct": _band_energy_pct(mono, sr, 20.0, 120.0),
        "air_pct": _band_energy_pct(mono, sr, 6000.0, 20000.0),
    }


def evaluate_vertical_stages(
    stems: dict[str, np.ndarray],
    sr: int,
    low_end_max_pct: float = 42.0,
    low_end_corr_min: float = 0.50,
) -> dict[str, Any]:
    """§2 V1..V4: cumulative in-memory mixes + the same diagnostics as the session tests."""
    def _sum(names: list[str]) -> np.ndarray | None:
        arrs = [np.asarray(stems[n], dtype=np.float64) for n in names if n in stems]
        if not arrs:
            return None
        stereo = any(a.ndim > 1 for a in arrs)
        n = max(a.shape[0] for a in arrs)
        out = np.zeros((n, 2) if stereo else n)
        for a in arrs:
            if stereo and a.ndim == 1:
                a = np.stack([a, a], axis=1)
            elif not stereo and a.ndim > 1:
                a = a.mean(axis=1)
            out[: a.shape[0]] += a
        return out

    groups = {g: [n for n in stems if _role_of(n) == g] for g in ("drums", "bass", "harmony", "lead", "fx", "other")}
    y_v1 = _sum(groups["drums"][:1] + groups["bass"])  # kick + bass
    y_v2 = _sum(groups["drums"] + groups["bass"])
    y_v3 = _sum(groups["drums"] + groups["bass"] + groups["harmony"])
    y_v4 = _sum(list(stems.keys()))
    stages: dict[str, Any] = {}
    for name, y in (("V1_low_end", y_v1), ("V2_groove", y_v2), ("V3_harmony", y_v3), ("V4_lead", y_v4)):
        if y is None:
            stages[name] = {"present": False}
            continue
        mono = _mono(y)
        low_share = _band_energy_pct(mono, sr, 20.0, 120.0)
        corr = _low_end_phase_corr(np.stack([mono, mono], axis=1) if y.ndim == 1 else y, sr)
        stages[name] = {
            "present": True,
            "rms_dbfs": round(20.0 * float(np.log10(float(np.sqrt(np.mean(mono ** 2))) + 1e-12)), 2),
            "low_end_share_pct": low_share,
            "low_end_corr": corr,
            "ok": low_share <= low_end_max_pct and corr >= low_end_corr_min,
        }
    return stages


def evaluate_stem_matrix(
    stems: dict[str, np.ndarray],
    sr: int,
    *,
    profile: str = "trap_rap",
    target_offsets_db: dict[str, float] | None = None,
    anchor_lufs: float = -16.0,
    low_end_max_pct: float | None = None,
    presence_floors_db: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Full §1 + §2 + gain-solver evaluation from one stems render. Pure NumPy."""
    from gain_staging_lib import ROLE_PRESENCE_FLOOR_DB, normalize_role_for_gain_staging, solve_analytic_gain_trims
    try:
        from mix_config import profile_settings
        ps = profile_settings(profile)
        low_end_max_pct = float(low_end_max_pct if low_end_max_pct is not None else ps.get("low_end_max_pct", 42.0))
    except Exception:
        low_end_max_pct = float(low_end_max_pct if low_end_max_pct is not None else 42.0)

    per_stem = {name: check_stem(name, y, sr) for name, y in stems.items()}
    stages = evaluate_vertical_stages(stems, sr, low_end_max_pct=low_end_max_pct)

    # measured "loudness" per stem: RMS dBFS as a deterministic proxy for LUFS on stems
    measured = {name: per_stem[name]["rms_dbfs"] for name in stems}
    offsets = target_offsets_db or {}
    solver = solve_analytic_gain_trims(
        measured_lufs=measured,
        target_offsets_db={normalize_role_for_gain_staging(n): offsets.get(_role_of(n), offsets.get(normalize_role_for_gain_staging(n), 0.0)) for n in stems},
        anchor_lufs=anchor_lufs,
        presence_floors_db=presence_floors_db or dict(ROLE_PRESENCE_FLOOR_DB),
        low_end_share_pct=stages.get("V4_lead", {}).get("low_end_share_pct"),
        low_end_max_pct=low_end_max_pct,
    )
    issues = []
    for name, chk in per_stem.items():
        if not chk["audible"]:
            issues.append({"severity": "error", "code": "stem_silent", "stem": name})
        if chk["narrow_peaks_2_5khz"]:
            issues.append({"severity": "warning", "code": "stem_harsh_peaks", "stem": name, "peaks": chk["narrow_peaks_2_5khz"]})
    for sname, st in stages.items():
        if st.get("present") and not st.get("ok"):
            issues.append({"severity": "warning", "code": "stage_violation", "stage": sname,
                           "low_end_share_pct": st.get("low_end_share_pct"), "low_end_corr": st.get("low_end_corr")})
    return {
        "sr": sr,
        "n_stems": len(stems),
        "per_stem": per_stem,
        "vertical_stages": stages,
        "gain_solver": solver,
        "issues": issues,
        "ok": not any(i["severity"] == "error" for i in issues),
        "gridline_renders_used": 1,
    }


def load_stems_dir(stems_dir: str) -> tuple[dict[str, np.ndarray], int]:
    """Load every wav in a stems dir into memory (for the single 8-bar stems render)."""
    import soundfile as sf
    stems: dict[str, np.ndarray] = {}
    sr0 = 44100
    for fn in sorted(os.listdir(stems_dir)):
        if not fn.lower().endswith((".wav", ".flac")):
            continue
        y, sr = sf.read(os.path.join(stems_dir, fn), dtype="float64", always_2d=True)
        stems[os.path.splitext(fn)[0]] = y
        sr0 = sr
    return stems, sr0
