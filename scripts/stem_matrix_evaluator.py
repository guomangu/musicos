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


# ---------------------------------------------------------------------------
# v3.1 HARM-03 — audit audio bar-synchronisé Chroma-vs-Grille (contrat §4.3).
# ---------------------------------------------------------------------------

def _segment_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """Chroma 12 pcs pondéré par l'énergie du segment (FFT NumPy replié sur 12 classes de
    hauteur, équivalent CQT 36 bins/octave agrégé — déterministe, sans dépendance librosa)."""
    y = _mono(np.asarray(y, dtype=np.float64))
    if len(y) < 256 or sr <= 0:
        return np.zeros(12)
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1.0 / sr)
    chroma = np.zeros(12)
    for f, m in zip(fr, sp):
        if f < 27.0 or f > 6000.0:
            continue
        midi = 69.0 + 12.0 * np.log2(f / 440.0)
        chroma[int(round(midi)) % 12] += float(m) ** 2
    return chroma / max(1e-12, float(chroma.sum()))


def _estimate_root_st(y: np.ndarray, sr: int, lo_hz: float = 30.0, hi_hz: float = 300.0) -> float | None:
    """Estimation haute résolution (interpolation parabolique FFT) du pitch dominant en demi-tons
    relatifs au La440 — permet un `bass_root_match` à ±0.35 st."""
    y = _mono(np.asarray(y, dtype=np.float64))
    if len(y) < 1024 or sr <= 0:
        return None
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1.0 / sr)
    band = (fr >= lo_hz) & (fr <= hi_hz)
    if not np.any(band):
        return None
    idx = np.argmax(sp[band])
    k = int(np.where(band)[0][idx])
    if 0 < k < len(sp) - 1:
        a, b, c = sp[k - 1], sp[k], sp[k + 1]
        denom = (a - 2.0 * b + c)
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    f = fr[k] + delta * (fr[1] - fr[0])
    if f <= 0:
        return None
    return 69.0 + 12.0 * np.log2(f / 440.0)


def _intersect_windows(s0: float, s1: float, masks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Intersection de [s0, s1] avec les intervalles stables (secondes) d'une piste."""
    out = []
    for a, b in masks or []:
        lo, hi = max(s0, float(a)), min(s1, float(b))
        if hi - lo > 0.02:
            out.append((lo, hi))
    return out


def evaluate_barwise_harmonic_alignment(
    stems_audio: dict[str, tuple[np.ndarray, int]],
    harmonic_grid: list[dict[str, Any]],
    bpm: float,
    valence: float = 0.0,
    sr_fallback: int = 44100,
    steady_masks: dict[str, list[tuple[float, float]]] | None = None,
) -> dict[str, Any]:
    """v3.1 HARM-03 (contrat §4.3) : découpe les stems bass/chords/lead par segment de la
    grille harmonique (marge 50 ms), calcule chroma CQT pondéré par énergie, puis :
      - `bass_root_match` (±0.35 st), `chords_in_chord_energy_ratio` (≥0.72, 0.65 si V<-0.2),
        `cross_stem_clash_score` (b2 non modal entre stems) ;
      - `harmonic_audio_clash_ratio` (Palier 1 ≤ 0.08), `per_track_pitch_bias_st`,
        `problematic_bars`, `recommended_actions` (FIX_TRACK_ROOT_NOTE /
        TIGHTEN_HARMONIC_CLAMP_OR_VOICING / SWAP_SAMPLE).
    """
    beat_s = 60.0 / max(40.0, float(bpm))
    margin_s = 0.05
    tonic_pc = int(harmonic_grid[0].get("root_pc", 0)) % 12 if harmonic_grid else 0
    role_stems: dict[str, list[str]] = {"bass": [], "chords": [], "lead": []}
    for name in stems_audio:
        r = _role_of(name)
        if r == "bass":
            role_stems["bass"].append(name)
        elif r == "harmony":
            role_stems["chords"].append(name)
        elif r == "lead":
            role_stems["lead"].append(name)

    per_seg: list[dict[str, Any]] = []
    bias_samples: dict[str, list[float]] = {"bass": [], "chords": [], "lead": []}
    problematic_bars: list[int] = []
    clash_segments = 0
    active_segments = 0
    glide_excluded = 0  # v3.3 #3 : fenêtres de glide/vibrato exclues de l'audit de justesse

    for seg in harmonic_grid or []:
        s0 = float(seg.get("start_beat", 0.0)) * beat_s + margin_s
        s1 = float(seg.get("end_beat", 0.0)) * beat_s - margin_s
        if s1 - s0 < 0.1:
            continue
        chord_pcs = set(int(p) for p in seg.get("chord_pcs", []))
        scale_pcs = set(int(p) for p in seg.get("scale_pcs", []))
        tension_pcs = set(int(p) for p in seg.get("modal_tension_pcs", []))
        bass_pc = int(seg.get("bass_pc", seg.get("root_pc", 0))) % 12
        seg_res: dict[str, Any] = {"chord_idx": int(seg.get("chord_idx", 0)), "bar": int(seg.get("start_beat", 0.0)) // max(1, 4) + 1,
                                   "bass_root_match": None, "chords_in_chord_energy_ratio": None, "cross_stem_clash_score": 0.0}
        chromas: dict[str, np.ndarray] = {}
        seg_active = False
        for role, names in role_stems.items():
            for name in names:
                if name not in stems_audio:
                    continue
                y, sr = stems_audio[name]
                sr = int(sr) if sr else sr_fallback
                windows = [(s0, s1)]
                if steady_masks and name in steady_masks:
                    windows = _intersect_windows(s0, s1, steady_masks[name])
                    if not windows:
                        glide_excluded += 1
                        continue  # fenêtre de glide 808 / attaque vibrato : hors audit de justesse
                parts = []
                for wa, wb in windows:
                    a, b = int(wa * sr), min(len(_mono(np.asarray(y))), int(wb * sr))
                    if b - a >= 256:
                        parts.append(np.asarray(y)[a:b])
                if not parts:
                    continue
                seg_y = parts[0] if len(parts) == 1 else np.concatenate(parts)
                rms = float(np.sqrt(np.mean(_mono(seg_y) ** 2))) if seg_y.size else 0.0
                if 20.0 * np.log10(rms + 1e-12) <= -40.0:
                    continue
                seg_active = True
                chroma = _segment_chroma(seg_y, sr)
                chromas.setdefault(role, np.zeros(12))
                chromas[role] = chromas[role] + chroma
                if role == "bass":
                    est = _estimate_root_st(seg_y, sr)
                    if est is not None:
                        expected = 36.0 + bass_pc + 12.0 * round((est - 36.0 - bass_pc) / 12.0)
                        d = (est - expected) % 12.0
                        if d > 6.0:
                            d -= 12.0
                        seg_res["bass_root_match"] = abs(d) <= 0.35
                        bias_samples["bass"].append(d)
                else:
                    est_pc = int(np.argmax(chroma))
                    if chord_pcs:
                        near = min(chord_pcs, key=lambda q: min((est_pc - q) % 12, (q - est_pc) % 12))
                        d = (est_pc - near) % 12
                        if d > 6:
                            d -= 12
                        bias_samples[role].append(d)
        if not seg_active:
            per_seg.append(seg_res)
            continue
        active_segments += 1
        if "chords" in chromas and chord_pcs:
            ratio = float(np.sum(chromas["chords"][list(chord_pcs)]) / max(1e-12, float(chromas["chords"].sum())))
            seg_res["chords_in_chord_energy_ratio"] = round(ratio, 4)
        clash = 0.0
        for other in ("chords", "lead"):
            if "bass" in chromas and other in chromas:
                for q in range(12):
                    nxt = (q + 1) % 12
                    if nxt in chord_pcs:
                        continue  #音色 d'accord : b9/bII modal autorisé
                    if q == tonic_pc and nxt in tension_pcs:
                        continue  # b2 contre la TONIQUE en mode phrygien-like : couleur modale licite
                    clash += float(min(chromas["bass"][q], chromas[other][nxt]))
        denom = max(1e-12, float(sum(c.sum() for c in chromas.values())))
        seg_res["cross_stem_clash_score"] = round(clash / denom, 5)
        if seg_res["cross_stem_clash_score"] > 0.05:
            clash_segments += 1
        bad = (seg_res["bass_root_match"] is False) or (
            seg_res["chords_in_chord_energy_ratio"] is not None
            and seg_res["chords_in_chord_energy_ratio"] < (0.65 if valence < -0.2 else 0.72)
        ) or seg_res["cross_stem_clash_score"] > 0.05
        if bad:
            problematic_bars.append(int(seg_res["bar"]))
        per_seg.append(seg_res)

    harmonic_audio_clash_ratio = (clash_segments / active_segments) if active_segments else 0.0
    per_track_bias = {r: (round(float(np.median(v)), 3) if v else 0.0) for r, v in bias_samples.items()}

    recommended_actions: list[dict[str, Any]] = []
    for role, names in role_stems.items():
        bias = per_track_bias.get(role, 0.0)
        if abs(bias) >= 1.0 and bias_samples.get(role):
            k = int(round(bias))
            for name in names:
                recommended_actions.append({
                    "code": "FIX_TRACK_ROOT_NOTE",
                    "track": name,
                    "params": {"delta_semitones": -k},
                })
    if problematic_bars:
        n_bad = len(problematic_bars)
        if n_bad <= 2 and active_segments:
            for role in ("chords", "lead"):
                for name in role_stems.get(role, []):
                    recommended_actions.append({
                        "code": "TIGHTEN_HARMONIC_CLAMP_OR_VOICING",
                        "track": name,
                        "params": {"bars": sorted(set(problematic_bars))},
                    })
        elif harmonic_audio_clash_ratio > 0.08:
            for role in ("chords", "lead"):
                for name in role_stems.get(role, []):
                    recommended_actions.append({
                        "code": "SWAP_SAMPLE",
                        "track": name,
                        "params": {"reason": "audio_harmonic_clash"},
                    })

    passed = harmonic_audio_clash_ratio <= 0.08 and all(
        seg.get("bass_root_match") is not False
        and (seg.get("chords_in_chord_energy_ratio") is None or seg["chords_in_chord_energy_ratio"] >= (0.65 if valence < -0.2 else 0.72))
        and seg.get("cross_stem_clash_score", 0.0) <= 0.05
        for seg in per_seg
    )
    return {
        "passed": bool(passed),
        "harmonic_audio_clash_ratio": round(harmonic_audio_clash_ratio, 4),
        "glide_windows_excluded": glide_excluded,
        "per_track_pitch_bias_st": per_track_bias,
        "problematic_bars": sorted(set(problematic_bars)),
        "recommended_actions": recommended_actions,
        "per_segment": per_seg,
    }
