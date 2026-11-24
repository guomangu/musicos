#!/usr/bin/env python3
"""
Inter-Project Catalog & EP/Album/DJ-Set Coherence Auditor (`catalog_coherence.py`, Phase 6 §3.5).

Scale S5 / Axis H (Album / EP / DJ Set, 0 GPU):
  Reads `<project>/renders/master_fingerprint.json` (~1 KB each) across a directory
  of projects and audits:
    1. Sound Homogeneity: integrated LUFS spread <= 1.5 dB, pink slope spread <= 1.2 dB/oct,
       and Bark 24-band pairwise spectral consistency.
    2. Compositional Diversity: alerts when two tracks share BPM (±4 BPM), same key, and
       same harmonic degree progression.
    3. Camelot Transition Compatibility (EP / DJ Set / Album): scores harmonic distance <= 1
       on the circle of fifths and computes an optimal tracklist order.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import os
import re
from typing import Any

import numpy as np

# Standard Camelot Wheel mapping (tonic semitone 0..11 + mode ring A=minor / B=major)
_NOTE_TO_SEMITONE: dict[str, int] = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}

_MINOR_CAMELOT: dict[int, str] = {
    8: "1A",   # G# / Ab minor
    3: "2A",   # D# / Eb minor
    10: "3A",  # A# / Bb minor
    5: "4A",   # F minor
    0: "5A",   # C minor
    7: "6A",   # G minor
    2: "7A",   # D minor
    9: "8A",   # A minor
    4: "9A",   # E minor
    11: "10A", # B minor
    6: "11A",  # F# / Gb minor
    1: "12A",  # C# / Db minor
}

_MAJOR_CAMELOT: dict[int, str] = {
    11: "1B",  # B major
    6: "2B",   # F# / Gb major
    1: "3B",   # C# / Db major
    8: "4B",   # G# / Ab major
    3: "5B",   # D# / Eb major
    10: "6B",  # A# / Bb major
    5: "7B",   # F major
    0: "8B",   # C major
    7: "9B",   # G major
    2: "10B",  # D major
    9: "11B",  # A major
    4: "12B",  # E major
}


def key_to_camelot(key_str: str) -> str:
    """Convert key strings like `'C minor'`, `'F#m'`, `'Ab major'`, `'A_dorian'` into Camelot code."""
    s = str(key_str or "").strip()
    if not s:
        return ""
    if re.fullmatch(r"(1[0-2]|[1-9])[ABab]", s):
        return s.upper()

    norm = s.replace("_", " ").replace("-", " ").strip()
    # Handle shorthand like F#m or Bbm or Cm
    m_short = re.match(r"^([A-Ga-g][#b♭♯]?)(m|min|minor|maj|major)?(?:\s+(.*))?$", norm, flags=re.IGNORECASE)
    if not m_short:
        return ""
    root_raw = m_short.group(1).replace("♭", "b").replace("♯", "#")
    root_up = root_raw[0].upper() + root_raw[1:].lower()
    if len(root_up) == 2 and root_up[1] == "b":
        key_lut = root_up[0] + "B"
    else:
        key_lut = root_up.upper()
    semi = _NOTE_TO_SEMITONE.get(key_lut)
    if semi is None:
        return ""

    qual = " ".join(x for x in (m_short.group(2) or "", m_short.group(3) or "") if x).lower().strip()
    minor_kw = ("m", "min", "minor", "aeolian", "dorian", "phrygian", "harmonic", "melodic", "locrian")
    major_kw = ("maj", "major", "ionian", "lydian", "mixolydian")
    if any(re.search(rf"\b{k}\b", qual) for k in minor_kw) or qual == "m":
        return _MINOR_CAMELOT[semi]
    if any(re.search(rf"\b{k}\b", qual) for k in major_kw):
        return _MAJOR_CAMELOT[semi]
    # Default plain note letter to Major unless 'm' present
    return _MAJOR_CAMELOT[semi]


def camelot_distance(cam_a: str, cam_b: str) -> int | None:
    """Compute harmonic distance on the Camelot Wheel (`0`=identical, `1`=harmonic compatible)."""
    ma = re.fullmatch(r"(1[0-2]|[1-9])([AB])", str(cam_a or "").strip().upper())
    mb = re.fullmatch(r"(1[0-2]|[1-9])([AB])", str(cam_b or "").strip().upper())
    if not ma or not mb:
        return None
    n1, r1 = int(ma.group(1)), ma.group(2)
    n2, r2 = int(mb.group(1)), mb.group(2)
    diff = abs(n1 - n2)
    wheel = min(diff, 12 - diff)
    return wheel + (0 if r1 == r2 else 1)


def build_master_fingerprint(
    *,
    project_name: str,
    project_dir: str,
    mix_profile: str,
    bpm: float,
    key: str,
    degree_progression: list[Any] | str | None,
    integrated_lufs: float,
    true_peak_dbtp: float,
    crest_factor_db: float,
    pink_slope_db_per_oct: float,
    bark24: list[float],
    duration_s: float,
) -> dict[str, Any]:
    """Compile lightweight `master_fingerprint.json` (~1 KB) for S5/H catalog checks."""
    if isinstance(degree_progression, list):
        prog_norm = "-".join(str(x).strip() for x in degree_progression if str(x).strip())
    else:
        prog_norm = str(degree_progression or "").strip()

    cam = key_to_camelot(key)
    return {
        "schema_version": 1,
        "project_name": str(project_name),
        "project_dir": os.path.abspath(os.path.expanduser(project_dir)),
        "mix_profile": str(mix_profile),
        "bpm": round(float(bpm), 2),
        "key": str(key or ""),
        "camelot_code": cam,
        "degree_progression": prog_norm,
        "integrated_lufs": round(float(integrated_lufs), 2),
        "true_peak_dbtp": round(float(true_peak_dbtp), 2),
        "crest_factor_db": round(float(crest_factor_db), 2),
        "pink_slope_db_per_oct": round(float(pink_slope_db_per_oct), 3),
        "bark24": [round(float(x), 5) for x in bark24[:24]],
        "duration_s": round(float(duration_s), 2),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def discover_fingerprints(root_dir: str) -> list[dict[str, Any]]:
    """Find and load all `renders/master_fingerprint.json` files inside `root_dir`."""
    abs_root = os.path.abspath(os.path.expanduser(root_dir))
    if not os.path.isdir(abs_root):
        raise FileNotFoundError(f"Catalog root directory not found: {abs_root}")

    found_paths: list[str] = []
    # Support passing either a catalog parent folder (containing multiple projects)
    # or scanning recursively for renders/master_fingerprint.json
    direct = os.path.join(abs_root, "renders", "master_fingerprint.json")
    if os.path.isfile(direct):
        found_paths.append(direct)
    for cur, dirs, files in os.walk(abs_root):
        dirs.sort()
        if "master_fingerprint.json" in files:
            cand = os.path.join(cur, "master_fingerprint.json")
            if cand not in found_paths:
                found_paths.append(cand)

    items: list[dict[str, Any]] = []
    for p in sorted(found_paths):
        with open(p, encoding="utf-8") as fh:
            obj = json.load(fh)
        if isinstance(obj, dict):
            obj.setdefault("project_name", os.path.basename(os.path.dirname(os.path.dirname(p))))
            if not obj.get("camelot_code") and obj.get("key"):
                obj["camelot_code"] = key_to_camelot(str(obj["key"]))
            obj["fingerprint_path"] = p
            items.append(obj)
    return items


def _transition_cost(t1: dict[str, Any], t2: dict[str, Any], mode: str) -> tuple[float, dict[str, Any]]:
    cam1 = str(t1.get("camelot_code") or key_to_camelot(str(t1.get("key") or "")))
    cam2 = str(t2.get("camelot_code") or key_to_camelot(str(t2.get("key") or "")))
    cdist = camelot_distance(cam1, cam2)
    bpm_diff = abs(float(t1.get("bpm", 120.0)) - float(t2.get("bpm", 120.0)))
    lufs_diff = abs(float(t1.get("integrated_lufs", -14.0)) - float(t2.get("integrated_lufs", -14.0)))

    cam_pen = 0.0 if cdist is None else (0.0 if cdist <= 1 else (cdist - 1) * 4.0)
    bpm_wt = 0.12 if mode == "dj_set" else (0.06 if mode == "ep" else 0.03)
    cost = cam_pen + bpm_wt * bpm_diff + 0.3 * max(0.0, lufs_diff - 1.0)
    compatible = (cdist is None or cdist <= 1) and (bpm_diff <= (14.0 if mode == "dj_set" else 30.0))
    info = {
        "from": t1.get("project_name"),
        "to": t2.get("project_name"),
        "from_camelot": cam1 or None,
        "to_camelot": cam2 or None,
        "camelot_distance": cdist,
        "bpm_delta": round(bpm_diff, 2),
        "lufs_delta": round(lufs_diff, 2),
        "compatible": compatible,
    }
    return cost, info


def _optimize_tracklist(tracks: list[dict[str, Any]], mode: str) -> tuple[list[int], list[dict[str, Any]]]:
    n = len(tracks)
    if n <= 1:
        return list(range(n)), []

    cost_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                c, _ = _transition_cost(tracks[i], tracks[j], mode)
                cost_mat[i, j] = c

    if n <= 8:
        best_order: list[int] = list(range(n))
        best_cost = 1e18
        for perm in itertools.permutations(range(n)):
            c = sum(cost_mat[perm[k], perm[k + 1]] for k in range(n - 1))
            if c < best_cost - 1e-9:
                best_cost = c
                best_order = list(perm)
    else:
        # Multi-start greedy nearest-neighbor for larger catalogs
        best_order = list(range(n))
        best_cost = 1e18
        for start in range(n):
            used = {start}
            order = [start]
            tot = 0.0
            while len(order) < n:
                cur = order[-1]
                nxt = min((j for j in range(n) if j not in used), key=lambda j: cost_mat[cur, j])
                tot += cost_mat[cur, nxt]
                used.add(nxt)
                order.append(nxt)
            if tot < best_cost:
                best_cost = tot
                best_order = order

    edges = [_transition_cost(tracks[best_order[k]], tracks[best_order[k + 1]], mode)[1] for k in range(n - 1)]
    return best_order, edges


def _norm_progression(prog: Any) -> str:
    if isinstance(prog, list):
        toks = [str(x).strip().lower() for x in prog if str(x).strip()]
        return "-".join(toks)
    return re.sub(r"\s+", "", str(prog or "").strip().lower())


def audit_catalog_coherence(
    root_dir: str = "",
    *,
    mode: str = "ep",
    fingerprints: list[dict[str, Any]] | None = None,
    lufs_spread_max_db: float = 1.5,
    pink_slope_spread_max: float = 1.2,
    out_path: str = "",
) -> dict[str, Any]:
    """
    Audit inter-project catalog coherence across 1..N `master_fingerprint.json` tracks.
    Modes: `ep`, `album`, `dj_set`.
    """
    eff_mode = mode if mode in ("ep", "album", "dj_set") else "ep"
    tracks = list(fingerprints) if fingerprints is not None else discover_fingerprints(root_dir)

    if len(tracks) < 2:
        return {
            "mode": eff_mode,
            "track_count": len(tracks),
            "verdict": "SKIP" if len(tracks) == 0 else "PASS",
            "score": 1.0,
            "sound_homogeneity": {"status": "SKIP", "passed": True},
            "compositional_diversity": {"status": "SKIP", "passed": True, "collisions": []},
            "transition_compatibility": {"status": "SKIP", "passed": True},
            "recommended_tracklist": [t.get("project_name") for t in tracks],
            "deviations": [],
        }

    # 1. Sound Homogeneity
    lufs_vals = [float(t.get("integrated_lufs", -14.0)) for t in tracks]
    slope_vals = [float(t.get("pink_slope_db_per_oct", -4.0)) for t in tracks]
    lufs_spread = round(max(lufs_vals) - min(lufs_vals), 3)
    slope_spread = round(max(slope_vals) - min(slope_vals), 3)

    bark_cosines: list[float] = []
    bark_vecs = []
    for t in tracks:
        bv = np.asarray(t.get("bark24") or [], dtype=np.float64)
        if len(bv) == 24 and np.linalg.norm(bv) > 1e-8:
            bark_vecs.append(bv / np.linalg.norm(bv))
    for i in range(len(bark_vecs)):
        for j in range(i + 1, len(bark_vecs)):
            bark_cosines.append(float(np.dot(bark_vecs[i], bark_vecs[j])))
    min_bark_cos = round(min(bark_cosines), 4) if bark_cosines else 1.0

    lufs_ok = lufs_spread <= float(lufs_spread_max_db)
    slope_ok = slope_spread <= float(pink_slope_spread_max)
    homog_ok = lufs_ok and slope_ok
    homog_score = 1.0
    if not lufs_ok:
        homog_score -= min(0.45, 0.15 * (lufs_spread - float(lufs_spread_max_db)))
    if not slope_ok:
        homog_score -= min(0.35, 0.15 * (slope_spread - float(pink_slope_spread_max)))
    homog_score = round(max(0.20, homog_score), 3)

    # 2. Compositional Diversity (alert if two tracks share BPM ±4, same key, and same progression)
    collisions: list[dict[str, Any]] = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            ti, tj = tracks[i], tracks[j]
            bpm_close = abs(float(ti.get("bpm", 0.0)) - float(tj.get("bpm", 999.0))) <= 4.0
            ci = str(ti.get("camelot_code") or key_to_camelot(str(ti.get("key") or "")))
            cj = str(tj.get("camelot_code") or key_to_camelot(str(tj.get("key") or "")))
            same_key = (bool(ci) and ci == cj) or (
                bool(ti.get("key")) and str(ti.get("key")).strip().lower() == str(tj.get("key")).strip().lower()
            )
            pi = _norm_progression(ti.get("degree_progression"))
            pj = _norm_progression(tj.get("degree_progression"))
            same_prog = bool(pi) and pi == pj
            if bpm_close and same_key and same_prog:
                collisions.append({
                    "track_a": ti.get("project_name"),
                    "track_b": tj.get("project_name"),
                    "bpm_a": ti.get("bpm"),
                    "bpm_b": tj.get("bpm"),
                    "key": ti.get("key") or ci,
                    "degree_progression": pi,
                })
    div_ok = len(collisions) == 0
    div_score = 1.0 if div_ok else round(max(0.30, 0.72 - 0.15 * len(collisions)), 3)

    # 3. Camelot Transition Compatibility & Optimal Tracklist Ordering
    best_idx_order, opt_edges = _optimize_tracklist(tracks, eff_mode)
    recommended_names = [str(tracks[i].get("project_name")) for i in best_idx_order]
    compat_count = sum(1 for e in opt_edges if e["compatible"])
    cam_ok_ratio = compat_count / float(max(1, len(opt_edges)))
    # Also compute current order edges so user sees before/after
    cur_edges = [_transition_cost(tracks[k], tracks[k + 1], eff_mode)[1] for k in range(len(tracks) - 1)]
    cur_compat = sum(1 for e in cur_edges if e["compatible"])

    # Consider transition gate strict in ep/dj_set modes if optimal sequence still has Camelot breaks
    camelot_strict = eff_mode in ("ep", "dj_set")
    trans_ok = (compat_count == len(opt_edges)) if camelot_strict else (cam_ok_ratio >= 0.5)
    trans_score = round(0.4 + 0.6 * cam_ok_ratio, 3)

    deviations: list[dict[str, Any]] = []
    if not homog_ok:
        deviations.append({
            "scale": "S5_context",
            "axis": "H",
            "location": "catalog_mastering",
            "metric": f"lufs_spread={lufs_spread}dB, slope_spread={slope_spread}",
            "diagnosis": "Master LUFS or pink slope inconsistency across EP/album tracks",
            "next_action": "MATCH_CATALOG_LOUDNESS_AND_SLOPE",
            "blocking": False,
            "severity": 0.75,
        })
    if not div_ok:
        c0 = collisions[0]
        deviations.append({
            "scale": "S5_context",
            "axis": "H",
            "location": f"{c0['track_a']} vs {c0['track_b']}",
            "metric": f"bpm={c0['bpm_a']}/{c0['bpm_b']}, key={c0['key']}, prog={c0['degree_progression']}",
            "diagnosis": "Two tracks share ±4 BPM, same key, and identical degree progression",
            "next_action": "DIVERSIFY_KEY_OR_TEMPO",
            "blocking": False,
            "severity": 0.70,
        })
    if camelot_strict and not trans_ok:
        bad_edges = [e for e in opt_edges if not e["compatible"]]
        b0 = bad_edges[0] if bad_edges else {}
        deviations.append({
            "scale": "S5_context",
            "axis": "H",
            "location": f"{b0.get('from')} -> {b0.get('to')}",
            "metric": f"camelot_dist={b0.get('camelot_distance')} (>1)",
            "diagnosis": "No tracklist order satisfies Camelot distance <= 1 for all transitions",
            "next_action": "TRANSPOSE_OR_REORDER_TRACKLIST",
            "blocking": False,
            "severity": 0.55,
        })

    overall_score = round(0.40 * homog_score + 0.35 * div_score + 0.25 * trans_score, 3)
    verdict = "PASS" if (homog_ok and div_ok and trans_ok) else "WARN"
    if not lufs_ok or len(collisions) > 0:
        verdict = "FIX"

    report = {
        "mode": eff_mode,
        "root_dir": os.path.abspath(os.path.expanduser(root_dir)) if root_dir else "",
        "track_count": len(tracks),
        "verdict": verdict,
        "score": overall_score,
        "sound_homogeneity": {
            "passed": homog_ok,
            "score": homog_score,
            "lufs_spread_db": lufs_spread,
            "lufs_spread_max_db": lufs_spread_max_db,
            "pink_slope_spread_db_oct": slope_spread,
            "pink_slope_spread_max": pink_slope_spread_max,
            "bark24_min_cosine": min_bark_cos,
        },
        "compositional_diversity": {
            "passed": div_ok,
            "score": div_score,
            "collisions": collisions,
        },
        "transition_compatibility": {
            "passed": trans_ok,
            "score": trans_score,
            "original_compatible_edges": f"{cur_compat}/{len(cur_edges)}",
            "optimal_compatible_edges": f"{compat_count}/{len(opt_edges)}",
            "optimal_transitions": opt_edges,
        },
        "recommended_tracklist": recommended_names,
        "deviations": deviations,
    }

    if out_path.strip():
        abs_out = os.path.abspath(os.path.expanduser(out_path))
        os.makedirs(os.path.dirname(abs_out), exist_ok=True)
        with open(abs_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        report["report_path"] = abs_out

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="S5/H Catalog & EP/Album/DJ-Set Coherence Audit.")
    ap.add_argument("dir_projets", help="Root folder containing project directories with renders/master_fingerprint.json")
    ap.add_argument("--mode", default="ep", choices=["ep", "album", "dj_set"])
    ap.add_argument("--out", default="", help="Optional output JSON report path")
    args = ap.parse_args()

    res = audit_catalog_coherence(args.dir_projets, mode=args.mode, out_path=args.out)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
