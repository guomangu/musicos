#!/usr/bin/env python3
"""
Local Project Sample Catalogger (`index_project_samples.py`).

Scans ONLY `<project_dir>/Samples/` (user-supplied original samples) and optional
`<project_dir>/samples/` derivatives — guaranteeing strict exclusivity (Invariant #1:
zero external samples ever added).

Writes `<project_dir>/samples_catalog.json` with:
  - compatible roles & role fit scores
  - `playable_chromatically`, `sample_nature`, `detected_note`, `midi_root`,
    `recommended_gl_fine_tune_ct`
  - concise 1-line `timbre_card` (<= 140 chars)
  - suggested `sample_prep_lib.py` recipe (e.g. best `slice_s` onset windows if
    `melodic_sequence`, `stabilize_low_mid` if wide phase, fades/LPF if needed).

Usage:
  .venv/bin/python scripts/index_project_samples.py <project_dir> [--with-clap]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import librosa
import numpy as np
import soundfile as sf

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
INDEX_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-index")
for _d in (SCRIPTS_DIR, INDEX_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from audio_metrics import (
    VALID_ROLES,
    analyze_single_file,
    compute_comfort_score,
    compute_role_score,
    estimate_inharmonicity,
    loop_hpss_hnr,
    mean_chroma_cqt,
    validate_midi_root_octave,
)
from indexer import (
    AUDIO_EXTS,
    DEFAULT_ANCHORS_NPZ,
    _infer_category_and_pack,
    _parse_metadata_from_name,
    build_semantic_card_from_clap,
    build_timbre_card_from_dsp,
    load_semantic_anchors,
)
from sample_prep_lib import measure_fft_tuning
from blueprint_safety import classify_loop_nature


def _detect_isolable_slices(path: str, max_slices: int = 2) -> list[dict[str, Any]]:
    """Detect up to `max_slices` clean single-note onset windows `[start_s, end_s]` inside a melodic sequence."""
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = np.mean(y, axis=1)
        dur = len(mono) / float(sr)
        if dur < 0.25:
            return []
        o_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=256)
        frames = librosa.onset.onset_detect(
            onset_envelope=o_env,
            sr=sr,
            hop_length=256,
            delta=0.16,
            wait=max(4, int(0.10 * sr / 256)),
        )
        times = [float(t) for t in librosa.frames_to_time(frames, sr=sr, hop_length=256)]
        if not times or times[0] > 0.12:
            times.insert(0, 0.0)
        times.append(dur)

        candidates: list[tuple[float, float, float]] = []
        for t0, t1 in zip(times[:-1], times[1:]):
            seg_dur = t1 - t0
            if seg_dur < 0.18:
                continue
            end_s = min(t1 - 0.015, t0 + 0.85)
            if end_s - t0 < 0.16:
                continue
            i0 = max(0, int(t0 * sr))
            i1 = min(len(mono), int(end_s * sr))
            seg = mono[i0:i1]
            rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
            if rms < 0.01:
                continue
            # Prefer moderately long, energetic, stable segments
            score = rms * min(1.0, (end_s - t0) / 0.55)
            candidates.append((score, round(t0, 3), round(end_s, 3)))

        candidates.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for idx, (_sc, s0, s1) in enumerate(candidates[:max_slices], start=1):
            out.append({
                "slice_index": idx,
                "slice_s": [s0, s1],
                "duration_s": round(s1 - s0, 3),
            })
        return out
    except Exception:
        return []


def _suggest_prep_recipes(
    filename: str,
    abs_path: str,
    dsp: dict[str, Any],
    needs_fade: bool,
    best_role: str,
) -> list[dict[str, Any]]:
    """Build concrete `sample_prep_lib.py` recipes tailored to detected audio issues."""
    stem = os.path.splitext(filename)[0].lower().replace(" ", "_")
    nature = str(dsp.get("sample_nature") or "unknown")
    r_harsh = float(dsp.get("r_harsh") or 0.0)
    lm_corr = dsp.get("low_mid_corr")
    recipes: list[dict[str, Any]] = []

    if nature == "melodic_sequence":
        slices = _detect_isolable_slices(abs_path, max_slices=2)
        if slices:
            for sl in slices:
                spec: dict[str, Any] = {
                    "src": filename,
                    "out": f"{stem}_note{sl['slice_index']}.wav",
                    "slice_s": sl["slice_s"],
                    "fade_in_ms": 2.0,
                    "fade_out_ms": 22.0,
                }
                if r_harsh > 0.30:
                    spec["lpf_hz"] = 5800.0
                if lm_corr is not None and float(lm_corr) < 0.55:
                    spec["stabilize_low_mid"] = {"cutoff_hz": 650.0, "side_keep_low": 0.18, "side_keep_high": 0.80}
                recipes.append({
                    "reason": f"melodic_sequence -> isoler 1 note stable ({sl['slice_s'][0]}s..{sl['slice_s'][1]}s)",
                    "sample_prep_spec": spec,
                })
            return recipes

    # General hygiene recipe
    spec_h: dict[str, Any] = {"src": filename, "out": f"{stem}_clean.wav"}
    reasons: list[str] = []
    if needs_fade:
        spec_h["fade_in_ms"] = 2.0
        spec_h["fade_out_ms"] = 18.0
        reasons.append("fade anti-clic")
    if r_harsh > 0.32 and best_role not in ("hihat_perc",):
        spec_h["lpf_hz"] = 6000.0 if best_role in ("chord_keys", "pad_atmosphere") else 7500.0
        reasons.append(f"LPF anti-dureté (r_harsh={r_harsh:.2f})")
    if lm_corr is not None and float(lm_corr) < 0.55 and best_role not in ("hihat_perc",):
        spec_h["stabilize_low_mid"] = {"cutoff_hz": 650.0, "side_keep_low": 0.18, "side_keep_high": 0.80}
        reasons.append(f"stabilisation M/S <650Hz (corr={float(lm_corr):.2f})")
    if nature == "poly_chord_or_dyad":
        reasons.append("accord polyphonique : utiliser en texture/one-shot fixe ou synthé gl.* pour mélodie chromatique")

    if reasons:
        recipes.append({
            "reason": "; ".join(reasons),
            "sample_prep_spec": spec_h,
        })
    return recipes


def _scan_project_audio(project_dir: str) -> tuple[list[tuple[str, str, str]], list[dict[str, str]]]:
    """Return (found [(abs, rel, origin)], skipped [{path, reason}]) inside `<proj>/Samples/` and `<proj>/samples/`.

    v3.1 ASSET-05: filtre `asset_hygiene.is_audio_file_safe` (whitelist d'extensions, taille,
    magic bytes) — les fichiers pièges (.wav.zpa, __MACOSX/._*, .DS_Store, Icon\r, < 1 Ko)
    sont rejetés proprement avec une raison, sans stacktrace.
    """
    from asset_hygiene import is_audio_file_safe
    abs_proj = os.path.abspath(os.path.expanduser(project_dir))
    found: list[tuple[str, str, str]] = []
    skipped: list[dict[str, str]] = []
    seen_real: set[str] = set()

    for sub, origin in (("Samples", "user_original"), ("samples", "derived")):
        sub_dir = os.path.join(abs_proj, sub)
        if not os.path.isdir(sub_dir):
            continue
        for root, dirnames, fns in os.walk(sub_dir, followlinks=True):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__MACOSX")
            for fn in sorted(fns):
                if fn.startswith("."):
                    skipped.append({"path": os.path.join(root, fn), "reason": "dotfile"})
                    continue
                full = os.path.join(root, fn)
                ok, reason = is_audio_file_safe(full)
                if not ok:
                    skipped.append({"path": full, "reason": reason})
                    continue
                real = os.path.realpath(full)
                if real in seen_real:
                    continue
                seen_real.add(real)
                rel = os.path.relpath(full, abs_proj)
                found.append((full, rel, origin))
    return found, skipped


def index_project_samples(
    project_dir: str,
    *,
    with_clap: bool = False,
    anchors_path: str = DEFAULT_ANCHORS_NPZ,
    write_catalog: bool = True,
) -> dict[str, Any]:
    """
    Scan `<project_dir>/Samples/` (and `<project_dir>/samples/`), analyze every audio file
    with deterministic DSP (+ optional CLAP if `with_clap=True` and `< 40` files),
    and write `<project_dir>/samples_catalog.json`.
    """
    abs_proj = os.path.abspath(os.path.expanduser(project_dir))
    if not os.path.isdir(abs_proj):
        raise FileNotFoundError(f"Project directory not found: {abs_proj}")

    files, skipped_files = _scan_project_audio(abs_proj)
    use_clap_eff = bool(with_clap and 0 < len(files) < 40)
    anchors = load_semantic_anchors(anchors_path) if use_clap_eff else None
    embedder = None
    if use_clap_eff and anchors is not None:
        try:
            from audio_metrics import ClapEmbedder

            embedder = ClapEmbedder(device="cpu")
        except Exception:
            embedder = None
            use_clap_eff = False

    entries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for abs_p, rel_p, origin in files:
        fn = os.path.basename(abs_p)
        try:
            entry, cvec = analyze_single_file(abs_p, embedder=embedder, with_clap=bool(embedder is not None), max_seconds=12.0)
            dsp = entry["dsp"]
            dur = float(entry.get("duration_sec") or 0.0)
            category, _pack, is_loop = _infer_category_and_pack(rel_p, fn, dur)

            # v3.0: octave sanity on midi_root + loop typing (HPSS/HNR) + CQT chroma + inharmonicity
            midi_root = dsp.get("midi_root")
            octave_corrected = False
            loop_nature = None
            chroma_cqt = None
            harm_ratio = None
            loop_hnr_db = None
            inharmonicity = 0.0
            try:
                y_m, sr_m = librosa.load(abs_p, sr=22050, mono=True, duration=12.0)
                spec_m = np.abs(librosa.stft(y_m, n_fft=2048, hop_length=512))
                fr_m = librosa.fft_frequencies(sr=sr_m, n_fft=2048)
                low_ratio = float(np.sum(spec_m[fr_m < 150.0])) / (float(np.sum(spec_m)) + 1e-12)
                midi_root, octave_corrected = validate_midi_root_octave(
                    midi_root, dsp.get("spectral_centroid_hz"), low_ratio
                )
                if is_loop:
                    harm_ratio, loop_hnr_db, harm_comp = loop_hpss_hnr(y_m, sr_m)
                    loop_nature = classify_loop_nature(harm_ratio, loop_hnr_db)
                    if loop_nature == "melodic_loop":
                        chroma_cqt = mean_chroma_cqt(harm_comp, sr_m)
                inharmonicity = estimate_inharmonicity(y_m, sr_m, dsp.get("detected_f0_hz"))
                # v3.2 TimbralCohesion : DRR natif (wetness/tail) pour pondérer les sends FX
                wet = dsp.get("wetness_score")
                if wet is None:
                    from audio_metrics import compute_wetness_tail
                    wet = compute_wetness_tail(y_m, sr_m)["wetness_score"]
            except Exception:
                pass

            # BPM : d'abord le nom de fichier, puis estimation librosa pour les boucles longues
            bpm_tag, _key_tag = _parse_metadata_from_name(f"{rel_p} {fn}")
            bpm_val = float(bpm_tag) if bpm_tag else None
            if bpm_val is None and (is_loop or dur >= 2.0):
                try:
                    y_mono, sr_m = librosa.load(abs_p, sr=22050, mono=True, duration=12.0)
                    if len(y_mono) >= sr_m * 1.5:
                        tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr_m)
                        t_val = float(np.atleast_1d(tempo)[0])
                        if 55.0 <= t_val <= 210.0:
                            bpm_val = round(float(t_val), 1)
                except Exception:
                    pass

            role_scores = {
                role: compute_role_score(dsp, dur, role)
                for role in VALID_ROLES
            }
            ranked_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
            best_role = ranked_roles[0][0]
            compatible_roles = [r for r, sc in ranked_roles if sc >= 0.45][:3] or [best_role]

            comfort = compute_comfort_score(dsp, bool(entry.get("needs_fade")), best_role)
            fft_info = measure_fft_tuning(abs_p)
            gl_fine_tune = (
                int(dsp["recommended_fine_tune_ct"])
                if dsp.get("recommended_fine_tune_ct") is not None
                else int(fft_info.get("recommended_gl_fine_tune_ct") or 0)
            )
            playable = bool(dsp.get("playable_chromatically"))

            if cvec is not None and anchors is not None:
                _v16, timbre_card, top_tags, defect_tags = build_semantic_card_from_clap(
                    cvec,
                    anchors,
                    category=category,
                    sample_nature=dsp.get("sample_nature"),
                    detected_note=dsp.get("detected_note"),
                    playable_chromatically=playable,
                    comfort_score=comfort,
                )
            else:
                timbre_card, top_tags, defect_tags = build_timbre_card_from_dsp(
                    category=category,
                    sample_nature=dsp.get("sample_nature"),
                    detected_note=dsp.get("detected_note"),
                    centroid_hz=dsp.get("spectral_centroid_hz"),
                    attack_ms=dsp.get("attack_ms"),
                    duration_sec=dur,
                    r_harsh=dsp.get("r_harsh"),
                    r_harm=dsp.get("r_harm"),
                    roughness=dsp.get("roughness"),
                    low_mid_corr=dsp.get("low_mid_corr"),
                    playable_chromatically=playable,
                    comfort_score=comfort,
                )

            recipes = _suggest_prep_recipes(fn, abs_p, dsp, bool(entry.get("needs_fade")), best_role)
            entries.append({
                "filepath": abs_p,
                "rel_path": rel_p,
                "filename": fn,
                "source_origin": origin,
                "category": category,
                "is_loop": bool(is_loop),
                "bpm": bpm_val,
                "duration_sec": round(dur, 3),
                "compatible_roles": compatible_roles,
                "role_scores": role_scores,
                "comfort_score": comfort,
                "sample_nature": dsp.get("sample_nature"),
                "playable_chromatically": playable,
                "detected_note": dsp.get("detected_note"),
                "f0_hz": dsp.get("detected_f0_hz"),
                "midi_root": midi_root,
                "midi_root_octave_corrected": octave_corrected,
                "loop_nature": loop_nature,
                "harmonic_energy_ratio": harm_ratio,
                "hnr_db": loop_hnr_db,
                "chroma_cqt": chroma_cqt,
                "inharmonicity": inharmonicity,
                "wetness_score": float(wet),
                "tail_duration_ms": float(dsp.get("tail_duration_ms", 0.0) or 0.0),
                "recommended_gl_fine_tune_ct": gl_fine_tune,
                "centroid_hz": dsp.get("spectral_centroid_hz"),
                "attack_ms": dsp.get("attack_ms"),
                "r_harsh": dsp.get("r_harsh"),
                "low_mid_corr": dsp.get("low_mid_corr"),
                "phase_corr": dsp.get("phase_corr"),
                "bark_24": dsp.get("bark_24"),
                "hard_reject": bool(entry.get("hard_reject")),
                "reject_reasons": entry.get("reject_reasons") or [],
                "timbre_card": timbre_card,
                "top_tags": top_tags,
                "defect_tags": defect_tags,
                "suggested_prep_recipes": recipes,
            })
        except Exception as exc:
            errors.append({"filepath": abs_p, "error": str(exc)})

    has_user_originals = any(e["source_origin"] == "user_original" for e in entries)
    catalog = {
        "project_dir": abs_proj,
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_samples_locked": has_user_originals,
        "external_samples_forbidden": has_user_originals,
        "used_clap": bool(embedder is not None),
        "sample_count": len(entries),
        "error_count": len(errors),
        "skipped_files": skipped_files,
        "skipped_count": len(skipped_files),
        "samples": entries,
        "errors": errors,
    }

    catalog_path = os.path.join(abs_proj, "samples_catalog.json")
    if write_catalog:
        with open(catalog_path, "w", encoding="utf-8") as fh:
            json.dump(catalog, fh, indent=2, ensure_ascii=False)
    catalog["catalog_path"] = catalog_path if write_catalog else None
    return catalog


def main() -> None:
    ap = argparse.ArgumentParser(description="Index project-local Samples/ into samples_catalog.json (exclusivity safe).")
    ap.add_argument("project_dir", help="Path to project directory containing Samples/ (and optional samples/).")
    ap.add_argument("--with-clap", action="store_true", help="Enable CLAP anchor cards when < 40 files.")
    ap.add_argument("--no-write", action="store_true", help="Do not write samples_catalog.json.")
    args = ap.parse_args()

    res = index_project_samples(args.project_dir, with_clap=args.with_clap, write_catalog=not args.no_write)
    summary = {
        "project_dir": res["project_dir"],
        "catalog_path": res["catalog_path"],
        "user_samples_locked": res["user_samples_locked"],
        "sample_count": res["sample_count"],
        "samples": [
            {
                "filename": s["filename"],
                "source_origin": s["source_origin"],
                "compatible_roles": s["compatible_roles"],
                "playable_chromatically": s["playable_chromatically"],
                "recommended_gl_fine_tune_ct": s["recommended_gl_fine_tune_ct"],
                "timbre_card": s["timbre_card"],
                "prep_recipes_count": len(s["suggested_prep_recipes"]),
            }
            for s in res["samples"]
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
