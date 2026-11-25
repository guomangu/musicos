#!/usr/bin/env python3
"""
Episodic User Taste Memory (`taste_memory.py`, Phase 7 P4).

Captures validated mixes (`quality_gate.passed == True`) into `memory/user_taste_profile.json`
(deduplicated by `project_id`, capped at 50 entries, weight 2.0 on `--approve-taste`) and
provides Euclidean `(V, T, E, D)` weighted prior queries for `target_fingerprint.py` in 0 tokens.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import tempfile
from typing import Any

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
DEFAULT_TASTE_PATH = os.path.join(ROOT_DIR, "memory", "user_taste_profile.json")
MAX_TASTE_ENTRIES = 50

_CLOSE_PROFILES: dict[str, tuple[str, ...]] = {
    "trap_rap": ("pop_rnb", "edm_club"),
    "pop_rnb": ("trap_rap", "lofi_chill"),
    "edm_club": ("trap_rap", "pop_rnb"),
    "rock_metal": ("pop_rnb",),
    "lofi_chill": ("jazz_acoustic", "pop_rnb"),
    "jazz_acoustic": ("lofi_chill", "classical_ambient"),
    "classical_ambient": ("jazz_acoustic",),
}

_DEFAULT_EMOTION: dict[str, float] = {
    "valence": 0.0,
    "tension": 0.35,
    "energy": 0.65,
    "depth": 0.50,
}


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _normalize_emotion(ev: dict[str, Any] | None) -> dict[str, float]:
    src = ev or {}
    return {
        "valence": round(_clamp(float(src.get("valence", _DEFAULT_EMOTION["valence"])), -1.0, 1.0), 4),
        "tension": round(_clamp(float(src.get("tension", _DEFAULT_EMOTION["tension"])), 0.0, 1.0), 4),
        "energy": round(_clamp(float(src.get("energy", _DEFAULT_EMOTION["energy"])), 0.0, 1.0), 4),
        "depth": round(_clamp(float(src.get("depth", _DEFAULT_EMOTION["depth"])), 0.0, 1.0), 4),
    }


def _load_json_safe(path: str) -> dict[str, Any]:
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                obj = json.load(fh)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _load_taste_store(path: str) -> dict[str, Any]:
    obj = _load_json_safe(path)
    entries = obj.get("entries")
    if not isinstance(entries, list):
        entries = []
    return {"version": 1, "entries": [e for e in entries if isinstance(e, dict)]}


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    abs_p = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(abs_p)
    os.makedirs(parent, exist_ok=True)
    fd, tmp_p = tempfile.mkstemp(prefix=".taste_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_p, abs_p)
    finally:
        if os.path.exists(tmp_p):
            try:
                os.unlink(tmp_p)
            except OSError:
                pass


def _derive_project_id(abs_proj_dir: str) -> str:
    try:
        rel = os.path.relpath(abs_proj_dir, ROOT_DIR)
        if rel and not rel.startswith(".."):
            return rel.replace(os.sep, "/")
    except Exception:
        pass
    return os.path.basename(abs_proj_dir) or abs_proj_dir


def record_validated_taste(
    project_dir: str,
    blueprint_path: str = "",
    fingerprint_path: str = "",
    approved_by_user: bool = False,
    *,
    taste_path: str | None = None,
    extra_metrics: dict[str, Any] | None = None,
    pq_score: float | None = None,
    musical_prefs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Record a validated song fingerprint (`quality_gate.passed == True`) into `user_taste_profile.json`.
    Deduplicates by `project_id`, keeps up to 50 most recent entries, and assigns `weight=2.0`
    when `approved_by_user=True` (otherwise `1.0`).

    Phase 8 Q4.2 quality guard: a mediocre `PQ < 7.8` is never auto-recorded; only an explicit
    user approval (`approved_by_user=True`, `--approve-taste`) can force the entry.
    """
    abs_proj = os.path.abspath(os.path.expanduser(project_dir))
    eff_bp_path = os.path.abspath(os.path.expanduser(blueprint_path)) if blueprint_path else os.path.join(abs_proj, "song_blueprint.json")
    eff_fp_path = os.path.abspath(os.path.expanduser(fingerprint_path)) if fingerprint_path else os.path.join(abs_proj, "renders", "master_fingerprint.json")
    eff_taste_path = os.path.abspath(os.path.expanduser(taste_path)) if taste_path else DEFAULT_TASTE_PATH

    bp = _load_json_safe(eff_bp_path)
    fp = _load_json_safe(eff_fp_path)
    tp = _load_json_safe(os.path.join(abs_proj, "song_target_profile.json"))
    extra = extra_metrics or {}

    if not fp and not extra:
        return {
            "recorded": False,
            "reason": f"Missing fingerprint: {eff_fp_path}",
            "taste_path": eff_taste_path,
        }

    eff_pq = pq_score
    if eff_pq is None:
        for src in (extra, fp, tp):
            if src.get("pq_score") is not None:
                eff_pq = float(src["pq_score"])
                break
            q_bert = (src.get("audiobox_aesthetics") or {}).get("PQ") if isinstance(src.get("audiobox_aesthetics"), dict) else None
            if q_bert is not None:
                eff_pq = float(q_bert)
                break
    if eff_pq is not None and float(eff_pq) < 7.8 and not approved_by_user:
        return {
            "recorded": False,
            "reason": f"pq_score={float(eff_pq):.2f} < 7.8 (use --approve-taste to force)",
            "taste_path": eff_taste_path,
            "pq_score": round(float(eff_pq), 2),
        }

    project_id = str(extra.get("project_id") or fp.get("project_id") or _derive_project_id(abs_proj))
    mix_profile = str(extra.get("mix_profile") or fp.get("mix_profile") or bp.get("mix_profile") or "trap_rap").strip()
    ev = _normalize_emotion(extra.get("emotion_vector") or fp.get("emotion_vector") or bp.get("emotion_vector"))
    bpm = round(float(extra.get("bpm") or fp.get("bpm") or bp.get("bpm") or 120.0), 2)

    bark_raw = extra.get("bark24_mean") or fp.get("bark24_mean") or fp.get("bark24") or tp.get("bark24_target") or []
    bark24_mean: list[float] = []
    if isinstance(bark_raw, list) and len(bark_raw) == 24:
        arr = [max(0.0, float(x)) for x in bark_raw]
        s = sum(arr)
        bark24_mean = [round(x / s, 6) for x in arr] if s > 1e-12 else [round(1.0 / 24.0, 6)] * 24

    stereo_w = extra.get("stereo_width_by_band") or fp.get("stereo_width_by_band") or tp.get("stereo_width_by_band")
    stereo_clean: dict[str, float] | None = None
    if isinstance(stereo_w, dict):
        stereo_clean = {
            band: round(float(stereo_w[band]), 4)
            for band in ("sub", "low_mid", "high_mid", "air")
            if band in stereo_w
        } or None

    slope = extra.get("pink_slope_db_per_oct", fp.get("pink_slope_db_per_oct"))
    crest = extra.get("crest_factor_db", fp.get("crest_factor_db"))
    onset_d = extra.get("onset_density_per_sec", fp.get("onset_density_per_sec", tp.get("target_onset_density_per_sec")))
    lufs = extra.get("integrated_lufs", fp.get("integrated_lufs", tp.get("target_lufs")))
    breath = extra.get("breathing_ratio", fp.get("breathing_ratio", tp.get("breathing_ratio")))

    ts_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_weight = 2.0 if approved_by_user else 1.0

    store = _load_taste_store(eff_taste_path)
    entries = list(store["entries"])
    prev_w = 0.0
    kept: list[dict[str, Any]] = []
    for ent in entries:
        if str(ent.get("project_id") or "") == project_id:
            prev_w = max(prev_w, float(ent.get("weight", 1.0)))
        else:
            kept.append(ent)

    eff_weight = max(new_weight, prev_w) if not approved_by_user and prev_w > 1.0 else new_weight
    entry: dict[str, Any] = {
        "project_id": project_id,
        "timestamp": ts_iso,
        "mix_profile": mix_profile,
        "emotion_vector": ev,
        "bpm": bpm,
        "weight": round(float(eff_weight), 2),
    }
    if slope is not None:
        entry["pink_slope_db_per_oct"] = round(float(slope), 3)
    if bark24_mean:
        entry["bark24_mean"] = bark24_mean
    if stereo_clean:
        entry["stereo_width_by_band"] = stereo_clean
    if crest is not None:
        entry["crest_factor_db"] = round(float(crest), 2)
    if onset_d is not None:
        entry["onset_density_per_sec"] = round(float(onset_d), 3)
    if lufs is not None:
        entry["integrated_lufs"] = round(float(lufs), 2)
    if breath is not None:
        entry["breathing_ratio"] = round(float(breath), 3)
    # v3.3 #4 — mémoire de goût MUSICALE (au-delà du mixage) : préférences de l'utilisateur
    _mp = musical_prefs or extra.get("musical_preferences")
    if isinstance(_mp, dict) and _mp:
        entry["musical_preferences"] = {
            k: _mp.get(k) for k in (
                "mode", "progression_id", "comping_styles", "timbre_archetypes",
                "groove_template_ids", "melodic_appeal_index", "swing", "tempo_bpm", "user_tags",
            ) if _mp.get(k) is not None
        }

    kept.append(entry)
    kept.sort(key=lambda e: (str(e.get("timestamp") or ""), float(e.get("weight", 1.0))), reverse=True)
    kept = kept[:MAX_TASTE_ENTRIES]

    _atomic_write_json(eff_taste_path, {"version": 1, "entries": kept})
    return {
        "recorded": True,
        "taste_path": eff_taste_path,
        "project_id": project_id,
        "weight": entry["weight"],
        "total_entries": len(kept),
        "entry": entry,
    }


def _weighted_avg_scalar(items: list[tuple[dict[str, Any], float]], key: str) -> float | None:
    num = 0.0
    den = 0.0
    for ent, w in items:
        if ent.get(key) is not None:
            num += float(ent[key]) * w
            den += w
    return (num / den) if den > 1e-12 else None


def query_taste_prior(
    mix_profile: str,
    emotion_vector: dict[str, Any] | None = None,
    top_k: int = 3,
    *,
    taste_path: str | None = None,
) -> dict[str, Any] | None:
    """
    Query top-k validated taste entries matching `mix_profile` (or close fallback profiles),
    weighted by Euclidean similarity over `(valence, tension, energy, depth)` and entry weight.
    Returns `None` cleanly if no matching history exists.
    """
    eff_taste_path = os.path.abspath(os.path.expanduser(taste_path)) if taste_path else DEFAULT_TASTE_PATH
    store = _load_taste_store(eff_taste_path)
    entries = store["entries"]
    if not entries:
        return None

    eff_prof = (mix_profile or "trap_rap").strip()
    exact = [(e, 1.0) for e in entries if str(e.get("mix_profile") or "").strip() == eff_prof]
    match_type = "exact_profile"
    if exact:
        pool = exact
    else:
        close_set = set(_CLOSE_PROFILES.get(eff_prof, ()))
        pool = [(e, 0.5) for e in entries if str(e.get("mix_profile") or "").strip() in close_set]
        match_type = "close_profile"
    if not pool:
        return None

    q_ev = _normalize_emotion(emotion_vector)
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for ent, prof_mult in pool:
        e_ev = _normalize_emotion(ent.get("emotion_vector"))
        dist = math.sqrt(
            (q_ev["valence"] - e_ev["valence"]) ** 2
            + (q_ev["tension"] - e_ev["tension"]) ** 2
            + (q_ev["energy"] - e_ev["energy"]) ** 2
            + (q_ev["depth"] - e_ev["depth"]) ** 2
        )
        sim = 1.0 / (1.0 + dist)
        base_w = max(0.05, float(ent.get("weight", 1.0)))
        eff_w = base_w * sim * prof_mult
        scored.append((eff_w, sim, ent))

    scored.sort(key=lambda t: (t[0], str(t[2].get("timestamp") or "")), reverse=True)
    top = scored[: max(1, int(top_k))]
    items = [(ent, eff_w) for eff_w, _sim, ent in top]
    tot_w = sum(eff_w for eff_w, _sim, _ent in top) or 1.0

    # Weighted 24-bin Bark average
    bark24_mean: list[float] | None = None
    bark_accum = [0.0] * 24
    bark_w = 0.0
    for ent, w in items:
        b = ent.get("bark24_mean") or ent.get("bark24")
        if isinstance(b, list) and len(b) == 24:
            for i in range(24):
                bark_accum[i] += max(0.0, float(b[i])) * w
            bark_w += w
    if bark_w > 1e-12:
        b_norm = [v / bark_w for v in bark_accum]
        b_sum = sum(b_norm)
        if b_sum > 1e-12:
            bark24_mean = [round(v / b_sum, 6) for v in b_norm]

    # Weighted stereo width by band
    stereo_out: dict[str, float] | None = None
    sw_accum: dict[str, float] = {}
    sw_w: dict[str, float] = {}
    for ent, w in items:
        sw = ent.get("stereo_width_by_band")
        if isinstance(sw, dict):
            for band in ("sub", "low_mid", "high_mid", "air"):
                if band in sw and sw[band] is not None:
                    sw_accum[band] = sw_accum.get(band, 0.0) + float(sw[band]) * w
                    sw_w[band] = sw_w.get(band, 0.0) + w
    if sw_w:
        stereo_out = {band: round(sw_accum[band] / sw_w[band], 4) for band in sw_accum if sw_w[band] > 1e-12}

    slope_avg = _weighted_avg_scalar(items, "pink_slope_db_per_oct")
    crest_avg = _weighted_avg_scalar(items, "crest_factor_db")
    onset_avg = _weighted_avg_scalar(items, "onset_density_per_sec")
    lufs_avg = _weighted_avg_scalar(items, "integrated_lufs")
    breath_avg = _weighted_avg_scalar(items, "breathing_ratio")
    bpm_avg = _weighted_avg_scalar(items, "bpm")
    mean_sim = sum(eff_w * sim for eff_w, sim, _ent in top) / tot_w

    prior: dict[str, Any] = {
        "source": "user_taste_memory",
        "mix_profile": eff_prof,
        "match_type": match_type,
        "n_entries_used": len(top),
        "project_ids": [str(ent.get("project_id") or "") for ent, _ in items],
        "mean_similarity": round(float(mean_sim), 4),
    }
    if slope_avg is not None:
        prior["pink_slope_db_per_oct"] = round(slope_avg, 3)
    if bark24_mean is not None:
        prior["bark24_mean"] = bark24_mean
    if stereo_out is not None:
        prior["stereo_width_by_band"] = stereo_out
    if crest_avg is not None:
        prior["crest_factor_db"] = round(crest_avg, 2)
    if onset_avg is not None:
        prior["onset_density_per_sec"] = round(onset_avg, 3)
    if lufs_avg is not None:
        prior["integrated_lufs"] = round(lufs_avg, 2)
    if breath_avg is not None:
        prior["breathing_ratio"] = round(breath_avg, 3)
    if bpm_avg is not None:
        prior["bpm"] = round(bpm_avg, 2)
    return prior


def query_musical_taste_prior(
    emotion_vector: dict[str, Any] | None,
    mix_profile: str,
    taste_path: str | None = None,
    max_bonus_pct: float = 15.0,
) -> dict[str, Any] | None:
    """v3.3 #4 — mémoire de goût musicale : agrège les `musical_preferences` des entrées
    validées les plus proches (profil + vecteur d'émotion). Le bonus reste borné à
    `max_bonus_pct` (<= 15 %) pour personnaliser sans enfermer dans la répétition."""
    prior = query_taste_prior(mix_profile, emotion_vector, top_k=3, taste_path=taste_path)
    if prior is None:
        return None
    eff_taste_path = os.path.abspath(os.path.expanduser(taste_path)) if taste_path else DEFAULT_TASTE_PATH
    store = _load_taste_store(eff_taste_path)
    entries = store.get("entries") or []
    prefs = [e.get("musical_preferences") for e in entries if isinstance(e.get("musical_preferences"), dict)]
    if not prefs:
        return None

    def _most_common(key: str) -> Any:
        from collections import Counter
        vals = [p.get(key) for p in prefs if p.get(key) is not None]
        if not vals:
            return None
        if isinstance(vals[0], list):
            flat = [x for v in vals for x in v]
            return Counter(flat).most_common(3) or None
        return Counter(vals).most_common(1)[0][0]

    out: dict[str, Any] = {
        "bonus_max_pct": max_bonus_pct,
        "n_entries_with_prefs": len(prefs),
        "mode": _most_common("mode"),
        "progression_id": _most_common("progression_id"),
        "comping_styles": _most_common("comping_styles"),
        "timbre_archetypes": _most_common("timbre_archetypes"),
        "groove_template_ids": _most_common("groove_template_ids"),
        "user_tags": _most_common("user_tags"),
    }
    mai_vals = [float(p["melodic_appeal_index"]) for p in prefs if p.get("melodic_appeal_index") is not None]
    if mai_vals:
        out["melodic_appeal_index_target"] = round(sum(mai_vals) / len(mai_vals), 4)
    swing_vals = [float(p["swing"]) for p in prefs if p.get("swing") is not None]
    if swing_vals:
        out["swing_preferred"] = round(sum(swing_vals) / len(swing_vals), 4)
    bpm_vals = [float(p["tempo_bpm"]) for p in prefs if p.get("tempo_bpm") is not None]
    if bpm_vals:
        out["tempo_bpm_preferred"] = round(sum(bpm_vals) / len(bpm_vals), 1)
    return out
