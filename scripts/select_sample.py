#!/usr/bin/env python3
"""
Unified Hybrid Sample Selector (SQLite `samples.db` + `sample_dsp_metrics` + Golden References).
Zero synchronous CLAP indexing in MCP: if CLAP embeddings are absent (`clap_embedding IS NULL`),
falls back immediately to pure deterministic DSP/psychoacoustic scoring with
`warning="clap_index_missing_fallback_dsp"`.

Usage:
  python scripts/select_sample.py \
    --role lead_melody \
    --target-key "E minor" \
    --top-k 5 \
    --no-clap \
    --format json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Any

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
INDEX_DIR = os.path.join(ROOT_DIR, "mcp-servers", "audio-index")
for _d in (SCRIPTS_DIR, INDEX_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from audio_metrics import (
    HARSH_THRESHOLD_BY_ROLE,
    VALID_ROLES,
    compute_masking_penalty,
    compute_pitch_shift,
    extract_dsp_metrics,
    load_and_normalize_audio,
    parse_key_tonic_pc,
    reference_aesthetic_score,
    score_candidate,
)
from fetch_samples_api import fetch_and_filter_samples
from indexer import (
    DEFAULT_ANCHORS_NPZ,
    DEFAULT_DB_PATH,
    build_timbre_card_from_dsp,
    load_semantic_anchors,
)

DEFAULT_OUT_DB = DEFAULT_DB_PATH
DEFAULT_REFS_PATH = os.path.join(INDEX_DIR, "golden_embeddings.npz")


def _build_anchor_query_vec(
    anchors: dict[str, Any] | None,
    query_tokens: list[str],
    valence: float | None,
    role: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build positive and defect unit vectors from pre-encoded `semantic_anchors.npz` (0 PyTorch)."""
    if anchors is None:
        return None, None
    emb = anchors["embeddings"]
    axes = anchors["axes"]
    ids = anchors["ids"]
    prompts = anchors["prompts"]
    labels = anchors["fr_labels"]

    pos_idx: list[int] = []
    def_idx: list[int] = []
    for i, ax in enumerate(axes):
        if ax == "defect_flags":
            def_idx.append(i)
            continue
        hay = f"{ids[i]} {prompts[i]} {labels[i]}".lower()
        if query_tokens and any(tok in hay for tok in query_tokens):
            pos_idx.append(i)
        elif valence is not None and ax == "mood":
            if valence <= -0.3 and any(k in ids[i] for k in ("sad", "dark", "aggressive", "tense", "hypnotic")):
                pos_idx.append(i)
            elif valence >= 0.3 and any(k in ids[i] for k in ("uplifting", "energetic", "dreamy", "chill", "epic")):
                pos_idx.append(i)
        elif not query_tokens and valence is None and ax == "texture" and ids[i] == "tex_clean_studio":
            pos_idx.append(i)

    # Add role-matching instrument anchor if no explicit token hit
    if not pos_idx:
        role_kw = {
            "kick": "kick",
            "bass_sub": "808",
            "snare_clap": "snare",
            "hihat_perc": "hihat",
            "lead_melody": "lead",
            "pad_atmosphere": "pad",
            "chord_keys": "piano",
        }.get(role, "")
        if role_kw:
            for i, ax in enumerate(axes):
                if ax == "instrument_family" and role_kw in prompts[i].lower():
                    pos_idx.append(i)

    pos_vec: np.ndarray | None = None
    if pos_idx:
        v = np.mean(emb[pos_idx], axis=0)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-12:
            pos_vec = (v / nrm).astype(np.float32)

    def_vec: np.ndarray | None = None
    if def_idx:
        v = np.mean(emb[def_idx], axis=0)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-12:
            def_vec = (v / nrm).astype(np.float32)
    return pos_vec, def_vec


def _fit(val: float, lo: float | None, hi: float | None, margin: float) -> float:
    if lo is not None and val < lo:
        r = 1.0 - (lo - val) / margin
        return 0.0 if r < 0.0 else r
    if hi is not None and val > hi:
        r = 1.0 - (val - hi) / margin
        return 0.0 if r < 0.0 else r
    return 1.0


def _fast_comfort_score(
    r_harsh: float,
    peak_harsh_db: float,
    roughness: float,
    needs_fade: bool,
    role: str,
) -> float:
    score = 1.0
    tau = HARSH_THRESHOLD_BY_ROLE.get(role, 0.30)
    if r_harsh > tau:
        d = ((r_harsh - tau) / max(0.15, 1.0 - tau)) * 1.1
        score -= 0.45 if d > 0.45 else d
    if peak_harsh_db > 10.0:
        d = (peak_harsh_db - 10.0) * 0.04
        score -= 0.35 if d > 0.35 else d
    rough_thresh = 0.38 if role == "hihat_perc" else 0.18
    if roughness > rough_thresh:
        d = (roughness - rough_thresh) * 0.9
        score -= 0.30 if d > 0.30 else d
    if needs_fade:
        score -= 0.08
    return 0.0 if score < 0.0 else (1.0 if score > 1.0 else round(score, 4))


def _fast_role_score(
    b_sub: float,
    b_bass: float,
    b_lowmid: float,
    b_mid: float,
    b_harsh: float,
    b_air: float,
    att: float,
    cf: float,
    rh: float,
    sfm: float,
    rho: float,
    pk_h: float,
    flux: float,
    dur: float,
    role_id: str,
) -> float:
    if role_id == "kick":
        gate = _fit(b_sub + b_bass, 0.65, None, 0.25) * _fit(dur, None, 1.0, 0.6)
        other_mean = (
            _fit(att, 0.0, 15.0, 20.0)
            + _fit(cf, 10.0, 22.0, 6.0)
            + _fit(rh, 0.10, 0.50, 0.30)
            + _fit(sfm, None, 0.15, 0.20)
            + _fit(rho, 0.85, None, 0.35)
        ) * 0.2
    elif role_id == "bass_sub":
        gate = _fit(b_sub + b_bass, 0.70, None, 0.25) * _fit(rh, 0.75, None, 0.30)
        other_mean = (
            _fit(att, 5.0, 80.0, 60.0)
            + _fit(cf, 3.0, 12.0, 6.0)
            + _fit(sfm, None, 0.05, 0.10)
            + _fit(rho, 0.80, None, 0.35)
            + _fit(b_harsh, None, 0.10, 0.15)
        ) * 0.2
    elif role_id == "snare_clap":
        gate = (
            _fit(b_lowmid + b_mid + b_harsh, 0.60, None, 0.25)
            * _fit(b_sub, None, 0.05, 0.10)
            * _fit(dur, None, 1.2, 0.8)
        )
        other_mean = (
            _fit(att, 0.0, 20.0, 25.0)
            + _fit(cf, 12.0, 24.0, 6.0)
            + _fit(rh, None, 0.45, 0.35)
            + _fit(sfm, 0.05, 0.45, 0.20)
        ) * 0.25
    elif role_id == "hihat_perc":
        gate = _fit(b_air + b_harsh, 0.75, None, 0.25) * _fit(b_sub + b_bass, None, 0.05, 0.10)
        other_mean = (
            _fit(att, 0.0, 15.0, 20.0)
            + _fit(cf, 10.0, 25.0, 6.0)
            + _fit(rh, None, 0.30, 0.35)
            + _fit(sfm, 0.15, None, 0.12)
        ) * 0.25
    elif role_id == "lead_melody":
        gate = _fit(b_lowmid + b_mid + b_harsh, 0.65, None, 0.25) * _fit(rh, 0.70, None, 0.30)
        other_mean = (
            _fit(att, 5.0, 100.0, 80.0)
            + _fit(cf, 4.0, 14.0, 6.0)
            + _fit(sfm, None, 0.08, 0.12)
            + _fit(b_harsh, None, 0.32, 0.18)
            + _fit(pk_h, None, 12.0, 6.0)
        ) * 0.2
    elif role_id == "pad_atmosphere":
        gate = (
            _fit(b_lowmid + b_mid, 0.50, None, 0.25)
            * _fit(rh, 0.80, None, 0.30)
            * _fit(b_sub, None, 0.10, 0.12)
        )
        other_mean = (
            _fit(att, 120.0, None, 100.0)
            + _fit(cf, 3.0, 10.0, 5.0)
            + _fit(sfm, None, 0.10, 0.12)
            + _fit(flux, None, 0.28, 0.25)
            + _fit(b_harsh, None, 0.25, 0.18)
        ) * 0.2
    elif role_id == "chord_keys":
        gate = _fit(b_bass + b_lowmid + b_mid, 0.70, None, 0.25) * _fit(rh, 0.75, None, 0.30)
        other_mean = (
            _fit(att, 5.0, 60.0, 60.0)
            + _fit(cf, 6.0, 16.0, 6.0)
            + _fit(sfm, None, 0.08, 0.12)
            + _fit(b_harsh, None, 0.28, 0.18)
        ) * 0.25
    else:
        return 0.5

    s = gate * (0.45 + 0.55 * other_mean)
    return 0.0 if s < 0.0 else (1.0 if s > 1.0 else round(s, 4))


def _select_best_from_sqlite(
    db_path: str,
    role: str,
    refs_path: str = DEFAULT_REFS_PATH,
    mix_path: str = "",
    target_key: str = "",
    top_k: int = 5,
    min_score: float = 0.0,
    query_filter: str = "",
    auto_fetch: bool = False,
    user_samples_locked: bool = False,
    fetch_query: str = "",
    with_clap: bool = True,
    monophonic_only: bool = False,
    playable_chromatically: bool | None = None,
    max_centroid_hz: float | None = None,
    max_r_harsh: float | None = None,
    min_low_mid_corr: float | None = None,
    bpm_range: tuple[float, float] | None = None,
    valence: float | None = None,
) -> dict[str, Any]:
    if user_samples_locked and auto_fetch:
        auto_fetch = False
    abs_db = os.path.abspath(os.path.expanduser(db_path))
    if not os.path.isfile(abs_db):
        raise FileNotFoundError(f"Unified SQLite index not found: {abs_db}")

    conn = sqlite3.connect(abs_db)
    has_dsp_table = (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sample_dsp_metrics'").fetchone() is not None
    )
    has_cards_table = (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sample_semantic_cards'").fetchone() is not None
    )
    has_any_clap = (
        conn.execute("SELECT EXISTS(SELECT 1 FROM samples WHERE clap_embedding IS NOT NULL)").fetchone()[0] == 1
    )
    if not has_any_clap and has_cards_table:
        has_any_clap = (
            conn.execute("SELECT EXISTS(SELECT 1 FROM sample_semantic_cards WHERE clap_vec IS NOT NULL)").fetchone()[0] == 1
        )

    warning: str | None = None
    eff_clap = bool(with_clap and has_any_clap)
    if with_clap and not has_any_clap:
        warning = "clap_index_missing_fallback_dsp"

    ref_matrix: np.ndarray | None = None
    if eff_clap and refs_path and os.path.isfile(refs_path):
        try:
            rz = np.load(refs_path)
            if role in rz:
                ref_matrix = rz[role]
        except Exception:
            ref_matrix = None

    anchors = load_semantic_anchors(DEFAULT_ANCHORS_NPZ) if eff_clap else None
    tokens_early = [t for t in query_filter.lower().split() if t] if query_filter else []
    anchor_pos_vec, anchor_def_vec = _build_anchor_query_vec(anchors, tokens_early, valence, role) if eff_clap else (None, None)

    mix_bark: list[float] | None = None
    mix_evaluated = False
    if mix_path and os.path.isfile(mix_path):
        ml = load_and_normalize_audio(mix_path)
        md = extract_dsp_metrics(ml["y_stereo"], ml["mono_norm"], sr=ml["sr"])
        mix_bark = md["bark_24"]
        mix_evaluated = True

    clauses = ["s.ext NOT IN ('.mid', '.midi')"]
    params: list[Any] = []
    if has_dsp_table:
        clauses.append("(d.hard_reject = 0 OR d.hard_reject IS NULL)")
        eff_playable = True if monophonic_only else playable_chromatically
        if eff_playable is True:
            clauses.append("d.playable_chromatically = 1")
        elif eff_playable is False:
            clauses.append("d.playable_chromatically = 0")
        if max_centroid_hz is not None:
            clauses.append("d.centroid_hz <= ?")
            params.append(float(max_centroid_hz))
        if max_r_harsh is not None:
            clauses.append("d.r_harsh <= ?")
            params.append(float(max_r_harsh))
        if min_low_mid_corr is not None:
            clauses.append("d.low_mid_corr >= ?")
            params.append(float(min_low_mid_corr))
    else:
        if max_centroid_hz is not None:
            clauses.append("s.spectral_centroid_hz <= ?")
            params.append(float(max_centroid_hz))

    if bpm_range is not None:
        lo_b, hi_b = float(bpm_range[0]), float(bpm_range[1])
        clauses.append("s.bpm IS NOT NULL AND s.bpm >= ? AND s.bpm <= ?")
        params.extend([lo_b, hi_b])

    tokens = [t for t in query_filter.lower().split() if t] if query_filter else []
    if tokens:
        sub_likes = ["LOWER(s.path || ' ' || s.tags) LIKE ?" for _ in tokens]
        clauses.append("(" + " OR ".join(sub_likes) + ")")
        params.extend([f"%{tok}%" for tok in tokens])

    where_sql = " AND ".join(clauses)
    total_candidates = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    role_order_expr = {
        "kick": "(COALESCE(d.band_sub,0) + COALESCE(d.band_bass,0)) - 0.6*COALESCE(d.r_harsh,0) - 0.3*MAX(0, s.duration_sec - 1.0)",
        "bass_sub": "(COALESCE(d.band_sub,0) + COALESCE(d.band_bass,0)) + 0.5*COALESCE(d.r_harm,0) - 0.6*COALESCE(d.r_harsh,0)",
        "snare_clap": "(COALESCE(d.band_lowmid,0) + COALESCE(d.band_mid,0) + COALESCE(d.band_harsh,0)) - 0.5*COALESCE(d.r_harsh,0)",
        "hihat_perc": "(COALESCE(d.band_air,0) + COALESCE(d.band_harsh,0)) - 0.5*COALESCE(d.r_harsh,0)",
        "lead_melody": "(COALESCE(d.r_harm,0) + 0.4*COALESCE(d.playable_chromatically,0) + COALESCE(d.band_mid,0)) - 0.7*COALESCE(d.r_harsh,0)",
        "pad_atmosphere": "(COALESCE(d.band_lowmid,0) + COALESCE(d.band_mid,0) + 0.5*COALESCE(d.r_harm,0)) - 0.7*COALESCE(d.r_harsh,0)",
        "chord_keys": "(COALESCE(d.band_bass,0) + COALESCE(d.band_lowmid,0) + COALESCE(d.band_mid,0) + 0.5*COALESCE(d.r_harm,0)) - 0.7*COALESCE(d.r_harsh,0)",
    }.get(role, "-COALESCE(d.r_harsh, 0)")
    pre_limit = max(top_k * 24, 150) if (has_dsp_table and mix_bark is None) else 0
    limit_sql = f" ORDER BY ({role_order_expr}) DESC LIMIT {int(pre_limit)}" if pre_limit > 0 else ""

    if has_dsp_table:
        sql = f"""
            SELECT
                s.id, s.path, s.duration_sec, s.spectral_centroid_hz, s.musical_key,
                d.sample_nature, d.playable_chromatically, d.f0_hz, d.recommended_root_midi,
                d.recommended_fine_tune_ct, d.detected_note, d.centroid_hz, d.r_harsh,
                d.peak_harsh_db, d.roughness, d.sfm, d.r_harm, d.spectral_flux, d.hnr_db,
                d.low_mid_corr, d.phase_corr, d.crest_factor_db, d.attack_ms,
                d.f0_std_semitones, d.onset_count, d.needs_fade,
                d.band_sub, d.band_bass, d.band_lowmid, d.band_mid, d.band_harsh, d.band_air
            FROM samples s
            LEFT JOIN sample_dsp_metrics d ON d.sample_id = s.id
            WHERE {where_sql}{limit_sql}
        """
    else:
        sql = f"""
            SELECT
                s.id, s.path, s.duration_sec, s.spectral_centroid_hz, s.musical_key,
                NULL, NULL, NULL, NULL,
                0, NULL, s.spectral_centroid_hz, 0.0,
                0.0, 0.0, 0.05, 0.5, 0.1, NULL,
                1.0, 1.0, 10.0, COALESCE(s.attack_ms, 10.0),
                0.0, 1, 0,
                NULL, NULL, NULL, NULL, NULL, NULL
            FROM samples s
            WHERE {where_sql}
        """

    rows = conn.execute(sql, params).fetchall()
    is_perc = role in ("kick", "snare_clap", "hihat_perc")
    target_pc = parse_key_tonic_pc(target_key) if target_key else None
    if is_perc:
        w1, w2, w3, w4 = 0.25, 0.55, 0.20, 0.05
    else:
        w1, w2, w3, w4 = 0.30, 0.45, 0.25, 0.20
    if not eff_clap:
        # Pure DSP weight distribution when CLAP is absent/disabled
        if is_perc:
            w1, w2, w3 = 0.30, 0.70, 0.0
        else:
            w1, w2, w3 = 0.38, 0.62, 0.0

    scored_items: list[tuple[float, tuple[Any, ...], float, float, bool]] = []
    for r in rows:
        (
            sid, path, dur, s_cent, s_key,
            nat, play_int, f0_hz, root_midi,
            ft_ct, det_note, d_cent, r_harsh,
            pk_h, rough, sfm, r_harm, flux, hnr_db,
            lm_corr, phase_corr, cf_db, att_ms,
            f0_std, onsets, needs_fade_int,
            b_sub, b_bass, b_lowmid, b_mid, b_harsh, b_air,
        ) = r

        ph = 1.0 if phase_corr is None else float(phase_corr)
        if role in ("kick", "bass_sub") and ph < 0.70:
            continue

        playable = bool(play_int) if play_int is not None else True
        if monophonic_only and not playable:
            continue

        dur_f = float(dur or 0.0)
        cent_f = float(d_cent if d_cent is not None else (s_cent or 1000.0))
        rh_val = 0.0 if r_harsh is None else float(r_harsh)
        pk_val = 0.0 if pk_h is None else float(pk_h)
        rg_val = 0.0 if rough is None else float(rough)
        sfm_val = 0.05 if sfm is None else float(sfm)
        r_harm_val = 0.5 if r_harm is None else float(r_harm)
        flux_val = 0.1 if flux is None else float(flux)
        cf_val = 10.0 if cf_db is None else float(cf_db)
        att_val = 10.0 if att_ms is None else float(att_ms)
        nf = bool(needs_fade_int)

        s_comfort = _fast_comfort_score(rh_val, pk_val, rg_val, nf, role)
        bright_warn = False
        if valence is not None and valence < 0.0 and role in ("lead_melody", "chord_keys", "pad_atmosphere"):
            if cent_f > 2500.0:
                pen = min(0.25, (cent_f - 2500.0) / 10000.0)
                s_comfort = round(max(0.0, s_comfort - pen), 4)
                bright_warn = True

        if b_sub is not None:
            s_role = _fast_role_score(
                float(b_sub), float(b_bass), float(b_lowmid), float(b_mid),
                float(b_harsh), float(b_air), att_val, cf_val, r_harm_val, sfm_val,
                ph, pk_val, flux_val, dur_f, role,
            )
        else:
            s_role = 0.5

        penalty = 0.28 if (role in ("lead_melody", "chord_keys", "bass_sub") and not playable) else 0.0
        key_bonus = 0.0
        if target_pc is not None and not is_perc and root_midi is not None:
            diff = (target_pc - (int(root_midi) % 12)) % 12
            sh = diff if diff <= 6 else 12 - diff
            if sh == 0:
                key_bonus = 0.05
            elif sh <= 2:
                key_bonus = 0.02
            else:
                key_bonus = -0.10  # root > 2 st outside the song scale: dissonance risk

        base_score = w1 * s_comfort + w2 * s_role + key_bonus - penalty
        if base_score < min_score and not eff_clap:
            continue
        scored_items.append((base_score, r, s_comfort, s_role, bright_warn))

    scored_items.sort(key=lambda x: x[0], reverse=True)
    # Re-rank top candidates with optional Bark masking / precomputed CLAP vectors & semantic cards (0 PyTorch!)
    pool = scored_items[: max(top_k * 5, 40)] if (mix_bark is not None or eff_clap) else scored_items[: max(1, top_k)]

    extra_by_id: dict[int, tuple[bytes | None, str | None, bytes | None, str | None, str | None, str | None]] = {}
    if pool:
        ids_str = ",".join(str(int(item[1][0])) for item in pool)
        sel_bark = "d.bark_24_json" if has_dsp_table else "NULL"
        join_dsp = "LEFT JOIN sample_dsp_metrics d ON d.sample_id = s.id" if has_dsp_table else ""
        if has_cards_table:
            sel_card = "c.clap_vec, c.timbre_card, c.top_tags_json, c.defect_tags_json"
            join_card = "LEFT JOIN sample_semantic_cards c ON c.sample_id = s.id"
        else:
            sel_card = "NULL, NULL, NULL, NULL"
            join_card = ""
        q_extra = (
            f"SELECT s.id, s.clap_embedding, {sel_bark}, {sel_card} "
            f"FROM samples s {join_dsp} {join_card} WHERE s.id IN ({ids_str})"
        )
        for eid, blob32, bark_js, blob16, t_card, t_tags, d_tags in conn.execute(q_extra).fetchall():
            extra_by_id[int(eid)] = (blob32, bark_js, blob16, t_card, t_tags, d_tags)

    conn.close()

    final_ranked: list[dict[str, Any]] = []
    missing_clap_in_pool = False
    for base_score, r, s_comfort, s_role, bright_warn in pool:
        (
            sid, path, dur, s_cent, s_key,
            nat, play_int, f0_hz, root_midi,
            ft_ct, det_note, d_cent, r_harsh,
            pk_h, rough, sfm, r_harm, flux, hnr_db,
            lm_corr, phase_corr, cf_db, att_ms,
            f0_std, onsets, needs_fade_int,
            *_bands,
        ) = r
        playable = bool(play_int) if play_int is not None else True
        blob32, bark_js, blob16, t_card, t_tags_js, d_tags_js = extra_by_id.get(
            int(sid), (None, None, None, None, None, None)
        )
        p_mask = 0.0
        if mix_bark is not None and bark_js:
            try:
                p_mask = compute_masking_penalty(json.loads(bark_js), mix_bark)
            except Exception:
                p_mask = 0.0

        cvec: np.ndarray | None = None
        if eff_clap:
            if blob16 is not None:
                cvec = np.frombuffer(blob16, dtype=np.float16).astype(np.float32)
            elif blob32 is not None:
                cvec = np.frombuffer(blob32, dtype=np.float32)
            else:
                missing_clap_in_pool = True

        s_aesth = 0.5
        s_anchor: float | None = None
        if eff_clap and cvec is not None:
            if ref_matrix is not None:
                s_ref = reference_aesthetic_score(cvec, ref_matrix)
                if s_ref is not None:
                    s_aesth = s_ref
            if anchor_pos_vec is not None and len(cvec) == len(anchor_pos_vec):
                pos_sim = float(np.dot(cvec, anchor_pos_vec))
                def_sim = float(np.dot(cvec, anchor_def_vec)) if anchor_def_vec is not None else 0.0
                # Map cosine difference cleanly to [0, 1]
                s_anchor = round(float(max(0.0, min(1.0, 0.5 + (pos_sim - def_sim) * 2.0))), 4)
                if ref_matrix is None:
                    s_aesth = s_anchor
                else:
                    s_aesth = round(0.6 * s_aesth + 0.4 * s_anchor, 4)

        if eff_clap:
            penalty = 0.28 if (role in ("lead_melody", "chord_keys", "bass_sub") and not playable) else 0.0
            s_final = max(0.0, min(1.0, w1 * s_comfort + w2 * s_role + w3 * s_aesth - w4 * p_mask - penalty))
        else:
            s_final = max(0.0, min(1.0, base_score - w4 * p_mask))
        s_final = round(float(s_final), 4)
        if s_final < min_score:
            continue

        top_tags: list[dict[str, Any]] = []
        defect_tags: list[dict[str, Any]] = []
        if t_card:
            try:
                top_tags = json.loads(t_tags_js) if t_tags_js else []
                defect_tags = json.loads(d_tags_js) if d_tags_js else []
            except Exception:
                pass
        else:
            t_card, top_tags, defect_tags = build_timbre_card_from_dsp(
                category=role,
                sample_nature=nat,
                detected_note=det_note,
                centroid_hz=d_cent if d_cent is not None else s_cent,
                attack_ms=att_ms,
                duration_sec=dur,
                r_harsh=r_harsh,
                r_harm=r_harm,
                roughness=rough,
                low_mid_corr=lm_corr,
                playable_chromatically=playable,
                comfort_score=s_comfort,
            )

        warnings_list: list[str] = []
        if needs_fade_int:
            warnings_list.append("needs_fade: apply 5ms linear fade-in/out on load")
        if bright_warn:
            warnings_list.append(
                f"bright_for_dark_valence(centroid={d_cent}Hz > 2500Hz): apply gl.sampler Filter=Low pass, Cutoff<=6000Hz"
            )
        if p_mask > 0.65:
            warnings_list.append(f"high_spectral_masking({p_mask:.2f}): consider EQ carve/HPF or candidate #2")
        if role in ("lead_melody", "chord_keys", "bass_sub") and not playable:
            warnings_list.append(
                f"DISSONANCE_RISK: sample is '{nat}' (onsets={onsets}, f0_std={f0_std}st)"
            )
        if not is_perc and lm_corr is not None and float(lm_corr) < 0.50:
            warnings_list.append(f"wide_low_mid_phase(corr={lm_corr}): apply M/S <650Hz stabilization")

        pitch_info = (
            compute_pitch_shift(det_note, target_key)
            if not is_perc
            else {"detected_note": det_note, "recommended_semitone_shift": 0}
        )
        pitch_info.update({
            "f0_hz": f0_hz,
            "midi_root": root_midi,
            "recommended_fine_tune_ct": int(ft_ct or 0),
            "sample_nature": nat or "unknown",
            "playable_chromatically": playable,
            "onset_count": onsets,
            "f0_std_semitones": f0_std,
            "hnr_db": hnr_db,
            "low_mid_corr": lm_corr,
        })
        if pitch_info.get("shift_warning"):
            warnings_list.append(pitch_info.pop("shift_warning"))
        else:
            pitch_info.pop("shift_warning", None)

        tier = "excellent" if s_final >= 0.70 else ("acceptable" if s_final >= 0.55 else "rejected")
        final_ranked.append({
            "filepath": path,
            "final_score": s_final,
            "tier": tier,
            "duration_sec": dur,
            "needs_fade": bool(needs_fade_int),
            "timbre_card": t_card,
            "top_tags": top_tags[:5],
            "defect_tags": defect_tags[:3],
            "scores": {
                "comfort": s_comfort,
                "role_fit": s_role,
                "aesthetic": s_aesth,
                "semantic_anchor": s_anchor,
                "masking_penalty": p_mask,
            },
            "pitch_info": pitch_info,
            "dsp_summary": {
                "spectral_centroid_hz": round(float(d_cent if d_cent is not None else (s_cent or 0.0)), 1),
                "r_harsh": round(float(0.0 if r_harsh is None else r_harsh), 4),
                "low_mid_corr": None if lm_corr is None else round(float(lm_corr), 3),
                "playable_chromatically": playable,
            },
            "warnings": warnings_list,
        })

    final_ranked.sort(key=lambda x: x["final_score"], reverse=True)
    top = final_ranked[: max(1, top_k)]
    for i, item in enumerate(top, start=1):
        item["rank"] = i

    if with_clap and missing_clap_in_pool and warning is None:
        warning = "clap_index_missing_fallback_dsp"

    online_fetch: dict[str, Any] | None = None
    if auto_fetch and not user_samples_locked and (not top or top[0]["final_score"] < 0.70):
        fq = fetch_query.strip() or f"{role.replace('_', ' ')} {target_key}".strip()
        online_fetch = fetch_and_filter_samples(
            query=fq,
            role=role,
            limit=top_k,
            mix_path=mix_path,
            target_key=target_key,
            min_score=max(0.55, min_score),
            with_clap=False,
            monophonic_only=monophonic_only,
        )

    out: dict[str, Any] = {
        "query": {
            "role": role,
            "target_key": target_key or None,
            "mix_evaluated": mix_evaluated,
            "total_candidates": total_candidates,
            "passed_hard_filter": len(rows),
            "backend": "sqlite",
        },
        "recommendations": top,
    }
    if warning:
        out["warning"] = warning
    if online_fetch is not None:
        out["online_fallback"] = online_fetch
    return out


def select_best_from_db(
    db_path: str = DEFAULT_DB_PATH,
    role: str = "lead_melody",
    refs_path: str = DEFAULT_REFS_PATH,
    mix_path: str = "",
    target_key: str = "",
    top_k: int = 5,
    min_score: float = 0.0,
    query_filter: str = "",
    auto_fetch: bool = False,
    fetch_query: str = "",
    with_clap: bool = True,
    monophonic_only: bool = False,
    playable_chromatically: bool | None = None,
    max_centroid_hz: float | None = None,
    max_r_harsh: float | None = None,
    min_low_mid_corr: float | None = None,
    bpm_range: tuple[float, float] | None = None,
    valence: float | None = None,
    user_samples_locked: bool = False,
) -> dict[str, Any]:
    """
    Select best samples using unified SQLite (`samples.db`) by default.
    Never triggers synchronous CLAP indexing; falls back immediately to pure DSP
    if CLAP embeddings are missing.
    """
    abs_db = os.path.abspath(os.path.expanduser(db_path)) if db_path else DEFAULT_DB_PATH
    if not abs_db.endswith(".json") or not os.path.isfile(abs_db):
        if abs_db.endswith(".json"):
            abs_db = DEFAULT_DB_PATH
        return _select_best_from_sqlite(
            db_path=abs_db,
            role=role,
            refs_path=refs_path,
            mix_path=mix_path,
            target_key=target_key,
            top_k=top_k,
            min_score=min_score,
            query_filter=query_filter,
            auto_fetch=auto_fetch,
            fetch_query=fetch_query,
            with_clap=with_clap,
            monophonic_only=monophonic_only,
            playable_chromatically=playable_chromatically,
            max_centroid_hz=max_centroid_hz,
            max_r_harsh=max_r_harsh,
            min_low_mid_corr=min_low_mid_corr,
            bpm_range=bpm_range,
            valence=valence,
            user_samples_locked=user_samples_locked,
        )

    # Legacy JSON fallback (never runs synchronous CLAP indexing)
    with open(abs_db, encoding="utf-8") as fh:
        db = json.load(fh)
    samples = db.get("samples", [])
    npz_path = db.get("npz_path")
    vecs: np.ndarray | None = None
    warning: str | None = None
    if with_clap and npz_path and os.path.isfile(npz_path):
        vecs = np.load(npz_path)["embeddings"]
    elif with_clap:
        warning = "clap_index_missing_fallback_dsp"

    ref_matrix: np.ndarray | None = None
    if with_clap and vecs is not None and refs_path and os.path.isfile(refs_path):
        try:
            rz = np.load(refs_path)
            if role in rz:
                ref_matrix = rz[role]
        except Exception:
            ref_matrix = None

    mix_bark: list[float] | None = None
    mix_evaluated = False
    if mix_path and os.path.isfile(mix_path):
        ml = load_and_normalize_audio(mix_path)
        md = extract_dsp_metrics(ml["y_stereo"], ml["mono_norm"], sr=ml["sr"])
        mix_bark = md["bark_24"]
        mix_evaluated = True

    tokens = [t for t in query_filter.lower().split() if t] if query_filter else []
    passed_hard = 0
    scored_list: list[dict[str, Any]] = []

    for entry in samples:
        if entry.get("hard_reject"):
            continue
        dsp = entry.get("dsp", {})
        if max_centroid_hz is not None and float(dsp.get("spectral_centroid_hz") or 0.0) > max_centroid_hz:
            continue
        if max_r_harsh is not None and float(dsp.get("r_harsh") or 0.0) > max_r_harsh:
            continue
        if min_low_mid_corr is not None and float(dsp.get("low_mid_corr") or 1.0) < min_low_mid_corr:
            continue
        if playable_chromatically is True and not dsp.get("playable_chromatically"):
            continue
        passed_hard += 1
        if tokens:
            fp_l = entry["filepath"].lower()
            if not any(tok in fp_l for tok in tokens):
                continue

        c_idx = entry.get("clap_index")
        c_vec = vecs[int(c_idx)] if (vecs is not None and c_idx is not None and 0 <= int(c_idx) < len(vecs)) else None
        res = score_candidate(
            entry=entry,
            role_id=role,
            clap_vec=c_vec,
            embedder=None,
            ref_matrix=ref_matrix,
            mix_bark=mix_bark,
            target_key=target_key or None,
            monophonic_only=monophonic_only,
            valence=valence,
        )
        if res.get("rejected") or res["final_score"] < min_score:
            continue
        scored_list.append(res)

    scored_list.sort(key=lambda x: x["final_score"], reverse=True)
    top = scored_list[: max(1, top_k)]
    for i, item in enumerate(top, start=1):
        item["rank"] = i

    out: dict[str, Any] = {
        "query": {
            "role": role,
            "target_key": target_key or None,
            "mix_evaluated": mix_evaluated,
            "total_candidates": len(samples),
            "passed_hard_filter": passed_hard,
            "backend": "json",
        },
        "recommendations": [
            {
                "rank": item["rank"],
                "filepath": item["filepath"],
                "final_score": item["final_score"],
                "tier": item["tier"],
                "duration_sec": item["duration_sec"],
                "needs_fade": item["needs_fade"],
                "scores": item["scores"],
                "pitch_info": item["pitch_info"],
                "warnings": item["warnings"],
            }
            for item in top
        ],
    }
    if warning:
        out["warning"] = warning
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Select best psychoacoustically verified samples for a track role.")
    ap.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to unified samples.db (or legacy JSON)")
    ap.add_argument("--refs", default=DEFAULT_REFS_PATH, help="Path to golden_embeddings.npz")
    ap.add_argument("--role", required=True, choices=list(VALID_ROLES))
    ap.add_argument("--mix-path", default="", help="Optional current_mix.wav for Bark masking check.")
    ap.add_argument("--target-key", default="", help="Optional target musical key (e.g. 'E minor').")
    ap.add_argument("--query", default="", help="Optional filename/path/tag keyword filter.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--auto-fetch", action="store_true", help="Fallback to online APIs if best local score < 0.70.")
    ap.add_argument("--fetch-query", default="", help="Custom query for --auto-fetch.")
    ap.add_argument("--mono-only", action="store_true", help="Require single-note monophonic samples.")
    ap.add_argument("--max-centroid-hz", type=float, default=None)
    ap.add_argument("--max-r-harsh", type=float, default=None)
    ap.add_argument("--min-low-mid-corr", type=float, default=None)
    ap.add_argument("--no-clap", action="store_true")
    ap.add_argument("--format", default="json", choices=["json", "text"])
    args = ap.parse_args()

    res = select_best_from_db(
        db_path=args.db,
        role=args.role,
        refs_path=args.refs,
        mix_path=args.mix_path,
        target_key=args.target_key,
        top_k=args.top_k,
        min_score=args.min_score,
        query_filter=args.query,
        auto_fetch=args.auto_fetch,
        fetch_query=args.fetch_query,
        with_clap=not args.no_clap,
        monophonic_only=args.mono_only,
        max_centroid_hz=args.max_centroid_hz,
        max_r_harsh=args.max_r_harsh,
        min_low_mid_corr=args.min_low_mid_corr,
    )

    if args.format == "json":
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print(f"Role: {args.role} | Candidates passed hard filter: {res['query']['passed_hard_filter']}/{res['query']['total_candidates']}")
        for r in res["recommendations"]:
            sc = r["scores"]
            print(
                f"  #{r['rank']} [{r['final_score']:.3f} {r['tier']}] "
                f"(comfort={sc['comfort']:.2f} role={sc['role_fit']:.2f} aesth={sc['aesthetic']:.2f} mask={sc['masking_penalty']:.2f}) "
                f"note={r['pitch_info'].get('detected_note')} -> {r['filepath']}"
            )
            for w in r["warnings"]:
                print(f"      WARN: {w}")


if __name__ == "__main__":
    main()
