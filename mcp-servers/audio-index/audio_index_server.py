"""Serveur MCP d'indexation et de recherche sémantique/musicale de samples (Essentia + CLAP)."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
from mcp.server.mcpserver import MCPServer

from indexer import (
    DEFAULT_ANCHORS_NPZ,
    DEFAULT_DB_PATH,
    DEFAULT_SAMPLES_DIR,
    NOTE_TO_PC,
    STUDIO_DIR,
    index_directory,
    init_db,
    load_semantic_anchors,
)

SCRIPTS_DIR = os.path.join(STUDIO_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

mcp = MCPServer(
    name="audio-index",
    instructions=(
        "Mémoire sonore locale sur ~/Music/AI-Studio/samples (pack/) + évaluation psychoacoustique "
        "(confort auditif, rugosité, dureté, masquage Bark, CLAP) et récupération de samples à la volée "
        "via Freesound API v2 et Lots of Sounds API."
    ),
)

# Traduction de descripteurs FR/EN courants vers catégories, profils et cibles acoustiques
FR_EN_SYNONYMS: dict[str, list[str]] = {
    "sec": ["dry", "punchy", "tight", "short"],
    "punchy": ["punchy", "hard", "knock", "tight", "kick"],
    "sombre": ["dark", "sad", "minor", "deep", "low", "sub", "ambient"],
    "triste": ["sad", "melancholic", "minor", "emotional", "lofi"],
    "ambiant": ["ambient", "pad", "atmosphere", "atmos", "texture", "sustained"],
    "chaud": ["warm", "analog", "lofi", "vintage", "tape"],
    "brillant": ["bright", "air", "crisp", "high", "sharp"],
    "lourd": ["heavy", "sub", "808", "deep", "bass"],
    "guitare": ["guitar", "gtr", "lofi"],
    "batterie": ["drum", "drums", "kick", "snare", "hat"],
    "caisse": ["snare", "clap"],
    "voix": ["vocal", "acapella", "voice", "vox", "chant"],
    "boucle": ["loop"],
    "nappe": ["pad", "ambient", "sustained"],
    "rapide": ["fast", "hat", "trap", "roll"],
}

BRIGHTNESS_ALIASES: dict[str, str] = {
    "sombre": "bass",
    "dark": "low_mid",
    "sub": "sub",
    "lourd": "sub",
    "chaud": "low_mid",
    "warm": "low_mid",
    "brillant": "high_mid",
    "bright": "high_mid",
    "air": "air",
    "crisp": "air",
}

_CLAP_TEXT_CACHE: dict[str, Any] = {}


def _get_clap_text_encoder() -> tuple[Any, Any] | None:
    """Charge paresseusement uniquement l'encodeur texte CLAP pour économiser la RAM."""
    if "encoder" in _CLAP_TEXT_CACHE:
        return _CLAP_TEXT_CACHE["encoder"]
    try:
        import torch
        from transformers import ClapProcessor, ClapTextModelWithProjection

        model_id = "laion/clap-htsat-unfused"
        processor = ClapProcessor.from_pretrained(model_id, local_files_only=True)
        text_model = ClapTextModelWithProjection.from_pretrained(model_id, local_files_only=True)
        text_model.eval()
        _CLAP_TEXT_CACHE["encoder"] = (processor, text_model, torch)
        return _CLAP_TEXT_CACHE["encoder"]
    except Exception:
        _CLAP_TEXT_CACHE["encoder"] = None
        return None


def _normalize_key(key_str: str | None) -> tuple[int, str] | None:
    if not key_str:
        return None
    m = re.match(r"^\s*([A-G](?:#|b)?)\s*(maj|major|min|minor|m)?\s*$", key_str, re.IGNORECASE)
    if not m:
        return None
    root_raw = m.group(1)[0].upper() + (m.group(1)[1:] if len(m.group(1)) > 1 else "")
    pc = NOTE_TO_PC.get(root_raw)
    if pc is None:
        return None
    qual = (m.group(2) or "major").lower()
    mode = "minor" if qual in ("m", "min", "minor") else "major"
    return pc, mode


def _key_compatibility_score(sample_key: str | None, target_key: str | None) -> float:
    """Score harmonique [0..1] : même tonalité = 1.0, relative maj/min = 0.9, quinte = 0.75."""
    t = _normalize_key(target_key)
    s = _normalize_key(sample_key)
    if t is None or s is None:
        return 0.0
    t_pc, t_mode = t
    s_pc, s_mode = s
    if t_pc == s_pc and t_mode == s_mode:
        return 1.0
    # Relative majeure/mineure
    if (t_mode == "minor" and s_mode == "major" and (t_pc + 3) % 12 == s_pc) or (
        t_mode == "major" and s_mode == "minor" and (t_pc + 9) % 12 == s_pc
    ):
        return 0.9
    # Quinte / quarte de même mode
    if t_mode == s_mode and ((s_pc - t_pc) % 12 in (5, 7)):
        return 0.75
    return 0.0


def _bpm_match_score(sample_bpm: float | None, target_bpm: float | None, tol: float) -> float:
    if target_bpm is None or sample_bpm is None:
        return 0.0
    diffs = [
        abs(sample_bpm - target_bpm),
        abs(sample_bpm * 2.0 - target_bpm),
        abs(sample_bpm / 2.0 - target_bpm),
    ]
    best = min(diffs)
    if best > tol:
        return -1.0
    return max(0.0, 1.0 - (best / max(tol, 1.0)))


def _expand_query_tokens(query: str) -> list[str]:
    raw_tokens = [t for t in re.split(r"[\s,;:/\-_]+", query.lower()) if len(t) >= 2]
    expanded = list(raw_tokens)
    for tok in raw_tokens:
        expanded.extend(FR_EN_SYNONYMS.get(tok, []))
    return list(dict.fromkeys(expanded))


ROLE_TO_CATEGORIES: dict[str, tuple[str, ...]] = {
    "kick": ("kick",),
    "bass_sub": ("808", "bass"),
    "snare_clap": ("snare", "clap"),
    "hihat_perc": ("hat", "open_hat", "perc"),
    "lead_melody": ("synth", "piano", "guitar", "vocal", "loop", "fx", "other"),
    "pad_atmosphere": ("synth", "piano", "guitar", "loop", "fx", "other"),
    "chord_keys": ("piano", "guitar", "synth", "loop", "other"),
}


def _row_to_dict(row: sqlite3.Row, score: float | None = None) -> dict[str, Any]:
    keys = row.keys()
    centroid = row["dsp_centroid_hz"] if "dsp_centroid_hz" in keys and row["dsp_centroid_hz"] is not None else row["spectral_centroid_hz"]
    out = {
        "path": row["path"],
        "filename": row["filename"],
        "pack_name": row["pack_name"],
        "category": row["category"],
        "is_loop": bool(row["is_loop"]),
        "duration_sec": row["duration_sec"],
        "bpm": row["bpm"],
        "musical_key": row["musical_key"],
        "spectral_centroid_hz": centroid,
        "brightness_band": row["brightness_band"],
        "decay_profile": row["decay_profile"],
        "peak_db": row["peak_db"],
        "gridline_plugin_hint": "gl.seltmann" if row["is_loop"] else "gl.sampler",
    }
    for col in (
        "sample_nature",
        "playable_chromatically",
        "f0_hz",
        "recommended_root_midi",
        "recommended_fine_tune_ct",
        "r_harsh",
        "hnr_db",
        "low_mid_corr",
        "timbre_card",
    ):
        if col in keys and row[col] is not None:
            out[col] = bool(row[col]) if col == "playable_chromatically" else row[col]
    if score is not None:
        out["score"] = round(score, 4)
    return out


def _get_conn(db_path: str = "") -> sqlite3.Connection:
    target = os.path.abspath(os.path.expanduser(db_path)) if db_path.strip() else DEFAULT_DB_PATH
    if target == DEFAULT_DB_PATH and not os.path.isfile(DEFAULT_DB_PATH) and os.path.isdir(DEFAULT_SAMPLES_DIR):
        index_directory(DEFAULT_SAMPLES_DIR, DEFAULT_DB_PATH, with_clap=False, with_dsp=True)
    return init_db(target)


@mcp.tool()
def search_samples(
    query: str = "",
    category: str = "",
    role: str = "",
    bpm: float | None = None,
    bpm_tolerance: float = 8.0,
    bpm_min: float | None = None,
    bpm_max: float | None = None,
    musical_key: str = "",
    is_loop: bool | None = None,
    brightness: str = "",
    playable_chromatically: bool | None = None,
    max_centroid_hz: float | None = None,
    max_r_harsh: float | None = None,
    min_low_mid_corr: float | None = None,
    use_clap: bool = True,
    limit: int = 10,
    db_path: str = "",
) -> str:
    """
    Recherche hybride instantanée sur SQLite unifié (`samples.db` + `sample_dsp_metrics`) :
    filtre SQL multi-critères (`role`, `playable_chromatically`, `max_centroid_hz`,
    `max_r_harsh`, `min_low_mid_corr`, `musical_key`, `bpm` / `bpm_min..bpm_max`) et
    scoring CLAP optionnel si les embeddings sont déjà en cache (zéro timeout CPU).
    """
    conn = _get_conn(db_path)
    has_dsp = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sample_dsp_metrics'").fetchone() is not None
    has_cards = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sample_semantic_cards'").fetchone() is not None
    clauses = ["s.ext != '.mid' AND s.ext != '.midi'"]
    params: list[Any] = []

    if category:
        clauses.append("s.category = ?")
        params.append(category.lower().strip())
    elif role and role.strip().lower() in ROLE_TO_CATEGORIES:
        cats = ROLE_TO_CATEGORIES[role.strip().lower()]
        placeholders = ",".join("?" for _ in cats)
        clauses.append(f"s.category IN ({placeholders})")
        params.extend(cats)

    if is_loop is not None:
        clauses.append("s.is_loop = ?")
        params.append(1 if is_loop else 0)
    if brightness:
        b_norm = BRIGHTNESS_ALIASES.get(brightness.lower().strip(), brightness.lower().strip())
        clauses.append("s.brightness_band = ?")
        params.append(b_norm)
    if bpm_min is not None:
        clauses.append("s.bpm IS NOT NULL AND s.bpm >= ?")
        params.append(float(bpm_min))
    if bpm_max is not None:
        clauses.append("s.bpm IS NOT NULL AND s.bpm <= ?")
        params.append(float(bpm_max))

    card_sel = ", c.timbre_card, c.clap_vec AS card_clap_vec" if has_cards else ""
    card_join = " LEFT JOIN sample_semantic_cards c ON c.sample_id = s.id" if has_cards else ""

    if has_dsp:
        clauses.append("COALESCE(d.hard_reject, 0) = 0")
        if playable_chromatically is True:
            clauses.append("d.playable_chromatically = 1")
        elif playable_chromatically is False:
            clauses.append("d.playable_chromatically = 0")
        if max_centroid_hz is not None:
            clauses.append("COALESCE(d.centroid_hz, s.spectral_centroid_hz, 0.0) <= ?")
            params.append(float(max_centroid_hz))
        if max_r_harsh is not None:
            clauses.append("COALESCE(d.r_harsh, 0.0) <= ?")
            params.append(float(max_r_harsh))
        if min_low_mid_corr is not None:
            clauses.append("COALESCE(d.low_mid_corr, 1.0) >= ?")
            params.append(float(min_low_mid_corr))
        select_cols = (
            "s.*, d.sample_nature, d.playable_chromatically, d.f0_hz, "
            "d.recommended_root_midi, d.recommended_fine_tune_ct, "
            f"d.centroid_hz AS dsp_centroid_hz, d.r_harsh, d.hnr_db, d.low_mid_corr{card_sel}"
        )
        join_sql = f"LEFT JOIN sample_dsp_metrics d ON d.sample_id = s.id{card_join}"
    else:
        if max_centroid_hz is not None:
            clauses.append("COALESCE(s.spectral_centroid_hz, 0.0) <= ?")
            params.append(float(max_centroid_hz))
        select_cols = f"s.*{card_sel}"
        join_sql = card_join

    where_sql = " AND ".join(clauses)
    rows = conn.execute(f"SELECT {select_cols} FROM samples s {join_sql} WHERE {where_sql}", params).fetchall()
    conn.close()

    tokens = _expand_query_tokens(query)
    has_clap_vecs = any(
        (r["clap_embedding"] is not None) or (has_cards and r["card_clap_vec"] is not None)
        for r in rows
    )
    warning: str | None = None
    clap_text_vec: np.ndarray | None = None
    if use_clap and not has_clap_vecs:
        warning = "clap_index_missing_fallback_dsp"
    elif use_clap and query.strip() and has_clap_vecs:
        # Zero-PyTorch runtime path: build query vector from pre-encoded semantic_anchors.npz
        anchors = load_semantic_anchors(DEFAULT_ANCHORS_NPZ)
        if anchors is not None and tokens:
            emb = anchors["embeddings"]
            matched_idx = [
                i
                for i, (aid, pr, lbl) in enumerate(zip(anchors["ids"], anchors["prompts"], anchors["fr_labels"]))
                if anchors["axes"][i] != "defect_flags"
                and any(tok in f"{aid} {pr} {lbl}".lower() for tok in tokens)
            ]
            if matched_idx:
                v = np.mean(emb[matched_idx], axis=0)
                nrm = float(np.linalg.norm(v))
                if nrm > 1e-12:
                    clap_text_vec = (v / nrm).astype(np.float32)

    scored: list[tuple[float, sqlite3.Row]] = []
    for r in rows:
        bpm_s = _bpm_match_score(r["bpm"], bpm, bpm_tolerance) if bpm is not None else 0.0
        if bpm_s < 0:
            continue

        key_s = _key_compatibility_score(r["musical_key"], musical_key) if musical_key else 0.0
        if musical_key and r["musical_key"] and key_s == 0.0:
            if r["category"] not in ("kick", "snare", "clap", "hat", "open_hat", "perc", "fx"):
                key_s = -0.35

        lex_s = 0.0
        if tokens:
            tags = f"{r['tags']} {r['timbre_card'] if (has_cards and r['timbre_card']) else ''}".lower()
            hits = sum(1 for tok in tokens if tok in tags)
            lex_s = hits / max(len(tokens), 1)
            if r["category"] in tokens:
                lex_s += 0.45

        clap_s = 0.0
        if clap_text_vec is not None:
            if has_cards and r["card_clap_vec"] is not None:
                vec = np.frombuffer(r["card_clap_vec"], dtype=np.float16).astype(np.float32)
            elif r["clap_embedding"] is not None:
                vec = np.frombuffer(r["clap_embedding"], dtype=np.float32)
            else:
                vec = None
            if vec is not None and len(vec) == len(clap_text_vec):
                clap_s = float(np.dot(clap_text_vec, vec))

        total_score = (lex_s * 1.4) + (clap_s * 2.2) + (bpm_s * 0.8) + (key_s * 0.9)
        if not tokens and bpm is None and not musical_key:
            total_score = 1.0
        scored.append((total_score, r))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = [_row_to_dict(r, score=s) for s, r in scored[: max(1, min(limit, 50))]]
    payload: dict[str, Any] = {"count": len(top), "results": top}
    if warning:
        payload["warning"] = warning
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def find_compatible_samples(
    bpm: float,
    musical_key: str,
    category: str = "",
    is_loop: bool | None = None,
    limit: int = 12,
) -> str:
    """Trouve les samples compatibles en tempo (y compris demi/double tempo) et tonalité (relative/quinte)."""
    return search_samples(
        query=category,
        category=category,
        bpm=bpm,
        bpm_tolerance=6.0,
        musical_key=musical_key,
        is_loop=is_loop,
        use_clap=False,
        limit=limit,
    )


@mcp.tool()
def get_sample_info(path: str) -> str:
    """Retourne les caractéristiques détaillées (métadonnées + DSP) d'un sample indexé."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    conn = _get_conn()
    row = conn.execute(
        "SELECT s.*, d.sample_nature, d.playable_chromatically, d.f0_hz, "
        "d.recommended_root_midi, d.recommended_fine_tune_ct, "
        "d.centroid_hz AS dsp_centroid_hz, d.r_harsh, d.hnr_db, d.low_mid_corr "
        "FROM samples s LEFT JOIN sample_dsp_metrics d ON d.sample_id = s.id WHERE s.path = ?",
        (abs_path,),
    ).fetchone()
    conn.close()
    if row is None:
        return json.dumps({"error": f"Sample non indexé : {abs_path}"}, ensure_ascii=False)
    return json.dumps(_row_to_dict(row), ensure_ascii=False, indent=2)


@mcp.tool()
def list_sample_packs() -> str:
    """Résume les packs de samples indexés, leurs catégories, métriques DSP et embeddings CLAP."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    clap_count = conn.execute("SELECT COUNT(*) FROM samples WHERE clap_embedding IS NOT NULL").fetchone()[0]
    dsp_count = conn.execute("SELECT COUNT(*) FROM sample_dsp_metrics").fetchone()[0]
    packs = [
        dict(r)
        for r in conn.execute(
            "SELECT pack_name, COUNT(*) AS files FROM samples GROUP BY pack_name ORDER BY files DESC"
        ).fetchall()
    ]
    cats = [
        dict(r)
        for r in conn.execute(
            "SELECT category, COUNT(*) AS files FROM samples GROUP BY category ORDER BY files DESC"
        ).fetchall()
    ]
    conn.close()
    return json.dumps(
        {
            "samples_dir": DEFAULT_SAMPLES_DIR,
            "db_path": DEFAULT_DB_PATH,
            "total_files": total,
            "dsp_metrics_count": dsp_count,
            "clap_embeddings": clap_count,
            "packs": packs,
            "categories": cats,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def reindex_samples(force: bool = False, with_clap: bool = False) -> str:
    """Ré-indexe les samples du dossier samples/ dans la base SQLite unifiée (DSP pur, zéro timeout CLAP)."""
    stats = index_directory(DEFAULT_SAMPLES_DIR, DEFAULT_DB_PATH, force=force, with_clap=False, with_dsp=True)
    if with_clap:
        stats["warning"] = "clap_indexing_disabled_in_mcp_use_cli: .venv/bin/python scripts/index_samples.py --enrich-clap --batch-size 64"
    return json.dumps(stats, ensure_ascii=False, indent=2)


@mcp.tool()
def search_synth_presets(
    query: str = "",
    category: str = "",
    brightness: str = "",
    use_clap: bool = True,
    limit: int = 10,
) -> str:
    """Recherche sémantique (CLAP + Essentia) parmi les presets d'usine des synthés natifs Gridline (gl.*)."""
    import numpy as np
    from preset_indexer import init_presets_table

    conn = _get_conn()
    init_presets_table(conn)

    clauses: list[str] = []
    params: list[Any] = []
    if category.strip():
        clauses.append("LOWER(category) = LOWER(?)")
        params.append(category.strip())
    if brightness.strip():
        clauses.append("LOWER(brightness_band) = LOWER(?)")
        params.append(brightness.strip())

    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(f"SELECT * FROM synth_presets {where_sql}", params).fetchall()
    conn.close()
    if not rows:
        return json.dumps(
            {
                "count": 0,
                "results": [],
                "hint": "Aucun preset indexé dans synth_presets : lance reindex_synth_presets(with_clap=True).",
            },
            ensure_ascii=False,
            indent=2,
        )

    tokens = _expand_query_tokens(query)
    q_vec = None
    if use_clap and query.strip() and any(r["clap_embedding"] is not None for r in rows):
        enc = _get_clap_text_encoder()
        if enc is not None:
            processor, text_model, torch = enc
            eng_query = " ".join(tokens) if tokens else query
            t_in = processor(text=[f"{query} ({eng_query})"], return_tensors="pt", padding=True)
            with torch.no_grad():
                out = text_model(**t_in)
                feats = getattr(out, "text_embeds", getattr(out, "pooler_output", out[0]))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            q_vec = feats.squeeze(0).cpu().numpy().astype(np.float32)

    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        score = 1.0
        tags = (row["tags"] or "").lower()
        if tokens:
            hits = sum(1 for t in tokens if t in tags)
            score += hits * 0.45
        if q_vec is not None and row["clap_embedding"] is not None:
            vec = np.frombuffer(row["clap_embedding"], dtype=np.float32)
            if len(vec) == len(q_vec):
                score += max(0.0, float(np.dot(q_vec, vec))) * 3.0
        if not tokens and q_vec is None:
            score += max(0.0, -float(row["inharmonicity"] or 0.0))
        if score > 0.05:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = [
        {
            "plugin": r["plugin_id"],
            "plugin_name": r["plugin_name"],
            "preset": r["preset_name"],
            "category": r["category"],
            "brightness_band": r["brightness_band"],
            "spectral_centroid_hz": r["spectral_centroid_hz"],
            "inharmonicity": r["inharmonicity"],
            "rms_db": r["rms_db"],
            "score": round(s, 3),
        }
        for s, r in scored[: max(1, min(limit, 40))]
    ]
    return json.dumps({"count": len(out), "results": out}, ensure_ascii=False, indent=2)


@mcp.tool()
def reindex_synth_presets(with_clap: bool = True, force: bool = False) -> str:
    """Rend et indexe les presets des synthés natifs Gridline (gl.*) dans samples.db."""
    from preset_indexer import index_gridline_presets

    return json.dumps(index_gridline_presets(db_path=DEFAULT_DB_PATH, with_clap=with_clap, force=force), ensure_ascii=False, indent=2)


@mcp.tool()
def evaluate_sample_quality(
    path_or_dir: str,
    role: str = "lead_melody",
    mix_path: str = "",
    target_key: str = "",
    top_k: int = 10,
    with_clap: bool = False,
    monophonic_only: bool = False,
) -> str:
    """
    Évalue un sample ou un dossier de samples en DSP déterministe (0 timeout CLAP synchrone).
    Si `with_clap=True`, utilise exclusivement les embeddings CLAP pré-calculés dans `samples.db`
    et bascule immédiatement sur le scoring DSP pur avec `warning="clap_index_missing_fallback_dsp"`
    si aucun embedding n'est en cache.
    """
    from audio_metrics import (
        VALID_ROLES,
        analyze_single_file,
        extract_dsp_metrics,
        load_and_normalize_audio,
        score_candidate,
    )
    from select_sample import DEFAULT_REFS_PATH

    role_id = role if role in VALID_ROLES else "lead_melody"
    abs_target = os.path.abspath(os.path.expanduser(path_or_dir))
    files: list[str] = []
    if os.path.isfile(abs_target):
        files = [abs_target]
    elif os.path.isdir(abs_target):
        for root, dirnames, fns in os.walk(abs_target, followlinks=True):
            dirnames[:] = sorted(d for d in dirnames if d not in ("__MACOSX", "sfz") and not d.startswith("."))
            for fn in sorted(fns):
                if os.path.splitext(fn)[1].lower() in {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}:
                    files.append(os.path.join(root, fn))
    else:
        return json.dumps({"error": f"Chemin introuvable : {abs_target}"}, ensure_ascii=False)

    mix_bark = None
    if mix_path and os.path.isfile(os.path.expanduser(mix_path)):
        ml = load_and_normalize_audio(mix_path)
        mix_bark = extract_dsp_metrics(ml["y_stereo"], ml["mono_norm"], sr=ml["sr"])["bark_24"]

    cached_clap: dict[str, np.ndarray] = {}
    if with_clap and os.path.isfile(DEFAULT_DB_PATH):
        try:
            conn = sqlite3.connect(DEFAULT_DB_PATH)
            for row in conn.execute("SELECT path, clap_embedding FROM samples WHERE clap_embedding IS NOT NULL").fetchall():
                cached_clap[row[0]] = np.frombuffer(row[1], dtype=np.float32)
            conn.close()
        except Exception:
            pass

    ref_matrix: np.ndarray | None = None
    if with_clap and cached_clap and os.path.isfile(DEFAULT_REFS_PATH):
        try:
            rz = np.load(DEFAULT_REFS_PATH)
            if role_id in rz:
                ref_matrix = rz[role_id]
        except Exception:
            ref_matrix = None

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    missing_clap = False

    for fp in files:
        try:
            entry, _ = analyze_single_file(fp, embedder=None, with_clap=False)
            if entry["hard_reject"]:
                rejected.append({"filepath": fp, "reasons": entry["reject_reasons"]})
                continue
            c_vec = cached_clap.get(os.path.abspath(fp))
            if with_clap and c_vec is None:
                missing_clap = True
            scored = score_candidate(
                entry,
                role_id=role_id,
                clap_vec=c_vec,
                embedder=None,
                ref_matrix=ref_matrix,
                mix_bark=mix_bark,
                target_key=target_key or None,
                monophonic_only=monophonic_only,
            )
            if scored.get("rejected"):
                rejected.append({"filepath": fp, "reasons": [scored.get("reject_reason")]})
                continue
            accepted.append(scored)
        except Exception as exc:
            rejected.append({"filepath": fp, "reasons": [str(exc)]})

    accepted.sort(key=lambda x: x["final_score"], reverse=True)
    for i, item in enumerate(accepted[: max(1, top_k)], start=1):
        item["rank"] = i

    out_payload: dict[str, Any] = {
        "role": role_id,
        "total_evaluated": len(files),
        "passed_count": len(accepted),
        "rejected_count": len(rejected),
        "recommendations": accepted[: max(1, top_k)],
        "rejected_samples": rejected[:10],
    }
    if with_clap and missing_clap:
        out_payload["warning"] = "clap_index_missing_fallback_dsp"
    return json.dumps(out_payload, ensure_ascii=False, indent=2)


@mcp.tool()
def select_best_samples(
    role: str,
    target_key: str = "",
    mix_path: str = "",
    query: str = "",
    db_path: str = "",
    top_k: int = 5,
    min_score: float = 0.55,
    auto_fetch: bool = False,
    monophonic_only: bool = False,
    playable_chromatically: bool | None = None,
    max_centroid_hz: float | None = None,
    max_r_harsh: float | None = None,
    min_low_mid_corr: float | None = None,
    bpm_min: float | None = None,
    bpm_max: float | None = None,
    use_clap: bool = True,
    valence: float | None = None,
    user_samples_locked: bool = False,
) -> str:
    """
    Sélectionne les meilleurs samples depuis la base unifiée `samples.db` (+ `sample_dsp_metrics`)
    sans jamais déclencher d'indexation CLAP CPU synchrone.
    Si `use_clap=True` et que `clap_embedding IS NULL`, bascule immédiatement sur le scoring DSP pur
    et retourne `warning="clap_index_missing_fallback_dsp"`.
    Si `user_samples_locked=True` (samples fournis par l'utilisateur), `auto_fetch` est ignoré :
    aucun sample externe n'est jamais téléchargé (Invariant #1).
    """
    from select_sample import DEFAULT_REFS_PATH, select_best_from_db

    target_db = db_path.strip() or DEFAULT_DB_PATH
    bpm_range = (bpm_min, bpm_max) if (bpm_min is not None and bpm_max is not None) else None
    res = select_best_from_db(
        db_path=target_db,
        role=role,
        refs_path=DEFAULT_REFS_PATH,
        mix_path=mix_path,
        target_key=target_key,
        top_k=top_k,
        min_score=min_score,
        query_filter=query,
        auto_fetch=auto_fetch,
        with_clap=use_clap,
        monophonic_only=monophonic_only,
        playable_chromatically=playable_chromatically,
        max_centroid_hz=max_centroid_hz,
        max_r_harsh=max_r_harsh,
        min_low_mid_corr=min_low_mid_corr,
        bpm_range=bpm_range,
        valence=valence,
        user_samples_locked=user_samples_locked,
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


@mcp.tool()
def fetch_samples_online(
    query: str,
    role: str = "lead_melody",
    provider: str = "auto",
    target_key: str = "",
    mix_path: str = "",
    duration_min: float = 0.1,
    duration_max: float = 5.0,
    cc0_only: bool = True,
    limit: int = 5,
    min_score: float = 0.55,
    out_dir: str = "",
    similar_to_fs_id: int | None = None,
    monophonic_only: bool = False,
) -> str:
    """
    Récupère des samples à la volée via Freesound API v2 et/ou Lots of Sounds API,
    les normalise en WAV 24-bit/44.1kHz, applique le filtre psychoacoustique DSP pur (zéro timeout CLAP),
    indexe les samples retenus dans `samples.db` et renvoie les candidats classés prêts pour gl.sampler.
    """
    from fetch_samples_api import DEFAULT_OUT_DIR, fetch_and_filter_samples

    dst = os.path.abspath(os.path.expanduser(out_dir)) if out_dir.strip() else DEFAULT_OUT_DIR
    res = fetch_and_filter_samples(
        query=query,
        role=role,
        provider=provider,
        duration_min=duration_min,
        duration_max=duration_max,
        cc0_only=cc0_only,
        limit=limit,
        out_dir=dst,
        mix_path=mix_path,
        target_key=target_key,
        min_score=min_score,
        with_clap=False,
        similar_to_fs_id=similar_to_fs_id,
        monophonic_only=monophonic_only,
    )
    if res.get("accepted") and dst.startswith(os.path.abspath(DEFAULT_SAMPLES_DIR)):
        try:
            index_directory(DEFAULT_SAMPLES_DIR, DEFAULT_DB_PATH, force=False, with_clap=False, with_dsp=True)
        except Exception:
            pass
    return json.dumps(res, ensure_ascii=False, indent=2)


@mcp.tool()
def analyze_sample_dsp(
    filepath: str,
    target_key: str = "",
    include_vocal_signature: bool = True,
) -> str:
    """
    Analyse DSP déterministe pure (CPU, sans modèle IA) d'un sample audio :
    - Nature : mono_oneshot vs melodic_sequence vs poly_chord_or_dyad vs percussion vs ambient_bed
    - Accordage exact (f0 Hz, midi_root, cents_offset, recommended_fine_tune_ct A4=440 Hz)
    - Stabilité F0, polyphonie, inharmonicité, HNR (Praat), corrélation stéréo 200-800 Hz
    - Signature vocale optionnelle (est_voix_probable, style_vocal, formants F1-F3, attaques/s, jitter/shimmer).
    """
    from audio_metrics import analyze_single_file, analyze_vocal_signature, compute_pitch_shift

    abs_p = os.path.abspath(os.path.expanduser(filepath))
    if not os.path.isfile(abs_p):
        return json.dumps({"error": f"Fichier introuvable : {abs_p}"}, ensure_ascii=False)

    entry, _ = analyze_single_file(abs_p, with_clap=False)
    dsp = entry["dsp"]
    out: dict[str, Any] = {
        "filepath": abs_p,
        "duration_sec": entry["duration_sec"],
        "sample_rate": entry["sample_rate"],
        "hard_reject": entry["hard_reject"],
        "reject_reasons": entry["reject_reasons"],
        "needs_fade": entry["needs_fade"],
        "pitch_and_purity": {
            "sample_nature": dsp.get("sample_nature"),
            "playable_chromatically": dsp.get("playable_chromatically"),
            "detected_note": dsp.get("detected_note"),
            "f0_hz": dsp.get("detected_f0_hz"),
            "midi_root": dsp.get("midi_root"),
            "cents_offset": dsp.get("cents_offset"),
            "recommended_fine_tune_ct": dsp.get("recommended_fine_tune_ct"),
            "f0_std_semitones": dsp.get("f0_std_semitones"),
            "onset_count": dsp.get("onset_count"),
            "polyphony_ratio": dsp.get("polyphony_ratio"),
            "hnr_db": dsp.get("hnr_db"),
            "inharmonic_warnings": dsp.get("inharmonic_warnings"),
        },
        "spectral_and_stereo": {
            "phase_corr": dsp.get("phase_corr"),
            "low_mid_corr": dsp.get("low_mid_corr"),
            "attack_ms": dsp.get("attack_ms"),
            "crest_factor_db": dsp.get("crest_factor_db"),
            "r_harsh": dsp.get("r_harsh"),
            "roughness": dsp.get("roughness"),
            "r_harm": dsp.get("r_harm"),
            "bands": dsp.get("bands"),
        },
    }
    if target_key.strip():
        out["key_alignment"] = compute_pitch_shift(dsp.get("detected_note"), target_key)
    if include_vocal_signature:
        out["vocal_signature"] = analyze_vocal_signature(abs_p)
    return json.dumps(out, ensure_ascii=False, indent=2)


@mcp.tool()
def propose_palettes(
    mix_profile: str = "trap_rap",
    valence: float = -0.3,
    tension: float = 0.4,
    energy: float = 0.7,
    depth: float = 0.5,
    target_key: str = "C minor",
    roles: list[str] | None = None,
    project_samples_dir: str | None = None,
    allow_gl_synths: bool = True,
    num_palettes: int = 3,
    db_path: str = "",
    bpm: float = 0.0,
    bpm_tolerance: float = 6.0,
) -> str:
    """
    Génère 3 palettes sonores complémentaires (4–6 rôles) en < 50 ms sans PyTorch (Bark24 + relais d'attaques + ancres).
    Si `project_samples_dir` est renseigné, n'utilise STRICTEMENT que les samples du projet (0 import externe).
    `bpm` pénalise/exclut les boucles dont le BPM détecté est incompatible (> 2x `bpm_tolerance`).
    Chaque palette fournit `blueprint_tracks` (avec `root_note`/`fine_tune_ct`) à recopier dans `song_blueprint.json`.
    """
    from palette_builder import propose_palettes as _propose_palettes

    res = _propose_palettes(
        mix_profile=mix_profile,
        emotion_vector={"valence": valence, "tension": tension, "energy": energy, "depth": depth},
        target_key=target_key,
        roles=roles,
        project_samples_dir=project_samples_dir,
        allow_gl_synths=allow_gl_synths,
        num_palettes=num_palettes,
        db_path=db_path.strip() or DEFAULT_DB_PATH,
        bpm=(float(bpm) if bpm else None),
        bpm_tolerance=float(bpm_tolerance),
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


@mcp.tool()
def index_project_samples(
    project_dir: str,
    with_clap: bool = False,
) -> str:
    """
    Indexe exclusivement `<project_dir>/Samples/` (et `samples/` dérivés) dans `<project_dir>/samples_catalog.json`
    avec `timbre_card`, `playable_chromatically`, `recommended_gl_fine_tune_ct` et recettes `sample_prep_lib.py`.
    """
    from index_project_samples import index_project_samples as _index_project_samples

    res = _index_project_samples(project_dir=project_dir, with_clap=with_clap, write_catalog=True)
    return json.dumps(res, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
