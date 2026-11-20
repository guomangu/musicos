"""Serveur MCP d'indexation et de recherche sémantique/musicale de samples (Essentia + CLAP)."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
from mcp.server.mcpserver import MCPServer

from indexer import DEFAULT_DB_PATH, DEFAULT_SAMPLES_DIR, NOTE_TO_PC, index_directory, init_db

mcp = MCPServer(
    name="audio-index",
    instructions=(
        "Mémoire sonore locale sur ~/Music/AI-Studio/samples (pack/) : "
        "recherche sémantique (CLAP + descripteurs acoustiques FR/EN), filtrage par BPM, "
        "tonalité harmonique (Essentia), brillance spectrale et compatibilité musicale."
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


def _row_to_dict(row: sqlite3.Row, score: float | None = None) -> dict[str, Any]:
    out = {
        "path": row["path"],
        "filename": row["filename"],
        "pack_name": row["pack_name"],
        "category": row["category"],
        "is_loop": bool(row["is_loop"]),
        "duration_sec": row["duration_sec"],
        "bpm": row["bpm"],
        "musical_key": row["musical_key"],
        "spectral_centroid_hz": row["spectral_centroid_hz"],
        "brightness_band": row["brightness_band"],
        "decay_profile": row["decay_profile"],
        "peak_db": row["peak_db"],
        "gridline_plugin_hint": "gl.seltmann" if row["is_loop"] else "gl.sampler",
    }
    if score is not None:
        out["score"] = round(score, 4)
    return out


def _get_conn() -> sqlite3.Connection:
    if not os.path.isfile(DEFAULT_DB_PATH):
        index_directory(DEFAULT_SAMPLES_DIR, DEFAULT_DB_PATH)
    return init_db(DEFAULT_DB_PATH)


@mcp.tool()
def search_samples(
    query: str = "",
    category: str = "",
    bpm: float | None = None,
    bpm_tolerance: float = 8.0,
    musical_key: str = "",
    is_loop: bool | None = None,
    brightness: str = "",
    use_clap: bool = True,
    limit: int = 10,
) -> str:
    """
    Recherche intelligente de samples par description sémantique (FR/EN + CLAP),
    catégorie (kick, snare, clap, hat, 808, guitar, piano, synth, vocal, fx, loop),
    BPM, tonalité et brillance spectrale.
    """
    conn = _get_conn()
    clauses = ["ext != '.mid' AND ext != '.midi'"]
    params: list[Any] = []

    if category:
        clauses.append("category = ?")
        params.append(category.lower().strip())
    if is_loop is not None:
        clauses.append("is_loop = ?")
        params.append(1 if is_loop else 0)
    if brightness:
        b_norm = BRIGHTNESS_ALIASES.get(brightness.lower().strip(), brightness.lower().strip())
        clauses.append("brightness_band = ?")
        params.append(b_norm)

    where_sql = " AND ".join(clauses)
    rows = conn.execute(f"SELECT * FROM samples WHERE {where_sql}", params).fetchall()
    conn.close()

    tokens = _expand_query_tokens(query)
    clap_text_vec: np.ndarray | None = None
    if use_clap and query.strip() and any(r["clap_embedding"] is not None for r in rows):
        enc = _get_clap_text_encoder()
        if enc is not None:
            processor, text_model, torch = enc
            eng_query = " ".join(tokens) if tokens else query
            t_in = processor(text=[f"{query} ({eng_query})"], return_tensors="pt", padding=True)
            with torch.no_grad():
                out = text_model(**t_in)
                t_feat = getattr(out, "text_embeds", getattr(out, "pooler_output", out[0]))
                t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
            clap_text_vec = t_feat.squeeze(0).cpu().numpy().astype(np.float32)

    scored: list[tuple[float, sqlite3.Row]] = []
    for r in rows:
        bpm_s = _bpm_match_score(r["bpm"], bpm, bpm_tolerance) if bpm is not None else 0.0
        if bpm_s < 0:
            continue

        key_s = _key_compatibility_score(r["musical_key"], musical_key) if musical_key else 0.0
        if musical_key and r["musical_key"] and key_s == 0.0:
            # Pénalise sans exclure complètement les samples non-tonals (drums)
            if r["category"] not in ("kick", "snare", "clap", "hat", "open_hat", "perc", "fx"):
                key_s = -0.35

        lex_s = 0.0
        if tokens:
            tags = r["tags"]
            hits = sum(1 for tok in tokens if tok in tags)
            lex_s = hits / max(len(tokens), 1)
            if r["category"] in tokens:
                lex_s += 0.45

        clap_s = 0.0
        if clap_text_vec is not None and r["clap_embedding"] is not None:
            vec = np.frombuffer(r["clap_embedding"], dtype=np.float32)
            if len(vec) == len(clap_text_vec):
                clap_s = float(np.dot(clap_text_vec, vec))

        total_score = (lex_s * 1.4) + (clap_s * 2.2) + (bpm_s * 0.8) + (key_s * 0.9)
        if not tokens and bpm is None and not musical_key:
            total_score = 1.0
        scored.append((total_score, r))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = [_row_to_dict(r, score=s) for s, r in scored[: max(1, min(limit, 50))]]
    return json.dumps({"count": len(top), "results": top}, ensure_ascii=False, indent=2)


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
    """Retourne les caractéristiques détaillées d'un sample indexé."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    conn = _get_conn()
    row = conn.execute("SELECT * FROM samples WHERE path = ?", (abs_path,)).fetchone()
    conn.close()
    if row is None:
        return json.dumps({"error": f"Sample non indexé : {abs_path}"}, ensure_ascii=False)
    return json.dumps(_row_to_dict(row), ensure_ascii=False, indent=2)


@mcp.tool()
def list_sample_packs() -> str:
    """Résume les packs de samples indexés, leurs catégories et le nombre d'embeddings CLAP."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    clap_count = conn.execute("SELECT COUNT(*) FROM samples WHERE clap_embedding IS NOT NULL").fetchone()[0]
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
            "clap_embeddings": clap_count,
            "packs": packs,
            "categories": cats,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def reindex_samples(force: bool = False, with_clap: bool = False) -> str:
    """Ré-indexe les samples du dossier samples/ dans la base SQLite locale."""
    stats = index_directory(DEFAULT_SAMPLES_DIR, DEFAULT_DB_PATH, force=force, with_clap=with_clap)
    return json.dumps(stats, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
