#!/usr/bin/env python3
"""
Psychoacoustic Sample Selector (Layers 1-4 + Golden References + Optional Online API Fallback).

Usage:
  python scripts/select_sample.py \
    --db ./mcp-servers/audio-index/sample_index.json \
    --refs ./mcp-servers/audio-index/golden_embeddings.npz \
    --role lead_melody \
    --mix-path ./current_mix.wav \
    --target-key "E minor" \
    --top-k 5 \
    --format json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_metrics import (
    ClapEmbedder,
    VALID_ROLES,
    extract_dsp_metrics,
    load_and_normalize_audio,
    score_candidate,
)
from fetch_samples_api import fetch_and_filter_samples
from index_samples import DEFAULT_OUT_DB, index_directory_psychoacoustic

DEFAULT_REFS_PATH = "/home/gamo/Bureau/musicos/mcp-servers/audio-index/golden_embeddings.npz"


def select_best_from_db(
    db_path: str,
    role: str,
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
) -> dict[str, Any]:
    abs_db = os.path.abspath(os.path.expanduser(db_path))
    if not os.path.isfile(abs_db):
        raise FileNotFoundError(f"Psychoacoustic index not found: {abs_db}. Run scripts/index_samples.py first.")

    db = json.load(open(abs_db, encoding="utf-8"))
    samples = db.get("samples", [])
    npz_path = db.get("npz_path")
    vecs: np.ndarray | None = None
    if with_clap and npz_path and os.path.isfile(npz_path):
        vecs = np.load(npz_path)["embeddings"]

    ref_matrix: np.ndarray | None = None
    if with_clap and refs_path and os.path.isfile(refs_path):
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

    embedder = ClapEmbedder(device="cpu") if (with_clap and vecs is not None and len(vecs) > 0) else None

    tokens = [t for t in query_filter.lower().split() if t] if query_filter else []
    passed_hard = 0
    scored_list: list[dict[str, Any]] = []

    for entry in samples:
        if entry.get("hard_reject"):
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
            embedder=embedder,
            ref_matrix=ref_matrix,
            mix_bark=mix_bark,
            target_key=target_key or None,
            monophonic_only=monophonic_only,
        )
        if res.get("rejected"):
            continue
        if res["final_score"] < min_score:
            continue
        scored_list.append(res)

    scored_list.sort(key=lambda x: x["final_score"], reverse=True)
    top = scored_list[: max(1, top_k)]
    for i, item in enumerate(top, start=1):
        item["rank"] = i

    online_fetch: dict[str, Any] | None = None
    if auto_fetch and (not top or top[0]["final_score"] < 0.70):
        fq = fetch_query.strip() or f"{role.replace('_', ' ')} {target_key}".strip()
        online_fetch = fetch_and_filter_samples(
            query=fq,
            role=role,
            limit=top_k,
            mix_path=mix_path,
            target_key=target_key,
            min_score=max(0.55, min_score),
            with_clap=with_clap,
            monophonic_only=monophonic_only,
        )

    out: dict[str, Any] = {
        "query": {
            "role": role,
            "target_key": target_key or None,
            "mix_evaluated": mix_evaluated,
            "total_candidates": len(samples),
            "passed_hard_filter": passed_hard,
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
    if online_fetch is not None:
        out["online_fallback"] = online_fetch
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Select best psychoacoustically verified samples for a track role.")
    ap.add_argument("--db", default=DEFAULT_OUT_DB, help="Path to sample_index.json")
    ap.add_argument("--refs", default=DEFAULT_REFS_PATH, help="Path to golden_embeddings.npz")
    ap.add_argument("--role", required=True, choices=list(VALID_ROLES))
    ap.add_argument("--mix-path", default="", help="Optional current_mix.wav for Bark masking check.")
    ap.add_argument("--target-key", default="", help="Optional target musical key (e.g. 'E minor').")
    ap.add_argument("--query", default="", help="Optional filename/path keyword filter.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--auto-fetch", action="store_true", help="Fallback to online APIs if best local score < 0.70.")
    ap.add_argument("--fetch-query", default="", help="Custom query for --auto-fetch.")
    ap.add_argument("--mono-only", action="store_true", help="Require single-note monophonic samples.")
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
                f"note={r['pitch_info'].get('detected_note')} shift={r['pitch_info'].get('recommended_semitone_shift'):+d}st "
                f"-> {r['filepath']}"
            )
            for w in r["warnings"]:
                print(f"      WARN: {w}")


if __name__ == "__main__":
    main()
