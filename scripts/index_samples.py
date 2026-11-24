#!/usr/bin/env python3
"""
Unified Audio Indexer CLI for AI-Studio (`samples.db` SQLite + optional CLAP enrichment).

Usage:
  # Fast deterministic DSP indexing into unified SQLite (0 GPU, < 50ms/file)
  .venv/bin/python scripts/index_samples.py --no-clap

  # Incremental CLAP enrichment in batches (CLI-only, never triggered synchronously by MCP)
  .venv/bin/python scripts/index_samples.py --enrich-clap --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

ROOT = os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_SERVER_DIR = os.path.join(ROOT, "mcp-servers", "audio-index")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SCRIPTS_DIR, INDEX_SERVER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from indexer import (
    DEFAULT_DB_PATH,
    DEFAULT_SAMPLES_DIR,
    enrich_clap_embeddings,
    index_directory,
    init_db,
    populate_semantic_cards,
)

DEFAULT_OUT_DB = DEFAULT_DB_PATH


def index_directory_psychoacoustic(
    input_dir: str = DEFAULT_SAMPLES_DIR,
    output_db: str = DEFAULT_DB_PATH,
    device: str = "cpu",
    with_clap: bool = False,
    force: bool = False,
    limit: int = 0,
    workers: int = 4,
) -> dict[str, Any]:
    """Unified SQLite indexing entrypoint (preserves backward-compatible signature)."""
    del device  # CLAP enrichment uses CPU in standalone incremental batches
    target_db = output_db
    if target_db.endswith(".json"):
        target_db = DEFAULT_DB_PATH
    return index_directory(
        samples_dir=os.path.abspath(os.path.expanduser(input_dir)),
        db_path=os.path.abspath(os.path.expanduser(target_db)),
        force=force,
        with_clap=with_clap,
        with_dsp=True,
        limit=limit,
        workers=workers,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified SQLite DSP & incremental CLAP sample indexer.")
    ap.add_argument("--input-dir", default=DEFAULT_SAMPLES_DIR, help="Root folder of samples to scan recursively.")
    ap.add_argument("--db-path", "--output-db", dest="db_path", default=DEFAULT_DB_PATH, help="Unified SQLite DB path.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device for CLAP.")
    ap.add_argument("--no-clap", action="store_true", help="Skip LAION-CLAP embeddings (fast DSP-only indexing + DSP timbre_cards).")
    ap.add_argument("--with-clap", action="store_true", help="Compute CLAP embeddings in batch offline and build CLAP timbre_cards.")
    ap.add_argument("--enrich-clap", action="store_true", help="Incrementally compute missing CLAP embeddings in SQLite.")
    ap.add_argument("--populate-cards", action="store_true", help="Only build/refresh sample_semantic_cards in pure NumPy (0 PyTorch).")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size for incremental CLAP enrichment.")
    ap.add_argument("--force", action="store_true", help="Re-analyze all files ignoring mtime/size cache.")
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of files to index.")
    ap.add_argument("--workers", type=int, default=4, help="Parallel CPU workers for DSP analysis.")
    args = ap.parse_args()

    target_db = DEFAULT_DB_PATH if args.db_path.endswith(".json") else os.path.abspath(os.path.expanduser(args.db_path))

    if args.populate_cards:
        conn = init_db(target_db)
        res = populate_semantic_cards(conn, force=args.force, limit=args.limit)
        conn.close()
    elif (args.enrich_clap or args.with_clap) and not args.no_clap:
        if not os.path.isfile(target_db) or args.force or args.with_clap:
            index_directory(
                samples_dir=os.path.abspath(os.path.expanduser(args.input_dir)),
                db_path=target_db,
                force=args.force,
                with_clap=False,
                with_dsp=True,
                limit=args.limit,
                workers=args.workers,
            )
        res = enrich_clap_embeddings(db_path=target_db, batch_size=max(1, args.batch_size), batch_limit=args.limit)
    else:
        res = index_directory(
            samples_dir=os.path.abspath(os.path.expanduser(args.input_dir)),
            db_path=target_db,
            force=args.force,
            with_clap=False,
            with_dsp=True,
            limit=args.limit,
            workers=args.workers,
        )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
