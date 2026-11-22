#!/usr/bin/env python3
"""
Pre-indexer (Layer 1 + Layer 3) for Psychoacoustic Sample Selection.

Usage:
  python scripts/index_samples.py --input-dir ./samples --output-db ./mcp-servers/audio-index/sample_index.json --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_metrics import ClapEmbedder, analyze_single_file

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
DEFAULT_INPUT_DIR = "/home/gamo/Bureau/musicos/samples"
DEFAULT_OUT_DB = "/home/gamo/Bureau/musicos/mcp-servers/audio-index/sample_index.json"


def _companion_npz_path(db_json_path: str) -> str:
    stem, _ = os.path.splitext(db_json_path)
    return stem + ".npz"


def index_directory_psychoacoustic(
    input_dir: str,
    output_db: str,
    device: str = "cpu",
    with_clap: bool = True,
    force: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    abs_in = os.path.abspath(os.path.expanduser(input_dir))
    abs_db = os.path.abspath(os.path.expanduser(output_db))
    abs_npz = _companion_npz_path(abs_db)
    os.makedirs(os.path.dirname(abs_db), exist_ok=True)

    existing_by_path: dict[str, dict[str, Any]] = {}
    existing_vecs: np.ndarray | None = None
    if not force and os.path.isfile(abs_db):
        try:
            old_data = json.load(open(abs_db, encoding="utf-8"))
            for item in old_data.get("samples", []):
                existing_by_path[item["filepath"]] = item
            if os.path.isfile(abs_npz):
                existing_vecs = np.load(abs_npz)["embeddings"]
        except Exception:
            existing_by_path = {}
            existing_vecs = None

    audio_files: list[str] = []
    for root, dirnames, files in os.walk(abs_in, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if d not in ("__MACOSX", "sfz") and not d.startswith("."))
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                audio_files.append(os.path.abspath(os.path.join(root, fn)))
    if limit > 0:
        audio_files = audio_files[:limit]

    embedder = ClapEmbedder(device=device) if with_clap else None
    samples_out: list[dict[str, Any]] = []
    vecs_out: list[np.ndarray] = []

    stats = {"total_scanned": len(audio_files), "reused": 0, "indexed": 0, "hard_rejected": 0, "errors": 0}

    for fp in audio_files:
        try:
            st = os.stat(fp)
            prev = existing_by_path.get(fp)
            if (
                prev is not None
                and abs(float(prev.get("file_mtime", -1.0)) - st.st_mtime) < 1e-3
                and int(prev.get("file_size", -1)) == st.st_size
            ):
                prev_idx = prev.get("clap_index")
                has_vec = (
                    existing_vecs is not None
                    and prev_idx is not None
                    and 0 <= int(prev_idx) < len(existing_vecs)
                )
                if not with_clap or prev.get("hard_reject") or has_vec:
                    entry = dict(prev)
                    if has_vec:
                        entry["clap_index"] = len(vecs_out)
                        vecs_out.append(existing_vecs[int(prev_idx)])
                    else:
                        entry["clap_index"] = None
                    samples_out.append(entry)
                    stats["reused"] += 1
                    if entry.get("hard_reject"):
                        stats["hard_rejected"] += 1
                    continue

            entry, vec = analyze_single_file(fp, embedder=embedder, with_clap=with_clap)
            if vec is not None:
                entry["clap_index"] = len(vecs_out)
                vecs_out.append(vec)
            samples_out.append(entry)
            stats["indexed"] += 1
            if entry.get("hard_reject"):
                stats["hard_rejected"] += 1
        except Exception as exc:
            stats["errors"] += 1
            print(f"[WARN] Skipped {fp}: {exc}", file=sys.stderr)

    db_payload = {
        "schema_version": 1,
        "input_dir": abs_in,
        "npz_path": abs_npz if vecs_out else None,
        "count": len(samples_out),
        "clap_count": len(vecs_out),
        "samples": samples_out,
    }
    with open(abs_db, "w", encoding="utf-8") as fh:
        json.dump(db_payload, fh, indent=2, ensure_ascii=False)

    if vecs_out:
        np.savez_compressed(abs_npz, embeddings=np.stack(vecs_out, axis=0).astype(np.float32))
    elif os.path.isfile(abs_npz):
        np.savez_compressed(abs_npz, embeddings=np.zeros((0, 512), dtype=np.float32))

    stats["output_db"] = abs_db
    stats["output_npz"] = abs_npz
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Index samples with psychoacoustic DSP metrics and LAION-CLAP embeddings.")
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Root folder of samples to scan recursively.")
    ap.add_argument("--output-db", default=DEFAULT_OUT_DB, help="Output JSON index path.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device for CLAP.")
    ap.add_argument("--no-clap", action="store_true", help="Skip LAION-CLAP embeddings for ultra-fast DSP-only indexing.")
    ap.add_argument("--force", action="store_true", help="Re-analyze all files ignoring mtime/size cache.")
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of files to index.")
    args = ap.parse_args()

    res = index_directory_psychoacoustic(
        input_dir=args.input_dir,
        output_db=args.output_db,
        device=args.device,
        with_clap=not args.no_clap,
        force=args.force,
        limit=args.limit,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
