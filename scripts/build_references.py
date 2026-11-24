#!/usr/bin/env python3
"""
Build Golden Reference CLAP Embeddings Matrix per Track Role.

Usage:
  python scripts/build_references.py --ref-dir ./golden_references --output-refs ./mcp-servers/audio-index/golden_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_metrics import ClapEmbedder, VALID_ROLES, load_and_normalize_audio

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
DEFAULT_REF_DIR = os.path.join(os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "golden_references")
DEFAULT_OUT_REFS = os.path.join(os.environ.get("AI_STUDIO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp-servers", "audio-index", "golden_embeddings.npz")


def build_golden_references(ref_dir: str, output_refs: str, device: str = "cpu") -> dict[str, Any]:
    abs_ref = os.path.abspath(os.path.expanduser(ref_dir))
    abs_out = os.path.abspath(os.path.expanduser(output_refs))
    os.makedirs(abs_ref, exist_ok=True)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)

    # Ensure role subdirectories exist
    for role in VALID_ROLES:
        os.makedirs(os.path.join(abs_ref, role), exist_ok=True)

    embedder = ClapEmbedder(device=device)
    arrays: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    for role in sorted(os.listdir(abs_ref)):
        r_dir = os.path.join(abs_ref, role)
        if not os.path.isdir(r_dir):
            continue
        vecs: list[np.ndarray] = []
        for fn in sorted(os.listdir(r_dir)):
            if os.path.splitext(fn)[1].lower() not in AUDIO_EXTS:
                continue
            fp = os.path.join(r_dir, fn)
            try:
                loaded = load_and_normalize_audio(fp)
                vecs.append(embedder.embed_audio(loaded["mono_norm"], sr=loaded["sr"]))
            except Exception as exc:
                print(f"[WARN] Failed reference {fp}: {exc}", file=sys.stderr)
        if vecs:
            arrays[role] = np.stack(vecs, axis=0).astype(np.float32)
            counts[role] = len(vecs)

    np.savez_compressed(abs_out, **arrays)
    return {"ref_dir": abs_ref, "output_refs": abs_out, "role_counts": counts}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CLAP golden reference matrices per role.")
    ap.add_argument("--ref-dir", default=DEFAULT_REF_DIR)
    ap.add_argument("--output-refs", default=DEFAULT_OUT_REFS)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    res = build_golden_references(args.ref_dir, args.output_refs, device=args.device)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
